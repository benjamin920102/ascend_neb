/* Thin ABI around SilasEigen/neb Apache-2.0 reference (not bundled).
 * Keep all consensus math in the upstream C reference. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include "nebhash2.h"
#include "blake3_min.h"

/* Exported in pow/ref/nebhash2.c; not declared by the public API header. */
extern int neb2_sample_grad(const neb2_model *, const uint8_t *, int, int32_t *, int8_t *);

#define KEP1_CONV_WEIGHTS 1360
#define KEP1_LINEAR_FACTORS (32 + 768 + 2 + 32)

void an_hash(const uint8_t *src, size_t len, uint8_t dst[32]) { blake3_hash(src, len, dst); }

int an_set_model(neb2_model *m, const uint8_t *packed, size_t size) {
    if (!m || !packed || size != 13000) return -1;
    memset(m, 0, sizeof(*m));
    m->arch = NEB2_KEP1;
    for (int i = 0; i < 26000; ++i) {
        unsigned int v = (i & 1) ? ((packed[i/2] >> 4) & 15) : (packed[i/2] & 15);
        m->w[i] = (int8_t)(v >= 8 ? (int)v - 16 : (int)v);
    }
    return 0;
}

void an_weights_hash(const neb2_model *m, uint8_t result[32]) {
    uint8_t residual[32];
    neb2_model_hashes(m, result, residual);
}

/* Build a compressed per-sample gradient table. For KEP1, each one-sample
 * convolution gradient fits signed int16; linear factors fit signed int8.
 * This is the same factorization used by upstream nebhash2_table_cuda.cu. */
int an_build_table(const neb2_model *model, const uint8_t *values,
                   const uint8_t *labels, uint32_t count,
                   int16_t *conv_out, int8_t *factor_out, int threads) {
    if (!model || model->arch != NEB2_KEP1 || !values || !labels || !conv_out || !factor_out || !count) return -1;
    int failed = 0;
    if (threads <= 0) threads = omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        int32_t *g = (int32_t*)malloc(26000 * sizeof(int32_t));
        int8_t fac[KEP1_LINEAR_FACTORS];
        if (!g) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp for schedule(dynamic, 16)
        for (uint32_t i = 0; i < count; ++i) {
            if (!g) continue;
            int r = neb2_sample_grad(model, values + (size_t)i * 784, labels[i], g, fac);
            if (r) {
                #pragma omp atomic write
                failed = 1;
            }
            if (r) continue;
            int16_t *row = conv_out + (size_t)i * KEP1_CONV_WEIGHTS;
            for (int k = 0; k < KEP1_CONV_WEIGHTS; ++k) {
                if (g[k] < INT16_MIN || g[k] > INT16_MAX) {
                    #pragma omp atomic write
                    failed = 1;
                    break;
                }
                row[k] = (int16_t)g[k];
            }
            memcpy(factor_out + (size_t)i * KEP1_LINEAR_FACTORS, fac, KEP1_LINEAR_FACTORS);
        }
        free(g);
    }
    return failed ? -2 : 0;
}

void an_batch_keys_indices(const uint8_t header[80], uint32_t start, uint32_t attempts,
                           uint32_t n_samples, uint8_t *keys, uint32_t *indices) {
    uint8_t h[80];
    memcpy(h, header, 80);
    for (uint32_t i = 0; i < attempts; ++i) {
        const uint32_t nonce = start + i;
        h[76] = (uint8_t)nonce; h[77] = (uint8_t)(nonce >> 8);
        h[78] = (uint8_t)(nonce >> 16); h[79] = (uint8_t)(nonce >> 24);
        blake3_hash(h, 80, keys + (size_t)i * 32);
        neb_share_indices(keys + (size_t)i * 32, n_samples, indices + (size_t)i * 16);
    }
}

void an_batch_tickets(const uint8_t *digests, const uint8_t *keys,
                      uint32_t count, uint8_t *tickets) {
    uint8_t message[64];
    for (uint32_t i = 0; i < count; ++i) {
        memcpy(message, digests + 32*(size_t)i, 32);
        memcpy(message + 32, keys + 32*(size_t)i, 32);
        blake3_hash(message, 64, tickets + 32*(size_t)i);
    }
}

typedef struct { const uint8_t *values; const uint8_t *labels; uint32_t n; } an_dataset_view;
static int get_sample(void *ctx, uint32_t idx, uint8_t *x) {
    const an_dataset_view *v = (const an_dataset_view*)ctx;
    if (idx >= v->n) return -1;
    memcpy(x, v->values + (size_t)idx*784, 784);
    return (int)v->labels[idx];
}
int an_reference_eval(const neb2_model *m, const uint8_t *values,
                      const uint8_t *labels, uint32_t n,
                      const uint8_t key[32], uint8_t digest[32], uint8_t ticket[32]) {
    if (!m || m->arch != NEB2_KEP1) return -1;
    int32_t *g = (int32_t*)malloc(26000 * sizeof(int32_t));
    if (!g) return -1;
    an_dataset_view view = { values, labels, n };
    neb2_dataset ds = { n, get_sample, &view };
    int rc = neb2_share_eval(m, &ds, key, g, digest, ticket, NULL, NULL);
    free(g);
    return rc;
}
