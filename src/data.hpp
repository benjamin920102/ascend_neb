#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "nebhash2.h"

extern "C" {
void an_hash(const uint8_t*,size_t,uint8_t[32]);
int an_set_model(neb2_model*,const uint8_t*,size_t);
void an_weights_hash(const neb2_model*,uint8_t[32]);
int an_build_table(const neb2_model*,const uint8_t*,const uint8_t*,uint32_t,int16_t*,int8_t*,int);
void an_batch_keys_indices(const uint8_t[80],uint32_t,uint32_t,uint32_t,uint8_t*,uint32_t*);
void an_batch_tickets(const uint8_t*,const uint8_t*,uint32_t,uint8_t*);
int an_reference_eval(const neb2_model*,const uint8_t*,const uint8_t*,uint32_t,const uint8_t[32],uint8_t[32],uint8_t[32]);
}

namespace neb {
struct DataSet {
  uint32_t arch=0,n=0;
  std::vector<uint8_t> values, labels;
};
struct GradientTable {
  uint32_t n=0;
  std::vector<int16_t> conv;
  std::vector<int8_t> factors;
};
std::vector<uint8_t> read_file(const std::string& path);
std::vector<DataSet> parse_datasets(const std::vector<uint8_t>& blob,bool allow_other);
GradientTable build_table(const neb2_model& model,const DataSet& ds,int threads);
std::string hex(const uint8_t* ptr,size_t n);
std::vector<uint8_t> unhex(const std::string& src);
std::array<uint8_t,32> hash(const std::vector<uint8_t>& value);
} // namespace neb
