#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace neb {
constexpr uint32_t kSamplesPerNonce = 16;
constexpr uint32_t kConvStride = 1360;
constexpr uint32_t kFacStride = 834;
constexpr uint32_t kWeights = 26000;
constexpr uint32_t kLaneCount = 8;
constexpr std::array<uint32_t,6> kNw = {40, 640, 40, 640, 24576, 64};
constexpr std::array<uint32_t,4> kConvOff = {0, 40, 680, 720};
constexpr std::array<uint32_t,2> kDenseCin = {768,32};
constexpr std::array<uint32_t,2> kDenseCout = {32,2};
constexpr std::array<uint32_t,2> kDenseFacOff = {0,800};

inline int32_t clip(int32_t a) { return a < -256 ? -256 : a > 256 ? 256 : a; }
inline std::array<uint32_t,8> digest(const int16_t* conv, const int8_t* fac,
                                      const uint32_t* indices, uint32_t n) {
    std::array<uint32_t,8> sums{};
    for (int layer=0; layer<6; ++layer) {
        for (uint32_t f=0; f<kNw[layer]; ++f) {
            int32_t gradient = 0;
            if (layer<4) {
                for (uint32_t s=0; s<16; ++s) {
                    const uint32_t row=indices[s];
                    if(row>=n) throw std::out_of_range("invalid sample index");
                    gradient += conv[static_cast<size_t>(row)*kConvStride+kConvOff[layer]+f];
                }
            } else {
                const int d=layer-4;
                const uint32_t o=f/kDenseCin[d], q=f%kDenseCin[d];
                for (uint32_t s=0; s<16; ++s) {
                    const uint32_t row=indices[s];
                    if(row>=n) throw std::out_of_range("invalid sample index");
                    const size_t at=static_cast<size_t>(row)*kFacStride+kDenseFacOff[d];
                    gradient += int32_t(fac[at+o])*int32_t(fac[at+kDenseCout[d]+q]);
                }
            }
            const uint32_t term=static_cast<uint32_t>(clip(gradient)) ^ (f + 7919u*static_cast<uint32_t>(layer+1));
            sums[f&7u]+=term;
        }
    }
    return sums;
}
inline void to_bytes(const std::array<uint32_t,8>& lanes, uint8_t* dst) {
    for(int d=0;d<8;++d)for(int b=0;b<4;++b) dst[d*4+b]=uint8_t(lanes[d]>>(8*b));
}
inline bool meets_target(const uint8_t ticket[32],const uint8_t target[32]) {
    for(int i=31;i>=0;--i) { if(ticket[i]!=target[i])return ticket[i]<target[i]; }
    return true;
}
} // namespace neb
