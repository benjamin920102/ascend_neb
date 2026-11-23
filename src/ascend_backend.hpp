#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <acl/acl.h>

namespace neb {
class AscendDevice {
 public:
  explicit AscendDevice(int device);
  ~AscendDevice();
  AscendDevice(const AscendDevice&)=delete;
  AscendDevice& operator=(const AscendDevice&)=delete;
  void upload(const std::vector<int16_t>& conv,const std::vector<int8_t>& fac,uint32_t rows);
  std::vector<std::array<uint32_t,8>> run(const std::vector<uint32_t>& indices);
  int device_id() const {return device_;}
 private:
  void* conv_=nullptr;
  void* fac_=nullptr;
  void* indices_=nullptr;
  void* digests_=nullptr;
  size_t index_capacity_=0;
  size_t digest_capacity_=0;
  int device_=-1;
  uint32_t rows_=0;
  aclrtStream stream_=nullptr;
};
}
