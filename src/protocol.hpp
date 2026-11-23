#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace neb {
struct Frame {uint8_t kind;std::vector<uint8_t> payload;};
class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& blob):v(blob){}
  uint8_t u8();uint16_t u16();uint32_t u32();uint64_t u64();uint64_t varint();
  std::vector<uint8_t> bytes(size_t n);std::vector<uint8_t> varbytes();
  void done() const;
 private:
  const std::vector<uint8_t>&v;size_t at=0;
};
class Framer {
 public:
  void append(const uint8_t* bytes,size_t n);
  bool pop(Frame& frame);
 private:
  std::vector<uint8_t> in_;
};
void put_u16(std::vector<uint8_t>&v,uint16_t x);
void put_u32(std::vector<uint8_t>&v,uint32_t x);
void put_u64(std::vector<uint8_t>&v,uint64_t x);
void put_varint(std::vector<uint8_t>&v,uint64_t x);
std::vector<uint8_t> frame(uint8_t kind,const std::vector<uint8_t>& payload);
std::vector<uint8_t> hello(const std::string& name);
std::vector<uint8_t> share(uint64_t id,uint32_t nonce);
std::vector<uint8_t> submit(uint64_t id,const std::vector<uint8_t>& extra,uint32_t time,uint32_t nonce);
struct Job {
  uint64_t id=0;
  bool clean=false;
  uint32_t version=0,time_start=0,time_window=0,bits=0;
  std::array<uint8_t,32> parent{},target{},share_target{},shares_root{};
  std::vector<uint8_t> prefix,suffix;
  uint8_t extra_len=0;
  std::vector<std::array<uint8_t,32>> branches;
  uint32_t tx_count=0,height=0,expiry=0;
  static Job decode(const std::vector<uint8_t>& payload);
  std::array<uint8_t,80> header() const;
};
} // namespace neb
