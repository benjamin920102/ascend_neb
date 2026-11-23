#include "protocol.hpp"
#include <array>
namespace neb {std::array<uint8_t,32> hash(const std::vector<uint8_t>& value);}
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace neb {
static constexpr size_t MAX_FRAME=2*1024*1024;
static void checked(bool ok){if(!ok)throw std::runtime_error("malformed work protocol packet");}
std::vector<uint8_t> Reader::bytes(size_t n){checked(n<=v.size()-at);auto a=v.begin()+at;at+=n;return {a,a+n};}
uint8_t Reader::u8(){return bytes(1)[0];}
uint16_t Reader::u16(){auto b=bytes(2);return uint16_t(b[0]|(b[1]<<8));}
uint32_t Reader::u32(){auto b=bytes(4);return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
uint64_t Reader::u64(){uint64_t x=u32();return x|(uint64_t(u32())<<32);}
uint64_t Reader::varint(){
  uint8_t x=u8();if(x<0xfd)return x;
  uint64_t v=x==0xfd?u16():x==0xfe?u32():u64();
  checked(v>=(x==0xfd?0xfdull:x==0xfe?0x10000ull:0x100000000ull));return v;
}
std::vector<uint8_t> Reader::varbytes(){auto n=varint();checked(n<=MAX_FRAME);return bytes(size_t(n));}
void Reader::done()const{checked(at==v.size());}
void put_u16(std::vector<uint8_t>&b,uint16_t x){for(int i=0;i<2;++i)b.push_back(uint8_t(x>>(8*i)));}
void put_u32(std::vector<uint8_t>&b,uint32_t x){for(int i=0;i<4;++i)b.push_back(uint8_t(x>>(8*i)));}
void put_u64(std::vector<uint8_t>&b,uint64_t x){for(int i=0;i<8;++i)b.push_back(uint8_t(x>>(8*i)));}
void put_varint(std::vector<uint8_t>&b,uint64_t x){
  if(x<0xfd)b.push_back(uint8_t(x));
  else if(x<=0xffff){b.push_back(0xfd);put_u16(b,uint16_t(x));}
  else if(x<=0xffffffff){b.push_back(0xfe);put_u32(b,uint32_t(x));}
  else {b.push_back(0xff);put_u64(b,x);}
}
std::vector<uint8_t> frame(uint8_t kind,const std::vector<uint8_t>& payload){
  if(kind<1||kind>7||payload.size()+1>MAX_FRAME)throw std::runtime_error("invalid frame");
  std::vector<uint8_t> b;put_u32(b,uint32_t(payload.size()+1));b.push_back(kind);
  b.insert(b.end(),payload.begin(),payload.end());return b;
}
std::vector<uint8_t> hello(const std::string& name){
  std::vector<uint8_t> b;put_u16(b,1);put_varint(b,name.size());b.insert(b.end(),name.begin(),name.end());return frame(1,b);
}
std::vector<uint8_t> share(uint64_t id,uint32_t nonce){
  std::vector<uint8_t> b;put_u64(b,id);put_u32(b,nonce);return frame(6,b);
}
std::vector<uint8_t> submit(uint64_t id,const std::vector<uint8_t>& extra,uint32_t time,uint32_t nonce){
  std::vector<uint8_t> b;put_u64(b,id);put_varint(b,extra.size());
  b.insert(b.end(),extra.begin(),extra.end());put_u32(b,time);put_u32(b,nonce);return frame(4,b);
}
void Framer::append(const uint8_t* data,size_t n){
  if(in_.size()+n>MAX_FRAME*2)throw std::runtime_error("work input buffer too large");
  in_.insert(in_.end(),data,data+n);
}
bool Framer::pop(Frame& f){
  if(in_.size()<4)return false;
  const size_t n=size_t(in_[0])|(size_t(in_[1])<<8)|(size_t(in_[2])<<16)|(size_t(in_[3])<<24);
  if(n<1||n>MAX_FRAME)throw std::runtime_error("invalid work frame length");
  if(in_.size()<n+4)return false;
  f.kind=in_[4];checked(f.kind>=1&&f.kind<=7);
  f.payload.assign(in_.begin()+5,in_.begin()+4+n);
  in_.erase(in_.begin(),in_.begin()+4+n);return true;
}
Job Job::decode(const std::vector<uint8_t>& b){
  Reader r(b);Job j;j.id=r.u64();j.clean=r.u8()!=0;
  j.version=r.u32();auto parent=r.bytes(32);std::copy(parent.begin(),parent.end(),j.parent.begin());
  j.time_start=r.u32();j.time_window=r.u32();j.bits=r.u32();
  auto t=r.bytes(32),s=r.bytes(32);
  std::copy(t.begin(),t.end(),j.target.begin());std::copy(s.begin(),s.end(),j.share_target.begin());
  j.prefix=r.varbytes();j.suffix=r.varbytes();j.extra_len=r.u8();checked(j.extra_len>=1&&j.extra_len<=32);
  const uint64_t count=r.varint();checked(count<=4096);
  for(uint64_t k=0;k<count;++k){auto h=r.bytes(32);std::array<uint8_t,32> v;std::copy(h.begin(),h.end(),v.begin());j.branches.push_back(v);}
  j.tx_count=r.u32();checked(j.tx_count>0);
  auto root=r.bytes(32);std::copy(root.begin(),root.end(),j.shares_root.begin());
  j.height=r.u32();j.expiry=r.u32();r.done();return j;
}
std::array<uint8_t,80> Job::header() const{
  std::vector<uint8_t> coinbase=prefix;coinbase.resize(coinbase.size()+extra_len,0);
  coinbase.insert(coinbase.end(),suffix.begin(),suffix.end());
  auto h=hash(coinbase);
  std::vector<uint8_t> acc={'l','e','a','f'};acc.insert(acc.end(),h.begin(),h.end());
  h=hash(acc);
  for(const auto &branch:branches){acc={'n','o','d','e'};acc.insert(acc.end(),h.begin(),h.end());
     acc.insert(acc.end(),branch.begin(),branch.end());h=hash(acc);}
  acc={'b','o','d','y'};put_u32(acc,tx_count);acc.insert(acc.end(),h.begin(),h.end());
  acc.insert(acc.end(),shares_root.begin(),shares_root.end());h=hash(acc);
  std::vector<uint8_t> bytes;put_u32(bytes,version);bytes.insert(bytes.end(),parent.begin(),parent.end());
  bytes.insert(bytes.end(),h.begin(),h.end());put_u32(bytes,time_start);put_u32(bytes,bits);put_u32(bytes,0);
  if(bytes.size()!=80)throw std::runtime_error("bad NEB header size");
  std::array<uint8_t,80> header;std::copy(bytes.begin(),bytes.end(),header.begin());return header;
}
}
