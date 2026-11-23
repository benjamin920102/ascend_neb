#include "protocol.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
// Stub for protocol tests only; hash() in production comes from official BLAKE3 bridge.
namespace neb {std::array<uint8_t,32> hash(const std::vector<uint8_t>& b){
   std::array<uint8_t,32> a{};for(size_t i=0;i<b.size();++i)a[i%32]^=b[i];return a;}}
static void require(bool b){if(!b)throw std::runtime_error("protocol host test failed");}
int main(){
  using namespace neb;
  auto a=hello("ascend_neb");
  Framer p;Frame f;
  p.append(a.data(),3);require(!p.pop(f));
  p.append(a.data()+3,a.size()-3);require(p.pop(f));
  require(f.kind==1);Reader r(f.payload);require(r.u16()==1);
  auto s=r.varbytes();r.done();require(std::string(s.begin(),s.end())=="ascend_neb");
  auto h=share(123456789ull,0x10203040);
  p.append(h.data(),h.size());require(p.pop(f)&&f.kind==6);
  Reader sr(f.payload);require(sr.u64()==123456789ull&&sr.u32()==0x10203040);sr.done();
  auto sub=submit(1,std::vector<uint8_t>(4,0),100,55);
  p.append(sub.data(),sub.size());require(p.pop(f)&&f.kind==4);
  Reader rr(f.payload);require(rr.u64()==1);require(rr.varbytes().size()==4);require(rr.u32()==100&&rr.u32()==55);rr.done();
  std::cout<<"PASS: native work framing/hello/share/submit\n";
}
