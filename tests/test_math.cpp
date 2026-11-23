#include "native_math.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void expect(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){
  using namespace neb;
  constexpr uint32_t rows=39;
  std::vector<int16_t> conv(static_cast<size_t>(rows)*kConvStride);
  std::vector<int8_t> fac(static_cast<size_t>(rows)*kFacStride);
  for(size_t i=0;i<conv.size();++i)conv[i]=int16_t(int(i*73%20001)-10000);
  for(size_t i=0;i<fac.size();++i)fac[i]=int8_t(int(i*17%16)-8);
  for(int t=0;t<48;++t){
    uint32_t indices[16];for(int j=0;j<16;++j)indices[j]=(t&1)?(t%rows):(t*17+j*13)%rows;
    const auto got=digest(conv.data(),fac.data(),indices,rows);
    std::array<uint32_t,8> oracle{};
    const int offs[6]={0,40,680,720,0,800};
    const int nows[6]={40,640,40,640,24576,64};
    for(int layer=0;layer<6;++layer){
      for(int i=0;i<nows[layer];++i){
        int sum=0;
        for(int s=0;s<16;++s){
          int row=int(indices[s]);
          if(layer<4)sum+=conv[row*1360+offs[layer]+i];
          else {
            int cout=layer==4?32:2,cin=layer==4?768:32;
            int o=i/cin,k=i%cin;
            sum+=int(fac[row*834+offs[layer]+o])*int(fac[row*834+offs[layer]+cout+k]);
          }
        }
        int clipped=sum<-256?-256:sum>256?256:sum;
        oracle[i%8]+=uint32_t(clipped) ^ uint32_t(i+7919*(layer+1));
      }
    }
    expect(oracle==got,"gradient/digest mismatch");
    uint8_t b[32];to_bytes(got,b);
    for(int k=0;k<8;++k){uint32_t v=uint32_t(b[4*k])|(uint32_t(b[4*k+1])<<8)|
      (uint32_t(b[4*k+2])<<16)|(uint32_t(b[4*k+3])<<24);
      expect(v==got[k],"little endian mismatch");}
  }
  uint8_t target[32]={},t[32]={};expect(meets_target(t,target),"LE target equal");
  t[0]=1;expect(!meets_target(t,target),"LE target greater");
  target[31]=1;expect(meets_target(t,target),"LE target less");
  uint32_t bad[16]={};bad[0]=rows;
  try{digest(conv.data(),fac.data(),bad,rows);throw std::runtime_error("missing bad idx check");}
  catch(const std::out_of_range&){}
  std::cout<<"PASS: 48 host INT32 digest tests, repeated samples, little-endian, bad-index\n";
  return 0;
}
