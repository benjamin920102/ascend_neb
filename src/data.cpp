#include "data.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace neb {
static constexpr const char* MAINNET_HASH="790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72";
static uint32_t u32(const std::vector<uint8_t>&b,size_t p){
  if(p+4>b.size())throw std::runtime_error("truncated NMD2 header");
  return uint32_t(b[p])|(uint32_t(b[p+1])<<8)|(uint32_t(b[p+2])<<16)|(uint32_t(b[p+3])<<24);
}
std::vector<uint8_t> read_file(const std::string& name){
  std::ifstream in(name,std::ios::binary);
  if(!in)throw std::runtime_error("cannot open "+name);
  return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
std::string hex(const uint8_t* ptr,size_t n){
  std::ostringstream out;out<<std::hex<<std::setfill('0');
  for(size_t i=0;i<n;++i)out<<std::setw(2)<<unsigned(ptr[i]);
  return out.str();
}
std::vector<uint8_t> unhex(const std::string& str){
  if(str.size()%2)throw std::runtime_error("hex length must be even");
  std::vector<uint8_t> result;result.reserve(str.size()/2);
  for(size_t i=0;i<str.size();i+=2){
    auto nibble=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;throw std::runtime_error("invalid hex");};
    result.push_back(uint8_t((nibble(str[i])<<4)|nibble(str[i+1])));
  }
  return result;
}
std::array<uint8_t,32> hash(const std::vector<uint8_t>& blob){
  std::array<uint8_t,32> h{};an_hash(blob.data(),blob.size(),h.data());return h;
}
std::vector<DataSet> parse_datasets(const std::vector<uint8_t>& blob,bool allow_other){
  if(!allow_other){auto h=hash(blob);if(hex(h.data(),32)!=MAINNET_HASH)
    throw std::runtime_error("mainnet dataset BLAKE3 mismatch; use --allow-other-dataset ONLY for tests");}
  if(blob.size()<8||memcmp(blob.data(),"NMDS",4)!=0)throw std::runtime_error("expected NMDS container");
  const uint32_t sets=u32(blob,4);
  if(sets<1||sets>16)throw std::runtime_error("invalid NMDS dataset count");
  size_t off=8;std::vector<DataSet> ds;
  for(uint32_t i=0;i<sets;++i){
    if(off+12>blob.size()||memcmp(blob.data()+off,"NMD2",4)!=0)
      throw std::runtime_error("expected NMD2 dataset record");
    DataSet d; d.arch=u32(blob,off+4);d.n=u32(blob,off+8);off+=12;
    if(d.arch!=1||d.n<1||d.n>1000000)throw std::runtime_error("only nonempty KEP1 supported");
    const size_t length=static_cast<size_t>(d.n)*785;
    if(length>blob.size()-off)throw std::runtime_error("truncated NMD2 samples");
    d.values.resize(static_cast<size_t>(d.n)*784);d.labels.resize(d.n);
    for(uint32_t j=0;j<d.n;++j){
      const auto* src=blob.data()+off+static_cast<size_t>(j)*785;
      if(std::any_of(src,src+784,[](uint8_t x){return x>7;})||src[784]>1)
        throw std::runtime_error("invalid KEP1 input/label range");
      memcpy(d.values.data()+static_cast<size_t>(j)*784,src,784);
      d.labels[j]=src[784];
    }
    off+=length;ds.push_back(std::move(d));
  }
  if(off!=blob.size())throw std::runtime_error("trailing bytes in NMDS");
  return ds;
}
GradientTable build_table(const neb2_model& model,const DataSet& ds,int threads){
  if(model.arch!=1||ds.arch!=1)throw std::runtime_error("only KEP1 supported");
  GradientTable out;out.n=ds.n;
  out.conv.resize(static_cast<size_t>(ds.n)*1360);
  out.factors.resize(static_cast<size_t>(ds.n)*834);
  if(an_build_table(&model,ds.values.data(),ds.labels.data(),ds.n,
      out.conv.data(),out.factors.data(),threads)!=0)
    throw std::runtime_error("reference gradient table failed (possibly out-of-range int16)");
  return out;
}
}
