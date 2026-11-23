#include "ascend_backend.hpp"
#include "native_math.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

extern "C" void neb_launch_digest(void*,void*,void*,void*,uint32_t,uint32_t,aclrtStream);

namespace neb {
static void check(aclError e,const char* action){
    if(e!=ACL_SUCCESS){ std::ostringstream msg;msg<<action<<" failed ACL status="<<e;throw std::runtime_error(msg.str()); }
}
static void release(void*&p){if(p) {aclrtFree(p);p=nullptr;}}
AscendDevice::AscendDevice(int device):device_(device){
    aclError init=aclInit(nullptr);
    if(init!=ACL_SUCCESS && init!=ACL_ERROR_REPEAT_INITIALIZE)check(init,"aclInit");
    check(aclrtSetDevice(device_),"aclrtSetDevice");
    check(aclrtCreateStream(&stream_),"aclrtCreateStream");
}
AscendDevice::~AscendDevice(){
    if(device_>=0)aclrtSetDevice(device_);
    if(stream_)aclrtSynchronizeStream(stream_);
    release(conv_);release(fac_);release(indices_);release(digests_);
    if(stream_)aclrtDestroyStream(stream_);
    if(device_>=0)aclrtResetDevice(device_);
}
void AscendDevice::upload(const std::vector<int16_t>& conv,const std::vector<int8_t>& fac,uint32_t rows){
    if(!rows||conv.size()!=static_cast<size_t>(rows)*kConvStride||
       fac.size()!=static_cast<size_t>(rows)*kFacStride)throw std::invalid_argument("bad KEP1 table");
    check(aclrtSynchronizeStream(stream_),"sync before table swap");
    release(conv_);release(fac_);rows_=0;
    check(aclrtMalloc(&conv_,conv.size()*sizeof(int16_t),ACL_MEM_MALLOC_HUGE_FIRST),"alloc conv GM");
    check(aclrtMalloc(&fac_,fac.size(),ACL_MEM_MALLOC_HUGE_FIRST),"alloc factor GM");
    check(aclrtMemcpy(conv_,conv.size()*sizeof(int16_t),conv.data(),conv.size()*sizeof(int16_t),ACL_MEMCPY_HOST_TO_DEVICE),"H2D conv");
    check(aclrtMemcpy(fac_,fac.size(),fac.data(),fac.size(),ACL_MEMCPY_HOST_TO_DEVICE),"H2D factors");
    rows_=rows;
}
std::vector<std::array<uint32_t,8>> AscendDevice::run(const std::vector<uint32_t>& indices){
    if(!rows_)throw std::runtime_error("upload gradient table first");
    if(indices.empty()||indices.size()%kSamplesPerNonce)throw std::invalid_argument("indices must be attempts x 16");
    const size_t attempts=indices.size()/kSamplesPerNonce;
    if(attempts>4096)throw std::invalid_argument("batch >4096: kernel uses eight blocks/attempt");
    for(auto x:indices)if(x>=rows_)throw std::out_of_range("bad sample index");
    const size_t input_bytes=indices.size()*sizeof(uint32_t);
    const size_t output_bytes=attempts*8*16*sizeof(uint32_t);
    if(input_bytes>index_capacity_){
        release(indices_);check(aclrtMalloc(&indices_,input_bytes,ACL_MEM_MALLOC_HUGE_FIRST),"alloc indices");
        index_capacity_=input_bytes;
    }
    if(output_bytes>digest_capacity_){
        release(digests_);check(aclrtMalloc(&digests_,output_bytes,ACL_MEM_MALLOC_HUGE_FIRST),"alloc lanes");
        digest_capacity_=output_bytes;
    }
    check(aclrtMemcpyAsync(indices_,index_capacity_,indices.data(),input_bytes,ACL_MEMCPY_HOST_TO_DEVICE,stream_),"indices H2D");
    neb_launch_digest(conv_,fac_,indices_,digests_,rows_,static_cast<uint32_t>(attempts),stream_);
    check(aclrtSynchronizeStream(stream_),"kernel launch/sync");
    std::vector<uint32_t> raw(attempts*8*16,0);
    check(aclrtMemcpy(raw.data(),output_bytes,digests_,output_bytes,ACL_MEMCPY_DEVICE_TO_HOST),"digest D2H");
    std::vector<std::array<uint32_t,8>> result(attempts);
    for(size_t a=0;a<attempts;++a)for(int l=0;l<8;++l)
       result[a][l]=raw[(a*8+l)*16];
    return result;
}
} // namespace neb
