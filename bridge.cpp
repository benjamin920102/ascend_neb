#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" size_t neb_tiling_size();
extern "C" int neb_tiling(void*, size_t, uint32_t*, size_t*);
extern "C" void neb_launch(uint32_t, aclrtStream, void*, void*, void*, void*, void*);
namespace {
thread_local std::string error;
std::once_flag init_flag;
void check(aclError code, const char* op) {
    if (code != ACL_SUCCESS) throw std::runtime_error(std::string(op) + ": " + std::to_string(code));
}
constexpr size_t A_BYTES = 16 * 64 / 2, B_BYTES = 64 * 64 / 2, C_BYTES = 16 * 64 * 4;
struct Engine {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void *a = nullptr, *b = nullptr, *c = nullptr, *workspace = nullptr, *tiling = nullptr;
    uint32_t blocks = 0;
    size_t workspace_size = 0;
    std::mutex lock;
    ~Engine() {
        if (context) aclrtSetCurrentContext(context);
        if (stream) aclrtSynchronizeStream(stream);
        for (void* ptr : {a, b, c, workspace, tiling}) if (ptr) aclrtFree(ptr);
        if (stream) aclrtDestroyStream(stream);
        if (context) aclrtDestroyContext(context);
        // No aclFinalize/reset: other contexts in the Python process may be alive.
    }
    void init(int device) {
        std::call_once(init_flag, [] { check(aclInit(nullptr), "aclInit"); });
        check(aclrtSetDevice(device), "aclrtSetDevice");
        check(aclrtCreateContext(&context, device), "aclrtCreateContext");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
        size_t ts = neb_tiling_size();
        std::vector<uint32_t> host((ts + 3) / 4, 0);
        if (neb_tiling(host.data(), ts, &blocks, &workspace_size))
            throw std::runtime_error("INT4 tiling generation failed");
        auto alloc = [](void** ptr, size_t n) { check(aclrtMalloc(ptr, n, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"); };
        alloc(&a, A_BYTES); alloc(&b, B_BYTES); alloc(&c, C_BYTES);
        alloc(&workspace, workspace_size ? workspace_size : 512);
        alloc(&tiling, ts);
        check(aclrtMemcpy(tiling, ts, host.data(), ts, ACL_MEMCPY_HOST_TO_DEVICE), "copy tiling");
    }
};
}
extern "C" const char* neb_last_error() { return error.c_str(); }
extern "C" void* neb_create(int device) {
    try {
        if (device < 0) throw std::runtime_error("device must be nonnegative");
        auto engine = std::make_unique<Engine>();
        engine->init(device);
        return engine.release();
    } catch (const std::exception& ex) { error = ex.what(); return nullptr; }
}
extern "C" int neb_matmul(void* handle, const uint8_t* a, const uint8_t* b, int32_t* c) {
    try {
        if (!handle || !a || !b || !c) throw std::runtime_error("null buffer");
        auto& e = *static_cast<Engine*>(handle);
        std::lock_guard<std::mutex> guard(e.lock);
        check(aclrtSetCurrentContext(e.context), "aclrtSetCurrentContext");
        check(aclrtMemcpy(e.a, A_BYTES, a, A_BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
        check(aclrtMemcpy(e.b, B_BYTES, b, B_BYTES, ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
        neb_launch(e.blocks, e.stream, e.a, e.b, e.c, e.workspace, e.tiling);
        check(aclrtSynchronizeStream(e.stream), "INT4 synchronize");
        check(aclrtMemcpy(c, C_BYTES, e.c, C_BYTES, ACL_MEMCPY_DEVICE_TO_HOST), "copy INT32 output");
        return 0;
    } catch (const std::exception& ex) { error = ex.what(); return -1; }
}
extern "C" void neb_destroy(void* handle) { delete static_cast<Engine*>(handle); }
