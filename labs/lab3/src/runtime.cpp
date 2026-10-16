// Provided: compile generated CUDA with NVRTC, load with the driver API, run plans.
#include "tc/compiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tc {

#define TC_CUDA(x) check_rt((x), #x)
#define TC_CU(x) check_drv((x), #x)
static void check_rt(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
static void check_drv(CUresult e, const char *what) {
    if (e != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(e, &msg);
        throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "?"));
    }
}

struct Runtime::Impl {
    std::unordered_map<std::string, CUfunction> cache; // source -> function
    std::vector<CUmodule> modules;
    std::string arch;
    int sms = 0;

    CUfunction compile(const std::string &src) {
        auto it = cache.find(src);
        if (it != cache.end()) return it->second;
        nvrtcProgram prog;
        if (nvrtcCreateProgram(&prog, src.c_str(), "tc_kernel.cu", 0, nullptr, nullptr) !=
            NVRTC_SUCCESS)
            throw std::runtime_error("nvrtcCreateProgram failed");
        const std::string arch_opt = "--gpu-architecture=" + arch;
        const char *opts[] = {arch_opt.c_str(), "-std=c++17", "-default-device"};
        const nvrtcResult rc = nvrtcCompileProgram(prog, 3, opts);
        if (rc != NVRTC_SUCCESS) {
            size_t n = 0;
            nvrtcGetProgramLogSize(prog, &n);
            std::string log(n, '\0');
            nvrtcGetProgramLog(prog, log.data());
            nvrtcDestroyProgram(&prog);
            throw std::runtime_error("NVRTC failed to compile the generated kernel:\n" + log +
                                     "\n----- source -----\n" + src);
        }
        size_t n = 0;
        nvrtcGetCUBINSize(prog, &n);
        std::string cubin(n, '\0');
        nvrtcGetCUBIN(prog, cubin.data());
        nvrtcDestroyProgram(&prog);
        CUmodule mod;
        CUfunction fn;
        TC_CU(cuModuleLoadData(&mod, cubin.data()));
        TC_CU(cuModuleGetFunction(&fn, mod, "tc_kernel"));
        modules.push_back(mod);
        return cache[src] = fn;
    }
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {
    TC_CUDA(cudaFree(nullptr)); // creates the primary context; the driver API then uses it
    cudaDeviceProp p{};
    TC_CUDA(cudaGetDeviceProperties(&p, 0));
    impl_->arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
    impl_->sms = p.multiProcessorCount;
}

Runtime::~Runtime() {
    for (CUmodule m : impl_->modules) cuModuleUnload(m);
}

int Runtime::compiled_kernels() const {
    return static_cast<int>(impl_->cache.size());
}

namespace {
struct DeviceState {
    std::unordered_map<int, float *> buf; // node id -> device buffer
    ~DeviceState() {
        for (auto &[id, p] : buf) cudaFree(p);
    }
};

float *alloc(DeviceState &st, int id, int64_t n) {
    float *p = nullptr;
    TC_CUDA(cudaMalloc(&p, std::max<int64_t>(n, 1) * sizeof(float)));
    st.buf[id] = p;
    return p;
}

void upload(const Graph &g, const Feeds &feeds, DeviceState &st) {
    for (int id : g.inputs()) {
        const Node &n = g.node(id);
        auto it = feeds.find(n.name);
        if (it == feeds.end())
            throw std::invalid_argument("missing feed for input '" + n.name + "'");
        if (it->second.shape != n.shape)
            throw std::invalid_argument("feed '" + n.name + "' has the wrong shape");
        std::vector<float> host(it->second.data.begin(), it->second.data.end());
        float *p = alloc(st, id, n.shape.numel());
        TC_CUDA(cudaMemcpy(p, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

struct Launch {
    CUfunction fn;
    unsigned grid_x, grid_y, block;
    std::vector<float *> ptrs; // ins..., out
    int64_t d0, d1, d2;        // dims
    int ndims;
};

std::vector<Launch> prepare(Runtime::Impl &rt, const Graph &g, const Plan &plan, DeviceState &st) {
    std::vector<Launch> out;
    for (const Kernel &k : plan.kernels) {
        Launch L{};
        L.fn = rt.compile(codegen(g, k));
        for (int r : k.reads) L.ptrs.push_back(st.buf.at(r));
        const Shape os = g.node(k.output).shape;
        L.ptrs.push_back(alloc(st, k.output, os.numel()));
        if (k.kind == KernelKind::Elementwise) {
            L.block = 256;
            L.grid_x = static_cast<unsigned>(
                std::min<int64_t>((os.numel() + 255) / 256, int64_t(rt.sms) * 8));
            L.grid_y = 1;
            L.d0 = os.rows, L.d1 = os.cols, L.ndims = 2;
        } else if (k.kind == KernelKind::Row) {
            // The row width is the column count of the widest node in the kernel.
            int64_t width = 1, rows = os.rows;
            for (int id : k.nodes) {
                const Node &n = g.node(id);
                for (int s : n.srcs) width = std::max(width, g.node(s).shape.cols);
                width = std::max(width, n.shape.cols);
                rows = std::max(rows, n.shape.rows);
            }
            L.block = 256;
            L.grid_x = static_cast<unsigned>(rows);
            L.grid_y = 1;
            L.d0 = rows, L.d1 = width, L.ndims = 2;
        } else {
            const Node *mm = nullptr;
            for (int id : k.nodes)
                if (g.node(id).op == Op::MatMul) mm = &g.node(id);
            const int64_t M = mm->shape.rows, N = mm->shape.cols,
                          K = g.node(mm->srcs[0]).shape.cols;
            L.block = 256;
            L.grid_x = static_cast<unsigned>((N + 63) / 64);
            L.grid_y = static_cast<unsigned>((M + 63) / 64);
            L.d0 = M, L.d1 = N, L.d2 = K, L.ndims = 3;
        }
        out.push_back(L);
    }
    return out;
}

void launch_all(std::vector<Launch> &ls) {
    for (Launch &L : ls) {
        std::vector<void *> args;
        for (float *&p : L.ptrs) args.push_back(&p);
        args.push_back(&L.d0);
        args.push_back(&L.d1);
        if (L.ndims == 3) args.push_back(&L.d2);
        TC_CU(cuLaunchKernel(L.fn, L.grid_x, L.grid_y, 1, L.block, 1, 1, 0, nullptr, args.data(),
                             nullptr));
    }
}
} // namespace

HostTensor Runtime::run(const Graph &g, const Plan &plan, const Feeds &feeds) {
    DeviceState st;
    upload(g, feeds, st);
    auto ls = prepare(*impl_, g, plan, st);
    launch_all(ls);
    TC_CUDA(cudaDeviceSynchronize());
    const Shape s = g.node(g.output()).shape;
    std::vector<float> host(s.numel());
    TC_CUDA(cudaMemcpy(host.data(), st.buf.at(g.output()), host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    return HostTensor{s, std::vector<double>(host.begin(), host.end())};
}

double Runtime::time_ms(const Graph &g, const Plan &plan, const Feeds &feeds, int repeats) {
    DeviceState st;
    upload(g, feeds, st);
    auto ls = prepare(*impl_, g, plan, st);
    for (int w = 0; w < 3; ++w) launch_all(ls);
    TC_CUDA(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    TC_CUDA(cudaEventCreate(&a));
    TC_CUDA(cudaEventCreate(&b));
    std::vector<float> times;
    for (int r = 0; r < repeats; ++r) {
        TC_CUDA(cudaEventRecord(a));
        launch_all(ls);
        TC_CUDA(cudaEventRecord(b));
        TC_CUDA(cudaEventSynchronize(b));
        float ms = 0;
        TC_CUDA(cudaEventElapsedTime(&ms, a, b));
        times.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace tc
