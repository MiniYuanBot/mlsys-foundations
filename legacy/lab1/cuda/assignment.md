# CUDA GEMM 实验

预计 **8–12 小时**。编译 `.cu`，实现共享内存与同步，检查硬件计数器，并解释优化在哪些形状下有效。先读 [setup.md](setup.md)，评分见 [grade.md](grade.md)。

提供的代码尚未经过 GPU 编译和实测。初始学生实现含 TODO，测试会失败；配置环境后需先验证参考实现。

## 文件与接口

| 文件 | 已提供 | 你的工作 |
|---|---|---|
| `CMakeLists.txt`、`gemm.cuh` | C++17/CUDA 构建、错误检查、统一接口 | 理解编译目标与 stream 契约 |
| `baselines.cu` | 可读的 naive GEMM 与 cuBLAS 强基线 | 解释行列布局，保持基线不变 |
| `main.cu` | 独立 CPU oracle、数据、边界测试、计时、CSV 输出 | 阅读测量契约；在独立分支补留出集 |
| `student.cu` | 网格、共享内存声明、TODO 提示 | 实现 tiled GEMM，所有边缘 tile 正确 |

矩阵均为连续 row-major：`A[M,K] × B[K,N] = C[M,N]`，FP32 输入/输出，覆盖写 C。cuBLAS 默认采用 column-major，因此参考调用计算 `Cᵀ[N,M] = Bᵀ[N,K] × Aᵀ[K,M]`，参数顺序是 B、A，leading dimensions 是 N、K、N；没有实际转置。不要通过只测方阵掩盖布局错误。

参考使用 [cuBLAS 的 pedantic FP32 模式](https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t)控制精度；这是当前精度约束内的库基线，不代表开启 TF32/低精度后 cuBLAS 的最高性能。它与低精度扩展必须分栏报告。

## 实验步骤

1. **理解与基线（1–2h）**：跑 `--test --reference-only`；手算 `fixed_dev`；画出一个 warp 对 A/B 的访问地址，说明 naive 重复读取与 coalescing。记录 naive 和 cuBLAS 的首份矩阵。
2. **系统实现（3–4h）**：完成 `student.cu` 三处 TODO；每线程协作加载，边界补零，tile 加载/复用间正确同步，最后有界写回。完成后再把 `implemented` 置为 `true`。禁止边缘线程在 barrier 前提前退出。
3. **受控优化（2–3h）**：保留 TILE=16 的可复现基线，比较 8/16/32 或实现寄存器 tiling。每次只改一个机制；用 profiler 或资源报告解释 bank conflict、寄存器压力、occupancy、访存量。TILE=32 有 1024 线程/block，需检查资源限制；更大 tile 不能机械套用当前线程布局。
4. **论文机制与留出（2–3h）**：预注册“扩大 tile 会减少全局访问，但在小 K/窄矩阵上可能因占用与边界浪费退化”的假设。用开发矩阵调参，再锁定代码，在三个未参与调优的形状上检验；记录反例，不要求得出更快的结果。见评分中的证据包。

核心阶段保持 FP32。**Tensor Cores/CUTLASS 扩展另计 4–8h**：运行 NVIDIA WMMA 示例，添加 FP16 输入、FP32 accumulation 的新实现，与同精度 cuBLAS 对比。明确转换时间是否计入、误差是否扩大，并查看 PTX/SASS 的矩阵指令。不能把低精度加速记作本实验 FP32 tiling 的收益。

## 测试与计时契约

```bash
# 在本目录，构建后执行；完整命令在 setup.md
./build/gemm_lab --test --reference-only  # 只检查提供的基线
./build/gemm_lab --test                   # 初始 TODO 必须返回非零
ctest --test-dir build --output-on-failure
./build/gemm_lab --benchmark --iterations 20
```

正确性：手算的 2×3×4 矩阵、零矩阵，以及下列每个 shape 的 seed=1/17；完整输出对照 CPU double 累加 oracle。误差约束 `abs(actual-ref) <= 2e-4 + 3e-4*abs(ref)`，NaN/Inf 直接失败。输出事先填 NaN，漏写会失败。

| M | N | K | 检查重点 |
|---:|---:|---:|---|
| 1 | 1 | 1 | 单元素 |
| 7 | 11 | 5 | 非方阵、三维边界 |
| 16 | 16 | 16 | 完整 tile |
| 31 | 37 | 19 | 所有维度都有尾块 |
| 64 | 65 | 33 | 窄矩阵与 K 尾块 |
| 129 | 17 | 65 | 长条、多个 block |

性能固定开发矩阵为 `(256,256,256)`、`(1024,1024,1024)`、`(257,509,131)`、`(512,2048,64)`、`(2048,256,128)`，seed=20261005。所有实现使用同一份输入；每个 shape 先做完整 cuBLAS 输出比较，另对每个实现的 128 个位置计算独立 CPU double oracle。大形状采用抽样 CPU 检查，不能声称完整 CPU 对照。

每实现每 shape 预热 5 次，然后 CUDA events 同 stream 记录 20 次调用的平均 GPU 时间，重复三轮并轮换实现顺序。编译、context 初始化、首次加载、分配和 memcpy 均在计时外。CSV 的 `mean_ms` 是一轮内平均值，不是单次请求 p50/p99；跨三轮报告 median/min/max。给出 TFLOP/s=`2MNK/(ms×10⁹)`、相对 naive 加速、相对 cuBLAS 效率；不要将 kernel-only 时间解释为端到端吞吐。

只有正确性成功才允许进入性能计时。`--reference-only` 输出 `STUDENT_NOT_EVALUATED`，不能作为结课通过；无 CUDA、编译失败和缺依赖属于环境阻塞，不得标为通过。`tests.py` 只负责从仓库评分器调用已编译的程序，不含 Python 算子实现，也不依赖 PyTorch。

## 阅读与参考代码

以下资料用于阅读和实现对照，本目录未复制外部解答。链接核验日期：2026-10-05。

| 阅读顺序 | 来源与精确入口 | 用在本实验的什么地方 |
|---|---|---|
| 基础 | [CMU LLM Systems HW1](https://github.com/llmsystem/llmsys_hw1)，`src/combine.cu`、`minitorch/cuda_kernel_ops.py`；[MIT](https://raw.githubusercontent.com/llmsystem/llmsys_hw1/main/LICENSE) | 对照 CUDA 算子与框架绑定；是已有实验入口，不是最新论文 |
| 基础 | [GPU MODE lecture_004](https://github.com/gpu-mode/lectures/tree/main/lecture_004)、[lecture_008](https://github.com/gpu-mode/lectures/tree/main/lecture_008)；仓库 [Apache-2.0](https://github.com/gpu-mode/lectures/blob/main/LICENSE) | 内存层次、性能排查 |
| 实现 | [NVIDIA WMMA 示例，固定 v12.8](https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)；文件内 BSD 3-Clause 条款 | 从 tile/共享内存走到 Tensor Core；从小尺寸开始 |
| 机制基础 | [FlashAttention，2022](https://arxiv.org/abs/2205.14135)、[FlashAttention-2，2023预印本/ICLR 2024](https://arxiv.org/abs/2307.08691) | 把本实验的数据复用、并行分工连接到 attention；这里只建立前置，不算复现 attention |
| 近年前沿 | [FlashAttention-3，2024](https://arxiv.org/abs/2407.08608) | 阅读异步搬运/计算重叠；提出哪些机制依赖 Hopper |
| 近年前沿 | [ThunderKittens，2024预印本/ICLR 2025](https://arxiv.org/abs/2410.20399)；[官方 educational_h100](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm/educational_h100)，MIT | 对比 CUDA C++ tile 抽象，H100 路径作为阅读或远程选修 |
| 近年前沿 | [TileLang，2025](https://arxiv.org/abs/2504.17577)；[examples/gemm/example_gemm.py](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)，[MIT](https://github.com/tile-ai/tilelang/blob/main/LICENSE) | 后续编译器实验：将手写 kernel 的 dataflow 与 layout/pipeline 调度分离 |

每篇精读只需交一张机制卡：瓶颈 → 机制 → 关键实验 → 隐含硬件条件 → 本实验能测什么/不能测什么。不要仅抄摘要或论文加速倍数。引用或改编代码时额外记录上游 commit 和文件许可证，保留相应声明。

## RTX 4060 与 WSL2 的运行范围

RTX 4060 是 [compute capability 8.9](https://developer.nvidia.com/cuda/gpus)，本项目默认编译 `sm_89`；本矩阵不需下载模型，工作集远小于 8GB。Ada 可以学习 CUDA tiling/WMMA；FA3 官方路径需要 H100/H800，TMA/WGMMA 与 Hopper warp specialization 不能在本机当作原论文实验执行。ThunderKittens 当前主线主要测试 Hopper/Blackwell，不作为 4060 必装依赖。

没有 profiler 权限时先保留失败信息，用 `ptxas` 资源报告和量化访存分析完成基础解释；明确标注未测的计数器。显卡同时承担桌面显示、温度/功耗、WSL 和驱动都会影响结果，记录运行条件；评分不绑定某个机器的绝对 TFLOP/s。
