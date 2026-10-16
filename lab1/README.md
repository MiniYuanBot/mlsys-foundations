# Lab 1：CUDA 执行模型与 GEMM —— 从 1% 到 cuBLAS

> 配套讲义：[L1 GPU 执行模型与 CUDA GEMM](../../mlsys-research-course/lectures/L1-cuda-gemm.md)（先读 §1–§2 再开始 Part A）
> 预计用时：2 周 × 约 7 小时实验　·　语言：CUDA C++17　·　硬件：NVIDIA GPU，sm_80 及以上（RTX 30/40 系列、A100、H100）

lab0 在 CPU 上把矩阵乘从 1 GFLOP/s 做到了接近 BLAS。这个 lab 把同一件事搬到 GPU 上做，但 GPU 的规则不同：几万个线程同时运行，32 个线程组成一个 warp，按同一条指令执行；内存要“合并”访问才快；shared memory 是你自己管理的缓存；Tensor Core 一条指令能完成一整块矩阵乘。

| 部分                    | 你要写什么                                                                    | 难度梯度                   | 对应的真实系统                                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** CUDA 基础         | grid-stride 向量加、warp shuffle 归约、shared memory 转置（含 bank conflict） | ① 代码填空（每个 5–20 行） | CUDA 编程的三个基本模式                                                                                                                      |
| **B** SGEMM 阶梯        | 合并访存 → shared memory 分块 → 1D 寄存器分块 → 2D 寄存器分块 + float4        | ② 核心实现 + 性能目标      | [siboehm 的 CUDA matmul](https://siboehm.com/articles/22/CUDA-MMM)、[MIT 6.S894](https://accelerated-computing.academy/fall25/labs/) lab 4–5 |
| **C** Tensor Core HGEMM | WMMA；再用 `ldmatrix` + `mma.sync` + `cp.async` 写一个三级流水线 kernel       | ③ 前沿设计的精简复现       | CUTLASS Ampere GEMM、FlashAttention-2 的内循环                                                                                               |

做完后你应当能回答：

- 为什么交换两个下标（B0→B1），速度就能差好几倍？
- shared memory 的 bank conflict 是什么？加一列 padding 为什么能消除它？
- 为什么 Tensor Core 比 FP32 单元快那么多？为了喂饱它，kernel 必须做哪些事？

---

## 0. 环境（约 1 小时，只做一次）

本 lab 在 **WSL2 Ubuntu** 里做。Windows 原生的 nvcc 依赖特定版本的 Visual Studio：作者机器上，CUDA 12.6 与 VS 2026 不兼容，编译器前端会直接崩溃。

1. 按 [setup.md](../setup.md) 的 “WSL2 与 GPU 驱动” 一节装好 WSL2 和 Windows 驱动。
2. 在 WSL 里、仓库根目录下执行 `bash tools/setup_env.sh --lab 1`（安装 CUDA Toolkit 12.6，约 3 GB，并把 `/usr/local/cuda/bin` 加进 PATH），新开一个终端后再执行 `bash tools/check_env.sh --lab 1`，每一项都必须是 OK。
3. 仓库可以留在 Windows 盘上（WSL 里的路径是 `/mnt/d/...`），但构建目录建议放到 Linux 文件系统里（比如 `-B ~/build/lab1`），编译会快很多。

在**仓库根目录**执行：

```bash
cmake -S lab1 -B build/lab1                         # default arch sm_89 (RTX 40xx)
cmake --build build/lab1 -j
ctest --test-dir build/lab1 --output-on-failure     # only the provided parts pass at first
```

其他 GPU 用 `-DCMAKE_CUDA_ARCHITECTURES=86`（RTX 30）、`80`（A100）、`90`（H100）。

常用命令：

```bash
build/lab1/test_b_sgemm B2                 # only tests whose name contains "B2"
compute-sanitizer build/lab1/test_a_basics # catches out-of-bounds and race conditions
cmake -S lab1 -B build/lab1 -DL1_PTXAS_VERBOSE=ON && cmake --build build/lab1 2>&1 | grep -A1 "Function properties"
                                           # registers / shared memory / spills per kernel
```

**每个 TODO 写完之后**，把对应启动函数里的 `constexpr bool implemented = false;` 改成 `true`。测试和 benchmark 遇到 `implemented = false` 的实现会直接跳过（报 NotImplemented）。忘记改这一行，就会看到 “TODO ... not implemented”。

## 1. 代码地图

```
lab1/
+-- include/l1/
|   +-- common.cuh        CUDA_CHECK, DeviceBuffer (RAII, poison()), time_ms (CUDA events)
|   +-- kernels.cuh       every launcher + tile-size constants (read first: it is the spec)
|   +-- ptx.cuh           cp.async / ldmatrix / mma.sync wrappers for Part C2
+-- src/
|   +-- core.cu           provided: naive transpose, naive SGEMM, cuBLAS wrappers
|   +-- basics_student.cu     TODO(A1-A3)
|   +-- sgemm_student.cu      TODO(B1-B4)
|   +-- hgemm_student.cu      TODO(C1, C2a-c)
+-- tests/                one test program per part (CPU double reference + cuBLAS)
+-- bench/                bench_basics (GB/s), bench_gemm (TFLOP/s, % of cuBLAS)
+-- scripts/plot_gemm.py
```

规则：**只改 `src/*_student.cu`**。所有矩阵都是行优先，计算 `C = A @ B`（覆盖写 C，不是累加）。测试会先把 C 填满 NaN，没写到的元素会以 NaN 的形式暴露出来。

---

## Part A：CUDA 基础（第 1 周前半，约 3 小时）

先读讲义 L1 §1（SIMT、warp、block、SM）和 §2（合并访存、shared memory 与 bank）。

| TODO | 内容               | 要点                                                           | 测试    |
| ---- | ------------------ | -------------------------------------------------------------- | ------- |
| A1   | `vector_add`       | grid-stride 循环；启动约 8×SM 个 block，而不是每个元素一个线程 | `A1_*`  |
| A2a  | `warp_reduce_sum`  | `__shfl_down_sync`，偏移依次为 16、8、4、2、1                  | `A2_*`  |
| A2b  | `reduce_kernel`    | 线程 → warp → block 三级归约；`__syncthreads()` 不能放进分支   | `A2_*`  |
| A3a  | `transpose_tiled`  | 32×32 tile；读、写都沿着行进行                                 | `A3a_*` |
| A3b  | `transpose_padded` | `tile[32][33]`：只改这一处                                     | `A3b_*` |

**测量**：

```bash
mkdir -p results/lab1
build/lab1/bench_basics > results/lab1/basics.csv
```

**通过标准**

- `test_a_basics` 全部 PASS。
- `vector_add` 与 `reduce_sum` 的带宽都 ≥ cudaMemcpy 的 **80%**。参考实现在 4060 Laptop 上约为 230 GB/s，超过了 cudaMemcpy 的 213 GB/s，即 109–112%。
- `transpose_padded` 比 `transpose_naive` 快，`transpose_tiled` 介于两者之间。用 bank conflict 解释这个顺序。
  参考实现的结果很有意思：在 8192×8192（256 MB，远大于 L2）时，tiled 与 padded 几乎一样（226 对 232 GB/s），因为瓶颈在 DRAM，bank conflict 被掩盖了；在 2048×2048（16 MB，能装进 32 MB 的 L2）时，padded 比 tiled 快 2.3 倍（953 对 414 GB/s）。用 `bench_basics --n 4194304 --side 2048` 复现这个现象，并解释原因。
- 想亲眼看到 bank conflict，可以用 Nsight Compute：`ncu --section MemoryWorkloadAnalysis_Tables build/lab1/bench_basics --side 4096`，比较 tiled 与 padded 的 “Shared Load Bank Conflicts”。如果 WSL 下 ncu 提示权限不足，在 Windows 的 NVIDIA 控制面板里开启 “允许所有用户访问 GPU 性能计数器”。

---

## Part B：SGEMM 优化阶梯（第 1 周后半 + 第 2 周前半，约 6 小时）

先读讲义 L1 §3。每一步都对应 [siboehm 的文章](https://siboehm.com/articles/22/CUDA-MMM) 中的一个 kernel，文章里有很好的图，**强烈建议**每写一步就对照着读一节。

| 步骤 | TODO                 | 做什么                                                  | siboehm 对应 | 参考实现，RTX 4060 Laptop，n=2048（TFLOP/s，占 cuBLAS 的 %） |
| ---- | -------------------- | ------------------------------------------------------- | ------------ | ------------------------------------------------------------ |
| B0   | 已提供               | `threadIdx.x` 走行：完全不合并                          | kernel 1     | 0.12（1.5%）                                                 |
| B1   | `sgemm_coalesced`    | 交换 x/y：`threadIdx.x` 走列                            | kernel 2     | 0.98（13%）：只改了一行，快 8 倍                             |
| B2   | `sgemm_smem`         | 32×32 shared memory 分块                                | kernel 3     | 1.23（16%）：提升不大，见讲义 §3.2                           |
| B3   | `sgemm_blocktile_1d` | 每个线程算 8 行（1D 寄存器分块）                        | kernel 4     | 3.9（50%）                                                   |
| B4   | `sgemm_blocktile_2d` | 每个线程算 8×8 + float4 访存 + A 转置存入 shared memory | kernels 5–6  | 7.8（~100%）；cuBLAS 为 7.8                                  |

每一步都按这个循环走：**写 kernel → 改 `implemented` → `test_b_sgemm Bx` → `bench_gemm --only sgemm` → 记录数字 → 用讲义里的 “每次 FMA 需要几次访存” 来解释**。

B3、B4 只接受 tile 整倍数的形状（launcher 会检查），这样 kernel 里不需要处理边界。真实库是怎么处理边界的，见文末可选挑战。

**测量与作图**

```bash
build/lab1/bench_gemm --only sgemm > results/lab1/sgemm.csv
python lab1/scripts/plot_gemm.py results/lab1/sgemm.csv --fp32-peak <your FP32 peak> --out results/lab1/sgemm.png
```

**通过标准**

- `test_b_sgemm` 全部 PASS，`compute-sanitizer` 无报错。
- n=2048 时，B4 ≥ cuBLAS SGEMM 的 **70%**（参考实现约 100%；挑战线 90%）。
- 报告中给出每一步的加速比，并用一句话说明原因，要具体到 “每个线程每次 FMA 读几次 global 或 shared memory”。

---

## Part C：Tensor Core HGEMM（第 2 周后半，约 5 小时）

先读讲义 L1 §4。输入是 fp16，累加和输出是 fp32，这正是大模型推理与训练里 GEMM 的主流配置。

**C1 WMMA（约 1 小时）**：用 `nvcuda::wmma` 的 C++ API。一个 warp 负责 32×32 的 C，即 2×2 个 16×16 的 fragment，直接从 global memory 加载。写完你会发现：这个“什么优化都没做”的 kernel，已经比 Part B 里所有的 FP32 kernel 都快。

**C2 `mma.sync` 三级流水线（约 4 小时，本 lab 的最高一级）**：这一级把 CUTLASS 的结构拆成三个 TODO：

| TODO | 函数               | 做什么                                                                                                    |
| ---- | ------------------ | --------------------------------------------------------------------------------------------------------- |
| C2a  | `c2_load_tile`     | 用 `cp.async` 把 A、B 的一个 K-tile 异步搬进 shared memory，不经过寄存器                                  |
| C2b  | `c2_compute_stage` | 用 `ldmatrix` 把 shared memory 中的数据按 Tensor Core 的 fragment 布局装进寄存器，再执行 16 条 `mma.sync` |
| C2c  | 主循环             | 3 级流水线：计算第 k 块时，第 k+1、k+2 块正在搬运                                                         |

`ptx.cuh` 已经封装好 PTX 指令，注释里写明了每个 lane 拿到哪些元素。难点在于**地址计算**：讲义 §4.3 会一步步从 fragment 布局推导出 `ldmatrix` 的地址公式。建议先画图，再写代码。

调试顺序：先让 `C2_mma_single_tile` 通过（K=32，流水线只有一级），这一步只验证 C2a、C2b；再跑 `C2_mma`，验证 C2c。

**测量**

```bash
build/lab1/bench_gemm --only hgemm > results/lab1/hgemm.csv
```

**通过标准**

- `test_c_hgemm` 全部 PASS。
- n=2048 时，C2 ≥ cuBLAS HGEMM 的 **60%**（参考实现约 85%，WMMA 版约 47%；挑战线 80%）。
- 报告中解释：padding（`BK+8`、`BN+8`）解决了什么问题？`STAGES` 从 3 改成 2 会怎样？实测一下。

**可选挑战（向前沿靠近）**

- **C3 swizzle**：用 XOR swizzle 代替 padding，省下 shared memory。这是 CUTLASS/CuTe 的做法。
- **C4 边界处理**：让 B4 或 C2 支持任意 M/N/K。用 predicated load，或者把边界 tile 单独交给一个慢速 kernel 处理。
- **C5 Hopper**：在 HPC 的 H100 上，按 [Colfax 的 WGMMA 教程](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 和 MIT 6.S894 lab 10，把 C2 改写成 TMA + WGMMA 版本。
- **C6 OJ**：把 C2 提交到 [LeetGPU](https://leetgpu.com/) 或 [GPU MODE](https://github.com/gpu-mode/reference-kernels) 的排行榜。

---

## 提交物

原始 CSV 和图存放在 `results/lab1/`，报告写在 `reports/lab1.md`（[模板](../templates/report.md)）。报告应包含：

1. 环境：GPU 型号（笔记本还是台式机）、SM 数、显存带宽、驱动、CUDA 版本；按讲义 §1.4 算出的 FP32 和 FP16 Tensor 峰值。
2. Part A：带宽表，以及 transpose 三个版本的对比与解释。
3. Part B、C：TFLOP/s 曲线和阶梯图（`plot_gemm.py`）；每一步的加速比，各用一句话解释。
4. 至少一个用 Nsight Compute 看到的现象，比如 bank conflict、occupancy 或寄存器溢出，附截图或数字。
5. 卡住最久的地方。

验收清单见 [grade.md](grade.md)。

> **笔记本 GPU 的测量注意事项**：插上电源，使用高性能模式，关掉其他占用 GPU 的程序。作者实测：满载时功耗稳定在 80 W 的墙上，SM 频率在 2.2–2.6 GHz 间波动（标称最高 3.1 GHz）。因此 n=4096 的数字比 n=2048 低几个百分点，而且波动更大。同一组实验要连续跑完，每个数字至少重复 3 次。性能目标以 n=2048 为准。

## 参考资料

- 讲义：[L1 GPU 执行模型与 CUDA GEMM](../../mlsys-research-course/lectures/L1-cuda-gemm.md)
- Simon Boehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（Part B 的主线读物）· [代码](https://github.com/siboehm/SGEMM_CUDA)
- [MIT 6.S894 Accelerated Computing](https://accelerated-computing.academy/fall25/labs/)：lab 4–6 是 GEMM，硬件和你的 GPU 同属 Ada 一代
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)、[PTX ISA: mma / ldmatrix / cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Modal GPU Glossary](https://modal.com/gpu-glossary)：随时查术语
- [CUTLASS](https://github.com/NVIDIA/cutlass)、[LeetCUDA](https://github.com/xlite-dev/LeetCUDA)（HGEMM 的各种 MMA 写法）
