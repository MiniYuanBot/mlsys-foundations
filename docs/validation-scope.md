# 脚手架验证范围

本页记录**脚手架**（提供的代码、测试与 benchmark）在什么环境下验证过。它不是学生完成 lab 的证明。

验证方法：作者先在仓库外写出完整的参考实现，确认全部测试通过、性能达到 README 中写的数字，再把学生部分替换成抛出 `TODO ... is not implemented yet` 的桩，并确认此时测试全部失败。参考实现不随仓库分发。

## lab0（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | Windows 11，MinGW-w64 GCC 13.2，CMake 3.29，`-O3 -march=native -fopenmp-simd` | 无警告 |
| 参考实现 + 测试 | 同上 | partA/partB/partC 全部通过 |
| 学生桩 + 测试 | 同上 | partA/partB/partC 全部失败（只有 `B0_naive_reference_sanity` 通过，这是预期的） |
| `bench_gemm` 参考实现 | Intel i9 Alder Lake（24 线程），n=1024 | naive 0.85，reordered 31，blocked 47，packed 113，parallel(24t) 369；n=2048 时 parallel 605 GFLOP/s |
| BLAS 基线 | NumPy 2.5 + MKL 2025 | 1 线程 129，24 线程 650 GFLOP/s |
| `bench_fusion` 参考实现 | 同上，n=2^24 | affine：unfused 340 ms，tiled 37 ms；gelu：unfused 2.7 s，tiled 0.79 s |
| 参考实现 + 测试 | WSL2 Ubuntu 24.04，GCC 13.3，CMake 3.28 | 全部通过；`bench_gemm` n=1024：packed 123，parallel(24t) 494 GFLOP/s |
| `MT_SANITIZE=ON`（ASan + UBSan） | 同上 | 全部通过，无报告 |

**未验证**：MSVC、Apple clang（arm64/NEON）。如果你在这些平台上遇到问题，请记录编译器版本与报错。

## lab1（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | WSL2 Ubuntu 24.04，CUDA 12.6，GCC 13.3，CMake 3.28，sm_89 | 参考实现与学生桩都无警告；所有 kernel 都没有寄存器溢出（B4 用 109 个寄存器，C2 用 102 个） |
| 参考实现 + 测试 | RTX 4060 Laptop（24 SM，32 MB L2，驱动 566.24） | partA/partB/partC 全部通过 |
| compute-sanitizer | 同上 | memcheck：三个测试程序均为 0 errors；racecheck（A3、B2、C2）：0 hazards |
| 学生桩 + 测试 | 同上 | 三组全部失败；只有提供的部分通过（A0、A3 naive、B0 naive/cuBLAS、C0 cuBLAS） |
| `bench_basics` 参考实现 | 同上，n=2^26 | cudaMemcpy 213 GB/s；vector_add 233，reduce_sum 240；transpose 8192²：naive 115 / tiled 226 / padded 232；2048²：122 / 414 / 953 |
| `bench_gemm` 参考实现 | 同上，n=2048，TFLOP/s（占 cuBLAS 的 %） | FP32：naive 0.12（1.5%）、coalesced 0.98（13%）、smem 1.23（16%）、1d 3.9（50%）、2d 7.8（~100%）、cuBLAS 7.8；FP16：wmma 13.9（47%）、mma 25.1（85%）、cuBLAS 29.5 |
| 功耗与频率 | 同上 | 满载时功耗卡在 80 W，SM 频率 2.2–2.6 GHz；n=4096 的结果比 n=2048 低 5–10% |

**未验证**：Windows 原生构建。CUDA 12.6 与 VS 2026（MSVC 14.51）不兼容，`cudafe++` 会崩溃；VS 2022（MSVC 14.3x）应当可以，但没有测试。也没有在其他 GPU（RTX 30、A100、H100）上验证过。

## lab2（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 C1 共享库 | WSL2，CUDA 12.6，sm_89 | 参考实现与学生桩都无警告；C1 kernel 用 123 个寄存器、27 KB smem，没有溢出 |
| 参考实现 + pytest | RTX 4060 Laptop，PyTorch 2.14.1+cu126，Triton 3.8.0 | 96 项全部通过 |
| 学生桩 + pytest | 同上 | 95 项失败，只有 `test_C1_rejects_bad_shape` 通过（参数检查在 TODO 之前执行） |
| softmax（4096×4096） | 同上 | torch.softmax 241 GB/s；A1 226（94%）；A2 234（97%）；unfused 60 |
| attention D=64、S=4096 | 同上，B=4、H=16 | SDPA flash 28.8 TFLOP/s；Triton B2 27.5（95%）；CUDA C1 27.1（94%）；SDPA math 0.32 |
| causal、S=4096 | 同上 | SDPA 27.8；Triton 25.4（91%，耗时为非 causal 的 54%）；CUDA 26.3（95%） |
| D=128、S=4096 | 同上 | 默认 64/32/4w：67%；128/64/8w：98%（作为 B4 调参题） |
| decode B=1、H=8、S=32768 | 同上 | 不切分 86 GB/s；自动切分 193 GB/s（2.3×）；SDPA 213 |

**未验证**：其他 GPU；Triton 解释器模式（`TRITON_INTERPRET=1`）：A1 全部通过；B1 中 S=1 的 3 项因 Triton 3.8 解释器自身的 bug（`ConstTensorWrapper` 缺少 `untyped_storage`）失败，同样的用例在 GPU 上通过。

## lab3（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | WSL2，GCC 13.3，CUDA 12.6（NVRTC 与 driver API，不需要 nvcc） | 参考实现与学生桩都无警告 |
| 参考实现 + 测试 | RTX 4060 Laptop | test_a 9/9、test_b 12/12、test_c 9/9 |
| 学生桩 + 测试 | 同上 | 全部失败，只有 `C0_canonical_distinguishes_sharing_from_copies` 通过（它只测已提供的代码） |
| `bench_fusion` | 8192×4096 | softmax：unfused 5 个 kernel 55 GB/s → fused 1 个 236；layernorm 36 → 239；rmsnorm 63 → 227；matmul_bias_gelu 4096×1024×4096：15.2 → 6.5 ms（5.3 TFLOP/s） |
| `bench_search` | softmax_matmul 8192×8192×64 | unfused 11.9 ms；搜索找到 6 个等价候选；模型排第一的实测 6.6 ms（实测第二），实测最快 5.8 ms（模型第四） |

## lab4（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | WSL2，Rust 1.99，cudarc 0.17.8，CUDA 12.6（NVRTC 与 cuBLAS） | 参考实现与学生桩都无警告 |
| 参考实现 + `cargo test --release` | RTX 4060 Laptop，Qwen3-0.6B（bf16） | 33/33：kernels 8、model 2、kv 10、scheduler 9、engine 4 |
| 与 HF transformers 5.18（fp32）对齐 | 同上，4 个 prompt（7–241 个 token） | teacher forcing 下最小余弦相似度 0.998–0.9999，top-1 一致率 0.996–1.0；greedy 24 个 token：3 个 prompt 全同，1 个前 13 个相同 |
| 学生桩 + 测试 | 同上 | 32 项失败，只有 `b0_paged_attention_naive_reference` 通过 |
| 单请求 decode | 同上 | 119 tok/s（带宽上限约 142） |
| bench：attention | 16 个请求 × 1024 + 64 | naive 19 → fast 224 tok/s（11.8×） |
| bench：batching | 64 个请求 × 256 + 32–224（输出长度随机），max_seqs 16 | static 363 → continuous 858 tok/s（2.4×）；TTFT p50 10.9 → 2.7 s |
| bench：前缀缓存 | 64 个请求 × 512（共享 448）+ 32 | 431 → 1325 tok/s（3.1×）；TTFT p50 2084 → 658 ms；块命中率 0.96 |
| bench：chunked prefill | 40 个请求 × 1536 + 128，Poisson 到达，rate = 2 | 不分块（2048）TPOT p99 61 ms；分块 256：53 ms，0 次抢占。没有准入控制时，分块会出现 63–201 次抢占，反而更慢（已写入 README） |

**未验证**：Qwen3-1.7B 的 HF 对齐（`make_reference.py --model Qwen3-1.7B` 可以生成参考，但测试默认只跑 0.6B）；与 vLLM 的对比（需要更新驱动）。

## lab5（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | WSL2，Rust 1.99，依赖 lab4 的**参考实现** | 参考实现与学生桩都无警告 |
| 参考实现 + `cargo test --release` | RTX 4060 Laptop，Qwen3-1.7B + 0.6B | verify 6/6、spec 5/5；拒绝采样的经验分布误差 0.0017；greedy 的 n-gram、draft、tree 都是 64/64 个 token 与 plain 相同 |
| 学生桩 + 测试（链接 lab4 参考实现） | 同上 | 11 项全部失败 |
| lab4 扩展点（tree attention 接口与额外 kernel 注册）之后重测 lab4 | 同上 | lab4 33/33 仍然通过 |
| bench（192 个 token） | plain 56 tok/s | n-gram k=8：chat 1.24×、code 2.05×、copy 2.85×；draft k=2 约 1.2×；tree [3,2,1,1] 每次 forward 产出 3.84–4.47 个 token，但加速只有 1.05–1.17×；采样 T=0.8 k=3：1.04–1.07×；code 上 greedy 在第 152 个 token 处因 bf16 打平而分叉 |

## lab6（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | WSL2，GCC 13.3，Open MPI 4.1.6 | 参考实现与学生桩都无警告 |
| 参考实现 + `ctest` | 同上，P = 1、2、3、4、8（test_coll）与 1、2、4（test_train） | 8/8 次运行全部通过 |
| 学生桩 + 测试 | 同上 | 8/8 次运行失败；只有提供的 chunk_range、MPI 基线、C0 同步 DDP 通过 |
| ping-pong | 2 个 rank，共享内存 | α ≈ 0.24 µs，β ≈ 19 GB/s，n_1/2 ≈ 4.5 KB |
| allreduce（P=4） | 同上 | 4 B：RD 0.6 µs、ring 1.7 µs；4 MB：ring 592 µs、MPI 718 µs、RD 1390 µs、star 2161 µs；α-β 模型约为实测的 1/2 到 1/3 |
| 训练（P=4，batch 16，3.2M 参数） | 同上 | sync 23.1 ms（comm_wait 6.2）；overlap 4 MB bucket 21.4 ms（comm_wait 3.7）；zero1 15.3 ms，优化器状态 6 MB（sync 为 24 MB）；comm_wait 在多次运行之间波动约 ±30% |

**未验证**：macOS（Open MPI 通过 Homebrew 安装）；`hpc/` 下的 NCCL 脚本（没有 HPC 账号）。

## 尚未重构的 lab

`legacy/` 下是重构前的 lab1–lab7（新 lab1 已替代旧 lab1 的 CUDA 部分），只用于迁移参考，不保证可以运行。
