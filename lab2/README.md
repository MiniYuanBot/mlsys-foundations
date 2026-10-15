# Lab 2：Attention kernel —— 从 online softmax 到 FlashAttention

> 配套讲义：[L2 Attention：从 online softmax 到 FlashAttention](../../mlsys-research-course/lectures/L2-attention.md)（先读 §1–§2 再开始 Part A）
> 预计用时：2 周 × 约 7 小时　·　语言：Triton（Python 语法的 GPU DSL）+ CUDA C++　·　硬件：RTX 4060（WSL2）

Attention 是 LLM 中最重要的算子：

$$\text{out} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

直接照公式计算，要把 S×S 的分数矩阵写进显存。S=8192 时，每个 head 就要 256 MB，而且 8 GB 显存只能“读一遍、写一遍”地来回搬它。FlashAttention 的核心想法是：**把 softmax 改写成可以分块、在线计算的形式，让分数矩阵永远只存在于片上存储里**。这个 lab 带你一步步推导并写出它。

| 部分                              | 你要写什么                                                                          | 难度梯度              | 对应的真实系统                                                                                                        |
| --------------------------------- | ----------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **A** Softmax                     | 融合的行 softmax；任意长度的 online softmax（同时求 log-sum-exp）                   | ① 填空，熟悉 Triton   | PyTorch `softmax` kernel、Triton 教程 02                                                                              |
| **B** FlashAttention-2（Triton）  | 单 tile 版 → 分块循环 + online 重缩放 → causal mask + 跳过空块 → 调参               | ② 核心实现 + 性能目标 | [FlashAttention-2](https://arxiv.org/abs/2307.08691)、Triton 教程 06                                                  |
| **C** 手写 Tensor Core + 推理解码 | C1：用 `mma.sync` 写 CUDA 版 FA2 前向（P 留在寄存器里）；C2：FlashDecoding split-KV | ③ 前沿设计的精简复现  | FlashAttention-2 的 CUDA 内循环、[FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)、FlashInfer |

做完后你应当能回答：

- online softmax 为什么是**精确**的，而不是近似？
- FlashAttention 省下的是计算还是访存？为什么它反而要做更多次 exp？
- decode 阶段为什么要切分 KV？什么时候切分没有用？

---

## 0. 环境与构建

用 `tools/setup_wsl.sh` 装好的虚拟环境（PyTorch 2.14 + Triton 3.8）即可，在 WSL 里执行：

```bash
mlsys                                              # alias: activates ~/venvs/mlsys
cd /mnt/d/__Coding__/github/mlsys-research-labs   # or wherever the repo is
cmake -S lab2 -B build/lab2 && cmake --build build/lab2 -j    # Part C1 shared library
python -m pytest lab2/tests -q                     # all fail at first, except one argument check
```

常用命令：

```bash
python -m pytest lab2/tests/test_b_flash.py -k B2 -v   # only tests whose name contains B2
TRITON_INTERPRET=1 python -m pytest lab2/tests/test_a_softmax.py -k A1   # run Triton on the CPU interpreter:
                                                                          # slow, but print() and pdb work
```

（Triton 3.8 的解释器有个 bug：S=1 这类单元素张量会报 `ConstTensorWrapper ... untyped_storage`。这是解释器自身的问题，用 `-k` 跳过这些用例即可。）

Triton 速查：`tl.program_id`、`tl.arange`、`tl.load`/`tl.store`（配合 `mask=`、`other=`）、`tl.max`/`tl.sum`（指定 `axis`）、`tl.dot`（至少 16×16）、`tl.where`、`tl.exp2`。完整文档见 [Triton language](https://triton-lang.org/main/python-api/triton.language.html)。

## 1. 代码地图

```
lab2/
+-- attn/
|   +-- reference.py          provided: fp32 PyTorch references (the tests use them)
|   +-- softmax_student.py    TODO(A1, A2)
|   +-- flash_student.py      TODO(B1, B2, B3) + launchers
|   +-- decode_student.py     TODO(C2a, C2b) + launcher
|   +-- cuda.py               provided: ctypes bridge to the C1 shared library
+-- csrc/
|   +-- flash_mma_student.cu  TODO(C1a, C1b, C1c)
|   +-- ptx.cuh               provided: ldmatrix / mma.sync wrappers (same as lab1)
+-- tests/                    pytest, one file per part
+-- bench/                    bench_softmax, bench_attention, bench_decode
```

规则：只改 `*_student.*`。每完成一个 TODO，就把对应的 `XX_DONE = False` 改成 `True`（CUDA 文件中是 `constexpr bool C1_DONE`）。

---

## Part A：Softmax（第 1 周前半，约 2.5 小时）

读讲义 §1（为什么要减去最大值）和 §2（online softmax 的推导）。

| TODO | 内容                    | 要点                                                       |
| ---- | ----------------------- | ---------------------------------------------------------- |
| A1   | `softmax_row_kernel`    | 一个 program 处理一整行；padding 位置填 `-inf`             |
| A2   | `softmax_online_kernel` | 第一遍维护 (m, l) 并重缩放，第二遍写出结果；同时写出 `lse` |

```bash
python lab2/bench/bench_softmax.py > results/lab2/softmax.csv
```

**通过标准**

- `test_a_softmax.py` 全部 PASS。
- 4096×4096 时，A1 ≥ `torch.softmax` 带宽的 **80%**，A2 ≥ 70%。参考实现分别为 94% 和 97%，unfused 的 PyTorch 写法只有约 25%。
- 报告中说明：A2 读了输入两遍，为什么在 1024×16384 时只比 A1 慢约 20%，而不是慢一倍？（提示：L2 有 32 MB。）

---

## Part B：FlashAttention-2 in Triton（第 1 周后半 + 第 2 周前半，约 6 小时）

读讲义 §3。四个 TODO 一级比一级难，**请按顺序做**：

| TODO | 内容                     | 要点                                                           | 测试   |
| ---- | ------------------------ | -------------------------------------------------------------- | ------ |
| B1   | 单 tile 版本（S ≤ 64）   | 不需要 online：一次算完 `softmax(QKᵀ)V`；熟悉 `tl.dot` 和 mask | `B1_*` |
| B2   | 分块循环 + online 重缩放 | 维护 `m_i`、`l_i`、`acc`；每个块都用 `alpha` 重缩放旧结果      | `B2_*` |
| B3   | causal                   | 先加 mask，确认正确后，再把循环上界改成对角线所在的块          | `B3_*` |
| B4   | 调参（无需写新代码）     | 为 D=128 找到一组 `block_m/block_n/num_warps`                  | 见下   |

调试技巧：`test_B2_peaky_scores_need_rescaling` 专门检查 alpha 重缩放（分数很尖锐时，每个块的最大值都不同）。B2 中如果只有这一个测试失败，说明重缩放写错了。

```bash
python lab2/bench/bench_attention.py > results/lab2/attn_d64.csv
python lab2/bench/bench_attention.py --causal --seqs 1024,4096,8192 > results/lab2/attn_causal.csv
python lab2/bench/bench_attention.py --D 128 --seqs 1024,4096 > results/lab2/attn_d128.csv
```

**通过标准**

- `test_b_flash.py` 全部 PASS。
- D=64、S=4096 时，B2 ≥ PyTorch SDPA FlashAttention 后端的 **70%**（参考实现 95%）；causal 的耗时 ≤ 非 causal 的 60%（参考实现约 54%）。
- **B4**：默认配置下，D=128 只能达到 SDPA 的约 67%。用 `flash_attention(..., block_m=, block_n=, num_warps=)` 做扫描，找到 ≥ 90% 的配置（存在能达到 98% 的配置），把它写进 `_block_sizes`，并在报告里用寄存器数和 occupancy 解释为什么这组配置更好。
- 与 `sdpa_math`（标准 PyTorch 实现，会物化 S×S 矩阵）比较，并解释 S=4096 时它慢了约 100 倍的原因。

---

## Part C：手写 Tensor Core FA2 + FlashDecoding（第 2 周，约 6 小时）

### C1 CUDA FlashAttention-2（约 4 小时，本 lab 的最高一级）

读讲义 §5。这一步把 lab1 Part C（`ldmatrix`、`mma.sync`、fragment 布局）和 Part B 的 online softmax 结合起来，全部在**寄存器**上完成。范围限定为 D=64、S 为 64 的倍数，每个 block 由 4 个 warp 负责 64 行 query。

| TODO | 内容                      | 关键                                                                                                                                     |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| C1a  | S = QKᵀ                   | K 在 shared memory 中按 `[key][d]` 存放，正好就是 mma B 操作数需要的布局，不需要转置；一条 `ldmatrix_x4` 覆盖两个 n-tile                 |
| C1b  | 寄存器上的 online softmax | 每个线程持有两行（g 和 g+8）的部分元素；行最大值和行和都要在 quad（4 个线程）之间用 `__shfl_xor_sync` 归约                               |
| C1c  | O += PV                   | **FA2 的关键技巧**：两个相邻 m16n8 累加器的布局，恰好等于一个 m16k16 的 A fragment，所以 P 直接在寄存器里转成 fp16，不经过 shared memory |

```bash
cmake --build build/lab2 -j && python -m pytest lab2/tests/test_c_cuda.py -v
```

先让 `test_C1_one_tile` 通过：只有一个 K tile 时不需要跨 tile 重缩放，可以单独验证 C1a 和 C1c。

### C2 FlashDecoding（约 2 小时）

读讲义 §6。decode 阶段每个序列只有一个新 token，`B×H` 个 program 远远填不满 GPU。把 KV 切成 `NSPLIT` 段并行处理，再用 log-sum-exp 合并结果。

| TODO | 内容                                                                             |
| ---- | -------------------------------------------------------------------------------- |
| C2a  | `decode_split_kernel`：在一段 KV 上做 online softmax，写出这一段的部分输出和 lse |
| C2b  | `decode_combine_kernel`：用 lse 加权合并各段结果                                 |

```bash
python lab2/bench/bench_decode.py > results/lab2/decode.csv
```

**通过标准**

- `test_c_cuda.py` 和 `test_c_decode.py` 全部 PASS；`compute-sanitizer --tool memcheck python -m pytest lab2/tests/test_c_cuda.py` 报告 0 errors。
- C1：D=64、S=4096 时 ≥ SDPA FlashAttention 后端的 **70%**（参考实现 94%，causal 95%）。
- C2：B=1、H=8、S=32768 时，自动切分比不切分（`num_splits=1`）快 ≥ **1.5×**（参考实现 2.3×），且带宽 ≥ cudaMemcpy 的 80%（参考实现 193 GB/s，lab1 测得的 memcpy 为 213 GB/s）。
- 报告中说明：为什么 B=16、H=32 时切分没有带来收益？

**可选挑战**

- **C3 FA2 反向传播**：Triton 实现，用 `lse` 重算 P，参考 CS336 作业 2 和 Triton 教程 06。
- **C4 C1 加 cp.async 双缓冲**：把 lab1 C2 的流水线套到 K/V 的加载上。
- **C5 paged decode**：让 C2 的 K/V 按 block table 间接寻址，这是 lab4 paged attention 的预习。

---

## 提交物

原始数据和图存放在 `results/lab2/`，报告写在 `reports/lab2.md`（[模板](../templates/report.md)），验收清单见 [grade.md](grade.md)。报告要包括：

1. 环境（`bash tools/check_env.sh` 的输出）；
2. 三张表：softmax 带宽、attention 的 TFLOP/s（D=64、causal、D=128）、decode 的 GB/s；
3. 上面各部分提出的问题；
4. 卡住最久的地方。

## 参考资料

- 讲义：[L2](../../mlsys-research-course/lectures/L2-attention.md)
- [FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Triton 教程：Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)、[Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)、[Triton-Puzzles](https://github.com/srush/Triton-Puzzles)（热身）
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)（约 100 行 CUDA，不用 Tensor Core，可与 C1 对比）
- [FlashDecoding 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)
