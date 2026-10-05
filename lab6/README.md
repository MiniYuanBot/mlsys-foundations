# Lab 6：集合通信与数据并行训练 —— 从 Sendrecv 到 ring allreduce、DDP 重叠与 ZeRO

> 配套讲义：[L6 集合通信与并行训练](../../mlsys-research-course/lectures/L6-distributed.md)
> 预计用时：2 周 × 约 7 小时　·　语言：C++17 + MPI（Open MPI）　·　硬件：任意多核 CPU，多进程运行在同一台机器上（WSL / Mac 均可）；可选在 HPC 多卡上跑 NCCL

一个只有单卡的学生怎么学分布式？其实集合通信的**算法**和**代价模型**与硬件无关：ring allreduce 在 8 个 CPU 进程之间和在 8 张 GPU 之间，走的是同一套步骤；α-β 模型也同样适用。差别只在 α 和 β 的数值上。这个 lab 用 MPI 的点对点原语（Send、Recv、Sendrecv）从头写出集合通信，测出每种算法的适用范围，再用它们实现数据并行训练中最重要的两项优化：**梯度 bucket 与计算重叠**（PyTorch DDP），以及**优化器状态分片**（ZeRO-1）。

| 部分 | 你要写什么 | 难度梯度 | 对应的真实系统 |
|---|---|---|---|
| **A** 点对点与基线 | ring shift；“星形” allreduce（全部汇到 rank 0） | ① 填空，熟悉 MPI | — |
| **B** 集合通信算法 | ring reduce-scatter / all-gather / allreduce（支持不整除的分块）；recursive doubling；α-β 模型拟合 | ② 核心算法 + 性能目标 | NCCL、Gloo、MPI 库的 allreduce |
| **C** 数据并行训练 | DDP 式的梯度 bucket（与反向传播重叠）；ZeRO-1（用你的 reduce-scatter + all-gather） | ③ 论文机制精简复现 | [PyTorch DDP](https://arxiv.org/abs/2006.15704)（VLDB'20）、[ZeRO](https://arxiv.org/abs/1910.02054)（SC'20） |

做完后你应当能回答：

- ring allreduce 的通信量为什么与 P 几乎无关？它的代价在哪里？
- 什么时候 recursive doubling 比 ring 快？交叉点能用 α-β 模型预测吗？
- bucket 太大或太小分别有什么问题？ZeRO-1 节省了什么，又付出了什么？

---

## 0. 构建

```bash
# WSL / Linux: sudo apt install openmpi-bin libopenmpi-dev   (tools/setup_wsl.sh already did)
# macOS:       brew install open-mpi
cmake -S lab6 -B build/lab6 && cmake --build build/lab6 -j
ctest --test-dir build/lab6 --output-on-failure        # runs every test with P = 1, 2, 3, 4, 8 ranks
mpirun -np 3 build/lab6/test_coll B1                   # one rank count, tests whose name contains B1
```

每个测试在所有 rank 上同时运行，只有所有 rank 都通过才算通过。代码一旦死锁（比如所有 rank 都卡在 `MPI_Send` 里），CTest 会在 120 秒后超时。

## 1. 代码地图

```
lab6/
+-- include/coll/coll.hpp     Comm, chunk_range, every collective (read first)
+-- include/coll/train.hpp    MLP with a flat gradient buffer, CommThread, Bucketer, Zero1Adam
+-- src/coll_core.cpp  src/train_core.cpp      provided
+-- src/coll_student.cpp      TODO(A1, A2, B1a-c, B2)
+-- src/train_student.cpp     TODO(C1, C2)
+-- tests/test_coll.cpp tests/test_train.cpp tests/mpitest.hpp
+-- bench/bench_coll.cpp      allreduce sweep + ping-pong
+-- bench/train.cpp           data-parallel training step breakdown
+-- scripts/alpha_beta.py     fit alpha-beta, compare with measurements
+-- hpc/                      optional: NCCL on PKU HPC (unverified template)
```

---

## Part A：点对点（约 1.5 小时）

| TODO | 内容 | 要点 |
|---|---|---|
| A1 | `ring_shift` | 用 `MPI_Sendrecv`：同时发给右邻居、收左邻居的数据 |
| A2 | `allreduce_star` | rank 0 依次接收 P−1 份、求和，再逐个发回 |

> ⚠️ 为什么一定要用 `MPI_Sendrecv`：如果每个 rank 都先 `MPI_Send` 再 `MPI_Recv`，消息较大时 Send 会等对方开始接收才返回。所有人都卡在 Send 里，形成环形等待，也就是死锁。`B1c_allreduce_ring_large` 用 16 MB 的消息专门检查这一点。

## Part B：集合通信算法（约 5 小时）

读讲义 §1–§3。**动手前，先在纸上画出 P=3 时 ring reduce-scatter 每一步的分块流向**（讲义 §2.2 有 P=4 的图）。

| TODO | 内容 | 要点 |
|---|---|---|
| B1a | `reduce_scatter_ring` | 第 s 步：发出块 (r−s−1) mod P，收到块 (r−s−2) mod P 并累加；结束后 rank r 持有块 r 的总和 |
| B1b | `allgather_ring` | 第 s 步：发出块 (r−s) mod P，收到块 (r−s−1) mod P |
| B1c | `allreduce_ring` | 两行：B1a + B1b |
| B2 | `allreduce_rd` | partner = rank XOR mask，每步交换整个缓冲区；P 不是 2 的幂时抛出异常 |

分块大小可以不相等（`chunk_range`）：n=7、P=3 时，三块分别是 3、2、2 个元素。测试会用 1、2、3、7、65537 这些不整除的长度来检查。

**测量**

```bash
mkdir -p results/lab6
mpirun -np 2 build/lab6/bench_coll --pingpong > results/lab6/pingpong.csv
mpirun -np 4 build/lab6/bench_coll > results/lab6/allreduce_p4.csv
mpirun -np 8 build/lab6/bench_coll > results/lab6/allreduce_p8.csv
python lab6/scripts/alpha_beta.py results/lab6/pingpong.csv results/lab6/allreduce_p4.csv --out results/lab6/alpha_beta.png
```

参考实现在 WSL（i9 Alder Lake，进程间走共享内存）上的结果：α ≈ 0.24 µs，β 约为 19 GB/s。P=4 时：

| 消息大小 | MPI_Allreduce | ring | recursive doubling | star |
|---|---|---|---|---|
| 4 B | 0.6 µs | 1.7 µs | **0.6 µs** | 0.6 µs |
| 64 KB | 15 µs | 21 µs | 22 µs | 41 µs |
| 4 MB | 718 µs | **592 µs** | 1390 µs | 2161 µs |
| 64 MB | 36.6 ms | **31.6 ms** | 74.0 ms | 75.9 ms |

**通过标准**

- `coll_np1` 到 `coll_np8` 全部 PASS。
- 消息 ≥ 1 MB 时，ring 的耗时 ≤ `MPI_Allreduce` 的 1.3 倍（参考实现在 1 MB 以上反而比 MPI 快 10–15%）。
- 消息 ≤ 256 B 时，recursive doubling 比 ring 快（参考实现快约 2.5 倍）。
- 报告中给出你的 α、β，用模型预测 ring 与 RD 的交叉点，并与实测对比。模型预测的耗时约为实测的 1/2 到 1/3，模型漏掉了什么？提示：讲义 §2.4 的 γ 项，以及缓存。

## Part C：数据并行训练（约 5 小时）

读讲义 §4–§5。`train.hpp` 中的 MLP 把所有梯度放在**一个连续的缓冲区**里，按反向传播完成的先后排列（最后一层在最前），所以“已就绪的梯度”总是缓冲区的一个前缀。`CommThread` 在后台线程上依次对提交的区间做 allreduce，训练线程同时继续算反向传播，这就是重叠。

| TODO | 内容 |
|---|---|
| C1 | `Bucketer`：每当一层的梯度就绪就累加；超过 bucket 容量就提交一个区间；`finish()` 提交剩余部分、等待完成、除以 P |
| C2 | `Zero1Adam`：reduce-scatter 梯度 → 本 rank 的块求平均 → 只在本块上做 Adam（m、v 只有 1/P 大小）→ all-gather 参数 |

`test_train` 用一个小 MLP 检查等价性：P 个 rank、每个 B 个样本训练出的参数，必须与单进程、P×B 个样本训练出的结果一致。

**测量**（3 个 1024 宽的隐藏层，约 3.2M 参数，12 MB 梯度）

```bash
for b in 16 64; do
  mpirun -np 4 build/lab6/train --mode sync    --batch $b
  for mb in 0.25 1 4 16; do mpirun -np 4 build/lab6/train --mode overlap --bucket-mb $mb --batch $b; done
  mpirun -np 4 build/lab6/train --mode zero1   --batch $b
done
```

参考实现（P=4，batch 16）：

| 模式 | step | compute | comm_wait | optimizer | 每个 rank 的优化器状态 |
|---|---|---|---|---|---|
| sync | 23.1 ms | 8.0 | 6.2 | 9.6 | 24 MB |
| overlap, 4 MB bucket | 21.4 ms | 8.8 | **3.7** | 9.8 | 24 MB |
| overlap, 16 MB bucket（一个 bucket） | 26.8 ms | 8.0 | 9.2 | 13.1 | 24 MB |
| zero1 | **15.3 ms** | 8.3 | — | 7.4（含 RS + AG） | **6 MB** |

**通过标准**

- `train_np1`、`train_np2`、`train_np4` 全部 PASS。
- 最好的 bucket 大小下，overlap 的 comm_wait ≤ sync 的 **70%**（参考实现约 60%）。
- ZeRO-1 每个 rank 的优化器状态 = sync 的 1/P。

**报告必答**

1. 这个 MLP 有 4 层，按反向顺序，梯度大小分别为 64 KB、4 MB、4 MB、4 MB。用这个数据解释：为什么 0.25 MB 和 4 MB 的 bucket 都产生 3 个 bucket，而 16 MB 只产生 1 个？为什么只有 1 个 bucket 时比 sync 还慢？
2. 在你的数据上，ZeRO-1 为什么是最快的？它的总通信量与 sync 相同（RS + AG = 一次 allreduce）。提示：比较 optimizer_ms 列。在 GPU 训练中，这个结论还成立吗？
3. 本机测得的 comm_wait 很嘈杂，波动可达 ±30%。说明你是怎样得到可信数字的：重复了几次、取什么统计量。

**可选挑战**

- **C3 ZeRO-2 / ZeRO-3**：梯度也只保留本块（ZeRO-2）；参数也分片，前向时再 all-gather（ZeRO-3 / FSDP）。测量额外的通信。
- **C4 张量并行**：把一个 Linear 层按列切到 P 个 rank 上（Megatron），前向需要一次 all-gather，反向需要一次 reduce-scatter。
- **C5 HPC 上的 NCCL**：见 [hpc/README.md](hpc/README.md)。

---

## 提交物与参考

报告写在 `reports/lab6.md`（[模板](../templates/report.md)），验收清单见 [grade.md](grade.md)。

- 讲义 [L6](../../mlsys-research-course/lectures/L6-distributed.md)
- Andrew Gibiansky, [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)：ring allreduce 的经典图解
- Thakur, Rabenseifner, Gropp, [Optimization of Collective Communication Operations in MPICH](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf)：α-β 代价模型与各种算法的出处
- [PyTorch DDP 论文](https://arxiv.org/abs/2006.15704)（bucket 与重叠）、[ZeRO](https://arxiv.org/abs/1910.02054)
- HuggingFace [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：DP、ZeRO、TP、PP 的交互图解
- [UCSD CSE 234 PA2](https://github.com/hao-ai-lab/cse234-w25-PA)：用 mpi4py 实现集合通信与张量并行，可以对照
