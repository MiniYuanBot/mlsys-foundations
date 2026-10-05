# 候选研究题目

每个题目都从**前面 lab 中实际测到的一个缺口**出发：你已经有基线代码、测量工具，并且知道问题确实存在。题目按方向分组，★ 越多表示离苗旭鹏老师组的方向越近。每题都给出了**第一天就能做的实验**和**止损条件**：如果第一天的实验就否定了问题的前提，换题，不要硬做。

> 选题前先做 30 分钟检索：在 arXiv、Google Scholar 和 [Hsword 的论文清单](https://github.com/Hsword/Awesome-Machine-Learning-System-Papers) 中搜索关键词，看问题是否已经被解决。被解决了也不要紧：复现它、找出它的适用边界，同样是有价值的工作。

---

## 编译器与 kernel

### T1 让超优化器的代价模型排对顺序 ★★★
- **缺口**：lab3 的 `bench_search` 中，代价模型排第一的候选实测只排第二，实测最快的那个模型排第四。
- **问题**：只加两三个解析项（序言在每个 N-tile 上的重复计算、行 kernel 对同一行的多次读取、wave quantization），能否让模型的前三名与实测一致？在多大比例的 kernel 上成立？
- **起点**：lab3 的 `estimate` 与 `bench_search`；再增加 5–10 个图（layernorm+matmul、gated MLP、attention 变体）。
- **基线**：当前的 Roofline 模型；“实测全部候选”作为上界（oracle）。
- **主指标**：前三名的排序一致率（Kendall τ）；被选中候选的实测耗时 / oracle 最快耗时。
- **第一天**：给 `estimate` 加上序言重复计算这一项，重跑 `bench_search`，看排序有没有变对。
- **止损**：加了三项之后排序仍然随机，说明主要误差来自缓存行为，题目要改成测量研究。
- **相关**：[Mirage](https://arxiv.org/abs/2405.05751)、[TVM/Ansor 的学习型代价模型](https://arxiv.org/abs/2006.06762)、[TLP](https://arxiv.org/abs/2211.03578)。

### T2 tinycomp 中的 Row + MatMul 融合（自动发现 FlashAttention）★★★
- **缺口**：lab3 能找到“重算 + 两个 kernel”的方案，但无法把 softmax 的归约与 matmul 放进同一个 kernel。
- **问题**：增加一个“在线归约 + matmul”模板和一条对应的改写规则，搜索能否自动生成单 kernel 的 attention？在 softmax_matmul 之外的图（如 log-sum-exp 加 matmul、带 mask 的变体）上是否也能发现？
- **起点**：lab3 的 `codegen_core`（新增模板）和 `search_student`（新增规则）；lab2 B2 是正确性与性能的参照。
- **基线**：lab3 当前的最优方案；lab2 的 Triton FA2。
- **第一天**：手写目标 kernel 的生成代码，确认 NVRTC 能编译，并且结果与 `evaluate` 一致。
- **止损**：模板只对 softmax_matmul 一种图有效，没有通用性，题目就不成立。
- **相关**：[Mirage](https://arxiv.org/abs/2405.05751) §thread-block graph、[FlexAttention](https://pytorch.org/blog/flexattention/)。

### T3 Ada GPU 上 HGEMM 的 tile 选择模型 ★
- **缺口**：lab1 的 C2 只用了一种 tile 配置（128×128×32、3 级流水），在不同形状上的表现未知。
- **问题**：在 4060 上，用简单的解析模型（occupancy、wave quantization、算术强度）选 tile，能否在 decode 形状（M ≤ 16）和 prefill 形状上都接近 cuBLAS？
- **起点**：lab1 C2，模板化 BM、BN、BK、STAGES；扫描 50 个形状。
- **第一天**：在 M = 1、8、16、128、4096 上对比 C2 与 cuBLAS，确认 M 小时差距确实大。
- **相关**：CUTLASS profiler、[stream-K](https://arxiv.org/abs/2301.03598)。

## 推理服务

### T4 按步耗时模型自适应选择 chunk 大小 ★★
- **缺口**：lab4 中 chunked prefill 只降低了约 13% 的 TPOT p99，而 chunk 大小是凭感觉定的（256、512）。
- **问题**：用 lab4 的数据拟合出步耗时模型 t = a·(prefill token 数) + b·(decode 序列数) + c·(KV 读取量) + d，再用它在每一步选择 chunk，使 TPOT 满足 SLO 的前提下 TTFT 最小。比固定 chunk 好多少？
- **起点**：lab4 的 `scheduler_student`、`bench`；在 engine 中记录每一步的组成和耗时。
- **基线**：固定 chunk 256、512、2048；[Sarathi-Serve](https://arxiv.org/abs/2403.02310) 的固定 token 预算策略。
- **主指标**：满足 TPOT p99 ≤ X ms 时的 TTFT p50，以及 goodput。
- **第一天**：记录 1000 个步的组成和耗时，拟合模型，看 R² 是否 > 0.9。
- **止损**：模型拟合得很差（R² < 0.7），说明步耗时还受到别的因素影响，先研究这些因素。
- **相关**：[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[DistServe](https://arxiv.org/abs/2401.09670)（goodput 定义）。

### T5 前缀缓存感知调度的公平性 ★★
- **缺口**：lab4 的前缀缓存让吞吐提升了 3.1 倍；如果再优先调度能命中缓存的请求，吞吐还能更高，但不共享前缀的请求可能被饿死。
- **问题**：设计一个兼顾命中率和等待时间的调度策略（例如按“等待时间 × 命中收益”打分），在混合负载下测量吞吐与尾延迟的帕累托前沿。
- **起点**：lab4 的 `scheduler_student`（waiting 队列的排序）、`bench`（增加混合负载）。
- **基线**：FIFO；“最长前缀优先”（SGLang 的 LPM）。
- **第一天**：实现 LPM，在 50% 共享、50% 不共享的负载下，测量不共享请求的 TTFT p99 是否显著变差。
- **相关**：[SGLang](https://arxiv.org/abs/2312.07104) §cache-aware scheduling、[Preble](https://arxiv.org/abs/2407.00023)。

### T6 4060 + Qwen3 的最优草稿：自推测 ★★★
- **缺口**：lab5 中，0.6B 草稿的成本比 c ≈ 0.47，加速被限制在 1.2 倍左右。
- **问题**：用 Qwen3-1.7B 自己的前 k 层（加上共享的 lm_head）作为草稿，c 和 α 怎样随 k 变化？是否存在一个 k，让加速超过 n-gram 在 chat 上的 1.24 倍？
- **起点**：lab5 的 `Session`（增加只跑前 k 层的 forward）；用 lab5 的期望加速公式预测。
- **基线**：plain、n-gram、0.6B 草稿、tree。
- **第一天**：对 k = 4、8、14，测量 α（用 tokens_per_target_call 反解）和 c，代入公式，预测最优 k。
- **止损**：所有 k 的预测 S 都 < 1.1，说明不经训练的浅层太弱，题目要改成“训练一个 1 层草稿头”（需要更多算力）。
- **相关**：[LayerSkip](https://arxiv.org/abs/2404.16710)、[Draft & Verify](https://arxiv.org/abs/2309.08168)、[SpecInfer](https://arxiv.org/abs/2305.09781)、[EAGLE](https://arxiv.org/abs/2401.15077)。

### T7 批量推理中什么时候应该关掉推测 ★★★
- **缺口**：lab5 只测了单请求。batch 增大时，GPU 越来越接近计算受限，推测的收益会减小，直至变成负收益。
- **问题**：把 lab5 的推测解码接入 lab4 的引擎，测量 batch = 1、4、16、64 时的收益，并用 lab4 的步耗时模型预测转折点。能否按负载动态开关推测？
- **起点**：lab4 的 `engine`（每个 decode 序列一次喂入 k+1 个 token）、lab5 的验证逻辑。
- **基线**：始终开启；始终关闭。
- **第一天**：不改调度器，只测量 batch 为 B、每序列 k+1 个 token 时的步耗时，看平坦区域在哪里结束。
- **相关**：[SpecInfer](https://arxiv.org/abs/2305.09781)（批量推测）、[TurboSpec](https://arxiv.org/abs/2406.14066)（按负载闭环控制推测长度）。

## 分布式训练

### T8 用动态规划求最优 bucket 划分 ★★
- **缺口**：lab6 的 bucket 大小是固定的（PyTorch 默认 25 MB），4 MB 和 0.25 MB 在 lab 中产生了同样的划分。
- **问题**：已知每层反向传播的耗时和 α-β 参数，用动态规划求出使暴露通信最小的划分。在层大小不均匀的模型（如 Transformer：attention、MLP、embedding）上，比固定大小好多少？
- **起点**：lab6 的 `Bucketer`（改成按给定的划分提交）、`train`（记录每层反向耗时）。
- **基线**：固定 1、4、25 MB；“每层一个 bucket”。
- **第一天**：记录每层的反向耗时和通信耗时，在纸上用 DP 算出最优划分，预测收益。
- **止损**：预测收益 < 3%，说明在这个规模上 bucket 划分不重要，换到更不均匀的模型或改题。
- **相关**：[PyTorch DDP](https://arxiv.org/abs/2006.15704) §bucket、[ByteScheduler](https://dl.acm.org/doi/10.1145/3341301.3359642)。

### T9 层级 allreduce 与拓扑感知（需要 HPC）★★
- **缺口**：lab6 只在单机的共享内存上测量。多卡、多节点时，链路是异构的（NVLink 与 InfiniBand）。
- **问题**：在 HPC 的 2 个节点 × 4 卡上，比较扁平 ring、层级（节点内 reduce-scatter → 节点间 allreduce → 节点内 all-gather）与 NCCL 默认算法；用 α-β 模型预测各自的优劣。
- **起点**：lab6 的 `hpc/`、nccl-tests；用 CUDA + NCCL 重写 lab6 的 ring。
- **相关**：[Thakur et al.](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf)、[Galvatron](https://arxiv.org/abs/2211.13878)（异构的代价建模）、[Helix](https://arxiv.org/abs/2406.01566)。

---

## 自己的题目

也可以提出自己的题目。用同样的格式写清楚：缺口（附上你测到的一张图）、问题、起点、基线、主指标、第一天的实验、止损条件、最接近的 3 项工作。写不出“第一天的实验”的题目，通常还不够具体。
