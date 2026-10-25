---
title: L4 LLM 推理系统
lab: lab4
weeks: 9-11
tags: [lecture, serving, kv-cache, paged-attention, batching, scheduling, prefix-cache]
---

# L4 LLM 推理系统

[← 课程大纲](../syllabus.md) · 上一讲：[L3 编译器](L3-compilers.md) · 配套实验：[lab4](../../labs/lab4/README.md) · 下一讲：[L5 推测解码](L5-speculative.md)

> [!abstract] 这一讲要解决的三个问题
>
> 1. **一个 token 是怎样被生成出来的？** prefill 和 decode 的计算特征为什么截然不同？
> 2. **KV cache 为什么是推理系统的核心资源？** PagedAttention 怎样像操作系统一样管理它？
> 3. **调度器怎样在吞吐量和延迟之间权衡？** continuous batching、chunked prefill、前缀缓存、PD 分离各自解决了什么问题？

本讲的数字都来自 lab4 在 RTX 4060 Laptop 上跑 Qwen3-0.6B 的实测结果。

| 节  | 内容                                                     | 对应 lab    |
| --- | -------------------------------------------------------- | ----------- |
| §1  | 一个 token 怎样流过 Qwen3                                | Part A      |
| §2  | prefill 与 decode；性能的天花板                          | Part A、B   |
| §3  | KV cache 与 PagedAttention                               | Part B1、B2 |
| §4  | 批处理：从 static 到 continuous                          | Part B3     |
| §5  | chunked prefill：别让 decode 被卡住                      | Part C1     |
| §6  | 前缀缓存：别重复算相同的 prompt                          | Part C2     |
| §7  | 演进脉络：PD 分离、Mooncake、KV 压缩、苗老师组的服务系统 | 选题参考    |

---

## §1 一个 token 怎样流过 Qwen3

Qwen3-0.6B：$\mathrm{hidden} = 1024$，28 层，16 个 query head、8 个 KV head（GQA），$\mathrm{head\_dim} = 128$，MLP 中间维度 3072，词表 151936。

```
token ids [T]
  |  embedding                                         [T, 1024]
  v
for each of 28 layers:
  x = RMSNorm(h + residual)  (fused, also updates residual)
  q = x Wq [T, 16*128]   k = x Wk [T, 8*128]   v = x Wv [T, 8*128]
  q, k = RMSNorm per head (QK-norm, new in Qwen3), then RoPE(position)
  write k, v into the KV cache                         <- what makes decoding cheap
  o = attention(q, all cached k, v of this sequence)   [T, 16*128]
  h = o Wo                                             [T, 1024]
  x = RMSNorm(h + residual)
  h = (SiLU(x Wgate) * (x Wup)) Wdown                  [T, 3072] -> [T, 1024]
x = RMSNorm(h + residual);  logits = x Wlm             [T, 151936]
next token = argmax(logits of the last position)
```

**GQA**（grouped-query attention）：16 个 query head 共用 8 组 K/V，每 2 个 query head 读同一组 K/V。KV cache 因此缩小一半。

**为什么要逐处对齐 bf16 舍入**：lab A1 的注释要求在 HF 舍入的位置同样舍入，比如 RMSNorm 在乘 weight 之前先转成 bf16。否则误差在 28 层中逐层累积，最终会改变 greedy 生成的结果。lab 的验收标准是：每个位置的 logits 与 HF fp32 的余弦相似度 $> 0.99$。

---

## §2 prefill 与 decode；性能的天花板

|                | prefill                                   | decode                                              |
| -------------- | ----------------------------------------- | --------------------------------------------------- |
| 每步处理       | 整个 prompt（几百到几千个 token）         | 每个序列 1 个 token                                 |
| GEMM 形状      | $[T, 1024] \times [1024, 3072]$，$T$ 很大 | $[1, 1024] \times [1024, 3072]$，矩阵 $\times$ 向量 |
| 瓶颈           | 计算（Tensor Core）                       | **读权重和 KV cache 的带宽**                        |
| 关心的延迟指标 | **TTFT**（time to first token）           | **TPOT**（time per output token）                   |

### 2.1 decode 的带宽上限

decode 每生成一个 token，都要把**全部权重**读一遍：Qwen3-0.6B 有 1.5 GB（bf16），4060 Laptop 的实测带宽约 213 GB/s（lab1 中 cudaMemcpy 的结果）：

$$
\text{单请求 decode 上限} \approx \frac{213\ \text{GB/s}}{1.5\ \text{GB}} \approx 142\ \text{tokens/s}
$$

lab 实测为 119 tok/s，达到上限的约 84%。剩下的差距来自 kernel 启动开销（每层十几个 kernel，28 层就有约 400 个）和小 kernel 用不满带宽。

**怎样突破这个上限？** 同时 decode 多个序列（batching）：读一遍权重就能服务 $B$ 个 token。在 $B$ 达到 Roofline 拐点之前，吞吐几乎随 $B$ 线性增长。这就是推理系统要做 batching 的根本原因。

### 2.2 KV cache 的显存账

每个 token 的 KV 占用：

$$
\underbrace{2}_{K\text{ and }V} \times \underbrace{28}_{\text{layers}} \times \underbrace{8}_{\text{KV heads}} \times 128 \times 2\ \text{bytes} = 112\ \text{KB}
$$

$$
\begin{aligned}
\text{KV for 1 sequence of 4096 tokens} &= 4096 \times 112\ \text{KB} = 458\ \text{MB} \\
\text{Token capacity of 2 GB KV cache} &= 18{,}700\ \text{tokens} \\
\text{Weights} &= 1.5\ \text{GB}
\end{aligned}
$$

例如可容纳 12 个长度为 $1.5\mathrm{k}$ token 的序列。

8 GB 的 4060 上，扣除权重和激活，剩下能用来存 KV 的显存只有几 GB，并发数直接受它限制。所以 KV cache 是推理系统中**最稀缺的资源**，下面的所有优化都围绕它展开。

---

## §3 KV cache 与 PagedAttention

### 3.1 连续分配的问题

最直接的做法：给每个请求按“最大长度”预留一整段连续显存。问题在于：

```
request A reserves 2048 slots, uses 300      [AAA.........................]   85% wasted
request B reserves 2048 slots, uses 1900     [BBBBBBBBBBBBBBBBBBBBBBBBBBB.]
request C cannot start: no contiguous 2048-slot hole left, though 40% is free
```

- **内部碎片**：输出长度事先不知道，只能按最大值预留；
- **外部碎片**：请求结束后留下的空洞大小不一；
- **无法共享**：相同的前缀（system prompt）各自存一份。

vLLM 论文测得，已有的系统中只有 20–40% 的 KV 显存真正存放了 token。

### 3.2 分页

借鉴操作系统的虚拟内存：把 KV cache 切成固定大小的**块**（lab 中每块 16 个 token），每个序列维护一张 **block table**，记录逻辑块到物理块的映射：

```
physical KV pool (blocks of 16 token slots):
  [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ] ...

seq A (37 tokens): block table [7, 3, 11]     positions 0-15 -> block 7, 16-31 -> 3, 32-36 -> 11
seq B (9 tokens):  block table [2]
token at position p of seq A lives at slot  table[p / 16] * 16 + p % 16
```

- 只有最后一块可能没用满，内部碎片 $< 1$ 块；
- 任何空闲块都能用，没有外部碎片；
- 块带有引用计数，多个序列可以指向同一个物理块（§6）。

代价是 attention kernel 读 K/V 时要多一次查表。lab B2 的 kernel 每处理一个 key 都要计算 `table[j / B] * B + j % B`。

### 3.3 attention kernel：每个 warp 负责一个 query

lab 的朴素版本用一个线程串行完成一个 (token, head) 的计算，用三遍 softmax，因此 K 要读两遍。B2 的版本改成：

```
one warp per (query token, head);  lane l owns dims 4l .. 4l+3 (8 bytes)
for each key j:
  s = warp_sum(q[4l..4l+3] . k_j[4l..4l+3])     # 32 lanes read one 256-byte K row: coalesced
  online softmax update of (m, l, acc[4])         # lab2 B2, one key at a time
out[4l..4l+3] = acc / l
```

lab 实测，decode 为主的负载下吞吐从 19 提升到 224 tok/s，快了 11.8 倍。vLLM 的 paged attention kernel 更进一步：一个 block 处理多个 key，并且做 split-K，也就是 lab2 C2 的 FlashDecoding 思路。

---

## §4 批处理：从 static 到 continuous

### 4.1 static batching

凑齐一批请求，一起 prefill，一起 decode，直到这批中**最长的**那个请求结束，再处理下一批：

```
time ->
req 0  [P][d][d][d]                               (finished, slot idle)
req 1  [P][d][d][d][d][d][d][d][d][d]
req 2  [P][d][d][d][d][d]                          (finished, slot idle)
req 3                                  [P][d]...  waits for the whole batch
```

输出长度差异很大时，大部分槽位都在空转，后来的请求要等整批结束才能开始。

### 4.2 continuous batching（[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)，OSDI'22）

**以迭代为调度单位**：每一步都重新决定这一步跑哪些序列。完成的序列立即退出，新请求立即加入：

```
time ->
req 0  [P][d][d][d]
req 1  [P][d][d][d][d][d][d][d][d][d]
req 2  [P][d][d][d][d][d]
req 3              [P][d][d][d][d][d]              joins as soon as req 0 leaves
req 4                       [P][d][d]...
```

一个关键细节：同一步中，各序列的长度不同，有的处于 prefill 阶段，有的处于 decode 阶段。lab 把这一步所有序列的 token 拼平成一个 $[T]$ 的数组（见 `batch.rs`），GEMM 一次算完，attention 再按各自的 block table 和位置分别处理。这正是 paged KV 让 continuous batching 变得简单的原因：每个序列的 KV 位置互相独立，不需要对齐。

lab 实测（64 个请求，输出长度在 32–224 之间均匀分布）：continuous 的吞吐是 static 的 2.4 倍，TTFT 中位数从 10.9 s 降到 2.7 s。

### 4.3 抢占

请求不断增长，KV 块终会用完。vLLM 的做法是**抢占**最后加入的那个序列：释放它的全部块，把它放回 waiting 队列的最前面，以后重新计算。另一种做法是把 KV 换出到 CPU 内存（swap）。重算的代价是 prefill 的计算量；swap 的代价是 PCIe 带宽。对短序列来说，重算通常更快。

---

## §5 chunked prefill：别让 decode 被卡住

### 5.1 问题：prefill 与 decode 抢占同一步

continuous batching 中，一个 1536 token 的 prefill 加入某一步后，这一步要算 1536 个 token，同一步中处于 decode 阶段的所有序列都要等它，它们的 TPOT 瞬间变成平常的好几倍：

```
step:     1    2    3       4                5    6
decode  [d]  [d]  [d]  [d + 1536 prefill]  [d]  [d]        <- step 4 is 5x longer
TPOT     15   15   15      75               15   15  ms
```

### 5.2 [Sarathi-Serve](https://arxiv.org/abs/2403.02310)（OSDI'24）的解法

把长 prompt 切成多块，每块不超过本步的 token 预算。每一步固定为“所有 decode + 一块 prefill”，步长因此稳定：

```
budget = 256 tokens per step
step:     1         2         3         4         5         6
decode  [d+256p]  [d+256p]  [d+256p]  [d+256p]  [d+256p]  [d+256p]   1536 = 6 chunks
TPOT      25        25        25        25        25        25
```

代价是：prefill 被拆成 6 步，TTFT 可能变长；每块 attention 都要重新读一遍前面已经缓存的 KV。

### 5.3 lab 中的两个发现

1. **准入控制是必需的**：如果只要空闲块够放第一块就接纳请求，就会有太多请求同时处于 prefill 只完成了一半的状态。之后的块需要更多显存，它们互相抢占，抢占次数高达 63–201 次，反而比不分块更慢。正确的做法是：接纳时，空闲块要能装下**整个** prompt。
2. **收益与模型大小有关**：lab 中 chunked prefill 让 TPOT p99 降低了约 13%（$61 \to 53\ \text{ms}$）。论文中 13B 模型的收益要大得多。原因是：0.6B 模型的 decode 步主要受启动开销和权重读取的限制，prefill 块增加的耗时，相对于已经很长的 decode 步耗时并不显著。模型越大，“一个长 prefill 步”和“一个 decode 步”的耗时差得越多，切块的收益也越大。

### 5.4 手算：从请求接纳到抢占重算

**题目说明**

用一个缩小的算例，把 §3 的分页 KV、§4 的连续批处理与抢占、§5 的分块 prefill 和准入控制串起来。

以下均为**手算假设参数**：KV 池共 7 块，每块容纳 4 个 token 位置（lab 中为 16）；每步 token 预算为 6，最多容纳 3 个 Running 请求。开启 continuous batching 和 chunked prefill，关闭前缀缓存。

|请求|Prompt 长度|要生成的输出 token 数|到达时间|
|---|---|---|---|
|A|7|6|第 1 步前|
|B|5|6|第 1 步前|
|C|6|4|第 1 步前|
|D|3|2|第 5 步前|

初始等待队列为 `[A, B, C]`。达到指定输出数量即结束，不考虑提前遇到 EOS。

记当前已有 token 总数为 $L$，已有有效 KV 的位置数为 $c$，未计算量为 $L-c$。采用以下规则：

1. 每步先按接纳顺序处理 `running`，每个请求安排 $\min(L-c,\text{剩余预算})$ 个 token，再按 FIFO 尝试接纳 `waiting`。
    
2. 接纳时，空闲块必须覆盖**完整已有序列尚缺少的块**，实际只分配当前分块需要的空间。队首无法接纳时，停止接纳后续请求。
    
3. Running 请求扩容失败时，抢占队尾请求：释放其 KV、令 $c=0$，保留已有 token，放回 Waiting 队首。
    
4. 每步先调度、后执行。处理完当前全部已有 token 后，才生成一个新输出；新输出的 KV 在后续把它作为输入时计算。结束请求的块在步末释放，下一步可用。
    

求各步的调度计划、预算余量和队列状态，并解释抢占、队首阻塞及重算过程。

**答案**

`A:6` 表示处理 A 的 6 个已有 token，并非生成 6 个输出。空闲块与队列均记录**步末状态**。

|步数|调度计划|剩余预算|空闲块|Running / Waiting|
|---|---|---|---|---|
|1|A:6|0|5|A / B,C|
|2|A:1，B:5|0|3|A,B / C|
|3|A:1，B:1，C:4|0|2|A,B,C / 空|
|4|A:1，B:1，C:2|2|0|A,B,C / 空|
|5|A:1，B:1，C:1|3|0|A,B,C / D|
|6|A:1，B:1|4|1|A,B / C,D|
|7|A:1，B:1|4|7|空 / C,D|
|8|C:6|0|5|C / D|
|9|C:2，D:3|1|4|C,D / 空|
|10|C:1，D:1|4|7|空 / 空|

**第 1～3 步：分块与混合批处理。** A 的 prompt 按 $6+1$ 处理，第 2 步复用已有块的空槽，完成 prefill 并生成首个输出。第 3 步 A、B 做 decode，C 做 prefill。可见，`Running` 表示已经接纳，不要求已经完成 prefill。

**第 4～5 步：两种资源限制。** 第 4 步 A、C 各新增一块，KV 池用满。第 5 步三个请求各处理一个 token，预算还剩 3，但 Running 已达数量上限，D 无法进入。

**第 6 步：抢占与准入控制。** 开始时 A、B、C 分别占 3、2、2 块。B 的计算进度从 8 增至 9，需要新增一块，于是抢占队尾 C。释放 C 的两块、给 B 一块后，只剩一块空闲。C 保留了“6 个 prompt＋2 个输出”，恢复完整序列需要两块，因此不能重新接纳；D 虽只需一块，也被队首 C 挡住。这说明准入检查覆盖完整序列，但并不为未来输出预留空间，后续 decode 仍可能触发抢占。

**第 7～10 步：恢复上下文。** A、B 在第 7 步末结束，C 从第 8 步开始，按 $6+2$ 重算已有序列。第 9 步处理的两个输出 token，其内容早已知道，可以像 prompt 一样一起计算，通过因果 attention 恢复 KV，再生成第 3 个输出。同一步接纳 D，第 10 步两者全部结束。

**计算量验算。** 无抢占时，每个请求需要计算“prompt 长度＋输出数量－1”个位置；最后生成的输出不必再次送入模型：

$$  
(7+6-1)+(5+6-1)+(6+4-1)+(3+2-1)=35  
$$

C 被抢占时丢失了 7 个位置的有效 KV，因此实际计算量为 $35+7=42$ 个 token 位置，与表中处理量之和一致。

---

## §6 前缀缓存：别重复算相同的 prompt

### 6.1 问题

实际服务中，大量请求共享相同的前缀：system prompt、多轮对话的历史、同一篇文档上的多个问题、few-shot 示例。每个请求都把它们 prefill 一遍，是巨大的浪费。

### 6.2 块级前缀缓存（vLLM APC）

为每个**写满**的块计算一个内容 hash，并将它与前一块的 hash 串成链：$H(\text{前一块的 hash}, \text{本块的 16 个 token})$。维护一张表：hash → 物理块。

```
request A: [sys sys sys sys | sys sys sys sys | sys q1 q1 q1 | q1 ...]
             block h1           block h2          block h3
request B: [sys sys sys sys | sys sys sys sys | sys q2 q2 q2 | ...]
             h1 hit -> reuse    h2 hit -> reuse   h3' miss -> compute from here
```

- 采用**链式** hash 的原因：一个块的 KV 不仅取决于它自己的 16 个 token，还取决于它前面的所有 token（attention 会看到它们）。只有前缀完全相同，KV 才相同。
- **最后一个 token 永远不复用**：需要它的 logits 来采样下一个 token，所以至少要重新计算一个 token。
- **延迟释放**：引用计数降到 0 的已缓存块不立即回收，而是进入 LRU 队列，只有显存不够时才被驱逐。下一个使用相同前缀的请求，很可能还能命中它。

lab 实测（512 token 的 prompt 中有 448 token 是共享的）：吞吐提升到原来的 3.1 倍，TTFT 中位数从 2084 ms 降到 658 ms，块命中率 96%。

### 6.3 RadixAttention（[SGLang](https://arxiv.org/abs/2312.07104)，NeurIPS'24）

把 hash 链换成一棵 **token 级的基数树**：

```
            root
             |
     "You are a helpful..."            (shared system prompt)
        /              \
  "Q: capital of"    "Q: translate"
    /      \               \
 "France"  "Japan"       "hello"
```

- 能匹配任意长度的前缀，不必以块为单位；
- 树的结构直接反映请求之间的共享关系，调度器可以优先调度能命中缓存的请求（cache-aware scheduling）；
- 适合多轮对话、树状搜索、agent 等前缀共享很复杂的场景。

lab 可选挑战 C3 要求你把块级 hash 换成基数树。

---

## §7 演进脉络

| 年份  | 工作                                                                                                                                                                                                          | 解决了上一步的什么问题                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 2022  | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI）                                                                                                                                      | 以迭代为调度单位的 continuous batching                                                              |
| 2023  | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)（SOSP）                                                                                                                                             | 分页管理 KV，消除碎片，支持共享；**lab B 的原型**                                                   |
| 2023  | [SpotServe](https://arxiv.org/abs/2311.15566)（ASPLOS'24，苗旭鹏等）                                                                                                                                          | 在会被随时回收的廉价抢占式实例上做推理服务，动态调整并行配置并迁移 KV                               |
| 2024  | [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)（NeurIPS）                                                                                                                                        | token 级的基数树前缀缓存与 cache-aware 调度                                                         |
| 2024  | [Sarathi-Serve](https://arxiv.org/abs/2403.02310)（OSDI）                                                                                                                                                     | chunked prefill，让 decode 不被 prefill 卡住；**lab C1 的原型**                                     |
| 2024  | [DistServe](https://arxiv.org/abs/2401.09670)（OSDI）、[Splitwise](https://arxiv.org/abs/2311.18677)（ISCA）                                                                                                  | **PD 分离**：prefill 和 decode 放在不同的 GPU 上，各自用最合适的并行方式，互不干扰；代价是要传输 KV |
| 2024  | [Helix](https://arxiv.org/abs/2406.01566)（ASPLOS'25，苗旭鹏等）                                                                                                                                              | 在由不同型号 GPU、不同网络组成的异构集群上，用最大流建模来规划模型的放置与请求的路由                |
| 2024  | [Mooncake](https://arxiv.org/abs/2407.00079)（FAST'25）                                                                                                                                                       | 以 KV cache 为中心的分离式架构：KV 存储池跨机共享，调度围绕 KV 的位置展开                           |
| 2024  | [NanoFlow](https://arxiv.org/abs/2408.12757)                                                                                                                                                                  | 在单 GPU 内把计算、访存、通信拆成重叠的流水线                                                       |
| 2023– | KV 压缩：[H2O](https://arxiv.org/abs/2306.14048)、[SnapKV](https://arxiv.org/abs/2404.14469)、[KVQuant](https://arxiv.org/abs/2401.18079)、[PQCache](https://arxiv.org/abs/2407.12820)（SIGMOD'25，苗旭鹏等） | 丢弃不重要的 token，或者把 KV 量化/乘积量化，用少量精度换取大得多的并发                             |
| 2025  | [SGLang 大规模 EP + PD 分离](https://lmsys.org/blog/2025-05-05-large-scale-ep/)                                                                                                                               | 在 DeepSeek-V3 这样的 MoE 模型上，把专家并行与 PD 分离结合起来                                      |

苗老师组的[推理服务综述](https://arxiv.org/abs/2312.15234)把这些工作按“算法”和“系统”两个维度归类，适合作为本讲的延伸阅读。

### 研究问题种子

- **调度的代价模型**：一步的耗时可以近似为 $a \times (\text{prefill token 数}) + b \times (\text{decode 序列数}) + c \times (\text{attention 读取的 KV 总量}) + d$ 吗？用 lab4 的数据拟合出这几个参数，再用模型来选择最优的 chunk 大小，能否比固定值更好？
- **前缀缓存与公平性**：cache-aware 调度会优先调度能命中缓存的请求。没有共享前缀的请求会不会因此被饿死？怎样设计调度策略，在命中率和公平性之间取得平衡？
- **小模型上的 PD 分离**：lab 中 chunked prefill 对 0.6B 模型的收益很小。PD 分离对小模型是否也没有意义？分离的收益与 $\text{prefill 步耗时} / \text{decode 步耗时}$ 之间是什么关系？

## 可视化资料汇总

| 资料                                                                                             | 形式                                                                   | 配合哪一节   |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------ |
| [Aleksa Gordić: Inside vLLM](https://www.aleksagordic.com/blog/vllm)                             | 长文，逐模块图解 vLLM v1：调度器、KV 管理器、前缀缓存、chunked prefill | **全讲主线** |
| [vLLM 发布博客](https://blog.vllm.ai/2023/06/20/vllm.html)                                       | PagedAttention 的分块与共享**动画**                                    | §3           |
| [Anyscale: continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) | static 与 continuous 的时间线图                                        | §4           |
| [HuggingFace: continuous batching](https://huggingface.co/blog/continuous_batching)              | 从零推导 continuous batching                                           | §4           |
| [Lequn Chen: Dissecting batching](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/)    | batching 的性能分析                                                    | §2、§4       |
| [SGLang 博客](https://lmsys.org/blog/2024-01-17-sglang/)                                         | RadixAttention 基数树的逐步示意图                                      | §6           |
| [DistServe 博客](https://hao-ai-lab.github.io/blogs/distserve/)                                  | goodput 与 PD 分离的图示                                               | §7           |
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)                                         | 约 1200 行 Python，结构与 lab4 几乎一一对应                            | 对照阅读     |

## 自测清单

- [ ] 能画出一层 Qwen3 的数据流，并说出 QK-norm、GQA 各自起什么作用
- [ ] 能算出单请求 decode 的带宽上限，并解释 batching 为什么能突破它
- [ ] 能算出给定模型每个 token 的 KV 字节数
- [ ] 能说明 PagedAttention 消除了哪两类碎片，以及它付出了什么代价
- [ ] 能画出 static 与 continuous batching 的时间线，说明后者为什么依赖 paged KV
- [ ] 能解释 chunked prefill 改善的是 TPOT，它的代价是什么，以及为什么需要准入控制
- [ ] 能说明前缀缓存为什么要用链式 hash，以及为什么最后一个 token 必须重新计算
- [ ] 能说出 PD 分离要解决的问题和它要付出的代价
