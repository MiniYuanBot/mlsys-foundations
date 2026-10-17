---
title: L5 推测解码
lab: lab5
weeks: 12
tags: [lecture, speculative-decoding, specinfer, rejection-sampling, token-tree]
---

# L5 推测解码

[← 课程大纲](../syllabus.md) · 上一讲：[L4 推理系统](L4-serving.md) · 配套实验：[lab5](../../labs/lab5/README.md) · 下一讲：[L6 集合通信与并行训练](L6-distributed.md)

> [!abstract] 这一讲要解决的三个问题
>
> 1. **为什么“猜几个 token 再一起验证”能变快？** 这个“免费午餐”来自硬件的哪一点？
> 2. **怎样做到无损？** greedy 解码时逐 token 相同；采样时，输出分布与原模型**完全一致**。
> 3. **草稿从哪里来、怎样组织？** n-gram、小模型、额外的预测头（Medusa、EAGLE）、token tree（SpecInfer）各有什么取舍？

| 节  | 内容                                      | 对应 lab      |
| --- | ----------------------------------------- | ------------- |
| §1  | 为什么验证 $k$ 个 token 和生成 1 个一样快 | Part A        |
| §2  | greedy 推测与期望加速公式                 | Part A3、报告 |
| §3  | 拒绝采样：无损的证明                      | Part B        |
| §4  | token tree 与 tree attention（SpecInfer） | Part C        |
| §5  | 草稿从哪里来：演进脉络                    | 选题参考      |

---

## §1 为什么验证 $k$ 个 token 和生成 1 个一样快

L4 §2 算过：decode 每一步都要把全部权重读一遍。对 Qwen3-1.7B 来说，这是 3.4 GB，在 4060 上约需 16 ms。lab5 实测 plain 解码为 56 tok/s，约 17.8 ms/token。这一步里 GEMM 的形状是 $[1, d] \times [d, d']$，算术强度约为 1 FLOP/Byte，远低于 Roofline 的拐点。

如果一次送进 $k+1$ 个 token，GEMM 就变成 $[k+1, d] \times [d, d']$：读的权重一样多，计算量变成 $k+1$ 倍，但计算单元本来就闲着。只要 $k+1$ 远小于拐点处的批大小（4060 上约为一百多），**耗时几乎不变**：

```
time of one target forward pass vs number of tokens fed (bandwidth-bound regime)

 time |
      |                                          ___----   compute-bound
      |                                 ___---
      |________________________________/                  <- flat: weights are read once
      +----+----+----+----+----+----+----+----+----> tokens fed
      1    2    4    8   16   32   64  128
```

推测解码就利用了这段平坦区域：先用便宜的办法猜出 $k$ 个 token，再用一次目标模型 forward 验证全部。猜对几个，就省下几次 forward。

---

## §2 greedy 推测与期望加速

### 2.1 一轮推测

```
committed: ... The cat sat          (last = "sat", its KV not yet computed)
draft:     on   the  mat             (k = 3 guesses)
target fed [sat, on, the, mat] in ONE pass, argmax at each position:
           on   the  rug  .
accept "on" (== on), "the" (== the), reject "mat" (!= rug), bonus = "rug"
new tokens: on the rug               (3 tokens from 1 target pass)
KV cache: positions of sat, on, the are valid; the KV written for "mat" is stale
          -> truncate; the next round overwrites it
```

不管猜对多少，每一轮至少产出 1 个 token（bonus），所以推测解码每次目标 forward 产出的 token 数**永远不会少于 plain**，只可能因为草稿成本而变慢。

lab A3 测试会检查：推测解码的输出与 plain greedy 逐 token 相同。

> [!note] 为什么偶尔不是 100% 相同
> 一次验证 $k+1$ 个 token 时，cuBLAS 可能为 $[k+1, d]$ 和 $[1, d]$ 两种形状选择不同的算法，bf16 的舍入顺序因此不同。在 logits 恰好打平的位置，argmax 可能翻转。lab 中 code prompt 在第 152 个 token 处分叉，就是这个原因。这不是算法错误，而是浮点的不确定性。

### 2.2 期望加速公式

设每个位置被接受的概率是 $\alpha$（假设各位置独立），草稿长度为 $k$。一轮产出的 token 数期望为：

$$
E[\text{tokens}] = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1-\alpha^{k+1}}{1-\alpha}
$$

设草稿模型单步的耗时是目标模型的 $c$ 倍，一轮的耗时约为 $1 + kc$（一次目标 forward 加 $k$ 步草稿）：

$$
S = \frac{1-\alpha^{k+1}}{(1-\alpha)(1+kc)}
$$

代入 lab5 的实测数据（0.6B 作草稿、1.7B 作目标，$c \approx 0.47$，$\alpha \approx 0.8$）：

| $k$ | $E[\text{tokens}]$ | $1 + kc$ | 预测 $S$ | 实测 $S$（chat） |
| --- | ------------------ | -------- | -------- | ---------------- |
| 2   | 2.44               | 1.94     | 1.26     | 1.20             |
| 3   | 2.95               | 2.41     | 1.22     | 1.14             |
| 5   | 3.69               | 3.35     | 1.10     | 1.03             |

$c = 0.47$ 的草稿太贵了。即便 $\alpha$ 高达 0.8，加速也只有约 1.2 倍。要想快，必须让 $c$ 很小：

- **n-gram / prompt lookup**：$c \approx 0$，但只有在文本会重复上文时 $\alpha$ 才高。lab 实测，copy 上 2.85 倍，code 上 2.05 倍，chat 上只有 1.24 倍；
- **EAGLE 的草稿头**：只有 1 层 Transformer，$c \approx 0.05$，$\alpha$ 仍然很高，这正是 EAGLE 能达到 3 倍以上加速的原因；
- **大小差距悬殊的模型对**：比如 70B 配 1B，$c \approx 0.02$。

---

## §3 拒绝采样：无损的证明

### 3.1 规则

采样解码时，目标模型在某个位置的分布是 $p$，草稿模型是 $q$，草稿 token $x \sim q$：

1. 以概率 $\min(1, p(x)/q(x))$ **接受** $x$；
2. 否则**拒绝**，从残差分布 $r(y) \propto \max(0, p(y) - q(y))$ 中重新采样一个 token，然后结束这一轮；
3. $k$ 个都被接受时，再从目标模型在下一个位置的分布中采样一个 bonus token。

### 3.2 证明：输出的分布恰好是 $p$

只需证明单个位置的情况，多个位置可以用归纳法推广。输出 token 为 $y$ 的概率由两部分组成：

$$
P(\text{out}=y) = \underbrace{q(y)\min\left(1,\frac{p(y)}{q(y)}\right)}_{\text{draft } y \text{ accepted}} + \underbrace{P(\text{reject})}_{\beta} \cdot r(y)
$$

第一项等于 $\min(q(y), p(y))$。拒绝的总概率为：

$$
\beta = 1 - \sum_x \min(p(x), q(x)) = \sum_x \max(0, p(x)-q(x))
$$

后一个等号成立，是因为 $\sum p = \sum q = 1$，所以 $\sum(p - \min(p, q)) = \sum \max(0, p - q)$。这恰好是残差分布 $r$ 的归一化常数，所以第二项等于 $\max(0, p(y) - q(y))$。两项相加：

$$
P(\text{out}=y) = \min(p(y), q(y)) + \max(0, p(y)-q(y)) = p(y) \qquad \blacksquare
$$

```
probability mass of each token y:

   p(y)  |######             accepted part = min(p, q)      (the overlap)
   q(y)  |#########          over-proposed: q > p -> rejected sometimes
         |
   p(y)  |#########          residual part = p - q           (only when p > q)
   q(y)  |####               under-proposed -> made up by resampling from r
```

lab B1 的测试用两条马尔可夫链，生成 20 万条长度为 3 的序列，参考实现的经验分布与精确分布的最大误差为 0.0017。

> [!question]- 自测 3.1：如果 $q = p$（草稿和目标完全相同），会发生什么？
> $\min(1, p/q) = 1$，所有草稿都被接受。lab 测试 `b1_identical_models_accept_everything` 验证的就是这一点。

> [!question]- 自测 3.2：如果拒绝后不从残差 $r$ 采样，而是直接从 $p$ 采样，会怎样？
> 输出分布变成 $\min(p, q) + \beta \cdot p \ne p$：那些被过度提议（$q > p$）的 token 会被重复计入，分布就偏了。残差采样之所以必要，正是为了补上 $p > q$ 那部分欠缺的概率质量。

> [!question]- 自测 3.3：temperature 应该加在哪里？
> $p$ 和 $q$ 都要用同一个 temperature 计算：$p = \operatorname{softmax}(z_{\mathrm{target}} / T)$，$q = \operatorname{softmax}(z_{\mathrm{draft}} / T)$。拒绝采样保证输出服从 $p$，也就是“目标模型在温度 $T$ 下的分布”。草稿用多少温度，只影响接受率，不影响正确性。

---

## §4 token tree 与 tree attention（SpecInfer）

### 4.1 链的问题

链式草稿中，第 1 个 token 一旦猜错，后面的猜测全部作废。可是草稿模型往往“差一点就猜对”：正确答案常常是它的第 2 或第 3 选择。

### 4.2 [SpecInfer](https://arxiv.org/abs/2305.09781)（ASPLOS'24，苗旭鹏等）

把草稿组织成一棵**树**，每层保留几个候选，目标模型一次 forward 验证整棵树。例如，分支宽度为 $[3, 2, 1]$ 时，节点数为：

$$
1 + 3 + 6 + 6 = 16
$$

```
Verified in ONE target pass

                       sat (root)
             /          |           \
           on          down         by
          /   \        /   \       /   \
        the    a     on    and   the   her
         |     |      |     |     |     |
        mat   rug    the   lay  door  side

target argmax:  after "sat" -> "on", after "on" -> "the", after "the" -> "rug"
accepted path:  on -> the,  bonus "rug"   (the chain [on, the, mat] gives the same here,
                                           but a chain [down, ...] would have lost everything)
```

关键在于 **tree attention**：每个节点只能看到前缀和它自己的祖先，不能看到兄弟和其他分支。实现时，用一个祖先位掩码代替因果掩码：

```
node:      0(sat) 1(on) 2(down) 3(by) 4(the) 5(a) ...
mask[4] =  bit 0 | bit 1 | bit 4          (sat, on, the)
mask[5] =  bit 0 | bit 1 | bit 5          (sat, on, a)
```

节点 $i$ 的 RoPE 位置和 KV 存储位置分别为：

$$
\begin{aligned}
\text{RoPE position}(i) &= \mathrm{prefix\_len} + \operatorname{depth}(i) \\
\text{KV storage}(i) &= \mathrm{prefix\_len} + i
\end{aligned}
$$

兄弟节点共享一个 RoPE 位置，每个节点都有独立的 KV 存储槽位。

接受一条路径之后，把路径上节点的 KV 从“存储位置”搬到“最终位置”（lab 提供的 `commit_tree_path`）。由于 RoPE 用的是深度而不是存储下标，搬动后的位置编码仍然正确。

lab C1 要你在 lab4 的 warp 级 attention 上加入这个掩码，C2 实现沿树的 greedy 接受规则。

### 4.3 lab 中的发现：树接受得更多，但不一定更快

|                                          | draft $k=3$  | tree $[3,2,1,1]$ |
| ---------------------------------------- | ------------ | ---------------- |
| 每次目标 forward 产出的 token 数（chat） | 2.91         | **3.84**         |
| 草稿 forward 次数                        | 198          | 249              |
| 实测加速                                 | $1.14\times$ | $1.05\times$     |

树确实使每次验证接受的 token 更多，但构建树需要更多次草稿 forward（每层一次）。在 $c \approx 0.47$ 时，这部分开销抵消了收益。SpecInfer 论文中的设置是：草稿模型比目标模型小得多（$c$ 很小），并且多个请求共享 GPU，这时树的收益才明显。**系统优化的收益依赖于具体的成本结构**，这正是 lab 要你用公式验证的原因。

---

## §5 演进脉络：草稿从哪里来

| 年份  | 工作                                                                                                          | 草稿来源                          | 解决了上一步的什么问题                                                       |
| ----- | ------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------- |
| 2023  | [Leviathan et al.](https://arxiv.org/abs/2211.17192)（ICML）、[Chen et al.](https://arxiv.org/abs/2302.01318) | 独立的小模型                      | 提出推测解码和无损的拒绝采样                                                 |
| 2023  | **[SpecInfer](https://arxiv.org/abs/2305.09781)**（ASPLOS'24，苗旭鹏等）                                      | 多个小模型 + token tree           | 用树组织候选，一次验证多个分支；tree attention；树上的多步推测采样           |
| 2023  | [Prompt lookup](https://github.com/apoorvumang/prompt-lookup-decoding)                                        | 上文中的 n-gram                   | 零成本草稿，适合代码编辑、RAG 等复制上文的场景                               |
| 2024  | [Medusa](https://arxiv.org/abs/2401.10774)（ICML）                                                            | 目标模型上额外的预测头            | 不需要单独的草稿模型；多个头分别预测 $+1$、$+2$、$+3$ 位置的 token，组合成树 |
| 2024  | [Lookahead](https://arxiv.org/abs/2402.02057)（ICML）                                                         | Jacobi 迭代 + n-gram 池           | 完全不需要训练，也不需要草稿模型                                             |
| 2024  | [EAGLE](https://arxiv.org/abs/2401.15077) / [EAGLE-2](https://arxiv.org/abs/2406.16858)                       | 在**特征层**上自回归的 1 层草稿头 | 在特征层预测比在 token 层更容易；EAGLE-2 根据置信度动态地扩展树              |
| 2025  | [EAGLE-3](https://arxiv.org/abs/2503.01840)                                                                   | 融合多层特征 + training-time test | 训练时就模拟推理时的多步草稿，解决了草稿越长越不准的问题                     |
| 2024– | [MagicDec](https://github.com/Infini-AI-Lab/MagicDec)                                                         | 稀疏 KV 的自推测                  | 长上下文中瓶颈在读 KV 而不在读权重，用“只看部分 KV 的自己”做草稿             |

另一个方向是**系统层面**：在批量推理中，batch 越大，GPU 越接近计算受限，§1 的“平坦区域”就越短，推测的收益也越小。怎样根据负载动态决定是否推测、推测多长，是生产系统（vLLM、SGLang）中一个仍在研究的问题。

### 研究问题种子

- **4060 上的最优草稿**：$c \approx 0.47$ 的 0.6B 草稿收益很小。用 Qwen3-1.7B 的前几层做自推测，$c$ 能降到多少，$\alpha$ 又会降到多少？在什么层数时 $S$ 最大？
- **n-gram 加树**：n-gram 草稿的 $c \approx 0$。如果用多个 n-gram 匹配组成一棵树，能否在 chat 场景中也提高 $\alpha$？
- **batch 与推测的权衡**：在 lab4 的引擎中加入推测解码。batch 增大时，推测从有益变成有害的转折点在哪里？能否用 L4 的步耗时模型预测这个点？

## 可视化资料汇总

| 资料                                                                                                               | 形式                           | 配合哪一节 |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------ | ---------- |
| [Google: Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) | 博客，原作者对历史和直觉的回顾 | §1、§2     |
| [Jay Mody: Speculative Sampling](https://jaykmody.com/blog/speculative-sampling/)                                  | 约 50 行 NumPy 实现 + 推导     | §3         |
| [HF: Assisted Generation](https://huggingface.co/blog/assisted-generation)                                         | 博客，带动画                   | §2         |
| [Lookahead decoding 博客](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)                                   | **Jacobi 迭代的 GIF 动画**     | §5         |
| [Medusa 项目页](https://sites.google.com/view/medusa-llm)                                                          | 多头预测与树注意力的示意图     | §4、§5     |
| [EAGLE 项目页](https://sites.google.com/view/eagle-llm)                                                            | 特征层草稿的示意图             | §5         |

## 自测清单

- [ ] 能用 Roofline 解释为什么验证 $k$ 个 token 的耗时和生成 1 个差不多
- [ ] 能手动走一遍 greedy 推测的一轮，包括 KV 的回滚
- [ ] 能写出期望加速公式，并用它解释“草稿太贵”的问题
- [ ] 能完整写出拒绝采样无损性的证明
- [ ] 能画出一棵 token tree 的祖先掩码，并说明 RoPE 位置为什么用深度
- [ ] 能比较 n-gram、小模型、Medusa、EAGLE 这几种草稿的 $c$ 和 $\alpha$
