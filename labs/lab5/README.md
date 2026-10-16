# Lab 5：specdec —— 推测解码：n-gram、草稿模型与 SpecInfer token tree

> 配套讲义：[L5 推测解码](../../course/lectures/L5-speculative.md)
> 预计用时：1 周 × 约 7 小时　·　语言：Rust + CUDA　·　模型：Qwen3-1.7B（目标模型）+ Qwen3-0.6B（草稿模型）　·　**前置：lab4 的 Part A、B 已完成**（lab5 直接调用你的引擎）

decode 是访存受限的：读一遍 3.4 GB 的权重才产出 1 个 token。然而，让目标模型**一次验证 k 个 token**，代价几乎和生成 1 个一样，因为权重同样只读一遍。推测解码就利用了这一点：先用便宜的办法猜出几个 token，再让大模型一次性检查。只要验证规则设计得当，输出与不做推测**完全相同**：greedy 解码时逐 token 一致；采样时服从同一个分布。

| 部分                      | 你要写什么                                                                      | 难度梯度                   | 对应的论文或系统                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **A** greedy 推测         | n-gram 草稿（prompt lookup）、链式 greedy 验证、一轮推测加 KV 回滚              | ① 填空                     | [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding)、[Lookahead](https://arxiv.org/abs/2402.02057) |
| **B** 草稿模型 + 拒绝采样 | 保证输出分布不变的 speculative sampling、采样模式下的一轮推测                   | ② 核心算法（带正确性证明） | [Leviathan et al.](https://arxiv.org/abs/2211.17192)（ICML'23）                                                                |
| **C** token tree          | tree attention kernel（在 lab4 kernel 的基础上加祖先掩码）、tree 的 greedy 验证 | ③ 前沿复现                 | **[SpecInfer](https://arxiv.org/abs/2305.09781)**（ASPLOS'24，苗旭鹏等）                                                       |

做完后你应当能回答：

- 为什么验证 k 个 token 的耗时和生成 1 个差不多？k 变大以后，这个说法什么时候不再成立？
- 拒绝采样为什么能保证输出分布与目标模型**完全一致**，而不只是近似？
- 0.6B 的草稿模型为什么只带来约 1.2 倍的加速，而“免费”的 n-gram 草稿在代码上能达到 2 倍？

---

## 0. 构建

lab5 是一个依赖 `../lab4` 的 Rust crate。`cargo` 会编译**你的** lab4 代码，所以你的 lab4 必须先通过 Part A、B 的测试。目标模型 Qwen3-1.7B 要另外下载（约 4 GB）：

```bash
bash tools/setup_env.sh --lab 5 && bash tools/check_env.sh --lab 5   # from labs/
```

```bash
cd lab5
cargo test --release --test verify          # CPU only: A1, A2, B1, C2
cargo test --release --test spec            # GPU: C1 kernel and end-to-end (loads 1.7B + 0.6B, ~5 GB)
cargo run --release --bin bench -- --tokens 192 > ../results/lab5/bench.csv
```

## 1. 代码地图

```
lab5/
+-- kernels/
|   +-- spec_provided.cu              gather/scatter of KV slots (tree commit)
|   +-- tree_attention_student.cu     TODO(C1)
+-- src/
|   +-- session.rs    provided: one sequence + one model + one KV cache; feed, truncate,
|   |                 feed_tree, commit_tree_path (read the invariant at the top)
|   +-- verify_student.rs   TODO(A1, A2, B1, C2)   pure CPU, unit-tested without a GPU
|   +-- spec_student.rs     TODO(A3, B2)           a speculative round on the GPU
|   +-- tree.rs decode.rs   provided: tree building with the draft, decoding loops
|   +-- ops5.rs rng.rs      provided
|   +-- bin/bench.rs
+-- tests/verify.rs  tests/spec.rs
```

**Session 的不变量**（所有的轮次都要维护它）：`num_computed == tokens.len() - 1`。最后一个 token 已经采样出来，但还没有送进模型。一轮推测会把 `[last, d1, ..., dk]` 一次性送进模型，这些位置的 KV 也都写进了 cache；被拒绝的那部分 KV 用 `truncate` “忘掉”即可，下一轮会覆盖它们。

---

## Part A：greedy 推测（约 2.5 小时）

| TODO | 内容                                                                                       |
| ---- | ------------------------------------------------------------------------------------------ |
| A1   | `ngram_propose`：在已有文本中找到最近一次出现的后缀 n-gram，把它后面的 k 个 token 作为草稿 |
| A2   | `greedy_accept`：逐个比较草稿与目标模型的 argmax，得到接受的个数加一个 bonus token         |
| A3   | `greedy_round`：feed → argmax → accept → 更新 tokens → `truncate`                          |

`tests/spec.rs` 中的 `a3_*` 会检查：推测解码的输出与普通 greedy 解码**逐 token 相同**，并且目标模型的 forward 次数确实减少了。

## Part B：草稿模型与拒绝采样（约 2 小时）

| TODO | 内容                                                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------------------------------------- |
| B1   | `speculative_sample`：以 min(1, p/q) 的概率接受草稿 token；被拒绝时从残差分布 max(0, p − q) 中采样；全部接受时再从 p 多采一个 |
| B2   | `sampling_round`：草稿模型逐个采样 k 个 token 并记下 q，目标模型一次给出 p，然后调用 B1                                       |

`b1_speculative_sampling_is_exact` 用两个小词表的马尔可夫链，分别充当“目标模型”和“草稿模型”，采样 20 万次，比较三元组的经验分布与目标模型的精确分布。参考实现的最大误差是 0.0017。**做完 B1 再读讲义 §3 的证明**，你会更清楚每一步为什么必不可少。

## Part C：SpecInfer token tree（约 2.5 小时）

链式草稿的问题在于，只要第一个 token 猜错，后面全部作废。SpecInfer 让草稿给出**一棵树**（每层保留几个候选），目标模型用 **tree attention** 一次验证所有分支：每个节点只看前缀和它自己的祖先。

| TODO | 内容                                                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------ |
| C1   | `tree_attention` kernel：从你 lab4 的 `paged_attention` 改出来；key 的集合 = 前缀的所有位置 + `tree_mask` 中置位的树节点 |
| C2   | `tree_accept`：从根出发，每一步走到 token 等于目标模型 argmax 的那个子节点                                               |

`c1_tree_attention_matches_causal_attention_on_paths` 对树上的每个节点，把它的路径重新排成连续的序列，再用 lab4 的因果 kernel 算一遍作为对照。树的构建（用草稿模型逐层取 top-k，每层一次 tree forward）和接受之后的 KV 搬移（`commit_tree_path`）都已提供，请读懂它们。

---

## 测量与通过标准

```bash
cargo run --release --bin bench -- --tokens 192 > ../results/lab5/bench.csv
```

参考实现在 RTX 4060 Laptop 上的结果（target 1.7B 单独解码为 56 tok/s）：

| 方法             | chat  | code  | copy      | 每次目标模型 forward 产出的 token 数（copy） |
| ---------------- | ----- | ----- | --------- | -------------------------------------------- |
| n-gram k=8       | 1.24× | 2.05× | **2.85×** | 3.10                                         |
| draft k=2        | 1.20× | 1.20× | 1.19×     | 2.59                                         |
| draft k=5        | 1.03× | 1.12× | 1.11×     | 4.27                                         |
| tree [3,2,1,1]   | 1.05× | 1.11× | 1.17×     | 4.47                                         |
| sample T=0.8 k=3 | 1.04× | 1.04× | 1.07×     | 3.25                                         |

**通过标准**

- `verify`（6 项）和 `spec`（5 项）全部 PASS。
- 所有 greedy 方法（n-gram、draft、tree）的 `same_as_plain` ≥ 128，即前 128 个 token 与普通解码相同。参考实现中，chat 和 copy 的 192 个 token 全部相同；code 在第 152 个 token 处因 bf16 恰好打平而分叉。
- n-gram 在 copy 上 ≥ **2×**，在 code 上 ≥ 1.5×；draft k=2 ≥ 1.1×。
- tree [3,2,1,1] 每次目标模型 forward 产出的 token 数，在三个 prompt 上都高于 draft k=3。

**报告必答**

1. 用讲义 §2 的期望加速公式 $S = \dfrac{1 - \alpha^{k+1}}{(1-\alpha)(1 + k c)}$ 预测 draft k=2、3、5 的加速比，并与实测对比。
   - **c**：草稿模型与目标模型单步耗时之比。分别计时 0.6B 和 1.7B 的 plain 解码即可，参考实现是 8.4 / 17.8 ≈ 0.47。
   - **α**：每个位置的接受概率。注意它**不是** bench 的 `acceptance` 列（那一列是接受数 / 提议数）。要从 `tokens_per_target_call` 反解：$\frac{1-\alpha^{k+1}}{1-\alpha}$ = 每轮产出的 token 数。参考实现在 chat 上 k=5 时反解得 α ≈ 0.80，预测 S ≈ 1.10，实测 1.03。
2. 为什么 tree 每次 forward 接受的 token 更多，实测速度却不比链式快？草稿模型的成本在其中起了什么作用？
3. 什么样的草稿最适合你的 4060 + Qwen3 这个组合？提示：EAGLE 的草稿头只有约 1 层，c ≈ 0.05。

**可选挑战**

- **C3 n-gram tree**：用多个 n-gram 匹配作为树的分支，零成本草稿加树验证，在 code 上能比单链 n-gram 更快吗？
- **C4 动态树**（EAGLE-2）：按草稿的置信度决定每个节点的分支数，而不是用固定的形状。
- **C5 树上的采样验证**：SpecInfer 的多步 speculative sampling，把 B1 推广到树上，并证明它是无损的。
- **C6 自推测**：去掉草稿模型，只用目标模型的前 N 层作为草稿（LayerSkip 的思路），看 c 能降到多少。

## 参考资料

- 讲义 [L5](../../course/lectures/L5-speculative.md)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)（Leviathan et al.）· [SpecInfer](https://arxiv.org/abs/2305.09781) · [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) / [EAGLE-2](https://arxiv.org/abs/2406.16858) / [EAGLE-3](https://arxiv.org/abs/2503.01840) · [Lookahead](https://arxiv.org/abs/2402.02057)
- Jay Mody, [Speculative Sampling（NumPy 实现）](https://jaykmody.com/blog/speculative-sampling/) · [HF assisted generation](https://huggingface.co/blog/assisted-generation)
