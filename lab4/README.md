# Lab 4：minillm —— 用 Rust + CUDA 写一个 LLM 推理引擎

> 配套讲义：[L4 LLM 推理系统](../../mlsys-research-course/lectures/L4-serving.md)
> 预计用时：3 周 × 约 7 小时　·　语言：Rust（控制平面）+ CUDA C++（kernel，由 NVRTC 在运行时编译）　·　模型：Qwen3-0.6B（真实权重）　·　硬件：RTX 4060（WSL2）

vLLM、SGLang 的核心其实不大：**一个 paged KV cache 加上一个按 token 预算做调度的循环**。这个 lab 让你从零写出这样一个引擎：加载 Qwen3-0.6B 的真实权重，输出与 Hugging Face 对齐，然后一步步加上 PagedAttention、continuous batching、chunked prefill 和前缀缓存，并在你的 4060 上逐项测出它们各自带来的收益。

```
requests -> Scheduler (B3, C1) -> BatchInput -> Qwen3::forward (A2) -> argmax -> tokens
               |                                    |
               +-- BlockManager (B1, C2) -----------+-- paged KV cache (GPU) <- paged_attention (B2)
```

| 部分 | 你要写什么 | 语言 | 难度梯度 |
|---|---|---|---|
| **A** 模型 | embedding、RMSNorm、fused add+RMSNorm、RoPE、SwiGLU、argmax 六个 kernel；28 层前向 | CUDA + Rust | ① 填空：熟悉 Transformer 推理的每一步 |
| **B** PagedAttention + 调度 | 带引用计数的块分配器、warp 级 paged attention kernel、continuous batching 调度器（含抢占） | Rust + CUDA | ② 核心实现 + 性能目标 |
| **C** 前沿机制 | chunked prefill（[Sarathi-Serve](https://arxiv.org/abs/2403.02310)）、自动前缀缓存（hash 链 + LRU 驱逐，vLLM APC / [SGLang RadixAttention](https://arxiv.org/abs/2312.07104) 的块级版本） | Rust | ③ 论文机制的精简复现 |

做完后你应当能回答：

- decode 为什么受限于显存带宽？单个请求的 tokens/s 上限是多少？
- PagedAttention 解决了什么问题？continuous batching 为什么必须依赖它？
- chunked prefill 和前缀缓存分别优化的是 TTFT、TPOT 还是吞吐？各自的代价是什么？

---

## 0. 环境与构建

`tools/setup_wsl.sh` 已经装好 Rust、PyTorch，并把模型下载到 `~/models`。在 WSL 中执行：

```bash
cd lab4
cargo build --release                          # first build ~1 min (downloads crates)
python scripts/make_reference.py               # once: HF reference logits -> ../build/lab4/ref/ (~2 min)
cargo test --release                           # nearly everything fails at first
cargo test --release --test kernels a1b        # a single test
cargo run --release --bin generate -- "Explain PagedAttention in two sentences."
```

kernel 源码由 `include_str!` 嵌入程序，启动时用 NVRTC 编译。`.cu` 文件里的编译错误会在**运行时**报出来，错误信息中带有文件名和行号。

> 💡 GPU 测试会加锁串行执行（8 GB 显存同时装不下好几份模型权重）。整套测试大约需要 1 分钟。

## 1. 代码地图

```
lab4/
+-- kernels/
|   +-- provided.cu             reshape_and_cache, paged_attention_naive, gather_rows
|   +-- basic_student.cu        TODO(A1a-f)
|   +-- attention_student.cu    TODO(B2)
+-- src/
|   +-- gpu.rs ops.rs           provided: CUDA context, NVRTC, typed launch wrappers, cuBLAS
|   +-- config.rs weights.rs    provided: config.json, safetensors -> GPU
|   +-- batch.rs                provided: one step's flattened inputs (read the doc comment!)
|   +-- kvcache.rs sequence.rs  provided: KV storage, per-request state
|   +-- engine.rs               provided: schedule -> forward -> sample -> update
|   +-- model_student.rs        TODO(A2)
|   +-- kv_student.rs           TODO(B1a-d, C2a-b)
|   +-- scheduler_student.rs    TODO(B3, C1)
|   +-- bin/generate.rs  bin/bench.rs
+-- tests/   kernels.rs model.rs kv.rs scheduler.rs engine.rs
+-- scripts/make_reference.py
```

Rust 只要会写结构体、`Vec`、`HashMap`、`match`、`Option` 就够用了。不熟的话，先读 [The Rust Book](https://doc.rust-lang.org/book/) 第 3–8 章（约 3 小时）。所有涉及 GPU 和 unsafe 的代码都已经写好。

---

## Part A：跑通模型（第 1 周，约 7 小时）

读讲义 §1（一个 token 怎样流过 Qwen3）。

### A1 六个 kernel（`kernels/basic_student.cu`）

| TODO | kernel | 要点 |
|---|---|---|
| A1a | `embedding` | 按行拷贝 |
| A1b | `rmsnorm` | block 内求平方和；**在乘 weight 之前先舍入到 bf16**（与 HF 一致） |
| A1c | `fused_add_rmsnorm` | 原地完成 residual += x 和 x = norm(residual)，3 个 kernel 合成 1 个 |
| A1d | `rope` | NeoX 风格：第 i 维与第 i+64 维配对；cos/sin 先舍入到 bf16 |
| A1e | `silu_mul` | SwiGLU |
| A1f | `argmax_rows` | 15 万维 vocab 的 block 归约；并列时取下标最小的 |

每个 kernel 都在 `tests/kernels.rs` 中与一个**在同样位置做 bf16 舍入**的 CPU 参考实现对比。注释中的舍入位置要严格照做，否则 A2 的结果对不上 HF。

### A2 前向传播（`src/model_student.rs`）

把 cuBLAS GEMM（`ops::linear`，已提供）和你的 kernel 拼成 28 层。注意 Qwen3 特有的 **QK-norm**：q、k 在做 RoPE 之前，要先按 head 各做一次 RMSNorm。

**通过标准**：`cargo test --release --test model` 通过。

- teacher forcing 下，每个位置的 logits 与 HF fp32 的余弦相似度 > 0.99，top-1 一致率 ≥ 90%。参考实现：余弦相似度 ≥ 0.998，一致率 99.6–100%。
- greedy 生成的前 8 个 token 与 HF 相同。参考实现中，4 个 prompt 有 3 个 24 个 token 全部相同，第 4 个前 13 个相同。分歧发生在 bf16 与 fp32 恰好打平的位置，这是正常的。
- `cargo run --release --bin generate` 能输出通顺的回答。

---

## Part B：PagedAttention 与 continuous batching（第 2 周，约 7 小时）

读讲义 §2–§4。

### B1 块管理器（`src/kv_student.rs`，纯 CPU）

KV cache 是一个由 `num_blocks` 个块组成的池子，每块存放 16 个 token 的 K/V。每个序列有一张 block table，把逻辑块映射到物理块。分配和释放都要维护引用计数，C2 的共享要靠它。

| TODO | 内容 |
|---|---|
| B1a | `allocate` |
| B1b | `release`：引用计数减到 0 才归还；重复释放要 panic |
| B1c | `ensure_capacity`：**要么全部分配成功，要么一个都不分配** |
| B1d | `free_sequence` |

### B2 Paged attention kernel（`kernels/attention_student.cu`）

`provided.cu` 中的 `paged_attention_naive` 让每个线程串行算完一个 (token, head)，并且对 K 读两遍。你要写的版本是**一个 warp 负责一个 (token, head)**：每个 lane 负责 4 个维度，用 8 字节的合并加载，配合 lab2 的 online softmax，只需一遍。

### B3 调度器（`src/scheduler_student.rs`，纯 CPU）

文件顶部的注释就是完整的规格说明（与 vLLM v1 的策略一致）：

- 先调度 running 中的序列，再按 FIFO 接纳 waiting 中的序列；
- 每一步的 token 数受预算限制；
- 显存不够时，抢占最后加入的那个序列（释放它的全部块，之后重算）；
- `static_batching` 模式作为基线：一批跑完之后才接纳下一批。

`tests/scheduler.rs` 在 CPU 上模拟引擎，逐步检查调度决策。

**测量**

```bash
cargo run --release --bin bench -- --label attn_naive --requests 16 --prompt 1024 --output 64 --attention naive --no-prefix-cache
cargo run --release --bin bench -- --label attn_fast  --requests 16 --prompt 1024 --output 64 --no-prefix-cache
cargo run --release --bin bench -- --label static     --requests 64 --prompt 256 --output 128 --vary-output --max-seqs 16 --static --no-prefix-cache
cargo run --release --bin bench -- --label continuous --requests 64 --prompt 256 --output 128 --vary-output --max-seqs 16 --no-prefix-cache
```

**通过标准**

- `kernels`、`kv`、`scheduler` 中的 B 系列测试，以及 `engine` 中的 `b3_*` 全部通过。
- 快速 attention 的输出吞吐 ≥ 朴素版本的 **5×**（参考实现 224 对 19 tok/s，11.8×）。
- continuous batching 的吞吐 ≥ static 的 **1.8×**（参考实现 858 对 363 tok/s，2.4×；TTFT 中位数从 10.9 s 降到 2.7 s）。
- 报告中回答：单请求 decode 的速度（`generate` 打印的 tok/s，参考实现约 119）离显存带宽给出的上限还有多远？上限的算法是：权重字节数 ÷ lab1 测得的 memcpy 带宽。

---

## Part C：chunked prefill 与前缀缓存（第 3 周，约 7 小时）

读讲义 §5–§6。

### C1 Chunked prefill（调度器中的 `chunked_prefill` 分支）

把长 prompt 切成多块，每块的大小不超过本步剩余的预算，让它与 decode 混在同一步中运行，从而避免 decode 被一个长 prefill 卡住。另外**必须做准入控制**：接纳一个序列时，空闲块要能容纳它的整个 prompt，而不只是第一块。作者最初的实现漏了这一点：chunked 模式同时接纳了太多只 prefill 了一半的请求，它们互相抢占，出现了 63–201 次抢占，结果反而比不分块更慢。

### C2 前缀缓存（`kv_student.rs` 中的 C2 部分）

| TODO | 内容 |
|---|---|
| C2a | `match_prefix`：对新请求按块计算链式 hash，命中的块直接复用（引用计数 +1），并且**永远不复用最后一个 token** |
| C2b | `register_full_blocks`：每一步之后，为新填满的块登记 hash |

引用计数降到 0 的已缓存块不会立即释放，而是进入 evictable 队列；只有空闲列表用完时，才按 LRU 顺序驱逐。

**测量**

```bash
# prefix caching: 448 of 512 prompt tokens are a shared system prompt
cargo run --release --bin bench -- --label prefix_off --requests 64 --prompt 512 --shared-prefix 448 --output 32 --no-prefix-cache
cargo run --release --bin bench -- --label prefix_on  --requests 64 --prompt 512 --shared-prefix 448 --output 32
# chunked prefill under Poisson arrivals (rate 1 and 2 req/s)
cargo run --release --bin bench -- --label no_chunk  --requests 40 --prompt 1536 --output 128 --rate 2 --max-tokens 2048 --no-chunked --no-prefix-cache
cargo run --release --bin bench -- --label chunk_256 --requests 40 --prompt 1536 --output 128 --rate 2 --max-tokens 256 --no-prefix-cache
```

**通过标准**

- 全部测试通过，包括 `engine` 中的 `c1_*`、`c2_*`，以及抢占测试。
- 前缀缓存：吞吐 ≥ 关闭时的 **2×**，TTFT 中位数 ≤ 关闭时的一半（参考实现 1325 对 431 tok/s，3.1×；TTFT 658 对 2084 ms；命中率 96%）。
- chunked prefill：在 rate = 1 和 rate = 2 下，TPOT p99 都低于不分块（参考实现 rate = 2 时 53 对 61 ms），并且 0 次抢占。
- 报告中讨论：chunked prefill 的收益为什么只有约 13%，没有 Sarathi-Serve 论文中那么大？提示：比较一个 1536-token 的 prefill 步与一个 decode 步的耗时，再想想 0.6B 模型与 13B 模型在这一比值上的差别。

**可选挑战**

- **C3 RadixAttention**：把块级的 hash 链换成 token 级的基数树，支持任意长度的前缀匹配（参考 SGLang 和 mini-sglang）。
- **C4 decode 的 CUDA Graph**：decode 步的 kernel 启动开销不可忽视。把固定 batch 大小的 decode 步录成 CUDA Graph，测量收益。
- **C5 Tensor Core prefill attention**：prefill 时用 lab2 C1 的 mma.sync FlashAttention 代替 warp 版本。
- **C6 与 vLLM 对比**：升级驱动后装上 vLLM，用同样的负载对比，并分析差距来自哪里。

---

## 提交物

原始 CSV 存放在 `results/lab4/`，报告写在 `reports/lab4.md`（[模板](../templates/report.md)），验收清单见 [grade.md](grade.md)。

## 参考资料

- 讲义 [L4](../../mlsys-research-course/lectures/L4-serving.md)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)、[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[SGLang](https://arxiv.org/abs/2312.07104)
- Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm)：vLLM v1 调度器与 KV 管理器的逐行图解，**与本 lab 的结构一一对应**
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（约 1200 行 Python，结构几乎相同，可以对照阅读）、[mini-sglang](https://github.com/sgl-project/mini-sglang)
- HF [modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)（A2 的标准答案在结构上就是这个文件）
- [cudarc](https://github.com/coreylowman/cudarc)（Rust 的 CUDA 绑定）
