# Lab 4 验收清单

## 自动测试（WSL2，`cd lab4`）

```bash
python scripts/make_reference.py      # once
cargo test --release --no-fail-fast
```

- [ ] `kernels`（8 项）、`model`（2 项）、`kv`（10 项）、`scheduler`（9 项）、`engine`（4 项）全部 PASS
- [ ] `model` 测试**确实运行了**（没有打印 `SKIP: no reference`）
- [ ] 只修改了 `kernels/*_student.cu` 和 `src/*_student.rs`
- [ ] `cargo run --release --bin generate` 能输出通顺的回答

## 性能（`bin/bench`，命令见 README；插电、关闭其他 GPU 程序）

- [ ] B2：快速 attention ≥ 朴素版本 5× 的输出吞吐
- [ ] B3：continuous ≥ static 1.8× 的输出吞吐
- [ ] C2：开启前缀缓存后吞吐 ≥ 2×，TTFT p50 ≤ 关闭时的 1/2
- [ ] C1：rate = 1 和 rate = 2 下，chunked prefill 的 TPOT p99 都低于不分块，且 0 次抢占

参考实现（RTX 4060 Laptop，Qwen3-0.6B）：B2 11.8×；B3 2.4×；C2 3.1× 且 TTFT 降为 1/3；C1 rate = 2 时 TPOT p99 从 61 ms 降到 53 ms；单请求 decode 约 119 tok/s。

## 报告 `reports/lab4.md`

- [ ] 画出一个 token 经过一层 Qwen3 的数据流，标出每个张量的形状和所用的 kernel
- [ ] 计算 Qwen3-0.6B 每个 token 的 KV cache 字节数；在 2 GB 的 cache 中能放多少个 token？
- [ ] 单请求 decode 的带宽上限，与实测 tok/s 对比，并解释差距的来源
- [ ] 四组 bench 的结果表（attention、static/continuous、prefix、chunked），每组用一句话解释“为什么”
- [ ] 解释准入控制为什么能消除 chunked prefill 中的抢占抖动
- [ ] 回答 C1 的讨论题：为什么收益比 Sarathi-Serve 论文中小？
- [ ] 写出“卡住最久的地方”和最终的解决办法

## 可选挑战

- [ ] C3 RadixAttention（token 级基数树）
- [ ] C4 decode 步的 CUDA Graph
- [ ] C5 Tensor Core prefill attention
- [ ] C6 与 vLLM 对比
