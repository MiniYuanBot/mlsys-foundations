# Lab 5 验收清单

## 自动测试（`cd lab5`；你的 lab4 Part A、B 必须已经通过）

```bash
cargo test --release --no-fail-fast
```

- [ ] `verify`（6 项，纯 CPU）全部 PASS，其中 `b1_speculative_sampling_is_exact` 打印的最大误差 < 0.006
- [ ] `spec`（5 项，GPU）全部 PASS
- [ ] 只修改了 `src/*_student.rs` 和 `kernels/tree_attention_student.cu`

## 性能（`cargo run --release --bin bench -- --tokens 192`）

- [ ] 所有 greedy 方法的 `same_as_plain` ≥ 128
- [ ] n-gram k=8：copy ≥ 2×，code ≥ 1.5×
- [ ] draft k=2 ≥ 1.1×（三个 prompt 的平均值）
- [ ] tree [3,2,1,1] 的 `tokens_per_target_call` 在三个 prompt 上都高于 draft k=3

参考实现（RTX 4060 Laptop）：n-gram k=8 在 copy 上 2.85×、code 上 2.05×；draft k=2 约 1.2×；tree [3,2,1,1] 每次 forward 产出 3.84–4.47 个 token，draft k=3 为 2.91–3.31。

## 报告 `reports/lab5.md`

- [ ] 完整的 bench 结果表
- [ ] 期望加速公式：用测得的 c 和由 `tokens_per_target_call` 反解出的 α，预测 draft k=2、3、5 的加速比，并与实测对比
- [ ] 用自己的话复述 B1 的正确性证明（讲义 §3），并说明 `b1_identical_models_accept_everything` 验证的是什么
- [ ] 解释 tree “每次接受更多、整体却不更快”的原因
- [ ] 讨论：怎样的草稿最适合这台机器和这个模型组合？
- [ ] 写出“卡住最久的地方”和最终的解决办法

## 可选挑战

- [ ] C3 n-gram tree
- [ ] C4 动态树（EAGLE-2）
- [ ] C5 树上的采样验证
- [ ] C6 自推测（用目标模型的前 N 层做草稿）
