# Lab 3 验收清单

## 自动测试

```bash
cmake --build build/lab3 -j && ctest --test-dir build/lab3 --output-on-failure
```

- [ ] `parta`（9 项）、`partb`（12 项）、`partc`（9 项）全部 PASS
- [ ] 只修改了 `src/*_student.cpp`

## 性能（`bench_fusion`、`bench_search`）

- [ ] softmax、layernorm、rmsnorm 各融合成 1 个 kernel，有效带宽 ≥ 180 GB/s（8192×4096）
- [ ] matmul_bias_gelu 融合后比 unfused 快 ≥ 2×
- [ ] softmax_matmul：搜索选出的方案比 unfused 快 ≥ 1.5×，且不物化任何 [M, K] 张量

参考实现（RTX 4060 Laptop）：带宽 227–239 GB/s；matmul_bias_gelu 快 2.3×；softmax_matmul 从 11.9 ms 降到 6.6 ms。

## 报告 `reports/lab3.md`

- [ ] `tc_dump softmax` 的完整输出，并为生成的源码逐段加注释，说明每个循环在做什么
- [ ] 手算 layernorm 在 unfused 与 fused 两种方式下各读写多少字节，并与 `estimate()` 对照
- [ ] 解释为什么 `exp(x)` 同时被 matmul 的 A 和尾声使用时必须物化（`B1_node_feeding_two_kernels_is_materialised`）
- [ ] `bench_search` 的完整输出；分析代价模型与实测排序不一致的原因，至少说出两个模型没有考虑的因素
- [ ] 用一段话说明：R3 是怎样被 C2 拒掉的？如果 C2 只用 1 个随机输入，会不会误判？为什么？
- [ ] 写出“卡住最久的地方”和最终的解决办法

## 可选挑战

- [ ] C4 Tensor Core GEMM 模板
- [ ] C5 改进代价模型，使排序与实测一致
- [ ] C6 Row kernel 与 MatMul kernel 合成一个（online softmax）
