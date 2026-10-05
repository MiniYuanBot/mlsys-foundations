# Lab 6 验收清单

## 自动测试

```bash
cmake --build build/lab6 -j && ctest --test-dir build/lab6 --output-on-failure
```

- [ ] `coll_np1`、`coll_np2`、`coll_np3`、`coll_np4`、`coll_np8` 全部 PASS（每次运行 10 个测试）
- [ ] `train_np1`、`train_np2`、`train_np4` 全部 PASS（每次运行 6 个测试）
- [ ] 只修改了 `src/*_student.cpp`

## 性能（`bench_coll`、`train`；每组至少跑 3 次，取中位数）

- [ ] 消息 ≥ 1 MB 时，ring 的耗时 ≤ `MPI_Allreduce` 的 1.3 倍（P=4 和 P=8）
- [ ] 消息 ≤ 256 B 时，recursive doubling 比 ring 快
- [ ] 最好的 bucket 大小下，overlap 的 comm_wait ≤ sync 的 70%（P=4，batch 16）
- [ ] ZeRO-1 每个 rank 的优化器状态 = sync 的 1/P

参考实现（WSL，i9）：ring 在 4 MB 时 592 µs，MPI 为 718 µs；4 B 时 RD 0.6 µs，ring 1.7 µs；comm_wait 从 6.2 ms 降到 3.7 ms；优化器状态从 24 MB 降到 6 MB。

## 报告 `reports/lab6.md`

- [ ] 画出 P=3 时 ring reduce-scatter 与 all-gather 每一步的分块流向
- [ ] α、β 的拟合结果，以及 `alpha_beta.py` 生成的图；ring 与 RD 交叉点的预测值与实测值
- [ ] 解释模型与实测相差 2–3 倍的原因（γ 项、内存拷贝、缓存）
- [ ] 训练表（sync、各 bucket 大小、zero1），并回答 README 中的 3 个问题
- [ ] 写出“卡住最久的地方”和最终的解决办法（死锁是怎样定位的？）

## 可选挑战

- [ ] C3 ZeRO-2 / ZeRO-3
- [ ] C4 张量并行的 Linear 层
- [ ] C5 在 HPC 上运行 nccl-tests，并回答 hpc/README.md 中的问题
