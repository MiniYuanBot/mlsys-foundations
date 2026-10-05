# 系统主线自评：100 分，B/S/R 分开报告

此评分已接入 `python -m tools.grade lab5 --tier systems`，对应上级 systems.rubric.json；默认base只评调度基础练习。本文B1/B2/S1等是子任务标签，全都属于全局S层。报告明确分别列出 B、S 和 R；没有运行的项记 `not-run` 或 `resource-blocked`。不能把空报告、README、TODO 静态检查、`--reference` 成功当学生测试成功。

| 项目 | 分数 | 证据与通过标准 |
|---|---:|---|
| B1 均分/单 rank 正确性 | 20 | world_size=1 与真实 2 rank 等分测试均通过；不同 PID、覆盖全部样本、5 步参数与 oracle 一致 |
| B2 非均分正确性 | 10 | 9 样本分成 5/4，真实 2 rank 测试通过；推导 sample-weighted gradient，解释平均-of-means 的反例 |
| S1 C++ MPI ring | 20 | 自写 point-to-point 两阶段算法 10；1/2/3 rank 与手算/MPI 对照一致 5；chunk 所有权图与死锁排查说明 5 |
| S2 测量与 trace | 25 | S/K/rank 矩阵与原始数据 10；正确区分 warm-up/计时/profile，重复3次 5；真实 collective 事件、API payload/理论 bytes 分开 10 |
| S3 bucket / overlap 解释 | 15 | 逐参数/bucket/DDP 对照且优化器语义一致 5；完成边界正确 5；时间线证据或明确无法量化的限制、负结果分析 5 |
| R 准备 | 10 | 一篇论文机制→源码位置→本实验可验证命题 5；两周问题、基线/消融/证伪条件与资源预算 5 |

B1、B2 必须满分；S1 至少 15、S2 至少 20、S3 至少 10、R准备至少 5；总分 ≥80 才算主线完成。不能用阅读分补齐 collective correctness。R 扩展真正执行情况另记：`proposal-only` / `mechanism-validated` / `scaled-reproduction`，写清 hardware。多 GPU 项不占 CPU 主线必过分，也不能以 CPU 主线满分声称完成了论文多 GPU 复现。

```bash
python -m unittest lab5.distributed.tests.RealCollectives.test_world_size_one -v
python -m unittest lab5.distributed.tests.RealCollectives.test_two_ranks_equal_shards -v
python -m unittest lab5.distributed.tests.RealCollectives.test_two_ranks_uneven_shards -v
ctest --test-dir build/lab5-mpi --output-on-failure
```

Python 测试启动实际 torchrun worker，不使用 mock；每一 rank 的每一步参数都与独立全 batch oracle 比较。单个 launcher 有 75 秒超时并清理子进程，Gloo process group 为 20 秒。MPI test 为 20 秒。依赖缺失、进程异常、超时和 TODO 都是失败/阻塞，不是 skip 后通过。

建议报告记录：命令与版本、测试完整输出、每 rank JSON、短 trace、未插桩 raw CSV、图、对照方案、失败记录、论文对照、研究提案。写出你试过且没有改善的一个因素；按严谨程度评分，不按是否得到加速评分。
