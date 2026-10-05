# lab4：负载、调度与请求回放

比较不同负载下的延迟与 SLO，再用固定的 token 和到达时间回放请求，研究调度与缓存行为。

## 内容

- [assignment.md](assignment.md)：基础负载矩阵、SLO 与分析要求。
- `plan.py`、`analyze.py`、`tests.py`：实验计划、指标计算与测试。
- [trace/](trace/README.md)：请求生成、数据审计、精确回放与调度器修改。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：缓存、调度与公平性拓展。
- [setup.md](setup.md)、[grade.md](grade.md)、[report.md](report.md)：环境、基础评分与报告。

## 使用顺序

1. 使用 lab3 的真实容量结果，确定负载和 SLO。
2. 保存实验计划，执行基础矩阵并分析请求级结果。
3. 进入 `trace/`，检查回放的 token、到达时间和实际工作量。
4. 完成调度器修改，用相同请求比较原始实现与改动。

正式测量前固定条件与指标。失败请求仍计入 SLO 分母；随机种子相同并不保证请求实际到达时间相同。
