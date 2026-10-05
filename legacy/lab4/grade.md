# lab4 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab4-workloads.md](../../mlsys-research-course/labs/lab4-workloads.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| slo | 自动 | 20 | 20 | `lab4.tests.SLO`：所有发出请求为分母、联合 SLO、goodput、边界等号 |
| matrix | 自动 | 10 | 10 | `lab4.tests.Matrix`：完整 12 条矩阵、有限 rate、交错顺序与固定参数 |
| design | 人工自评 | 20 | 20 | **运行前冻结设计**：真实 C 与数值化 0.3C/0.7C（5）；30 请求稳定 pilot，2×中位数 SLO 预先冻结（10）；统一 N、顺序与时间预算（5）。 |
| measurements | 人工自评 | 30 | 30 | **真实矩阵证据**：12 次原始请求/命令/配置（15）；失败与长度/时段校验、每轮汇总（10）；没有客户端并发截断到达过程（5）。 |
| analysis | 人工自评 | 20 | 10 | **图与结论**：两张图显示逐轮 p95 TTFT、输出吞吐与 SLO（10）；600–900 字含替代解释、尾部样本局限和下一步（10）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab4.tests -v
python -m tools.grade lab4
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab4 --assessment lab4/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab4.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

