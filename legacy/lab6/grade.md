# lab6 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab6-reproduction.md](../../mlsys-research-course/labs/lab6-reproduction.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| matrix | 自动 | 20 | 20 | `lab6.tests.Matrix`：12 条 A/B 区组、两 rate、仅 budget 改变 |
| preregister | 人工自评 | 20 | 20 | **预注册与文献定位**：原论文主张/图表定位及缩规模差异表（10）；p95 TPOT 主指标、支持/反驳/证据不足标准（10）。 |
| measurements | 人工自评 | 35 | 35 | **当周公平 A/B**：重新测 A，A/B 启动配置与逐次预热日志（10）；12 次原始 JSON/请求、固定 rate/N/SLO/模型（15）；失败保留、区组顺序与三轮波动（10）。 |
| report | 人工自评 | 25 | 15 | **机制报告**：主图及 TTFT/吞吐权衡图可重建（10）；2–3 页等量报告，两个限制与最小后续检验（10）；区分 TPOT/ITL 及受控观察/直接机制证据（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab6.tests -v
python -m tools.grade lab6
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab6 --assessment lab6/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab6.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

