# lab5 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab5-tail.md](../../mlsys-research-course/labs/lab5-tail.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| schedulers | 自动 | 20 | 20 | `lab5.tests.Schedulers`：手算 8/6、K=1、均匀负载、非整批 |
| conservation | 自动 | 20 | 20 | `lab5.tests.Conservation`：输入重放、下界、无丢失/重复/重叠、指标恒等式 |
| simulation | 人工自评 | 30 | 30 | **完整模拟证据**：配对输入 CSV、逐任务时序、24 行汇总（15）；两张注明独立等速槽位模型的图（10）；区分均匀重复确定性与长尾独立 seed（5）。 |
| directions | 人工自评 | 30 | 20 | **方向与研究边界**：三路线兴趣/工具/两项工作/资源证据表（15）；两周可验证问题与导师三个问题（10）；不将模拟收益称为 RL/GPU 实测加速（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab5.tests -v
python -m tools.grade lab5
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab5 --assessment lab5/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab5.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

