# lab0 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab0-pytorch.md](../../mlsys-research-course/labs/lab0-pytorch.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| training | 自动 | 20 | 20 | `lab0.tests.Training`：32 样本过拟合、参数/梯度更新、短末批次与 eval/no_grad |
| gradient | 自动 | 15 | 15 | `lab0.tests.Gradient`：float64 中心差分、误差 <1e-5、输入未修改 |
| checkpoint | 自动 | 10 | 10 | `lab0.tests.Checkpoint`：真实更新后的 checkpoint 重载一致性 |
| full_run | 人工自评 | 30 | 30 | **完整训练与复跑**：20 epoch 的两次固定配置训练（10）；epochs.csv/loss 图/配置/日志（10）；训练 loss 下降、测试准确率如实记录及异常定位（10）。测试集不得参与挑 seed。 |
| explanation | 人工自评 | 25 | 15 | **调试、理解与记录**：一个真实 bug 与最小复现（10）；解释五个训练动作及有限差分（10）；AI/来源、周报和 CUDA 状态（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab0.tests -v
python -m tools.grade lab0
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab0 --assessment lab0/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab0.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

