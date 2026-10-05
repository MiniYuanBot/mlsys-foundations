# lab2 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab2-softmax.md](../../mlsys-research-course/labs/lab2-softmax.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| contract | 自动 | 5 | 5 | `lab2.tests.Contract`：拒绝不支持的 shape/dtype/stride |
| gpu | 自动 | 35 | 35 | `lab2.tests.GPU`：开发矩阵、N=1、极值、尾部、有限输出、输入未修改 |
| measurements | 人工自评 | 35 | 35 | **三种实现的公平基准**：6×3×3=54 条 CUDA 记录（15）；同输入、FP32、输出分配、预热同步、编译单列（10）；正确性报告与两图可重建（10）。 |
| analysis | 人工自评 | 25 | 15 | **机制解释与边界**：解释 mask、BLOCK、稳定归约与工作分配（10）；500–800 字分析，包括不获胜形状/噪声（10）；来源与独立修改检查（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab2.tests -v
python -m tools.grade lab2
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab2 --assessment lab2/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab2.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

