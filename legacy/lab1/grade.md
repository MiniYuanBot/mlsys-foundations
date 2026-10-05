# lab1 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab1-attention.md](../../mlsys-research-course/labs/lab1-attention.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| cpu | 自动 | 20 | 20 | `lab1.tests.CPU`：六个形状、首位置性质与未来 token 不影响过去 |
| gpu | 自动 | 15 | 15 | `lab1.tests.GPU`：同一 FP32 契约在 CUDA 上通过 |
| measurements | 人工自评 | 40 | 40 | **完整 GPU 证据**：6×2×3=36 条 CUDA 记录（15）；20 预热/100 迭代/同步、独立进程与首调用隔离（10）；allocated/reserved/peak 与误差、原始文件（10）；单独 profiler trace 和后端观察（5）。 |
| analysis | 人工自评 | 25 | 15 | **图与解释**：可重建的耗时/显存两图含逐轮点或范围（10）；500–800 字分析、资源账本和限制（10）；下一步区分实验与来源记录（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab1.tests -v
python -m tools.grade lab1
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab1 --assessment lab1/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab1.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

