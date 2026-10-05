# lab2x 自评分（100 分）

依据 [课程 lab2x-kernel-challenge.md](../../mlsys-research-course/labs/lab2x-kernel-challenge.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| gpu | 自动 | 40 | 40 | `lab2x.tests.GPU`：开发/留出/N=1、z/y 双输出、零输入/零 gamma/符号性质及输入不变 |
| ablation | 人工自评 | 35 | 35 | **开发、消融与留出**：开发 54 条与留出 18 条共 72 条（15）；仅改变 num_warps=4→8 并冻结选择（10）；双输出误差与延迟图、相同计时口径（10）。 |
| analysis | 人工自评 | 25 | 15 | **解释与外部契约**：解释实际 N、padding、epsilon、z 写回（10）；500–800 字分析与来源（10）；OJ 完整契约差异表或真实提交证据（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab2x.tests -v
python -m tools.grade lab2x
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab2x --assessment lab2x/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab2x.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。本章本身是选修，不阻塞 lab3。

