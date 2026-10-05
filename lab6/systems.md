# lab6｜S系统层

入口：[实际任务与脚手架](research.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| correctness（自评） | 25 | 25 | 真实系统patch或算子实现及独立正确性/回归（25） |
| baseline（自评） | 20 | 15 | 固定上游/强基线/同工作量与质量、执行成本（20） |
| preregister（自评） | 15 | 10 | 原文主张、指标/矩阵/停止条件在数据前冻结（15） |
| controls（自评） | 25 | 20 | 至少一个消融和一个负对照、三轮与失败/噪声（25） |
| artifact（自评） | 15 | 10 | 源码diff、环境/配置/raw/图命令与局限（15） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab6 --tier systems
python -m tools.grade lab6 --tier systems --assessment lab6/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab6 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

