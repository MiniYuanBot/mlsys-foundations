# lab2｜S系统层

入口：[系统任务](attention/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| attention（自动） | 30 | 30 | `lab2.attention.tests.GPU`：真实GPU流式attention，尾部/causal/数值/tile性质 |
| streaming（自评） | 25 | 25 | online推导/代码、无全局S²中间、精度与工作分配（25） |
| measurement（自评） | 25 | 20 | 三实现27条、显存与误差、一个tile消融9条、后端（25） |
| holdout（自评） | 20 | 15 | 未见两形状18条、编译产物/资源分析、原始重建（20） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab2 --tier systems
python -m tools.grade lab2 --tier systems --assessment lab2/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab2 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

