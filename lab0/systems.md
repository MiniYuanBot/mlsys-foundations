# lab0｜S系统层

入口：[系统任务](native/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| native（自动） | 30 | 30 | `lab0.native.tests.Native`：C++ stride/非连续/边界/算子独立正确性 |
| ab（自评） | 25 | 20 | 保留同精度A/B与BLAS强基线，开发三shape各三轮、线程/分配口径（25） |
| mechanism（自评） | 25 | 15 | 布局/编译flag/资源账本、拷贝摊销与来源解释（25） |
| holdout（自评） | 20 | 15 | 未见非方阵/stride、代码diff、原始记录与重建（20） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab0 --tier systems
python -m tools.grade lab0 --tier systems --assessment lab0/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab0 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

