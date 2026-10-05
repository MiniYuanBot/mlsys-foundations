# lab7｜S系统层

入口：[实际任务与脚手架](research.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| holdout（自评） | 30 | 25 | 继承L06路线，未参与调参条件，三轮/错误/趋势判断（30） |
| artifact（自评） | 30 | 25 | fresh checkout重建图表、上游commit/patch/哈希与命令（30） |
| literature（自评） | 20 | 15 | 最接近三工作具体定位/重合/差异，不以未查到当不存在（20） |
| proposal（自评） | 20 | 15 | 两页提案、六任务、资源预算/停止条件、10分钟答辩（20） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab7 --tier systems
python -m tools.grade lab7 --tier systems --assessment lab7/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab7 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

