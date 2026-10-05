# lab4｜S系统层

入口：[系统任务](trace/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| arrival（自动） | 20 | 20 | `lab4.trace.tests.ReplayAudit`：缺失/重复/早发/延迟/失败的到达审计 |
| patch（自评） | 25 | 25 | 真实scheduler或cache控制patch、回归与源码定位（25） |
| workload（自评） | 30 | 25 | 真实token/时间trace、cache×overlap十二轮、冷暖/失败/计时一致（30） |
| mechanism（自评） | 25 | 15 | 无patch/off/on开销负对照、iteration证据与替代解释（25） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab4 --tier systems
python -m tools.grade lab4 --tier systems --assessment lab4/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab4 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

