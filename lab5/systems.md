# lab5｜S系统层

入口：[系统任务](distributed/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| collective（自动） | 30 | 30 | `lab5.distributed.tests.RealCollectives`：真实world_size1/2、均分/不均分、每步与独立oracle一致 |
| mpi（自评） | 20 | 15 | C++ MPI两阶段ring、1/2/3rank正确性、所有权/死锁图（20） |
| measurement（自评） | 25 | 20 | 消息/形状/rank矩阵三轮、真实trace、payload/理论流量区分（25） |
| overlap（自评） | 15 | 10 | 逐参数/bucket/DDP对照、同步完成边界与时间线或限制（15） |
| paper（自评） | 10 | 5 | FLUX/Comet/HybridFlow机制→源码→问题与预算（10） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab5 --tier systems
python -m tools.grade lab5 --tier systems --assessment lab5/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab5 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

