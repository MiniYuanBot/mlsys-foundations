# lab1｜S系统层

入口：[系统任务](cuda/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| cuda_correctness（自动） | 10 | 10 | `lab1.cuda.tests.NativeGPU`：真实CUDA学生kernel与CPU/cuBLAS对照 |
| implementation（自评） | 25 | 25 | shared-memory学生实现，协作load、barrier、tail与输出独立（25） |
| safety（自评） | 10 | 5 | sanitizer与新增反例（10）；无工具时准确标缺证据 |
| layout（自评） | 10 | 5 | 解释cuBLAS行列映射、fixture与数据流（10） |
| measurement（自评） | 20 | 15 | 五shape×三轮×三实现，同精度事件计时与强基线（20） |
| ablation（自评） | 15 | 10 | 三tile配置或声明单变量消融、ptxas/profiler资源证据（15） |
| holdout（自评） | 10 | 5 | 冻结后3个新shape、论文机制卡与边界（10） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab1 --tier systems
python -m tools.grade lab1 --tier systems --assessment lab1/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab1 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

