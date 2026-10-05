# lab3｜S系统层

入口：[系统任务](runtime/assignment.md) · [课程安排](../../mlsys-research-course/syllabus.md)。基础任务见 [assignment.md](assignment.md)，对应 rubric.json；本页列出系统任务的评分要求。

| ID | 分值 | 必过线 | 证据 |
|---|---:|---:|---|
| allocator（自动） | 45 | 45 | `lab3.runtime.tests.TestPagedKVContract`：分配/refcount/COW/释放/守恒/原子失败 |
| resources（自评） | 15 | 10 | 连续预留/分页同trace、页大小敏感性与物理/逻辑内存（15） |
| source（自评） | 10 | 5 | 固定真实引擎commit调用链、教学模型差异（10） |
| engine_patch（自评） | 20 | 20 | 真实引擎开关化patch、回归/工作负载/开销证据，只有diff不给通过（20） |
| paper（自评） | 10 | 5 | PagedAttention/SGLang机制、可证伪问题/控制/局限（10） |

复制systems.assessment.example.json为systems.assessment.json，按证据填分。总分≥80且所有必过线达到才完成；分值按子目录grade要求拆项，机检不验证人工材料真实性。没有CUDA/编译/MPI/模型环境标阻塞或待补，不能用阅读分补核心实现。

```bash
python -m tools.grade lab3 --tier systems
python -m tools.grade lab3 --tier systems --assessment lab3/systems.assessment.json
```

R研究层独立：读 [research.md](research.md)，运行 `python -m tools.grade lab3 --tier research` 查看所需证据。每模块提供R问题，但最终只需选一条在L06/L07完成完整研究循环；阅读本身不算R实验完成。

