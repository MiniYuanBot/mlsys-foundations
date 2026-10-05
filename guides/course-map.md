# 课程与实验地图

各章按基础阅读、代码练习、系统任务的顺序衔接。周次和完成要求见 [课程安排](../syllabus.md)，实验命令与评分在 [代码仓库](../../mlsys-research-labs/README.md)。

| 章节 | 基础讲义 | 基础题面 | 系统任务 |
|---|---|---|---|
| lab0 | [训练与张量](../modules/lab0-pytorch.md) | [训练检查](../labs/lab0-pytorch.md) | [C++ 布局](../../mlsys-research-labs/lab0/native/assignment.md) |
| lab1 | [性能测量](../modules/lab1-measurement.md) | [Attention](../labs/lab1-attention.md) | [CUDA GEMM](../../mlsys-research-labs/lab1/cuda/assignment.md) |
| lab2 | [GPU 算子](../modules/lab2-kernels.md) | [Softmax](../labs/lab2-softmax.md) | [流式 attention](../../mlsys-research-labs/lab2/attention/assignment.md) |
| lab3 | [推理服务](../modules/lab3-serving.md) | [服务基线](../labs/lab3-serving.md) | [KV 管理与引擎修改](../../mlsys-research-labs/lab3/runtime/assignment.md) |
| lab4 | [负载与调度](../modules/lab4-scheduling.md) | [负载矩阵](../labs/lab4-workloads.md) | [请求回放与调度观测](../../mlsys-research-labs/lab4/trace/assignment.md) |
| lab5 | [分布式与 RL](../modules/lab5-directions.md) | [长尾模拟](../labs/lab5-tail.md) | [Gloo 与 MPI](../../mlsys-research-labs/lab5/distributed/assignment.md) |
| lab6 | [复现实验](../modules/lab6-reproduction.md) | [对照方法](../labs/lab6-reproduction.md) | [研究项目](../../mlsys-research-labs/lab6/research.md) |
| lab7 | [研究提案](../modules/lab7-proposal.md) | [留出方法](../labs/lab7-proposal.md) | [验证与归档](../../mlsys-research-labs/lab7/research.md) |

[lab2x](../labs/lab2x-kernel-challenge.md) 是 lab2 之后的融合 RMSNorm 选修。阅读材料见 [论文路线](../readings/papers.md)，可复用的外部任务见 [开源实验](../readings/open-source-labs.md)。
