# 课程与代码对应

课程的周次、阅读和完成要求见 [课程安排](../../mlsys-research-course/syllabus.md)。下表用于查找本仓库的实现入口。

| 章节 | 基础练习 | 系统任务 |
|---|---|---|
| [lab0](../lab0/README.md) | 训练与梯度检查 | C++ 布局与算子 |
| [lab1](../lab1/README.md) | Attention 与测量 | CUDA GEMM 与 cuBLAS |
| [lab2](../lab2/README.md) | 行 softmax | 流式 attention |
| [lab2x](../lab2x/README.md) | 选修：融合 RMSNorm | 独立 kernel 练习 |
| [lab3](../lab3/README.md) | 服务基线与指标 | KV 状态机与引擎修改 |
| [lab4](../lab4/README.md) | 开放负载与 SLO | token trace 与调度观测 |
| [lab5](../lab5/README.md) | 长尾模型 | Gloo、MPI 与通信测量 |
| [lab6](../lab6/README.md) | token budget 对照方法 | 选择一个研究项目 |
| [lab7](../lab7/README.md) | 留出验证方法 | 延续项目、归档与提案 |

每个实验目录以 `README.md` 导航，`assignment.md` 写任务，`setup.md` 写环境，`grade.md` 写评分。各章 `systems.md` 汇总系统任务的验收，`research.md` 提供研究方向。

基础、系统、研究分别使用 `rubric.json`、`systems.rubric.json`、`research.rubric.json`，评分命令见 [grading.md](grading.md)。lab2x 使用独立的基础评分入口。

配套课程的文件哈希见 [course-sources.json](course-sources.json)；它记录工作区内容，可能包含尚未提交到 Git 的修改。
