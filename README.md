# MLSys Research Labs

MLSys 课程的实验代码、任务说明与自评工具。配套的课程安排和论文阅读在 [课程仓库](../mlsys-research-course/README.md)。

在 VS Code 中打开 [mlsys-research.code-workspace](mlsys-research.code-workspace)，可同时编辑课程和实验。两个仓库需保留现有目录名，并放在同一父目录下。

| 目录 | 内容 |
|---|---|
| [lab0](lab0/README.md) | 训练检查与 C++ 张量布局 |
| [lab1](lab1/README.md) | Attention 测量与 CUDA GEMM |
| [lab2](lab2/README.md) | Triton softmax 与流式 attention |
| [lab2x](lab2x/README.md) | 融合 RMSNorm 选修 |
| [lab3](lab3/README.md) | 推理服务与 KV 内存管理 |
| [lab4](lab4/README.md) | 请求负载、缓存与调度 |
| [lab5](lab5/README.md) | 分布式训练与通信 |
| [lab6](lab6/README.md)、[lab7](lab7/README.md) | 研究实验、留出验证与提案 |
| [docs](docs/README.md)、[templates](templates/README.md) | 实验规范、评分说明与记录模板 |
| [common](common/README.md)、[tools](tools/README.md)、[tests](tests/README.md) | 公共代码、命令工具与框架自检 |
| [portfolio](portfolio/README.md) | 结课材料索引 |

先看 [环境指南](setup.md)，再按 lab0–lab7 学习；lab2x 按需选做。每章先读 README，再进入 `assignment.md` 的任务步骤；环境和评分分别见 `setup.md`、`grade.md`。系统任务见各章 `systems.md`，研究拓展见 `research.md`。

依赖按章节自行安装。代码中的 TODO 留给学习者；测试通过范围见 [验证记录](docs/validation-scope.md)，成绩规则见 [测试与评分](docs/grading.md)。运行数据放在 `results/`，该目录不进 Git，需要自行归档。
