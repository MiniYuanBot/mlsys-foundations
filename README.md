# MLSys Research Labs

MLSys 自学课程的配套实验：8 个 lab，从 CPU 张量运行时一直做到 LLM 推理引擎和分布式训练。讲义和大纲在 [mlsys-research-course](../mlsys-research-course/README.md)。

## 开始

1. 把本仓库和 `mlsys-research-course` clone 到同一个父目录下，跨仓库链接依赖这个位置。
2. 按 [setup.md](setup.md) 配置环境：lab0 只需要 C++ 编译器；lab1 起需要 WSL2 + CUDA，运行一次 `bash tools/setup_wsl.sh` 就能装好，再用 `bash tools/check_env.sh` 检查。
3. 先读对应讲义，再按 lab 的 `README.md` 一步步做，用 `grade.md` 自查。

用 VS Code 打开 [mlsys-research.code-workspace](mlsys-research.code-workspace)，可以同时编辑两个仓库。

## 实验

| Lab                    | 内容                                                          | 语言         |
| ---------------------- | ------------------------------------------------------------- | ------------ |
| [lab0](lab0/README.md) | strided 张量、GEMM 优化、lazy 计算图与融合                    | C++          |
| [lab1](lab1/README.md) | CUDA 基础、SGEMM 优化、Tensor Core HGEMM                      | CUDA         |
| [lab2](lab2/README.md) | online softmax、FlashAttention-2、FlashDecoding               | Triton、CUDA |
| [lab3](lab3/README.md) | 张量编译器：融合、CUDA 代码生成、超优化搜索                   | C++          |
| [lab4](lab4/README.md) | Qwen3 推理引擎：PagedAttention、continuous batching、前缀缓存 | Rust、CUDA   |
| [lab5](lab5/README.md) | 推测解码：n-gram、拒绝采样、token tree                        | Rust、CUDA   |
| [lab6](lab6/README.md) | 集合通信、DDP 重叠、ZeRO-1                                    | C++、MPI     |
| [lab7](lab7/README.md) | 研究项目：实验统计、预注册、候选题目                          | Python       |

每个 lab 都按三级难度组织：

- 代码填空：熟悉整体架构；
- 核心实现：带有明确的性能目标；
- 前沿复现：论文设计的精简版。

## 怎么做一个 lab

- 只修改 `*_student.*` 文件。每个 TODO 都有同名测试，可以单独运行，例如 `build/lab0/test_a_tensor A5`。
- 原始数据放在 `results/`，报告写在 `reports/labN.md`，格式见 [templates/report.md](templates/report.md)。
- 卡住时先看讲义和 README 中的提示。参考答案在 [solutions/](solutions/README.md)，建议做完再看。

## 目录

| 路径                                                   | 内容                                      |
| ------------------------------------------------------ | ----------------------------------------- |
| `labN/`                                                | 任务说明、验收清单、源码、测试、benchmark |
| [solutions/](solutions/README.md)                      | 参考答案                                  |
| [templates/](templates/README.md)                      | 报告、预注册、周报模板                    |
| [docs/](docs/README.md)                                | 实验记录规范，以及参考实现的实测结果      |
| [tools/](tools/README.md)、[common/](common/README.md) | 环境安装与检查、运行记录、画图            |
| [tests/](tests/README.md)                              | 仓库自检                                  |
| [portfolio/](portfolio/README.md)                      | 结课材料索引                              |

## 代码格式化

由于强迫症加入了代码格式化模块。安装依赖、VS Code 配置和手动格式化命令见 [代码格式化说明](docs/formatting.md)。当然也可以不装啦:)
