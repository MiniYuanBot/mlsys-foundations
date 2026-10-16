# MLSys Foundations

一门从系统基础走到研究前沿的自学课程，涵盖张量运行时、GPU kernel、深度学习编译器、LLM 推理系统、推测解码和分布式训练。课程共 16 周，每周约 11 小时，配套 8 个 lab。

讲义、实验和参考答案都在这个仓库中：

| 目录 | 内容 |
| --- | --- |
| [course/](course/README.md) | 大纲、讲义、阅读资料、学习指南和课程模板 |
| [labs/](labs/README.md) | lab0–lab7 的任务说明、学生代码、测试、benchmark 和实验工具 |
| [solutions/](solutions/README.md) | 各 lab 的参考实现，与学生代码分开存放 |

## 开始

1. 读 [起步指南](course/guides/getting-started.md) 和 [16 周大纲](course/syllabus.md)。
2. 按 [环境配置](labs/setup.md) 安装 lab0 所需的依赖。全部 lab 共用一套环境，每开始一个新 lab 再补装它需要的工具。
3. 从 [L0 讲义](course/lectures/L0-tensor-runtime.md) 和 [lab0](labs/lab0/README.md) 开始，按各 lab 的 `grade.md` 自查。

实验文档中的命令默认在 `labs/` 目录执行。WSL / Linux / macOS 下，先从仓库根目录进入该目录：

```bash
cd labs
bash tools/setup_env.sh --lab 0
bash tools/check_env.sh --lab 0
```

Windows 原生支持 lab0、lab7，安装命令见 [环境配置](labs/setup.md#windows-原生lab0lab7)；lab1–lab6 在 WSL2 中完成。参考答案的对照和运行命令见 [solutions/README.md](solutions/README.md)，建议做完再看。

## 阅读与开发

用 VS Code 打开仓库根目录或 [mlsys-foundations.code-workspace](mlsys-foundations.code-workspace)，即可同时编辑课程、实验和参考答案；新终端默认进入 `labs/`。也可以把仓库根目录作为 Obsidian vault 打开。

- [详细讲义](course/lectures/README.md)：L0–L7 的原理、例子和自测题。
- [阅读资料](course/readings/README.md)：论文、代码、参考课程和可视化资料。
- [学习指南](course/guides/README.md)：环境、实验规范、论文阅读和研究路线。
- [实验记录规范](labs/docs/experiment-records.md)：运行记录、原始数据和图表归档。
- [代码格式化](labs/docs/formatting.md)：共享配置、工具安装和编辑器设置。

## 仓库自检

在仓库根目录执行：

```bash
python -m unittest discover -s labs/tests -v
```

自检覆盖目录结构、参考答案与学生文件的对应关系、Python 语法，以及课程、实验和参考答案之间的 Markdown 链接。各 lab 的正确性与性能验收仍按自己的 `README.md` 和 `grade.md` 执行。
