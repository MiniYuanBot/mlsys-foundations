# MLSys Research Labs

MLSys 自学课程的实验代码。课程大纲、讲义和阅读材料在相邻的 [课程仓库](../mlsys-research-course/README.md)。两个仓库要放在**同一个父目录**下，保持目录名不变，否则跨仓库链接会失效。在 VS Code 中打开 [mlsys-research.code-workspace](mlsys-research.code-workspace)，可以同时编辑两个仓库。

## 设计原则

- **写 infra，不写胶水**。你写的代码以 C++、CUDA、Rust 为主；Python 只用于跑基线、画图和充当对照（oracle）。
- **每个 lab 三级阶梯**：① 带 TODO 的代码填空，帮你熟悉整体架构 → ② 自己实现核心代码，并给出明确的性能目标 → ③ 复现前沿论文设计的精简版。
- **测试先行**：每个 TODO 都有对应的测试，测试名带 TODO 编号，可以单独运行。验收标准都写在每个 lab 的 `grade.md` 中。

## 实验列表

| Lab | 主题 | 语言 | 硬件 | 状态 |
|---|---|---|---|---|
| [lab0](lab0/README.md) | minitensor：strided 张量、GEMM 优化阶梯、lazy 融合 | C++17 | 任意 CPU | ✅ 已完成 |
| [lab1](lab1/README.md) | CUDA 执行模型与 GEMM：tiling → 寄存器分块 → Tensor Core | CUDA C++ | RTX 4060（WSL2） | ✅ 已完成 |
| lab2 | Attention kernel：online softmax → FlashAttention-2 → FlashDecoding / paged | Triton + CUDA | RTX 4060 | 🚧 |
| lab3 | 张量编译器：IR → 融合 → CUDA 代码生成 → mini-Mirage 搜索 | C++ + NVRTC | RTX 4060 | 🚧 |
| lab4 | 推理引擎：KV cache、paged 分配器、continuous batching、前缀缓存 | Rust 控制平面 + CUDA kernel | RTX 4060 | 🚧 |
| lab5 | 推测解码：n-gram 草稿 → 拒绝采样 → SpecInfer token tree | Rust + CUDA | RTX 4060 | 🚧 |
| lab6 | 集合通信与并行训练：ring allreduce → α-β 模型 → DDP 重叠 / ZeRO | C++ MPI（HPC 上用 NCCL） | CPU 多进程 / HPC | 🚧 |
| lab7 | 研究项目：从 lab2–lab6 中选一个方向扩展 | — | 视方向而定 | 🚧 |

`legacy/` 下是重构前的旧版 lab1–lab7，只作为迁移参考，之后会逐步删除。

## 目录

| 目录 | 内容 |
|---|---|
| `labN/` | 每个 lab：`README.md`（任务说明）、`grade.md`（验收清单）、源码、测试、benchmark |
| [templates](templates/README.md) | 报告、预注册、周报等模板 |
| [docs](docs/README.md) | 实验记录规范、脚手架的验证范围 |
| [common](common/README.md)、[tools](tools/README.md) | Python 辅助：运行记录、环境检查、画图 |
| [tests](tests/README.md) | 仓库自检（结构、链接，确认没有残留答案） |
| [portfolio](portfolio/README.md) | 结课材料索引 |
| `build/`、`results/`、`reports/` | 构建产物、原始数据、你的报告（前两者不进 Git） |

## 开始

1. 按 [setup.md](setup.md) 准备 lab0 需要的 C++ 工具链（约 30 分钟）。
2. 读课程仓库的 [课程大纲](../mlsys-research-course/syllabus.md) 和 [L0 讲义](../mlsys-research-course/lectures/L0-tensor-runtime.md)。
3. 进入 [lab0](lab0/README.md)。
