# 环境配置

工具链**按 lab 逐步安装**，不要一开始就把所有框架装齐。下表是全课程的总览，每个 lab 的 README 第 0 节会写出该 lab 需要的具体步骤。

## 你的机器分别做什么

| 机器 | 适合 | 不适合 |
|---|---|---|
| Windows + RTX 4060 8 GB | 全部 CUDA lab（lab1–lab5）；建议在 **WSL2** 中做 | 多 GPU 通信、Hopper 专属特性（TMA/WGMMA） |
| Mac mini（Apple Silicon，16 GB） | lab0（NEON 微内核）、lab6 的 MPI 多进程部分、读代码与写报告 | 任何 CUDA |
| 北大 HPC / 组里服务器 | lab6 的 NCCL 多卡实验；H100 上的可选挑战 | 日常开发与调试（排队太慢） |

## 工具链总览

| 工具 | 首次需要 | 安装 |
|---|---|---|
| C++17 编译器 + CMake ≥ 3.18 | lab0 | Windows：[MSYS2](https://www.msys2.org/) 的 `mingw-w64-ucrt-x86_64-gcc cmake`，或 Visual Studio 2022；WSL/Linux：`sudo apt install build-essential cmake`；macOS：`xcode-select --install && brew install cmake` |
| Python 3.10+、NumPy、matplotlib | lab0（基线和画图） | `python -m pip install numpy -r requirements-analysis.txt` |
| WSL2 + CUDA Toolkit 12.x | lab1 | 见下文 |
| Triton（随 PyTorch 安装） | lab2 | 在 WSL 里按 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 安装 CUDA 版 PyTorch |
| Rust（rustup）+ cudarc | lab4 | [rustup.rs](https://rustup.rs/)；lab4 README 会详细说明 |
| MPI（Open MPI 或 MPICH） | lab6 | WSL/Linux：`sudo apt install openmpi-bin libopenmpi-dev`；macOS：`brew install open-mpi` |

## WSL2 与 CUDA（lab1 开始前做）

1. 在 Windows PowerShell 执行 `wsl --install`，按提示重启，然后安装 Ubuntu。
2. Windows 上装好最新的 NVIDIA 驱动即可。**不要在 WSL 里安装 Linux 显卡驱动**。参见 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
3. 在 WSL 里执行 `nvidia-smi`，能看到 4060 就说明驱动正常。如果找不到命令，试试 `/usr/lib/wsl/lib/nvidia-smi`。
4. 按 NVIDIA 文档中 “WSL-Ubuntu” 一节安装 CUDA Toolkit，再用 `nvcc --version` 确认。注意：`nvidia-smi` 里显示的 CUDA 版本只表示驱动支持到哪个版本，并不代表已经装了 toolkit。
5. 把代码放在 WSL 的 Linux 文件系统里（如 `~/code/mlsys-research-labs`），不要放在 `/mnt/c` 或 `/mnt/d` 下，否则编译和 IO 会慢很多。两个仓库都 clone 到同一个父目录。

## 记录环境

每个 lab 开始时保存一次环境信息，写进报告的“环境”一节：

```bash
python -m tools.doctor --out results/environment/labN.json   # 解释器、包版本、nvidia-smi
```

## 卡住了怎么办

环境问题超过 30 分钟没解决：保存能复现问题的最短命令和完整报错，写进 [周报](templates/weekly-report.md)。超过 90 分钟就停止盲试，去问人或换一台机器。CPU 部分和读论文可以继续推进。
