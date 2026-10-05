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
| PyTorch + Triton | lab2 | `tools/setup_wsl.sh`（见下文） |
| Rust（rustup） | lab4、lab5 | `tools/setup_wsl.sh` |
| MPI（Open MPI 或 MPICH） | lab6 | WSL/Linux：`sudo apt install openmpi-bin libopenmpi-dev`；macOS：`brew install open-mpi` |

## WSL2 与 CUDA（lab1 开始前做）

1. 在 Windows PowerShell 执行 `wsl --install`，按提示重启，再安装 Ubuntu 24.04。
2. Windows 上装好最新的 NVIDIA 驱动即可。**不要在 WSL 里安装 Linux 显卡驱动**：Windows 驱动会自动透传进 WSL，所以 WSL 里 `nvidia-smi` 能用。但这**不代表**已经装了 CUDA Toolkit（`nvcc`）。参见 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
3. 在 WSL 里安装 Toolkit（需要 sudo，约 3 GB）：
   ```bash
   cd /tmp && wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-6
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   nvcc --version     # should print release 12.6
   ```
   一定要选 `wsl-ubuntu` 仓库，不要选普通的 `ubuntu2404`：后者会把 Linux 显卡驱动一起装上。
4. 代码位置：仓库可以就放在 Windows 盘上（WSL 里通过 `/mnt/d/...` 访问），但构建目录放到 Linux 文件系统里，编译会快很多：`cmake -S lab1 -B ~/build/lab1`。

> 为什么不用 Windows 原生的 nvcc？它要求特定版本的 MSVC 作为主机编译器。CUDA 12.6 支持 VS 2022（MSVC 14.3x）。作者机器上只有 VS 2026（MSVC 14.51），nvcc 的前端 `cudafe++` 会直接崩溃。WSL 里用 GCC 就没有这个问题。

## lab2–lab7 的一次性安装（WSL）

装好 CUDA 之后，在 WSL 里、仓库根目录下运行：

```bash
bash tools/setup_wsl.sh      # needs sudo once; ~15-30 min, ~12 GB download
source ~/.bashrc
bash tools/check_env.sh      # every required line must say OK
```

脚本会做五件事，可以重复运行（已装好的会跳过）：

| 步骤 | 内容 | 位置 |
|---|---|---|
| 1 | apt：`python3-venv`、`python3-dev`（Triton 运行时要编译 C 启动代码）、`libssl-dev`、Open MPI | 系统 |
| 2 | Python 虚拟环境：PyTorch（CUDA 12.6 版，自带 Triton）、numpy、matplotlib、pytest、transformers、safetensors、`hf` 命令 | `~/venvs/mlsys` |
| 3 | Rust stable（rustup） | `~/.cargo`、`~/.rustup` |
| 4 | `~/.bashrc`：CUDA 路径、cargo、`MLSYS_MODELS`，以及 `mlsys` 别名（用来激活虚拟环境） | |
| 5 | 模型：Qwen3-0.6B（lab4 主模型、lab5 草稿模型）和 Qwen3-1.7B（lab5 目标模型） | `~/models` |

说明：
- **为什么是 CUDA 12.6 版的 PyTorch**：驱动 566.x 最高支持 CUDA 12.7，PyTorch 的 cu128 及更新版本需要更新的驱动。以后升级了 Windows 驱动，也可以换用更新的版本。
- **不装 vLLM**：lab4 会自己实现推理引擎。如果想用 vLLM 当外部强基线，它的当前版本要求比 566.x 更新的驱动，需要先升级 Windows 驱动，再在**另一个**虚拟环境里安装。
- 如果下载 Hugging Face 模型太慢，先 `export HF_ENDPOINT=https://hf-mirror.com`，再运行脚本。
- **NCCL** 只在多 GPU 时有意义，lab6 会在 HPC 上使用，本机不需要装。

## 记录环境

每个 lab 开始时保存一次环境信息，写进报告的“环境”一节：

```bash
python -m tools.doctor --out results/environment/labN.json   # 解释器、包版本、nvidia-smi
```

## 卡住了怎么办

环境问题超过 30 分钟没解决：保存能复现问题的最短命令和完整报错，写进 [周报](templates/weekly-report.md)。超过 90 分钟就停止盲试，去问人或换一台机器。CPU 部分和读论文可以继续推进。
