# 环境配置

整个课程只用**一套环境**：一个 Python 虚拟环境 `mlsys`，再加上 C++ 编译器、CUDA、Rust 和 MPI。本页命令都在 `labs/` 目录执行。版本统一锁定在 [tools/env-versions.env](tools/env-versions.env)。环境**按 lab 分阶段安装**：每开始一个新 lab，就用更大的 lab 编号再运行一次安装脚本，它只补装还缺的依赖。

| 平台                                | 能做的 lab       | 安装与检查（在 `labs/` 目录执行）                                                                                                              |
| ----------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **WSL2 Ubuntu 24.04**（课程主环境） | 全部             | `bash tools/setup_env.sh --lab N`<br>`bash tools/check_env.sh --lab N`                                                                         |
| Linux（Ubuntu 24.04 + NVIDIA 显卡） | 全部             | 同上                                                                                                                                           |
| macOS（Apple Silicon）              | lab0、lab6、lab7 | 同上                                                                                                                                           |
| Windows 原生（PowerShell）          | lab0、lab7       | `powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab N`<br>`powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab N` |

`check_env` 只检查第 N 个 lab 需要的东西，每个 `[FAIL]` 下面都会给出修复命令。在 Windows 上，lab1–lab6 都在 WSL 里做（见[下文](#wsl2-与-gpu-驱动只做一次)）；在 Git Bash 里运行 `.sh` 脚本，会提示你改用 PowerShell 或 WSL。其他 Linux 发行版也能用 `setup_env.sh`，只是脚本不会替你装系统包，而是列出缺少的命令。

## 你的机器分别做什么

| 机器                             | 适合                                                             | 不适合                                    |
| -------------------------------- | ---------------------------------------------------------------- | ----------------------------------------- |
| Windows + RTX 4060 8 GB          | **全部 lab，都在 WSL2 里做**；lab0、lab7 也可以在 Windows 原生做 | 多 GPU 通信、Hopper 专属特性（TMA/WGMMA） |
| Mac mini（Apple Silicon，16 GB） | lab0（NEON 微内核）、lab6 的 MPI 部分、lab7                      | 任何 CUDA（lab1–lab5）                    |
| 北大 HPC / 组里服务器            | lab6 的 NCCL 多卡实验；H100 上的可选挑战                         | 日常开发与调试（排队太慢）                |

建议整个课程都用 WSL：lab0 在 WSL 里配好，后面的 lab 只是往同一套环境里补装。Windows 原生的 nvcc 要求特定版本的 MSVC（作者机器上 CUDA 12.6 和 VS 2026 不兼容），所以 Windows 原生只支持不用 CUDA 的 lab0 和 lab7。

## 各 lab 新增的内容

| N   | 新装的东西                                                                                    | 位置                   |
| --- | --------------------------------------------------------------------------------------------- | ---------------------- |
| 0   | C++ 编译器、CMake、direnv；虚拟环境：numpy、matplotlib、pytest                                | 系统、`~/venvs/mlsys`  |
| 1   | CUDA Toolkit 12.6（约 3 GB；只装 Toolkit，不装显卡驱动）                                      | `/usr/local/cuda-12.6` |
| 2   | PyTorch 2.14.1（cu126 版，自带 Triton 3.8.0）                                                 | `~/venvs/mlsys`        |
| 3   | 无（NVRTC 已随 CUDA Toolkit 安装）                                                            |                        |
| 4   | Rust 1.99.0、`libssl-dev`；transformers、safetensors、`hf` 命令；模型 Qwen3-0.6B（约 1.5 GB） | `~/.cargo`、`~/models` |
| 5   | 模型 Qwen3-1.7B（约 4 GB）                                                                    | `~/models`             |
| 6   | Open MPI                                                                                      | 系统                   |
| 7   | 无                                                                                            |                        |

各平台的系统包来源：WSL/Ubuntu 用 apt（需要 sudo），macOS 用 Homebrew，Windows 用 winget 安装 MSYS2（UCRT64 的 gcc、cmake、ninja）和 Python 3.12。如果 PATH 里已经有可用的编译器或 Python，脚本会直接用现成的。

脚本还会改一个 shell 配置文件：WSL/Linux 是 `~/.bashrc`，macOS 是 `~/.zshrc`，写入 direnv 钩子、`mlsys` 命令，以及 CUDA、cargo 和 `MLSYS_MODELS` 的环境变量；Windows 是 PowerShell 的 profile，写入 `mlsys` 命令。不想下载模型，可以加 `--no-models`。

## 虚拟环境放在哪里，怎么生效

虚拟环境放在**仓库外面**：`~/venvs/mlsys`，Windows 上是 `%USERPROFILE%\venvs\mlsys`。原因有三个：

- **同一个文件夹会被两个系统使用。** 仓库通常放在 Windows 盘上，Windows 原生通过 `D:\...` 访问，WSL 通过 `/mnt/d/...` 访问。虚拟环境和操作系统绑定：Linux 的虚拟环境是 `bin/python`，Windows 的是 `Scripts\python.exe`。放在仓库里的同一个 `.venv` 会被两边互相覆盖。
- **WSL 读写 Windows 盘很慢。** PyTorch 有好几 GB，装在 `/mnt/d` 上安装慢，`import torch` 也慢。放在 WSL 自己的文件系统里要快得多。
- **多个 clone 共用一份环境。** 学习用的 checkout 和参考答案副本共用同一份环境，不用重复下载 PyTorch。执行 `git clean -fdx` 也不会把环境删掉。

如果确实想把环境放进项目里，先设置 `MLSYS_VENV`（例如 `export MLSYS_VENV=$PWD/.venv`，并写进 shell 配置文件），再运行安装脚本。`.venv*` 已被 git 忽略。

**环境怎么生效。** `ModuleNotFoundError: No module named 'numpy'` 几乎都是同一个原因：当前用的 `python3` 是系统自带的（比如 `/usr/bin/python3`），不是课程环境里的。Ubuntu 24.04 也不允许直接往系统 Python 里 `pip install`（会报 `externally-managed-environment`），所以**不要**用 `pip install --user` 或 `--break-system-packages` 来“修”。

- **WSL / Linux / macOS**：仓库根目录的 `.envrc` 由 [direnv](https://direnv.net/) 执行。你 `cd` 进仓库时，它把课程环境放到 PATH 最前面，离开仓库时自动恢复。所以在仓库里，`python`、`python3`、`pytest` 都是课程环境里的版本，其他项目不受影响。第一次安装后要**新开一个终端**，钩子才会生效。在另一个 clone 里第一次 `cd` 进去时，direnv 会提示 `.envrc is blocked`，在仓库根目录执行一次 `direnv allow` 即可。
- **Windows PowerShell**：没有自动激活。每个新终端先执行一次 `mlsys`（等价于运行 `~\venvs\mlsys\Scripts\Activate.ps1`）。安装脚本在虚拟环境里额外放了一个 `python3.exe`，激活之后 README 里的 `python3 ...` 命令也能直接用，不会跑到 Microsoft Store 的占位程序上。为了允许激活脚本运行，安装脚本会把当前用户的执行策略设为 `RemoteSigned`（[Python 官方文档](https://docs.python.org/3/library/venv.html)推荐的做法）。
- **任何平台、仓库外面**：执行 `mlsys` 手动激活。Git Bash 里用 `source ~/venvs/mlsys/Scripts/activate`。

## WSL2 与 GPU 驱动（只做一次）

1. 在 Windows PowerShell 执行 `wsl --install`，按提示重启，再安装 Ubuntu 24.04。
2. Windows 上装好最新的 NVIDIA 驱动即可。**不要在 WSL 里安装 Linux 显卡驱动**：Windows 驱动会自动透传进 WSL，所以 WSL 里 `nvidia-smi` 能用。但这**不代表**已经装了 CUDA Toolkit（`nvcc`），Toolkit 由 `setup_env.sh --lab 1` 从 NVIDIA 的 `wsl-ubuntu` 仓库安装，这个仓库里不带显卡驱动。参见 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
3. 代码位置：仓库可以放在 Windows 盘上（WSL 里通过 `/mnt/d/...` 访问），但构建目录放到 Linux 文件系统里，编译会快很多，例如 `cmake -S lab1 -B ~/build/lab1`。

## Linux（原生）

显卡驱动要自己装，驱动版本要支持 CUDA 12.6（≥ 560），例如在 Ubuntu 上执行 `sudo ubuntu-drivers install` 后重启。然后运行 `bash tools/setup_env.sh --lab N`。在 Ubuntu 24.04 x86_64 上，脚本会从 NVIDIA 的 `ubuntu2404` 仓库安装 `cuda-toolkit-12-6`：这个包只有 Toolkit，不会替换你的驱动。其他发行版请按 [CUDA 12.6 下载页](https://developer.nvidia.com/cuda-12-6-0-download-archive)自行安装 Toolkit 到 `/usr/local/cuda-12.6`，再重跑脚本。

## macOS

先执行 `xcode-select --install`，再安装 [Homebrew](https://brew.sh)。然后运行 `bash tools/setup_env.sh --lab 0`（做 lab6 时用 `--lab 6`）。脚本会用 Homebrew 的 Python 3.12 创建 `~/venvs/mlsys`。在 macOS 上，传入 `--lab 1` 到 `--lab 5` 会直接报错退出。

## Windows 原生（lab0、lab7）

在 PowerShell 里、`labs/` 目录下执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 0
# open a NEW PowerShell window, cd back into labs/
mlsys
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 0
```

PATH 里没有 `g++`、`cmake`、`ninja` 时，脚本会用 winget 安装 MSYS2，并把 `C:\msys64\ucrt64\bin` 加进用户 PATH。构建 lab0 时要加 `-G Ninja`，否则装了 Visual Studio 的机器会改用 MSVC：`cmake -S lab0 -B build/lab0 -G Ninja -DCMAKE_BUILD_TYPE=Release`。AddressSanitizer 在 MinGW 上不可用，调试内存错误请去 WSL。

## 版本说明

- **为什么是 CUDA 12.6 版的 PyTorch**：驱动 566.x 最高支持 CUDA 12.7，PyTorch 的 cu128 及更新版本需要更新的驱动。以后升级了驱动，可以在 `tools/env-versions.env` 里一起改 `CUDA`、`TORCH` 和 `TORCH_INDEX`，再重跑安装脚本。
- **升级任何版本**都只改 `tools/env-versions.env`（Rust 在 `tools/formatter-versions.env`），并在同一个提交里更新本页。`check_env` 遇到与锁定版本不同的包会报 `[WARN]`：通常还能用，但要在报告里写明。
- **不装 vLLM**：lab4 会自己实现推理引擎。如果想用 vLLM 当外部强基线，它的当前版本要求比 566.x 更新的驱动，需要先升级驱动，再在**另一个**虚拟环境里安装，不要装进 `~/venvs/mlsys`。
- 如果下载 Hugging Face 模型太慢，先执行 `export HF_ENDPOINT=https://hf-mirror.com`，再运行脚本。
- **NCCL** 只在多 GPU 时有意义，lab6 会在 HPC 上使用，本机不需要装。

## 记录环境

每个 lab 开始时保存一次环境信息，写进报告的“环境”一节（Windows 原生把第一行换成 `check_env.ps1`）：

```bash
bash tools/check_env.sh --lab N > results/environment/labN-check.txt
python -m tools.doctor --out results/environment/labN.json   # interpreter, package versions, nvidia-smi
```

## 卡住了怎么办

1. 先跑 `check_env`，照 `[FAIL]` 下面的 `fix:` 处理。
2. 环境问题超过 30 分钟没解决：保存能复现问题的最短命令和完整报错，写进[周报](templates/weekly-report.md)。超过 90 分钟就停止盲试，去问人或换一台机器。CPU 部分和读论文可以继续推进。
