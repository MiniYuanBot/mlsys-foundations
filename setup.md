# 项目环境配置指南

本仓库交付的是配置指导与检查工具。以下安装命令由你按周执行；不要提前装齐所有框架。基于 [课程环境要求](../mlsys-research-course/guides/environment.md)，入口文档核对于 2026-10-05，最终以**你安装版本的帮助与 wheel 要求**为准。

## 1. 选择运行地点

| 工作 | 推荐位置 | 环境 |
|---|---|---|
| B00、B01 CPU 正确性 | Mac / Windows 原生 / WSL2 | `.venv-torch` |
| B01 GPU、B02、lab2x | 4060 8 GB 的 WSL2 Linux | WSL 的 `.venv-torch` |
| lab3/L04及L06/L07的服务路线 | 同一 WSL2 Linux | 独立 `.venv-vllm` |
| B05调度模拟、指标/矩阵/评分器 | 任意 Python 3.10+ | 标准库即可；作图另装 matplotlib |

建议按课程使用 Python 3.12；如果已有受相应 wheel 支持的解释器，也可使用并记录。不要把 Windows 虚拟环境拷到 WSL 或 Mac，也不要把两个 GPU 环境合并。

先检查 `python --version` 和 `python -c "import sys; print(sys.executable)"`。本次制作只使用现有 Python；没有更改系统 PATH、驱动或环境。

## 2. Windows → WSL2（进入 GPU 章节时再做）

在 Windows PowerShell 检查 `wsl --status`、`wsl -l -v`。尚无 WSL2 时，按 [Microsoft WSL 安装](https://learn.microsoft.com/windows/wsl/install) 执行安装并按提示重启，进入 Linux 终端。

Windows NVIDIA 驱动和 WSL GPU 路径按 [NVIDIA 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 核对；**不要在 WSL 安装 Linux NVIDIA 显示驱动**。在 WSL 运行：

```bash
nvidia-smi
# 若不在 PATH，检查官方指南提到的入口：
/usr/lib/wsl/lib/nvidia-smi
```

记录 4060 是桌面/笔记本版、驱动、显存与电源模式。`nvidia-smi` 的 CUDA 字样是驱动兼容信息，不等于已安装 CUDA toolkit。仅在实际工具链要求时安装匹配 toolkit，先查官方说明。

把代码放在 WSL 的 Linux 文件系统，如 `~/projects/mlsys-research-labs`。首次可从 `/mnt/d/__Coding__/github/mlsys-research-labs` 复制**源码**到新目录，或初始化 Git 后 clone；不要复制 Windows 环境/缓存/权重。后续明确哪份是工作副本，用 Git 同步，避免修改两份后混用结果。GPU 实验从 Linux 副本根目录执行，课程仓库可保持相邻目录以便阅读链接。

## 3. 第一周：PyTorch 环境

Mac / WSL Bash（项目根目录）：

```bash
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install --upgrade pip
```

Windows 原生仅做 CPU 任务时：

```powershell
py -3.12 -m venv .venv-torch
.\.venv-torch\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

没有 `py -3.12` 时先自行安装支持的 Python，或用实际解释器创建环境。若激活被策略阻止，可直接调用 `.\.venv-torch\Scripts\python.exe`，不必修改全局执行策略。

打开 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)，按 OS 与 CPU/CUDA 平台复制安装命令。**本仓库不写死未经你硬件验证的 CUDA wheel**。安装后：

```bash
python -m pip install -r requirements-analysis.txt
python -m tools.doctor --require torch --out results/environment/torch.json
python -m pip freeze > results/environment/torch-requirements.txt
python -m unittest lab0.tests -v
```

预期 doctor 为 ready、CPU 梯度有限；lab0 尚未填 TODO 时失败属于课程起点。lab1 GPU 前再运行：

```bash
python -m tools.doctor --require cuda --out results/environment/cuda.json
```

## 4. 第4周：Triton

仅在 WSL 的 torch 环境中检查 `python -c "import triton; print(triton.__version__)"`。优先保留 PyTorch 已带的兼容版本；缺失时查 [Triton 安装](https://triton-lang.org/main/getting-started/installation.html)，不要无目的升级所有依赖。

```bash
python -m tools.doctor --require triton --out results/environment/triton.json
```

doctor 只检查导入、CPU/CUDA 小计算；**不证明 Triton 编译已通过**。按 [官方向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 跑一个非 block 整倍数输入，保存源码/命令/输出，再进入 lab2。解释器可用于有限正确性排查，不能当 CUDA 计时。

## 5. 第6周：独立 vLLM 环境

按 [vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) 选择与驱动匹配的稳定 wheel。其 PyTorch/CUDA 组合单独维护，不覆盖 torch 环境。

```bash
# 新 WSL 终端，项目根目录
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
# 在这里执行官方文档中与你的硬件/驱动匹配的安装命令
python -m pip install -r requirements-analysis.txt
python -m tools.doctor --require vllm --out results/environment/vllm.json
python -m pip freeze > results/environment/vllm-requirements.txt
vllm serve --help > results/environment/vllm-serve-help.txt
vllm bench serve --help > results/environment/vllm-bench-help.txt
```

不要把注释当作已安装。doctor 不能替代模型 smoke。模型、固定 snapshot 与首次启动详见 [lab3/setup.md](lab3/setup.md)。本仓库命令生成器只写文本，不自行启动服务。

## 6. 版本冻结与排错

`requirements-analysis.txt` 只约束作图库；它不是 GPU 环境锁。成功 smoke 后保存每个环境的 `pip freeze`、Python 路径、安装命令/索引、驱动和帮助；同一实验矩阵不升级框架。重建时在**同 OS/硬件路径的新环境**按原 wheel 来源安装，再使用对应冻结清单；两环境的清单不能交叉使用。

显存不足：检查自己其他进程 → max-num-seqs 4→2 → 上下文 2048→1536→1024 并统一缩短输入 → 仅预留比例超过可用内存时再调整 memory utilization。所有对照沿用同一降级；重做容量标定。参考 [课程降级顺序](../mlsys-research-course/guides/environment.md)。

CUDA/Triton/服务失败超过 30 分钟保存最短命令与完整错误，90 分钟停止盲试。用 [周报](templates/weekly-report.md) 标明环境阻塞；CPU 和分析练习可继续，GPU 验收保持未完成。HPC/付费机器不是前置；按课程说明确认账号、卡型与队列后再安排，在计算节点运行性能测试。

## 原生/系统工具链（按章节自行准备）

16周安排见[课程主线](../mlsys-research-course/syllabus.md)。上面的torch/Triton/服务环境需再按S层任务补工具链，额外环境不一次安装：

| 任务 | 需要什么 | 配置入口 |
|---|---|---|
| S00 C++ | C++17/CMake/CTest，CPU可跑 | [native setup](lab0/native/setup.md) |
| S01 CUDA | WSL Linux编译器、匹配驱动的CUDA Toolkit/nvcc/cuBLAS、CMake | [CUDA setup](lab1/cuda/setup.md) |
| S02 attention | 兼容torch/Triton，编译/资源诊断工具按需 | [attention setup](lab2/attention/setup.md) |
| S03/S04 runtime | 固定commit的独立源码工作副本/环境，真实引擎回归 | [runtime setup](lab3/runtime/setup.md)、[trace setup](lab4/trace/setup.md) |
| S05 distributed | PyTorch CPU/Gloo；C++17/MPI；多GPU再NCCL | [distributed setup](lab5/distributed/setup.md) |
| R C++ runtime | 固定llama.cpp源码与模型/转换版本，按上游构建 | [研究方案](lab6/research.md) |

PyTorch wheel能运行CUDA不代表已有nvcc和C++编译工具链；S01需要Toolkit，按NVIDIA WSL指南避免安装Linux显示驱动。架构与工具链按实际设备记录，4060通常对应Ada路径，Hopper/Blackwell专用实验不能直接套用。每个上游项目独立环境，固定commit/许可证/编译器；不把它的依赖直接覆盖本仓库环境。build/和external/已忽略，模型/编译缓存不提交。工具不会自动安装这些组件。
