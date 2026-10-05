---
title: 设备、环境与算力使用
tags: [mlsys, environment]
---

# 设备、环境与算力使用

[返回大纲](../syllabus.md) · [实验规范](experiments.md)

## 设备分工

| 设备 | 本课程用途 | 不能据此推出的结论 |
|---|---|---|
| Mac mini，16 GB 统一内存 | 阅读、笔记、Python、lab0、CPU 正确性、lab5 模拟、数据分析；MPS 可作为选做 | Mac CPU/MPS 与 NVIDIA CUDA 的耗时不能直接拼成一条优化对比曲线 |
| Windows＋RTX 4060 8 GB | WSL2 Linux 下的 PyTorch CUDA、Triton、0.5B 模型推理 | 小模型单卡现象不自动推广到 70B、多卡或生产服务 |
| Intel UHD 集显 | 无必做任务 | 不把它当成第二块可用于 NCCL 的 CUDA GPU |
| 北大 HPC / 课题组服务器 | 后续外部有效性验证、多卡研究 | 未确认账号、卡型、驱动、队列、费用前，不假定可用 |
| 在线算子 OJ | 通过浏览器/平台工具做远端算子评测，详见 [平台手册](online-judge.md) | 远端卡型、权限和额度需核实；不能默认支持常驻 vLLM 服务或自定义 profiler |

必做实验采用小尺寸张量和 `Qwen/Qwen2.5-0.5B-Instruct`。8 GB 是课程设计约束，不是显存充足的保证。先运行检查点，再扩大规模。

TensorOJ 所查题目面向 Ascend C / 昇腾 NPU；本地 RTX 4060 的 CUDA 环境不能运行这类 NPU kernel。选择其拓展路线时优先使用平台指定模板与版本，Mac 可用于编辑、CPU 参考与提交准备。在线评测无法导出所需数据时，仍按原实验要求在自己的 CUDA 机器完成测量，未完成项如实标注。

## 第一周：先打通 Python 与 PyTorch

在 Mac 或 WSL 的一个工作目录中建独立环境。示例使用 Python 3.12；若本机没有这个解释器，先安装受当前官方版本支持的 Python，再把命令中的版本改成实际版本。不要在现有项目环境里覆盖依赖。

```bash
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install --upgrade pip
```

打开 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)，按当前操作系统与计算平台复制安装命令。Mac 使用对应的 macOS 命令；WSL 使用 Linux/CUDA 命令。不要凭本文猜 CUDA wheel 版本，也不要将 `nvidia-smi` 显示的 CUDA 兼容版本当作本地 toolkit 版本。

之后装实验记录工具，并保存依赖：

```bash
python -m pip install numpy matplotlib
mkdir -p results/environment
python -m pip freeze > results/environment/torch-requirements.txt
python --version > results/environment/python-version.txt
```

运行以下检查，保存输出。它只做小计算，不是性能测试。

```python
import platform
import torch
print('platform:', platform.platform())
print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
x = torch.randn(8, 8, requires_grad=True)
(x @ x.T).sum().backward()
print('CPU gradient finite:', bool(torch.isfinite(x.grad).all()))
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    y = torch.randn(128, 128, device='cuda')
    z = y @ y
    torch.cuda.synchronize()
    print('CUDA output finite:', bool(torch.isfinite(z).all()))
```

CPU 梯度通过即可开始 lab0。CUDA 未通过时把结果标记为阻塞，不用重新学机器学习。

## Windows 上的 CUDA 路径

按 [NVIDIA CUDA on WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 安装/更新支持 WSL 的 Windows NVIDIA 驱动，并使用 WSL2 Linux。指南明确区分 Windows 驱动与 WSL 内部的软件包：**不要在 WSL 中安装 Linux NVIDIA 显示驱动**。

在 WSL 内运行 `nvidia-smi`；若命令不在 PATH，按该指南检查 `/usr/lib/wsl/lib/nvidia-smi`。记录 GPU 名称、驱动、可用显存，以及 4060 是笔记本版还是桌面版。笔记本接电、固定电源模式。关掉占用 GPU 的游戏、浏览器重任务和其他训练进程，但不要擅自结束不认识的系统进程。

将实验代码与结果放在 WSL 自己的 Linux 文件系统中，避免把 Windows 挂载盘 I/O 混入测量。每次实验先确认显存占用；`nvidia-smi` 在 WSL 的部分监控功能有限，因此不把某个利用率数字作为唯一证据。

第4周使用 [Triton 官方向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 检查工具链。优先使用与 PyTorch 环境兼容的 Triton 依赖，避免无目的升级所有包。若教程报 API 不匹配，记录当前版本并查对应版本文档。

## 第6周：单独建 vLLM 环境

[vLLM GPU 安装文档](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) 以 Linux 为主；本课程选择 WSL2。Mac 上存在社区路径，但课程不要求你维护第二套服务栈。

新建 `.venv-vllm`，按该文档安装稳定发布版。vLLM 的二进制包依赖特定 PyTorch/CUDA 组合，不要为了复用 `.venv-torch` 强行覆盖。在新的 WSL 终端、项目根目录执行：

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
mkdir -p results/environment
```

然后从安装文档选择匹配实际驱动的安装路径，确认下载的是对应发布版 wheel。安装成功后执行并保存：

```bash
python -m pip freeze > results/environment/vllm-requirements.txt
vllm serve --help > results/environment/vllm-serve-help.txt
vllm bench serve --help > results/environment/vllm-bench-help.txt
```

使用 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 所对应的模型。记录模型仓库名、实际下载的 revision/commit、tokenizer revision 和下载时间。后续所有对照使用同一份权重与 tokenizer；不在实验中途自动换版本。

本课程的保守起始配置如下，推理服务 lab（lab4）会详细说明使用方法：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype half --max-model-len 2048 \
  --max-num-seqs 4 --gpu-memory-utilization 0.70 \
  --enforce-eager --enable-chunked-prefill \
  --max-num-batched-tokens 2048 --no-enable-prefix-caching
```

参数参照 [服务 CLI](https://docs.vllm.ai/en/stable/cli/serve/)。先检查安装版本的帮助是否包含这些选项。若不包含，查看对应版本文档，不要静默删掉实验中的关键开关。保存启动日志，确认最终生效配置。`--enforce-eager` 用于减少入门时图捕获相关的显存与启动复杂性；因此结果只代表此模式。

## 显存不足的处理顺序

1. 分辨是 Windows/其他进程占用、权重加载失败，还是 KV cache/并发不足；保存错误与可用显存。
2. 结束自己启动且不用的实验服务。每次只运行一台服务。检查是否误用了 7B 或更大模型。
3. 先将 `max-num-seqs` 从 4 降至 2；仍失败时将上下文降至 1536，再至 1024，同时缩短输入，确保输入＋最大输出不超过限制。lab7 的长输入条件必须相应改写。
4. 若启动日志显示预留比例超过当前可用显存，才降低 `gpu-memory-utilization`；这个参数降低后 KV cache 也变少，它不是所有 OOM 的通用解法。
5. 所有对照统一使用降级后的配置，重做容量标定。不要只缩小表现差的那个实验。

仍失败时，本周可完成论文、CPU 检查和负载设计，GPU 验收保持未完成。带着错误请组内成员确认服务器，而不是立即租长时间机器。

## 北大 HPC：先确认再安排实验

根据 [官方开户说明](https://hpc.pku.edu.cn/ug/account/)，科研账号与项目负责人、成员管理相关；教学用途另有安排。不能仅凭学生身份推断已有科研账号或免费额度。[未名二号说明](https://hpc.pku.edu.cn/ug/cluster/wm2/) 展示节点信息，但页面没有给出足以确认本课程所需 GPU 型号的完整规格。旧硬件页面也不能代表当前可分配设备。

向平台或导师确认：GPU 型号/显存/计算能力、驱动和容器支持、单卡/多卡权限、排队时长、最长作业时间、模型文件下载方式、计费归属及预算。当前 vLLM 的硬件要求需对照安装文档检查；显存大不代表架构一定兼容。

阅读 [Slurm 使用说明](https://hpc.pku.edu.cn/ug/guide/slurm/) 与 [sbatch 文档](https://hpc.pku.edu.cn/ug/guide/slurm/sbatch/)。**训练与性能测试在分配的计算节点运行，不在登录/传输节点运行。** 下面仅是结构示意，所有大写占位项需换成平台核准值；不要直接提交。

```bash
#!/bin/bash
#SBATCH --job-name=mlsys-smoke
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=smoke-%j.log
set -euo pipefail
source /ABSOLUTE/PATH/TO/VENV/bin/activate
nvidia-smi
python /ABSOLUTE/PATH/TO/PROJECT/smoke.py
```

具体 GPU 请求语法、分区、账户选项以实际平台为准；这里只演示一次短测试作业。`smoke.py` 可由本页前面的 PyTorch 环境检查代码另存得到。远端服务基准的客户端也应按平台规定运行，课程不要求在公网开放端口。费用按 [官方缴费说明](https://hpc.pku.edu.cn/ug/account/recharge/) 与实际账号确认，不在本课纲预估未知卡型单价。

## 环境验收与资料

- [ ] CPU 自动微分有限值检查通过。
- [ ] CUDA 矩阵乘检查通过，或已提交阻塞记录。
- [ ] 软件版本、驱动、卡型与内存限制已记录。
- [ ] 第6周另建 vLLM 环境，保留 CLI 帮助与启动日志。
- [ ] 没有把 HPC、多卡或付费资源当作前四周前提。

参考：[PyTorch 安装](https://pytorch.org/get-started/locally/) · [WSL CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) · [vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) · [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [北大 HPC](https://hpc.pku.edu.cn/)

## 系统实现补充

16周安排与各阶段资源见[主线](../syllabus.md)。按章节准备C++17/CMake、WSL CUDA Toolkit/nvcc/cuBLAS、MPI；PyTorch wheel不等于完整CUDA编译工具链。按配套[全局setup](../../mlsys-research-labs/setup.md)与子项目setup逐步由你配置，分开源码引擎环境。CPU/Gloo/MPI提供真实进程或通信语义，不能证明多GPU/NCCL吞吐。FA3/Hopper、Blackwell特定路径和完整RL大模型只作指定硬件阅读/远端选修。
