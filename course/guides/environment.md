---
title: 设备、环境与算力使用
tags: [mlsys, environment]
---

# 设备、环境与算力使用

[返回大纲](../syllabus.md) · [实验规范](experiments.md)

安装步骤全部写在 `labs/` 中的 [setup.md](../../labs/setup.md)，本页只讲设备分工、显存限制和 HPC。

## 设备分工

| 设备 | 本课程用途 | 不能据此推出的结论 |
|---|---|---|
| Windows＋RTX 4060 8 GB | 课程主环境，WSL2 里做全部 lab；lab0、lab7 也可在 Windows 原生做 | 小模型单卡现象不自动推广到 70B、多卡或生产服务 |
| Mac mini，16 GB 统一内存 | lab0、lab6 的 MPI 部分、lab7；阅读、笔记与数据分析 | Mac CPU 与 NVIDIA CUDA 的耗时不能拼成一条优化对比曲线 |
| Intel UHD 集显 | 无必做任务 | 不把它当成第二块可用于 NCCL 的 CUDA GPU |
| 北大 HPC / 课题组服务器 | lab6 的多卡实验、后续外部有效性验证 | 未确认账号、卡型、驱动、队列、费用前，不假定可用 |
| 在线算子 OJ | 通过浏览器/平台工具做远端算子评测，详见 [平台手册](online-judge.md) | 远端卡型、权限和额度需核实；单个算子题不能替代 lab4/lab5 的端到端实验 |

lab1–lab5 需要 NVIDIA GPU，lab6 需要 MPI。各平台能做的 lab：

- WSL2：全部 lab，是课程主环境。
- Linux（带 NVIDIA 显卡）：全部 lab。
- Windows 原生：lab0 和 lab7。
- Mac：lab0、lab6 和 lab7。

TensorOJ 所查题目面向 Ascend C / 昇腾 NPU；本地 RTX 4060 的 CUDA 环境不能运行这类 NPU kernel。选择其拓展路线时优先使用平台指定模板与版本，Mac 可用于编辑、CPU 参考与提交准备。在线评测无法导出所需数据时，仍按原实验要求在自己的 CUDA 机器完成测量，未完成项如实标注。

## 安装与检查

按 [setup.md](../../labs/setup.md) 做。每开始一个新 lab 运行一次 `setup_env`，它只补装这个 lab 新增的东西；`check_env` 逐项检查并给出修复命令。版本锁定在 `labs/` 中的 `labs/tools/env-versions.env`，不要凭 `nvidia-smi` 显示的 CUDA 版本猜 PyTorch wheel 该装哪个。

Windows 上装好支持 WSL 的 NVIDIA 驱动即可，**不要在 WSL 中安装 Linux 显卡驱动**，参见 [NVIDIA CUDA on WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。在 WSL 内运行 `nvidia-smi`；若命令不在 PATH，按该指南检查 `/usr/lib/wsl/lib/nvidia-smi`。记录 GPU 名称、驱动、可用显存，以及 4060 是笔记本版还是桌面版。笔记本接电、固定电源模式，关掉占用 GPU 的游戏和浏览器重任务，但不要擅自结束不认识的系统进程。

构建目录放在 WSL 自己的 Linux 文件系统里（例如 `-B ~/build/lab1`），避免把 Windows 挂载盘的 I/O 混进测量。每次实验先确认显存占用；`nvidia-smi` 在 WSL 的部分监控功能有限，因此不把某个利用率数字作为唯一证据。

## 模型

lab4 使用 Qwen3-0.6B，lab5 再加上 Qwen3-1.7B 作为目标模型，由 `setup_env` 下载到 `~/models`。记录模型仓库名、实际下载的 revision/commit、tokenizer revision 和下载时间。后续所有对照使用同一份权重与 tokenizer；不在实验中途换版本。

必做实验采用小尺寸张量和这两个模型。8 GB 是课程设计约束，不是显存充足的保证。先运行检查点，再扩大规模。

## 显存不足的处理顺序

1. 分辨是 Windows/其他进程占用、权重加载失败，还是 KV cache 不够；保存错误与当时的可用显存。
2. 结束自己启动且不用的进程。同一时刻只跑一个实验。检查是否误用了更大的模型。
3. 降低并发序列数，再缩短上下文长度，同时确保输入＋最大输出不超过限制。相应条件下的实验要一并改写，不能只缩小表现差的那一个。
4. 所有对照统一使用降级后的配置，重做容量标定。
5. 仍失败时，本周可完成论文、CPU 部分和负载设计，GPU 验收保持未完成。带着错误请组内成员确认服务器，而不是立即租长时间机器。

## 北大 HPC：先确认再安排实验

根据 [官方开户说明](https://hpc.pku.edu.cn/ug/account/)，科研账号与项目负责人、成员管理相关；教学用途另有安排。不能仅凭学生身份推断已有科研账号或免费额度。[未名二号说明](https://hpc.pku.edu.cn/ug/cluster/wm2/) 展示节点信息，但页面没有给出足以确认本课程所需 GPU 型号的完整规格。旧硬件页面也不能代表当前可分配设备。

向平台或导师确认：GPU 型号/显存/计算能力、驱动和容器支持、单卡/多卡权限、排队时长、最长作业时间、模型文件下载方式、计费归属及预算。显存大不代表架构一定兼容。

阅读 [Slurm 使用说明](https://hpc.pku.edu.cn/ug/guide/slurm/) 与 [sbatch 文档](https://hpc.pku.edu.cn/ug/guide/slurm/sbatch/)。**训练与性能测试在分配的计算节点运行，不在登录/传输节点运行。** `labs/` 中的 [lab6/hpc/](../../labs/lab6/hpc/README.md) 有一份 NCCL 作业脚本，其中的大写占位项需换成平台核准值。具体 GPU 请求语法、分区、账户选项以实际平台为准。费用按 [官方缴费说明](https://hpc.pku.edu.cn/ug/account/recharge/) 与实际账号确认，不在本课纲预估未知卡型单价。

## 环境验收

- [ ] 对应 lab 的 `check_env` 全部为 OK，输出保存进报告的“环境”一节。
- [ ] 软件版本、驱动、卡型与显存限制已记录。
- [ ] GPU 未通过时已提交阻塞记录，而不是反复盲试。
- [ ] 没有把 HPC、多卡或付费资源当作前几周的前提。

参考：[labs setup.md](../../labs/setup.md) · [WSL CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) · [PyTorch 安装](https://pytorch.org/get-started/locally/) · [北大 HPC](https://hpc.pku.edu.cn/)

## 系统实现补充

16 周安排与各阶段资源见[主线](../syllabus.md)。本课程的实验是自己写 C++/CUDA/Rust 实现，不依赖现成的推理框架：PyTorch 只用来做参考实现和画图，PyTorch wheel 也不等于完整的 CUDA 编译工具链。CPU/MPI 实验提供真实进程与通信语义，不能证明多 GPU/NCCL 吞吐。FA3/Hopper、Blackwell 特定路径只作指定硬件的阅读或远端选修。
