---
title: lab2｜GPU 访存、Triton 与算子融合
tags: [mlsys, module, gpu, triton]
weeks: [4, 5]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab2｜GPU 访存、Triton 与算子融合

[索引](../guides/course-map.md) · [前一模块](lab1-measurement.md) · [实验 lab2](../labs/lab2-softmax.md) · [下一模块](lab3-serving.md)

**时间：3h＋3h＋5h。** 先修：lab1 计时函数。目标是看懂一个内核的工作分配与数据移动，并完成稳定 softmax；不要求一周精通 CUDA 或实现 FlashAttention。

## A 课：用访存解释性能，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 浏览 [LCPU 课程日程](https://infra.seminars.lcpu.dev/schedule)，定位 Session 1 与 Session 2；列出自己不熟的 GPU 术语 | 不超过 8 个术语，避免漫无目的补课 |
| 20–60 | 读 [Session 2.1 GEMM 讲义](https://infra.seminars.lcpu.dev/slides/session0201.pdf) 的朴素实现、共享内存分块部分（PDF 约第 3–9 页） | 手画一个 tile 从全局内存到计算的流动 |
| 60–95 | 对矩阵乘和逐元素加法分别估算 FLOPs/byte；理解带宽与计算上限 | 两个算术强度估算及忽略项，不追求硬件峰值的精确复刻 |
| 95–105 | 休息 | — |
| 105–145 | 从 [CS336](https://cs336.stanford.edu/spring2025/) 的 GPU / kernels 讲义补 coalescing、register、shared memory 的用途 | 表格：存储层级、谁共享、访问开销/容量限制 |
| 145–180 | 回看 FlashAttention 的分块思想；用一个 4×4 attention 小例子讨论避免保存完整 score 的意义 | 3 分钟口头解释稿，标明自己还没实现的部分 |

### 课内题

先做第 1、2 题，沿用本节 GEMM 与访存草图；第 3 题选做。这里只做地址和成本模型，不要求编译 CUDA。

1. **【定量·画图】分块到底少读了多少数据？** 对 FP32 的 `A:(256,128)`、`B:(128,256)`，比较每个输出独立读取完整 dot product 的朴素实现，与输出 tile 边长 T=`16/32` 的共享内存实现。假设无跨 block 缓存复用，tile 内 A/B 元素各从全局内存读取一次并复用，累加器留在寄存器，输出只写一次。分别算 FLOPs、全局读写字节、算术强度和 block 数；画一个输出 tile 沿 K 维迭代时的“加载→同步→计算→同步”流程，说明两处同步各保护什么。参考 [NVIDIA GEMM 指南 Figure 1 和 Larger tiles 图](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。自检须保留输出写入项；图中的复用上限不等于真实 L2/DRAM 流量或性能保证。
2. **【推导·画图】连续线程是否真的合并访问？** 一次 load 中有 32 个 lane，每个读一个 FP32；基址按 128 字节对齐，以 32 字节 sector 计，忽略缓存命中。比较 lane `i` 访问元素 `i`、`32*i`、`i+1` 三种方式。列出地址与 sector 编号，画 lane→sector 连线，计算请求覆盖的不同 sector 数和有效字节占比；再给出一个通过改变线程到行列的映射改善第二种模式的办法。参考 [CUDA Best Practices 的 Coalesced Access 图解](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)。本题的 sector 计数属于给定访存模型，不直接等于所有架构上的 HBM 事务数。
3. **【定量·画图，选做】矩阵只多一行一列，为何时间可能跳变？** 比较输出 `256×256` 与 `257×257`，tile 分别为 `64×64`、`128×128`。假设设备有 4 个 SM，每个 SM 对两种配置都最多同时运行 2 个 block，所有 block 耗时相同，越界输出用 mask 丢弃但计算槽位仍占用。计算 block 数、wave 数、最后一 wave 的槽位利用率，以及有效输出/总 tile 槽位比例，画四种调度图。再列出要把这张图变成性能预测还缺少的资源信息。参考 [NVIDIA GEMM 指南的 Tile/Wave Quantization 图](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)；可对照 [Triton matmul 的 grouped ordering 图](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 思考 launch 顺序怎样影响复用，但不把逻辑顺序画成硬件必然执行的顺序。

## B 课：从向量加法走到 softmax，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–45 | 逐段读并运行 [Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；只跑一个小输入 | 标出 program id、offset、mask、load/store 的含义 |
| 45–80 | 将向量长度改成非 block 整数倍；先预测尾部索引，再观察 mask 的作用 | 一次边界测试，解释越界为何危险 |
| 80–90 | 休息 | — |
| 90–135 | 读 [Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 的稳定公式与逐行归约内核 | 按 load→max→exp→sum→divide→store 写伪代码 |
| 135–160 | 按 [在线 OJ 路线](../guides/online-judge.md)，用 Tensara 或 LeetGPU 的 Vector Addition 核对 Triton 模板并尝试最小评测；平台阻塞则改查 [LCPU Assignment 01](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment01/README.md) 的解释器调试说明 | 一份接口/设备/结果记录，或明确阻塞和本地替代，不追加整套作业 |
| 160–180 | 制定 lab2 的形状矩阵、参考实现和错误检查；说明哪些教程代码被借用 | 实验计划与代码来源注释 |

### 课内题

先做第 1、2 题，用来检查本节 softmax 伪代码；第 3 题选做。数值部分可在 CPU 复现，无 GPU 时保留 kernel 设计与手算结果。

1. **【定量·复现】尾部 mask 必须同时保护地址和归约。** 向量长度 N=`2050`、block=`1024` 时，列出最后一个 program 的有效与无效 offset 范围。另取一行 `[-1000,-1001,-999]`，内部补齐到 4 个元素，比较将补位填 0 与填 `-∞` 后执行稳定 softmax 的行最大值、分母及三个有效输出。用 CPU float64 复现，检查有效输出和为 1、加任意常数后结果不变；再测试 `[1000,999,1001]` 与未经稳定处理的实现。参考 [Triton 向量加法的 mask](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 和 [Fused Softmax 的按行补齐](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。交付索引图与误差表；不要实际取消 mask 去触发越界访问。
2. **【定量·画图】给融合的收益写一份可核查账本。** 令输入为 FP32、`M=128,N=1025`，朴素版本依次用 5 个 kernel 做 max、减最大值、exp、sum、divide，每步结果写回全局内存；行归约结果为 M 个元素，广播读取按每行一次计。画中间张量的数据流，逐步列读写量；再与“输入读一次、输出写一次，归约留在片上”的融合模型比较。假设带宽 `400 GB/s`、每个 kernel 启动成本 `5 μs`，按“启动成本之和＋字节数/带宽”估算两者时间与加速比，并注明该模型忽略的算术、缓存和资源限制。对照 [Triton softmax 教程末尾的 GB/s 曲线](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，说明为何按 `2×输入字节/耗时` 定义的有效带宽不是 profiler 测得的实际总流量，也不能预设融合一定胜过库函数。
3. **【推导·复现，选做】不保存完整概率行，能得到相同的加权和吗？** 给定 scores=`[1000,999,1001,998,1002]`、values=`[1,-1,2,0,3]`，按 `2+2+1` 分块读取。只保留已读 scores 的最大值 m、`l=Σexp(score−m)` 和 `o=Σexp(score−m)×value`，从这些定义推导新块到来时的更新式，特别处理最大值改变后的旧状态缩放。画状态流转图，列出每个块后的 `(m,l,o)`；用 CPU float64 与整行稳定 softmax 的加权和核对，并改变块顺序做误差检查。参考 [FlashAttention 的 Figure 1 与 Algorithm 1](https://arxiv.org/pdf/2205.14135)。这是一个标量 values 的教学缩小版，不包含论文的并行调度、反向传播和 GPU 性能结论。

## 本周实验与验收

[lab2](../labs/lab2-softmax.md) 只要求一个可解释的 kernel 和可信对照。你可以从官方教程改写，但报告应区分教程原有实现与你亲自完成的边界处理、形状实验和解释。若最喜欢这个模块，在 lab5 的方向表中提高 kernel/compiler 的兴趣评分。

借助 AI 实现时，优先选择 lab2 的 300 分钟增强表，增加性质测试、参数消融与留出形状。想做完整的第二个算子，可插入 [lab2x 融合残差与 RMSNorm](../labs/lab2x-kernel-challenge.md)；想体验不同硬件可选 [TensorOJ Ascend C 路线](../guides/online-judge.md)。完整 OJ 题面可能要求任意维度、多 dtype 或更大形状，不把本地 softmax 直接当作已满足远端规格。

## 资料汇总

必读：[LCPU 日程](https://infra.seminars.lcpu.dev/schedule)、[GEMM 讲义](https://infra.seminars.lcpu.dev/slides/session0201.pdf)、[Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Triton softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。选做：LCPU CUDA/GEMM 其他作业、CS336 Assignment 2；它们不是本周 5 小时实验的附加要求。

在线补充：[平台选择与题目路线](../guides/online-judge.md) · [AI 辅助实验](../guides/ai-assisted-labs.md)。
