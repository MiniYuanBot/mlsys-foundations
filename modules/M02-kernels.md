---
title: M02｜GPU 访存、Triton 与算子融合
tags: [mlsys, module, gpu, triton]
week: 3
status: not-started
---

# M02｜GPU 访存、Triton 与算子融合

[索引](../index.md) · [前一模块](M01-measurement.md) · [实验 L02](../labs/L02-softmax.md) · [下一模块](M03-serving.md)

**时间：3h＋3h＋5h。** 先修：L01 计时函数。目标是看懂一个内核的工作分配与数据移动，并完成稳定 softmax；不要求一周精通 CUDA 或实现 FlashAttention。

## A 课：用访存解释性能，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 浏览 [LCPU 课程日程](https://infra.seminars.lcpu.dev/schedule)，定位 Session 1 与 Session 2；列出自己不熟的 GPU 术语 | 不超过 8 个术语，避免漫无目的补课 |
| 20–60 | 读 [Session 2.1 GEMM 讲义](https://infra.seminars.lcpu.dev/slides/session0201.pdf) 的朴素实现、共享内存分块部分（PDF 约第 3–9 页） | 手画一个 tile 从全局内存到计算的流动 |
| 60–95 | 对矩阵乘和逐元素加法分别估算 FLOPs/byte；理解带宽与计算上限 | 两个算术强度估算及忽略项，不追求硬件峰值的精确复刻 |
| 95–105 | 休息 | — |
| 105–145 | 从 [CS336](https://cs336.stanford.edu/spring2025/) 的 GPU / kernels 讲义补 coalescing、register、shared memory 的用途 | 表格：存储层级、谁共享、访问开销/容量限制 |
| 145–180 | 回看 FlashAttention 的分块思想；用一个 4×4 attention 小例子讨论避免保存完整 score 的意义 | 3 分钟口头解释稿，标明自己还没实现的部分 |

课内题：为什么更大的 tile 既可能增加复用，又可能降低并行度？为什么“更高算术强度”不保证小张量更快？回答必须提及资源占用和启动/调度开销，不能只说“GPU 很复杂”。

## B 课：从向量加法走到 softmax，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–45 | 逐段读并运行 [Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；只跑一个小输入 | 标出 program id、offset、mask、load/store 的含义 |
| 45–80 | 将向量长度改成非 block 整数倍；先预测尾部索引，再观察 mask 的作用 | 一次边界测试，解释越界为何危险 |
| 80–90 | 休息 | — |
| 90–135 | 读 [Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 的稳定公式与逐行归约内核 | 按 load→max→exp→sum→divide→store 写伪代码 |
| 135–160 | 按 [在线 OJ 路线](../05-online-judge.md)，用 Tensara 或 LeetGPU 的 Vector Addition 核对 Triton 模板并尝试最小评测；平台阻塞则改查 [LCPU Assignment 01](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment01/README.md) 的解释器调试说明 | 一份接口/设备/结果记录，或明确阻塞和本地替代，不追加整套作业 |
| 160–180 | 制定 L02 的形状矩阵、参考实现和错误检查；说明哪些教程代码被借用 | 实验计划与代码来源注释 |

课内题：softmax 为何先减行最大值？把 masked 元素填 0 与填负无穷在 max/exp 中有什么差别？融合减少了哪些中间张量读写？与 PyTorch 自带 softmax 比，为什么不能预设一定获胜？

## 本周实验与验收

[L02](../labs/L02-softmax.md) 只要求一个可解释的 kernel 和可信对照。你可以从官方教程改写，但报告应区分教程原有实现与你亲自完成的边界处理、形状实验和解释。若最喜欢这个模块，在 M05 的方向表中提高 kernel/compiler 的兴趣评分。

借助 AI 实现时，优先选择 L02 的 300 分钟增强表，增加性质测试、参数消融与留出形状。想做完整的第二个算子，可插入 [L02X 融合残差与 RMSNorm](../labs/L02X-kernel-challenge.md)；想体验不同硬件可选 [TensorOJ Ascend C 路线](../05-online-judge.md)。完整 OJ 题面可能要求任意维度、多 dtype 或更大形状，不把本地 softmax 直接当作已满足远端规格。

## 资料汇总

必读：[LCPU 日程](https://infra.seminars.lcpu.dev/schedule)、[GEMM 讲义](https://infra.seminars.lcpu.dev/slides/session0201.pdf)、[Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Triton softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。选做：LCPU CUDA/GEMM 其他作业、CS336 Assignment 2；它们不是本周 5 小时实验的附加要求。

在线补充：[平台选择与题目路线](../05-online-judge.md) · [AI 辅助实验](../06-ai-assisted-labs.md)。
