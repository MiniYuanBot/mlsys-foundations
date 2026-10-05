---
title: L02｜实现并测量 Triton Fused Softmax
tags: [mlsys, lab, triton]
week: 3
status: not-started
---

# L02｜实现并测量 Triton Fused Softmax

[索引](../index.md) · [对应课程 M02](../modules/M02-kernels.md) · [前置测量工具](L01-attention.md)

## 问题与最小范围

问题：融合行内运算能否减少中间读写，收益是否随矩阵形状变化？环境为 WSL2、CUDA、Triton。参考 [向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 和 [Fused Softmax 官方教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。课程只实现一行或少量行的稳定 softmax，不追加 autotune、大 GEMM 或 FlashAttention。

可从教程改写，并明确标注来源。你的独立工作至少包括：解释工作分配、处理非整齐列数、建立三种实现的公平测量、分析结果边界。

借助 AI 时推荐使用下文的增强执行表：仍为一个 softmax kernel，增加验证和参数消融。更完整的第二个算子放在独立的 [L02X 挑战周](L02X-kernel-challenge.md)。

## 实现与输入

三种实现：A，显式 PyTorch 的减 max、exp、sum、除法；B，`torch.softmax(x, dim=-1)`；C，你的 Triton 融合实现。三者使用同一输入、FP32、no_grad、稳定公式。B 是强 baseline，不预设 C 比 B 快。

性能矩阵：行数 M∈{128,1024}，列数 N∈{128,513,1024}，6 个形状。输入 contiguous，seed=42。正确性额外检查 N=1，以及输入放大到标准差 50 的极值情况；不要把这些额外检查误算成固定性能矩阵。

## 300 分钟执行表

| 时间 | 任务 |
|---|---|
| 0–40 | 在现有环境运行向量加法，验证非 block 整倍数输入 |
| 40–110 | 写/改写 softmax kernel 与 Python wrapper，先过 128×128 |
| 110–160 | 做全部形状、N=1、极值与尾部 mask 正确性检查 |
| 160–220 | 用 L01 工具测量 6×3×3=54 条性能记录 |
| 220–265 | 画耗时/相对速度图，解释中间访存和启动开销 |
| 265–300 | 整理代码来源、实验说明、局限与周报 |

## AI 辅助增强版：替换上表，总计仍为 300 分钟

原表与本表二选一。增强版保留原正确性、54 条基准记录及全部基础验收；使用课内已完成的向量加法和 AI 草稿节省实现时间。

| 时间 | 任务 |
|---|---|
| 0–30 | 确认环境、输入契约与先验预测，复用课内小样例 |
| 30–90 | 完成/审查 kernel 与 wrapper，逐行解释 mask 和归约 |
| 90–145 | 原边界测试＋下面的性质检查；用独立参考定位错误 |
| 145–205 | 运行原 54 条＋参数消融 18 条＋留出 18 条，共 90 条记录 |
| 205–255 | 画逐形状性能和留出对照图，解释选择配置的依据 |
| 255–300 | 整理来源、AI 使用记录、限制与周报，做 10 分钟解释/修改检查 |

**性质检查。** 使用本实验标准正态范围内的有限 FP32 输入，补三类测试：每行加同一个常数 10，输出应近似不变；整行元素相等时应近似 1/N；对列做同一随机置换，输出应发生相同置换。沿用本实验误差起点，不能假设浮点运算逐位一致或推广到任意极端数值。额外检查 wrapper 对非 contiguous 输入明确报错，而非悄悄按 contiguous 读取；本实验不要求支持任意 stride。

**一次参数消融。** 原 C 固定 `num_warps=4`；只将它改成 8，在原 6 个形状上再跑三轮，共增加 18 条。若当前版本/设备不支持某设置，记录并改选其支持的两个值，运行前写清。其他编译参数、输入、精度、计时与输出分配口径全部固定，所有配置均先过正确性。

**一次留出检验。** 看完开发矩阵后，写下你倾向的配置及预测，再用 M=7、N∈{257,1537}、seed=2026 测 `torch.softmax` 和两个 Triton 配置，每条件三轮，增加 18 条。留出形状不参与之前的配置选择。只增加行 softmax 的形状，不同时改变 dtype 或归约轴。没有获胜配置也可通过；应分析形状依赖或证据不足。

增强交付另加：性质检查输出、消融/留出原始记录、配置选择依据和一个尚未解释的结果。90 条全部来自同一硬件；OJ 结果另表，不用远端总分代替这些记录。超过预算时优先保住原实验，消融或留出缺项标待补，不从矩阵中选择性删点。

OJ 迁移可查看 [平台题目路线](../05-online-judge.md)。本周不要求在 5h 内再完整解决任意维度 softmax；第二个算子或 Ascend C 体验在 X 周做。

## kernel 的具体要求

1. Python wrapper 验证二维 contiguous FP32 输入；输出 shape 相同；计算不小于 N 的 2 次幂 `BLOCK_SIZE`。
2. 一个 program 负责一行的简单版本即可：由 program id 定位行；由 arange 定位列；使用 stride 计算地址。
3. 对列索引 `<N` 做 mask；无效列加载为负无穷，使其不会改变 max，exp 后贡献 0。写回也要 mask。
4. 在行内依次做减最大值、指数、求和、归一化。不要先写出 exp 张量到全局内存再另起 kernel，否则不再是同一融合方案。
5. 解释选择的 num_warps/工作分配；可以沿用教程推荐设置，不要求在本周调到最优。首次 JIT 编译单独记录，不纳入稳定运行耗时。

## 正确性与测量

对 C 与 B 比较 `assert_close(rtol=1e-4, atol=1e-5)`，记录最大绝对误差、行和偏差与是否有限。N=1 的输出必须接近 1。极值输入仍应有限；错误先看稳定公式、mask 与行/列 stride。

性能测量复用 L01 的同步/预热方法，20 次预热、每轮 100 次、三轮；所有实现的输入分配排除在计时外，输出分配口径保持一致。小输入噪声过大时整体增迭代数。逐点计算 A/C 与 B/C 的延迟比，标注分子分母，避免把“加速比 0.8”错误解释为加速。

若估算有效带宽，可用最低逻辑读写量 `2*M*N*4` 字节除以时间，但必须命名“按一次读写估算的有效带宽”；这不是实际硬件流量测量。解释 A 的中间张量为何增加读写，B 为什么已经可能融合或高度优化。

## 交付与验收

交付：`softmax_triton.py`、`test_softmax.py`、`bench_softmax.py`、54 条原始记录、正确性报告、两张图、500–800 字分析。

- [ ] 所有性能形状与额外边界检查均通过。
- [ ] 能不看教程解释 mask、BLOCK_SIZE、行归约与数值稳定性。
- [ ] 三种实现使用相同精度、输入、预热和计时口径。
- [ ] 未把 JIT 编译成本混进某一个实现的稳定运行时间。
- [ ] 能说明至少一个融合有效或无效的形状；不要求战胜 PyTorch。

## 阻塞处理

Windows 原生环境报 Triton 问题先回到 WSL2 支持路径；不要无目的编译完整框架。参考 [LCPU 作业说明](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment01/README.md) 的解释器调试线索；解释器适用于部分正确性排查，不能提供 CUDA 性能结论。超过 90 分钟仍无法运行时保留最小案例，申请组内帮助，并把 GPU 测量标为待补。

资料回查：[Triton softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) · [向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) · [实验规范](../02-experiment-standard.md)
