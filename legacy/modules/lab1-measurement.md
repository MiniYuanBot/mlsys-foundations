---
title: lab1｜Transformer 资源账本与可靠测量
tags: [mlsys, module, measurement]
weeks: [2, 3]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab1｜Transformer 资源账本与可靠测量

[索引](../guides/course-map.md) · [前一模块](lab0-pytorch.md) · [实验 lab1](../labs/lab1-attention.md) · [下一模块](lab2-kernels.md)

**时间：3h＋3h＋5h。** 先修：lab0 的张量与梯度检查。核心问题：一段模型代码慢，究竟是算得多、搬得多，还是你量错了？

## A 课：从形状推导资源，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 从 [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/) 找 Lecture 2（PyTorch/resource accounting）与 Lecture 3（architectures），浏览讲义目录 | 定位参数、激活、FLOPs 三部分；不要求看完两段视频 |
| 20–60 | 对一个单层 Transformer 画数据流，给 Q/K/V、attention score、输出和 MLP 标形状 | 一张资源账本，变量 B、S、H、h、d 含义明确 |
| 60–95 | 推导两次 attention 矩阵乘的主项运算量，按一次乘加 2 FLOPs 计约 `4 B S² H`；估算 score 张量大小 | 分别把 S、B 翻倍，预测运算/中间张量如何变化 |
| 95–105 | 休息 | — |
| 105–145 | 选读 [FlashAttention](https://arxiv.org/abs/2205.14135) 的摘要、引言与 IO 动机；关注“不写出完整 score” | 一段 150–250 字说明：少搬数据为何可能比少算几次更重要 |
| 145–180 | 给定 B=1、S=512、h=4、d=64、FP32，计算 Q/K/V 和一个 score 张量字节数 | 手算表及单位；为 lab1 写先验趋势 |

### 课内题

先做第 1、2 题，把结果并入本节资源账本；第 3 题选做。全部可以手算，不要求运行大张量。

1. **【定量·画图】为 attention 标注形状和字节。** 固定 `B=2,S=1024,h=8,d=64`，只分析已得到 Q/K/V 后的前向 attention，不含投影、mask、dropout 或反向传播；Q/K/V 与输出为 FP16，完整 score 与 softmax 概率为 FP32。按一次乘加 2 FLOPs 计算两次矩阵乘的运算量，并给各张量分别标注 shape、元素数、MiB。画数据流，分别将 S、B 翻倍，标出哪些边的资源增长不同。自检必须说明缩放、指数与归约为何未计入矩阵乘 FLOPs。参考 [Transformer Explainer 的 QKV 与 masked attention 图](https://poloclub.github.io/transformer-explainer/)；可在交互界面查看一个 token 的 attention 行，但其中 GPT-2 的尺寸、dtype 和后端不能直接用于本题账本。
2. **【推导·画图】最大张量为何不等于峰值内存？** 沿用第 1 题，规定 Q/K/V 全程保留；softmax 分配概率矩阵时 score 仍存活，softmax 结束才释放 score；第二次矩阵乘分配输出时概率仍存活，计算结束才释放概率。忽略分配器缓存和其他 workspace，画各张量的存活时间线，计算各阶段 live bytes 与峰值。再对照 [FlashAttention 论文 Figure 1（PDF 第 2 页）](https://arxiv.org/pdf/2205.14135)，另画分块版本的 HBM↔SRAM 数据流，圈出不再完整写回的中间量。交付两张图与峰值表；图中的算法内存量不能冒充框架的 `reserved`，论文特定硬件的带宽与加速比也不照搬。
3. **【定量，选做】同一层在 prefill 与 decode 时还是同一种负载吗？** 沿用第 1 题的 B、h、d，对比 `Q 长度=1024，KV 长度=1024` 与 `Q 长度=1，KV 长度=1024`，不计新增 KV 写入。分别画 score 形状，计算两次矩阵乘 FLOPs；假设融合实现将 Q/K/V 各从 HBM 读一次、输出写一次，计算这一下界模型中的算术强度。给定假想设备算力 `60 TFLOP/s`、带宽 `600 GB/s`，在同一张 roofline 草图上标出两点，并列出这个模型遗漏的三项成本。参考 [NVIDIA GEMM 指南的 Math And Memory Bounds 与 Figure 1](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)；本题只估 attention，不能据此断言完整模型的 decode 瓶颈。

## B 课：计时、显存和 profiler，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–25 | 读 [实验规范](../guides/experiments.md) 的 GPU 计时与显存部分 | 写出“异步提交时间”与“完成时间”的区别 |
| 25–65 | 读 [benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)，为一个矩阵乘分别做无同步与有同步计时 | 两组数值与解释；CPU 环境只验证脚本，不声称重现 CUDA 异步现象 |
| 65–95 | 从 [Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 跑最小 trace，识别 CPU op、CUDA kernel、shape | 一个 trace 和前三个主要耗时操作 |
| 95–105 | 休息 | — |
| 105–145 | 写 lab1 测量方案：参考实现、正确性、预热、重复、记录字段；检查是否把输入分配放进计时 | 一份实验记录的“运行前”部分 |
| 145–180 | 完成课内题并开始实现显存/耗时采集函数 | 可复用的 `measure.py` 草稿 |

### 课内题

先做第 1、2 题，作为本节测量脚本的设计检查；第 3 题选做。以下时序与样本是教学数据，不需要 GPU 或安装 profiler。

1. **【定量·画图】把提交、排队和执行分开。** CPU 在 `[0,2]`、`[2,4]`、`[4,6]` 微秒依次提交 K1/K2/K3；同一 GPU stream 从 `t=2` 开始执行，三者持续 `8/12/4` 微秒，除此之外没有开销。画 CPU/GPU 双泳道时间线，计算只包住三个 launch 的主机计时、从 K1 真正开始到 K3 结束的设备区间，以及从 `t=0` 到全部完成的墙钟区间。标出同步应放在哪些边界，并解释“把每个 API 和 kernel 的时长相加”为什么可能重复计数。参考 [Nsight Systems 图 8–11 的同步与排队时间线](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)；它们是静态 trace 截图，不代表你的设备延迟。
2. **【定量·复现】五轮配对测量能支持多强的结论？** 五个独立进程轮次中，每轮分别预热并重复 100 次，得到参考实现中位数 `[10,12,11,13,10]` μs，候选实现中位数 `[9,10,10,11,9]` μs，同位置属于同一轮。算逐轮加速比、加速比的中位数和“两组中位数之比”，画保留配对关系的点线图；解释统计单位为什么是轮次，而不是 500 次调用。给出一个 AB/BA 交替顺序和结果表字段，至少包含 shape、dtype、实际 SDPA 后端、正确性误差及进程轮次。参考 [PyTorch Profiler 的 tracing 与 schedule](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：写出只采一段 trace 核实内核、关闭 profiler 后另行计时的流程，不用这五轮数据宣称跨机器显著收益。
3. **【推导·画图，选做】释放了变量，显存去了哪里？** 用简化的同步分配器模型：开始已 reserved `128 MiB`、无活动张量；分配 A=`64 MiB`，创建指向 A 的 `4 MiB` 逻辑切片 b，删除变量 A，再将 b 克隆为 c，最后删除 b。忽略分配粒度、额外引用和异步释放，空闲块留在缓存中，并假设可独立归还系统。逐步计算活动存储、逻辑张量字节总数与 reserved，画存活图；再预测调用 `empty_cache` 能释放哪部分，并说明真实分配器的 segment 布局为什么可能改变这个结果。对照 [PyTorch 显存可视化博客的 Memory Timeline 与 OOM 示例](https://pytorch.org/blog/understanding-gpu-memory-1/)，圈出类似的跨迭代存活分配。博客有静态时间线及交互查看器入口；无 GPU 时只读示例，不把分配器视图当成所有设备分配的全集。

提示：profiler 会带来额外开销；缓存与常驻分配会影响观测；实现名称相同不代表内核相同。查实际 trace，不根据 API 名字猜测。

## 本周实验与验收

[lab1](../labs/lab1-attention.md) 会固定 6 个张量配置。至少解释一个理论与实测不完全吻合的地方。第 3 周准备给导师/学长看的材料是“资源账本＋一张正确计时的图＋一个尚不能解释的问题”，不需要包装成创新。

## 资料汇总

必读：[CS336](https://cs336.stanford.edu/spring2025/)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。API 查阅：[scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[sdpa_kernel](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)。
