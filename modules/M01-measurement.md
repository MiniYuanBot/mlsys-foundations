---
title: M01｜Transformer 资源账本与可靠测量
tags: [mlsys, module, measurement]
week: 2
status: not-started
---

# M01｜Transformer 资源账本与可靠测量

[索引](../index.md) · [前一模块](M00-pytorch.md) · [实验 L01](../labs/L01-attention.md) · [下一模块](M02-kernels.md)

**时间：3h＋3h＋5h。** 先修：M00 的张量与梯度检查。核心问题：一段模型代码慢，究竟是算得多、搬得多，还是你量错了？

## A 课：从形状推导资源，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 从 [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/) 找 Lecture 2（PyTorch/resource accounting）与 Lecture 3（architectures），浏览讲义目录 | 定位参数、激活、FLOPs 三部分；不要求看完两段视频 |
| 20–60 | 对一个单层 Transformer 画数据流，给 Q/K/V、attention score、输出和 MLP 标形状 | 一张资源账本，变量 B、S、H、h、d 含义明确 |
| 60–95 | 推导两次 attention 矩阵乘的主项运算量，按一次乘加 2 FLOPs 计约 `4 B S² H`；估算 score 张量大小 | 分别把 S、B 翻倍，预测运算/中间张量如何变化 |
| 95–105 | 休息 | — |
| 105–145 | 选读 [FlashAttention](https://arxiv.org/abs/2205.14135) 的摘要、引言与 IO 动机；关注“不写出完整 score” | 一段 150–250 字说明：少搬数据为何可能比少算几次更重要 |
| 145–180 | 给定 B=1、S=512、h=4、d=64、FP32，计算 Q/K/V 和一个 score 张量字节数 | 手算表及单位；为 L01 写先验趋势 |

课内题：若一个 score 张量是 `B*h*S*S` 元素，为什么程序的峰值显存不一定等于它的大小？计算图、softmax 临时值、框架缓存分别会怎样影响？如果 PyTorch 自动调用了融合后端，你的理论对象又变成了什么？

## B 课：计时、显存和 profiler，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–25 | 读 [实验规范](../02-experiment-standard.md) 的 GPU 计时与显存部分 | 写出“异步提交时间”与“完成时间”的区别 |
| 25–65 | 读 [benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)，为一个矩阵乘分别做无同步与有同步计时 | 两组数值与解释；CPU 环境只验证脚本，不声称重现 CUDA 异步现象 |
| 65–95 | 从 [Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 跑最小 trace，识别 CPU op、CUDA kernel、shape | 一个 trace 和前三个主要耗时操作 |
| 95–105 | 休息 | — |
| 105–145 | 写 L01 测量方案：参考实现、正确性、预热、重复、记录字段；检查是否把输入分配放进计时 | 一份实验记录的“运行前”部分 |
| 145–180 | 完成课内题并开始实现显存/耗时采集函数 | 可复用的 `measure.py` 草稿 |

课内题：① 为什么一次 profiler 运行的耗时不直接当作最终性能结果？② 三轮各跑 100 次与 300 个独立试验是否相同？③ `reserved` 下降或 `nvidia-smi` 不变，各自能说明什么？④ 比较 SDPA 与显式 attention 时，要怎样记录后端，才不把框架选择误当作自己优化？

提示：profiler 会带来额外开销；缓存与常驻分配会影响观测；实现名称相同不代表内核相同。查实际 trace，不根据 API 名字猜测。

## 本周实验与验收

[L01](../labs/L01-attention.md) 会固定 6 个张量配置。至少解释一个理论与实测不完全吻合的地方。第 2 周准备给导师/学长看的材料是“资源账本＋一张正确计时的图＋一个尚不能解释的问题”，不需要包装成创新。

## 资料汇总

必读：[CS336](https://cs336.stanford.edu/spring2025/)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。API 查阅：[scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[sdpa_kernel](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)。
