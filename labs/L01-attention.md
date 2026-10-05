---
title: L01｜Attention 正确性、计时与显存
tags: [mlsys, lab, measurement]
week: 2
status: not-started
---

# L01｜Attention 正确性、计时与显存

[索引](../index.md) · [对应课程 M01](../modules/M01-measurement.md) · [实验规范](../02-experiment-standard.md)

> [!tip] AI 辅助增强
> 本周可选：FP16 正确性、计时与后端对照（30–45 分钟）。具体步骤与验收见 [增强实验手册](../06-ai-assisted-labs.md)。原实验提前完成时使用剩余时间，未完成则顺延；不额外叠加到本周 5h。

## 目标、环境与资料

写显式 causal attention，与 PyTorch SDPA 对照，检查形状扩展如何影响耗时和显存。CPU 用于正确性；主要性能表使用 4060 CUDA。只有 CPU 时先完成正确性与测量程序，GPU 表标“未执行”。

读 [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[后端选择](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)、[benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。只使用 forward、no_grad、dropout=0，不加入训练反向传播。

## 固定矩阵

Q/K/V 形状均为 `[B, h, S, d]`，h=4、d=64、FP32、seed=42、contiguous。B∈{1,4}，S∈{128,256,512}，共 6 个配置。性能实现为“显式 matmul＋mask＋softmax＋matmul”和“SDPA 自动后端”两种；每配置每实现三轮。

注意 SDPA 自动后端可能随 dtype、形状、设备变化。记录实际可观察到的内核/后端；不把 SDPA 名称直接写成 FlashAttention。必要时用 math 后端做额外正确性定位，但不把该临时定位结果与自动后端混为同一配置。

## 300 分钟执行表

| 时间 | 任务 |
|---|---|
| 0–45 | 手算形状、写显式 causal attention，完成小输入检查 |
| 45–90 | 对照 SDPA，检查全部 6 个配置的输出误差 |
| 90–150 | 编写/复用计时与显存采集，确认预热和同步 |
| 150–210 | 跑 6×2×3=36 条性能记录，保存原始结果 |
| 210–255 | 对一个中等配置采集 profiler，作两张图 |
| 255–300 | 写解释、限制和周报，准备第 2 周反馈材料 |

## 操作步骤

1. **参考实现。** 计算 `scores=Q@K.transpose(-2,-1)/sqrt(d)`，将未来位置 `j>i` 置负无穷，沿最后一维做 softmax，再乘 V。先用 B=1、h=1、S=4、d=2 检查：第 0 个输出位置只能看到第 0 个 V；所有 softmax 行和应接近 1。保留 mask 代码和手算示意。
2. **正确性。** 调 SDPA 时设置 `is_causal=True, dropout_p=0.0`。对六配置执行 `torch.testing.assert_close`，起点 `rtol=1e-4, atol=1e-5`，记录最大绝对误差。FP32 比较时保持矩阵乘精度配置一致；误差异常先检查 mask、缩放、精度和后端，不先放宽阈值。
3. **计时。** 输入张量先分配在目标设备；20 次预热后同步；每轮计 100 次完整 forward，结束同步，除以 100 得到 ms/调用。三轮均保留。短运算若计时太接近分辨率可统一增至 1000 次；两个实现一起改。
4. **显存。** 同步后记录输入驻留的 allocated 基线，重置峰值，调用一次并保持输出存活，再同步读取峰值。记录绝对峰值和相对基线增量，另记 reserved。不同实现尽量用新进程；不将 cached reserved 误报为算子临时内存。
5. **Profiler。** 对 B=4、S=256 单独采集短 trace，启用 shape 记录，识别主要 matmul、softmax 或融合内核。profiling 不混入主计时。若 CUDA trace 不可用，保留 CPU trace 并标明限制。
6. **作图。** 固定 B 分组，横轴 S，分别画耗时与峰值显存。每个点展示三轮或范围。用资源账本解释至少一个趋势，再指出一个不能仅靠理论主项解释的地方。

## 原始记录格式

`implementation,B,h,S,d,dtype,device,repeat,warmup,iterations,latency_ms,max_abs_error,allocated_baseline_bytes,peak_allocated_bytes,reserved_bytes,backend_note`

不要把不同设备的结果放在同一条实现对照线上。CPU 结果可另表用于调试，不作 GPU 优化结论。

## 交付与验收

交付：`attention.py`、`measure.py`、全部 36 条 CSV、环境/命令、correctness JSON、trace、两张图、500–800 字实验说明。

- [ ] causal mask 的第一位置性质和全部形状正确性通过。
- [ ] 没有对异步 CUDA 调用直接做无同步 wall time 作为正式结果。
- [ ] 预热/重复/后端/显存口径齐全，图能由 CSV 重建。
- [ ] 不要求 SDPA 必然更快；需要解释实际观察到的差异。
- [ ] 至少提出一个下一步可检验解释，未把猜测写成结论。

排错：输出不一致先用 CPU 小张量；NaN 查是否一整行被 mask；显存不释放查是否列表保存了所有输出；耗时漂移查后台任务与电源模式。不要先扩大 S 到可能 OOM 的规模。

资料回查：[SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) · [benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) · [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
