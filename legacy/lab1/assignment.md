# 基础练习：Attention 正确性与测量

运行环境：CPU 正确性 + CUDA 实测。以下命令均在项目根目录执行。

## 实现任务

`attention.py` 提供独立 SDPA 参考；补 `attention(q,k,v)` 的显式 scaled matmul、causal mask、softmax、乘 V。无需重写计时器。

先手算 B1/h1/S4/d2：第0个输出只取第0个V；未来位置不可影响过去。mask 应是 j>i，不要让整行被遮蔽。参考 [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 实验步骤

固定 B∈{1,4}、h4、S∈{128,256,512}、d64、FP32、seed42、contiguous、forward/no_grad/dropout0。六形状×两实现×三轮=36条。

```bash
python -m unittest lab1.tests.CPU -v
python -m unittest lab1.tests.GPU -v
python -m lab1.measure --implementation explicit --device cuda --out results/lab1/explicit
python -m lab1.measure --implementation sdpa --device cuda --out results/lab1/sdpa
python -m lab1.profile_attention --device cuda --out results/lab1/trace.json
python -m tools.plot results/lab1/explicit/raw.csv results/lab1/sdpa/raw.csv --x S --y latency_ms --group implementation B --xlabel "Sequence length" --ylabel "Latency (ms)" --title "lab1 CUDA FP32" --out results/lab1/latency.png
python -m tools.plot results/lab1/explicit/raw.csv results/lab1/sdpa/raw.csv --x S --y peak_allocated_bytes --group implementation B --xlabel "Sequence length" --ylabel "Peak allocated (bytes)" --title "lab1 CUDA memory" --out results/lab1/memory.png
```

每实现独立进程；每次18条，总36条。首调用单列，20次预热、100次迭代同步、三轮；如统一改1000迭代，两实现都改。CPU调试可把device改cpu，另存目录；显存字段为空是合理的，不能用于GPU显存图。

正确性起点 rtol1e-4/atol1e-5。阅读 `common/benchmark.py` 并解释同步和 baseline/peak/reserved，不要直接相信一个 latency 数字。trace 打开方式可用 Chrome/Perfetto 的本地 trace 查看器，记录实际算子/内核；无法确定后端写 unknown。SDPA auto 不等于已确认 FlashAttention。

交付：36条记录、误差JSON、trace、两图、500–800字[报告](report.md)、可重建命令与周报。正负性能差异都保留。

增强30–45分钟：仅 B4/S256 加 FP16；参考用 FP16输入转回FP32 后计算，起点 rtol1e-2/atol1e-3；记录误差、耗时、实际后端，不把精度和后端变化全归因于同一机制。
