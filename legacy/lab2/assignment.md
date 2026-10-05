# 基础练习：Triton Softmax

运行环境：WSL2/CUDA/Triton。以下命令均在项目根目录执行。

## 实现任务

先通过 lab1。已提供 eager 稳定 softmax、torch.softmax 参考、输入验证、Triton 行网格/加载/写回骨架与 benchmark。你要：

1. 补 `kernel.py` 中稳定 max/exp/sum/normalize，替换全零占位。
2. 在 `softmax_triton.py` 删除明确的 NotImplementedError，启用给出的 wrapper 启动代码。
3. 解释 padding=-inf 为什么不会污染归约，BLOCK 是不小于N的2次幂，输入不支持任意stride。

来源：[Triton 官方 softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。骨架不是现成正确答案；初始 wrapper 故意阻止返回占位零输出。

## 基础实验

FP32，M∈{128,1024}、N∈{128,513,1024}、seed42。A显式torch、B torch.softmax、C Triton4；6×3×3=54条。额外正确性N=1、std50极值与尾部；性能矩阵不额外计这些检查。

```bash
python -m unittest lab2.tests.Contract lab2.tests.GPU -v
python -m lab2.bench_softmax --implementation explicit --out results/lab2/explicit
python -m lab2.bench_softmax --implementation torch --out results/lab2/torch
python -m lab2.bench_softmax --implementation triton4 --out results/lab2/triton4
python -m tools.plot results/lab2/explicit/raw.csv results/lab2/torch/raw.csv results/lab2/triton4/raw.csv --x N --y latency_ms --group implementation M --xlabel Columns --ylabel "Latency (ms)" --title "lab2 FP32 CUDA" --out results/lab2/latency.png
```

每命令18条，共54条。第二张图请自己写分析脚本，逐形状计算 A/C 与 B/C，标明分子分母，展示三轮或范围；小于1即变慢。没有要求胜过PyTorch。参考与C容差1e-4/1e-5；不要靠放宽阈值“修复”mask。

交付：kernel与wrapper、54条原始数据/误差报告、两图、500–800字[报告](report.md)、计时与来源说明。最低逻辑带宽2MN×4/time是估算，不是实测内存流量。

## 拓展实验

```bash
python -m unittest lab2.tests.Properties -v
python -m lab2.bench_softmax --implementation triton8 --out results/lab2/triton8
python -m lab2.bench_softmax --implementation torch --split heldout --out results/lab2/heldout-torch
python -m lab2.bench_softmax --implementation triton4 --split heldout --out results/lab2/heldout-triton4
python -m lab2.bench_softmax --implementation triton8 --split heldout --out results/lab2/heldout-triton8
```

54+18消融+18留出=90条。留出M7/N257,1537/seed2026。已有GPU测试会检查这些形状的正确性；**留出的是性能调参数据**，不要提前运行留出benchmark再声称未参与配置选择。性质检查平移、常量、置换和拒绝非contiguous。增强结果单记，不改变主线100分。
