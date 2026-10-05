# 流式 Attention 实验

预计 10–14 小时，通过基础 softmax 测试后开始。Triton 使用 Python 语法描述 GPU 程序；本实验需要控制 tile、访存、归约和 attention 中间存储。

必须读 [FlashAttention](https://arxiv.org/abs/2205.14135) 的IO分析和算法1，以及 [FlashAttention-2](https://arxiv.org/abs/2307.08691) 的work partitioning。先推导online softmax在分块后如何重新缩放历史分子/分母，手算两块的结果，再填kernel。只有forward、causal、FP32、D32/64，无backward/dropout/GQA，实现范围不包含完整 FA2 库。

已有：契约、Q加载/网格提示、独立SDPA正确性、三实现benchmark。你补wrapper、K/V流式加载、online m/l/o更新、mask和输出；不能分配全局S×S矩阵。首位置性质、非整齐S、极大score、tile变化结果不变和输入不变都要通过。

```bash
python -m unittest lab2.attention.tests -v
python -m lab2.attention.bench --implementation explicit --out results/lab2/attention-explicit
python -m lab2.attention.bench --implementation sdpa --out results/lab2/attention-sdpa
python -m lab2.attention.bench --implementation streaming --out results/lab2/attention-streaming
```

开发S256/512/1024、B1/H2/D64、3实现×3轮=27条。固定FP32 IEEE计算作为入门契约；强基线SDPA的实际kernel必须记录，不能声称自己的简单FP32实现必然更快。测绝对/增量显存，验证S增长时是否避免二次方临时张量；仅看wall time无法证明IO改善。

只改变一个tile参数再测三形状×3轮=9条；选择后冻结预测，再使用heldout S383/1537，各实现三轮（18条）。图同时报告延迟、显存、误差，增加一个寄存器/occupancy或编译产物观察，无法取得写“未观测”。

研究拓展：与 [TileLang](https://github.com/tile-ai/tilelang) 的同一forward契约比较schedule/生成代码/编译成本；或研究小batch短序列时融合/分块的边界。先保持相同dtype/精度/输出再比较；FP16/Tensor Core是另一个声明的实验，不与IEEE FP32混作纯tile消融。FA3的Hopper特性不在4060上复现。论文与开源适配见[课程计划](../../../mlsys-research-course/syllabus.md)。
