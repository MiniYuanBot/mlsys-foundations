# 流式 Attention

用 Triton 实现分块 causal attention，检查 online softmax 的数值正确性和中间显存开销。

主要文件：

- `attention.py`：待完成的调用接口。
- `kernel.py`：待完成的 Triton kernel。
- `tests.py`：与 PyTorch SDPA 对照的正确性测试。
- `bench.py`：显式实现、SDPA 与流式实现的测量入口。

建议顺序：

1. 按 [setup.md](setup.md) 检查 GPU 与依赖。
2. 阅读 [assignment.md](assignment.md)，推导 online softmax 后实现代码。
3. 通过测试，运行开发集、tile 实验和留出集。
4. 按 [grade.md](grade.md) 检查结果。

范围限于 FP32 causal forward，head dimension 为 32 或 64。不得分配全局序列长度平方的中间矩阵；低精度扩展应单独比较。
