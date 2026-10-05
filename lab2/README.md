# lab2：Triton 归约与流式 Attention

从行 Softmax 练习 Triton 归约，再实现不保存完整注意力矩阵的流式 Attention。

## 内容

- [assignment.md](assignment.md)：Softmax 的实现、测试与性能实验。
- `kernel.py`、`softmax_triton.py`、`reference.py`、`tests.py`：内核骨架、启动代码、参考与测试。
- `bench_softmax.py`：基线、配置消融与留出测量。
- [attention/](attention/README.md)：分块 Attention、online softmax 与显存实验。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：分块、融合与编译器拓展。
- [setup.md](setup.md)、[grade.md](grade.md)、[report.md](report.md)：环境、基础评分与报告。

## 使用顺序

1. 完成 lab1 的 Attention 正确性练习。
2. 补齐 Softmax 内核，通过边界测试后测量性能。
3. 进入 `attention/`，完成流式实现、消融和留出实验。
4. 如需增加一次融合练习，可继续 [lab2x](../lab2x/README.md)。

GPU 实验使用 Linux 或 WSL2 下的 CUDA/Triton 环境。框架中的占位实现会主动报错，具体输入限制见各任务说明。
