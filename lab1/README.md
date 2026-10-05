# lab1：Attention 与 CUDA 矩阵乘法

先测清 Attention 的计算、显存和延迟，再实现 CUDA GEMM，理解线程、分块与共享内存如何影响性能。

## 内容

- [assignment.md](assignment.md)：显式 Attention、SDPA 对照与测量要求。
- `attention.py`、`measure.py`、`profile_attention.py`、`tests.py`：实现、计时、性能分析与测试。
- [cuda/](cuda/README.md)：CUDA GEMM 骨架、CPU 参考、naive CUDA 与 cuBLAS 基线。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：布局、流水线与论文拓展。
- [setup.md](setup.md)、[grade.md](grade.md)：基础环境与评分；[report.md](report.md)：实验报告。

## 使用顺序

1. 完成显式 Attention，用小输入验证因果掩码与数值结果。
2. 测量延迟和显存，检查性能分析记录。
3. 进入 `cuda/`，实现 GEMM，完成边界测试、基线比较与分析。

Attention 正确性可先在 CPU 上检查；CUDA GEMM 和 GPU 性能测量需要 NVIDIA GPU。比较结果时须记录精度与实际执行后端。
