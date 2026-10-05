# CUDA 矩阵乘法

实现共享内存 GEMM，学习 GPU 访存、线程同步和性能分析。

主要文件：

- `student.cu`：待完成的 tiled GEMM。
- `baselines.cu`：naive CUDA 与 cuBLAS 基线。
- `gemm.cuh`：接口和错误检查。
- `main.cu`：CPU 参考、边界测试与计时入口。
- `CMakeLists.txt`、`tests.py`：构建配置与评分器入口。

建议顺序：

1. 按 [setup.md](setup.md) 配置编译器并检查基线。
2. 阅读 [assignment.md](assignment.md)，完成实现和正确性测试。
3. 测量 tile 配置，分析资源占用，再验证留出形状。
4. 按 [grade.md](grade.md) 整理实验记录。

需要 NVIDIA GPU、CUDA Toolkit 和 C++17 编译器。初始 TODO 会使学生测试失败；仅通过基线测试不代表完成。GPU 测试和性能需要在自己的设备上验证。
