# C++ 张量布局

用 C++ 实现张量寻址与矩阵乘法，观察 stride、循环顺序和数据拷贝的成本。

主要文件：

- `tensor.hpp`：张量视图与接口约定。
- `student.cpp`：待完成的寻址和矩阵乘法。
- `main.cpp`：参考实现、正确性检查与计时入口。
- `CMakeLists.txt`、`tests.py`：构建配置与评分器入口。

建议顺序：

1. 按 [setup.md](setup.md) 编译。
2. 阅读 [assignment.md](assignment.md)，完成代码并通过正确性检查。
3. 保存优化前后的实现，再进行性能实验。
4. 按 [grade.md](grade.md) 整理结果和自评。

需要 C++17 编译器，不需要 GPU。参考实现使用 double 累加；正式性能比较应保持相同精度。
