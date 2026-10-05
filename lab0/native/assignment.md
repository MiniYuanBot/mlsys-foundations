# C++ 张量布局与执行成本

预计 6–8 小时。先通过 lab0 的基础测试，再完成这里的系统实验。

目标：解释张量的逻辑 shape、stride、storage 和物理访问，并在 C++ 中实现寻址和矩阵乘法。

已给：C++17 View契约、独立double累加参考、非连续/转置/末尾padding测试、计时入口。补 `student.cpp` 的逻辑寻址与matmul；先正确，再保留版本A并做循环顺序或packing的单变量优化B。不能让参考与实现共用可疑索引代码。

按 [setup](setup.md) 编译；运行：
```bash
build/lab0-native/tensor_lab --test
build/lab0-native/tensor_lab --benchmark > results/lab0/native.csv
```
事先创建results/lab0，或用全局capture保存命令/编译器/配置。默认性能形状64/128/256方阵，三轮、各10次、3次预热；随机输入在计时外，输出分配包含在内，两实现一致。参考double累加与学生float会有数值/计算差异，**这不是公平优化加速比**；正式A/B必须用自己保留的相同dtype/累加口径版本。默认表主要用于检查计时路径；增加同精度BLAS强基线时记录库与线程数，计入系统分。

交付：`student.cpp`、A/B代码diff、编译器与优化flag、至少三组shape×三轮数据、1个非连续用例；固定线程数、热身、校验和，解释layout/拷贝/缓存和分配成本。用自己的分析脚本绘图，比较前写预测。研究延伸：选择一个实际模型op，比较strided直接访问与先compact再算，找拷贝摊销拐点；留出非方阵，不能只测方阵。

阅读入口：[DLSys](https://dlsyscourse.org/)的NDArray后端思想、[TVM (OSDI 2018)](https://www.usenix.org/conference/osdi18/presentation/chen)。读TVM“计算与调度分离”部分，说明循环变化为什么不应改变语义。最近编译器阅读通过 [论文路线](../../../mlsys-research-course/syllabus.md) 接Mirage/TileLang；本实验只实现小算子，不包含完整编译器。

评分见 [grade.md](grade.md)。本目录提供的教学代码未复制外部实验解答。
