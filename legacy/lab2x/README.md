# lab2x：残差与 RMSNorm 融合

这是 lab2 之后的可选练习：独立组织 Triton 内核与启动代码，在一次融合计算中同时输出残差相加结果和 RMSNorm 结果。

## 内容

- [assignment.md](assignment.md)：计算契约、实验矩阵与交付要求。
- `fused_add_rmsnorm.py`：待实现接口；`reference.py`：独立参考。
- `tests.py`、`bench.py`：性质测试、基准、配置消融与留出测量。
- [setup.md](setup.md)、[grade.md](grade.md)、[report.md](report.md)：环境、评分与报告。

## 使用顺序

1. 完成 lab2 的行归约与融合练习，阅读本题输入输出契约。
2. 实现内核，通过边界与输入不变性测试。
3. 比较 eager、不同 warp 配置，保存配置选择后再测留出形状。
4. 整理误差、延迟和报告；有余力再做外部题目的适配。

本目录使用 NVIDIA CUDA/Triton。计时必须包含两个输出，且不能原位修改输入。
