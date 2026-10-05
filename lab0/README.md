# lab0：训练与张量布局

从一个小型训练任务入手，练习正确性检查，再用 C++ 理解张量布局与矩阵乘法。

## 内容

- [assignment.md](assignment.md)：训练、评估、数值梯度与 checkpoint 检查。
- `train_toy.py`、`check_grad.py`、`tests.py`：基础代码与测试。
- [native/](native/README.md)：C++ 张量视图、stride 和矩阵乘法。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：布局与性能的拓展问题。
- [setup.md](setup.md)、[grade.md](grade.md)：基础环境与评分；[report.md](report.md)：实验报告。

## 使用顺序

1. 阅读环境要求，完成训练与梯度检查。
2. 进入 `native/`，完成布局访问和矩阵乘法，再比较正确性与性能。
3. 整理测试结果和报告，按需选择研究问题。

基础练习可在 CPU 上完成；原生任务还需要 C++ 编译器。实验命令默认在项目根目录运行。
