# lab2x 环境

完全复用 [lab2环境](../lab2/setup.md)，不另装框架。开始前lab2主线应已通过；本章只做forward FP32、小形状。

```bash
python -m tools.doctor --require triton --out results/environment/lab2x.json
python -m unittest lab2x.tests -v
```

最初测试遇NotImplementedError是未完成kernel，区别于CUDA/Triton缺失的ENV skip。每个版本按同一输入契约输出两个张量；先检查一个小形状和N513，再跑全矩阵。

开发与留出在同一台硬件/版本；num_warps4/8是默认单变量消融。设备/版本不支持时，先预注册两个支持的值，并同步修改实验计划/测试/报告，不静默删除困难条件。

OJ不作为本地环境替代。[课程OJ手册](../../mlsys-research-course/guides/online-judge.md)中的Ascend路线使用平台规定的编译器/NPU模板，与本环境分开；如果改选那条路线，保存自己的替代任务和评分映射，不能将本章CUDA测试skip说成Ascend通过。

