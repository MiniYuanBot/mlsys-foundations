---
title: M00｜PyTorch 桥接与训练调试
tags: [mlsys, module, pytorch]
week: 1
status: not-started
---

# M00｜PyTorch 桥接与训练调试

[索引](../index.md) · [本周实验 L00](../labs/L00-pytorch.md) · [下一模块](M01-measurement.md)

**时间：A 课 180 分钟＋B 课 180 分钟＋L00 300 分钟。** 先修：Python、线性代数、反向传播概念。你不需要先看完一个深度学习视频系列。

## 本周结束时能做什么

能够独立写一个 PyTorch 训练循环，解释每个张量的 shape、dtype、device 与梯度状态；能从 loss 异常定位数据、梯度或更新问题；能保存并验证模型。指标不是“看过教程”，而是能修改代码后预判结果。

## A 课：张量与自动微分，180 分钟

| 时段 | 学习步骤与材料 | 当场留下的东西 |
|---|---|---|
| 0–15 | 读 [课程使用说明](../00-start.md)，建立笔记与代码两个目录，写本周目标 | 一条可验收目标与课表时间块 |
| 15–50 | 读 [Tensor 教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) 的创建、属性、运算、NumPy 互操作；用 CPU 跑形状例子 | `tensor_scratch.py`，每行重要运算写预测 shape |
| 50–85 | 构造 `x:(4,3)`、`w:(3,2)`、`b:(2,)`，手算 broadcasting 与矩阵乘；比较 transpose、contiguous、view/reshape | 一个表：shape、stride、是否共享底层存储；一次故意触发的维度错误 |
| 85–95 | 休息 | — |
| 95–145 | 读 [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)，运行前向、backward、清梯度；观察连续 backward 的累加 | 对 `y=(x*w).sum()` 手算梯度并核对；解释 `detach` 与 `no_grad` |
| 145–180 | 合上教程完成三道课内题；将不会的 API 定位到官方文档，不复制整段训练代码 | `M00-A.md`，包含答案、错误与修正 |

课内题：① `x` 为 4×3、`w` 为 3×2 时，loss 对 w 的梯度是什么形状，为什么？② 同一个小批量做两次 backward 而不清梯度会发生什么？用数值验证。③ 在 no_grad 中前向是否等于 `model.eval()`？各解决什么问题？用“梯度记录”和“层的训练行为”分别回答。

完成标准：能不用查教程写出 `requires_grad → forward → scalar loss → backward → grad` 的最小闭环。若不能，B 课第一段优先补这个闭环。

## B 课：训练循环与故障定位，180 分钟

| 时段 | 学习步骤与材料 | 当场留下的东西 |
|---|---|---|
| 0–20 | 按 [环境手册](../01-environment.md) 运行 CPU 检查；下载可后台进行 | Python/PyTorch 版本与 CPU 梯度输出 |
| 20–60 | 从 [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 进入 Build Model 与 Datasets/DataLoaders；只学 `nn.Module`、参数注册、mini-batch | 自己写 16→32→2 MLP 的类；画 batch 经过网络的形状 |
| 60–95 | 读 [优化循环教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)，逐行解释 loss、zero_grad、backward、step | 不看原文写 15–25 行训练核心代码 |
| 95–105 | 休息 | — |
| 105–145 | 分别故意漏掉 `step`、漏清梯度、给 CrossEntropyLoss 传错标签 dtype；一次只引入一种错误 | 三行“症状→检查→修正”表，不要求长时间训练 |
| 145–180 | 从 Basics 进入 Save/Load；保存 `state_dict`，重新建模型并加载；准备 L00 的数据划分与日志字段 | checkpoint 核对草稿与 [L00](../labs/L00-pytorch.md) 实验计划 |

课内作业：在代码旁解释五件事：为什么分类模型输出 logits；标签为什么用整数类别；训练/测试为什么分开；为什么推理常同时需要 eval 与 no_grad；为什么要比较加载前后的输出而不只检查文件存在。

检查提示：CrossEntropyLoss 自带需要的 log-softmax 处理；eval 不自动关闭梯度；`state_dict` 包含注册参数及相关 buffer。暂不要求实现复杂 Dataset、多卡训练或完整 Transformer。

## 本周实验与验收

执行 [L00](../labs/L00-pytorch.md)。课内草稿可直接用于实验，实验时间用于独立整理与验证。验收时请自己从空白文件写出训练核心，并口头解释一个你实际遇到的 bug；解释不清的代码标记为需复习。

## 资料汇总

必读：[Tensor](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)、[Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[优化循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[Basics 入口](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。选读：[CMU Deep Learning Systems](https://dlsyscourse.org/) 的自动微分相关材料；只有反向传播概念不清时使用，不要求完成其全部框架作业。
