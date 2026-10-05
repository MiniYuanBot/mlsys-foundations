---
title: lab0｜PyTorch 桥接与训练调试
tags: [mlsys, module, pytorch]
weeks: [1]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab0｜PyTorch 桥接与训练调试

[索引](../guides/course-map.md) · [本周实验 lab0](../labs/lab0-pytorch.md) · [下一模块](lab1-measurement.md)

**时间：A 课 180 分钟＋B 课 180 分钟＋lab0 300 分钟。** 先修：Python、线性代数、反向传播概念。你不需要先看完一个深度学习视频系列。

## 本周结束时能做什么

能够独立写一个 PyTorch 训练循环，解释每个张量的 shape、dtype、device 与梯度状态；能从 loss 异常定位数据、梯度或更新问题；能保存并验证模型。指标不是“看过教程”，而是能修改代码后预判结果。

## A 课：张量与自动微分，180 分钟

| 时段 | 学习步骤与材料 | 当场留下的东西 |
|---|---|---|
| 0–15 | 读 [课程使用说明](../guides/getting-started.md)，建立笔记与代码两个目录，写本周目标 | 一条可验收目标与课表时间块 |
| 15–50 | 读 [Tensor 教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) 的创建、属性、运算、NumPy 互操作；用 CPU 跑形状例子 | `tensor_scratch.py`，每行重要运算写预测 shape |
| 50–85 | 构造 `x:(4,3)`、`w:(3,2)`、`b:(2,)`，手算 broadcasting 与矩阵乘；比较 transpose、contiguous、view/reshape | 一个表：shape、stride、是否共享底层存储；一次故意触发的维度错误 |
| 85–95 | 休息 | — |
| 95–145 | 读 [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)，运行前向、backward、清梯度；观察连续 backward 的累加 | 对 `y=(x*w).sum()` 手算梯度并核对；解释 `detach` 与 `no_grad` |
| 145–180 | 合上教程完成前两道课内题，第 3 题选做；将不会的 API 定位到官方文档，不复制整段训练代码 | `lab0-A.md`，包含答案、错误与修正 |

### 课内题

先做第 1、2 题，复用本节的张量与 autograd 草稿；第 3 题选做。图可手绘，复现使用 CPU。

1. **【推导·画图】从索引走到地址。** 令 `a=torch.arange(24, dtype=torch.float32).reshape(4,6)`，`b=a[1:4:2,1:6:2]`，`c=b.t()`。在运行前列出三者的 shape、stride、storage offset，并算出 `c[2,1]` 相对底层存储起点的字节偏移。画出逻辑矩阵到 24 个存储单元的连线，再比较 `c.contiguous()` 的存储布局和复制字节数。交付地址表与一张图；用实际索引值、storage 指针及修改一个元素后的变化核对共享关系。参考 [PyTorch internals 的 stride/storage 图](https://blog.ezyang.com/2019/05/pytorch-internals/)；也可用作者的 [Stride Visualizer](https://ezyang.github.io/stride-visualizer/index.html) 调整形状与步长。后者是浏览器交互演示，不模拟缓存或 CUDA 访存；旧博客的内部类名不作为当前实现依据。
2. **【推导·复现】把梯度的两条来源画出来。** 固定 `x=[1,2]`、`w=[2,1]`，令 `z=x*w`、`u=z.sum()`、`loss=u*u+z[0]`。画前向 DAG，在每条反向边标出局部导数，手算 `w.grad`；再用 CPU float64 的 autograd 与中心差分核对，差分步长取 `1e-6`。保持参数不变，重新前向后再做一次 backward，分别验证清梯度与不清梯度的结果。交付带数值的 DAG 和误差表，特别检查共享节点上的梯度是否相加。参考 [官方 autograd 教程的计算图](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)；它展示依赖关系，不表示 GPU 的实际执行时间线。
3. **【复现，选做】自己处理一次反向传播中的共享节点。** 只实现标量的加法、乘法和反向拓扑遍历，使 `u=a*b; loss=u*u+u` 能自动求导，测试 `a=2,b=3` 及 `a=0,b=-2`。先画共享 `u` 的图，再写不超过 60 行的核心代码；与手算和中心差分比较，添加一个能发现“梯度覆盖而非累加”的测试。参考作者 [micrograd 的 Tracing / Visualization 图](https://github.com/karpathy/micrograd)，只读图和接口；无需安装 Graphviz，也不扩展到张量或完整训练框架。

完成标准：能不用查教程写出 `requires_grad → forward → scalar loss → backward → grad` 的最小闭环。若不能，B 课第一段优先补这个闭环。

## B 课：训练循环与故障定位，180 分钟

| 时段 | 学习步骤与材料 | 当场留下的东西 |
|---|---|---|
| 0–20 | 按 [环境手册](../guides/environment.md) 运行 CPU 检查；下载可后台进行 | Python/PyTorch 版本与 CPU 梯度输出 |
| 20–60 | 从 [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 进入 Build Model 与 Datasets/DataLoaders；只学 `nn.Module`、参数注册、mini-batch | 自己写 16→32→2 MLP 的类；画 batch 经过网络的形状 |
| 60–95 | 读 [优化循环教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)，逐行解释 loss、zero_grad、backward、step | 不看原文写 15–25 行训练核心代码 |
| 95–105 | 休息 | — |
| 105–145 | 分别故意漏掉 `step`、漏清梯度、给 CrossEntropyLoss 传错标签 dtype；一次只引入一种错误 | 三行“症状→检查→修正”表，不要求长时间训练 |
| 145–180 | 从 Basics 进入 Save/Load；保存 `state_dict`，重新建模型并加载；准备 lab0 的数据划分与日志字段 | checkpoint 核对草稿与 [lab0](../labs/lab0-pytorch.md) 实验计划 |

### 课内题

先做第 1、2 题，合并到本节故障定位与保存加载练习中；第 3 题选做。不需要下载数据集或新增可视化依赖。

1. **【定量·复现】用轨迹区分两种训练故障。** 对单个样本 `x=2,y=0`，取 `loss=(w*x-y)^2/2`、`w₀=1`、学习率 `0.1`，使用无 momentum、无 weight decay 的 SGD。手算连续两次更新的 loss、梯度和参数，再分别复现正常训练、漏掉 `step`、只在开始清一次梯度三种情况，每次都重新前向。画三组参数轨迹，指出最早在哪个观测量上能区分故障；不能仅凭 loss 下降判为正确。参考 [TensorBoard 官方教程第 5 节的训练曲线截图](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)，本题用纸笔或 CSV 绘图即可，平滑后的曲线不能替代逐步原始值。
2. **【画图·复现】训练状态与梯度记录是两条开关。** 用 `Linear(2,2)→BatchNorm1d(2)→Dropout(0.5)` 和固定输入 `[[1,2],[3,4],[5,6],[7,8]]`，从同一初始化复制四份独立模型，比较 `train/eval × 正常梯度/no_grad`。每个条件连续前向两次，只在该条件开始设置一次随机种子；记录输出的 `requires_grad`、两次输出差异及 BatchNorm 的 running mean 是否变化。画状态流程图，标明参数、buffer 和梯度分别在哪里读写；再保存并重建其中一个模型，在双方 `eval` 且输入相同时核对输出。参考 [训练教程的训练/验证循环](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) 与 [autograd 计算图](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。前者含视频与代码，后者是依赖图；随机输出偶然相同不能单独证明 dropout 未生效。
3. **【定量·复现，选做】为什么能加载权重，还不等于能续训？** 沿用第 1 题，改为 SGD momentum `0.9`，其余设置不变。对比连续训练三步、第一步后仅恢复模型参数、第一步后同时恢复模型与 optimizer 状态三条路径；手算第二步的 momentum buffer，画出 checkpoint 分叉后的参数轨迹。交付最小复现和逐步误差表，明确确定性标量例子省略了哪些真实训练状态。参考 [官方 Saving a General Checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) 的状态字段；不要把这个无随机性的例子推广成任意训练都可逐位复现。

检查提示：CrossEntropyLoss 自带需要的 log-softmax 处理；eval 不自动关闭梯度；`state_dict` 包含注册参数及相关 buffer。暂不要求实现复杂 Dataset、多卡训练或完整 Transformer。

## 本周实验与验收

执行 [lab0](../labs/lab0-pytorch.md)。课内草稿可直接用于实验，实验时间用于独立整理与验证。验收时请自己从空白文件写出训练核心，并口头解释一个你实际遇到的 bug；解释不清的代码标记为需复习。

## 资料汇总

必读：[Tensor](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)、[Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[优化循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[Basics 入口](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。选读：[CMU Deep Learning Systems](https://dlsyscourse.org/) 的自动微分相关材料；只有反向传播概念不清时使用，不要求完成其全部框架作业。
