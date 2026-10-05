---
title: lab0｜从零写可检查的小型训练实验
tags: [mlsys, lab, pytorch]
weeks: [1]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab0｜从零写可检查的小型训练实验

[索引](../guides/course-map.md) · [对应课程 lab0](../modules/lab0-pytorch.md) · [实验记录模板](../templates/experiment-record.md)

> [!tip] AI 辅助增强
> 本周可选：断点续训与连续训练的等价性检查（45–60 分钟）。具体步骤与验收见 [增强实验手册](../guides/ai-assisted-labs.md)。原实验提前完成时使用剩余时间，未完成则顺延；不额外叠加到本周 5h。

## 问题与实验边界

问题：当训练结果不对时，你能判断是数据、梯度、更新还是保存加载出了问题吗？本实验用合成二分类任务训练一个极小 MLP，避免下载数据集和显存问题。CPU 即可完成，Mac 与 Windows/WSL 均可；不比较两台机器的速度。

参考：[PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)、[Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[优化循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。这些资料用于查 API；训练脚本应由你组织。

## 固定配置

| 项目 | 必做配置 |
|---|---|
| 数据 | `torch.manual_seed(42)` 后生成标准正态 `X`，形状 2048×16 |
| 标签 | `y = (X[:,0] + 0.5 * X[:,1] > 0).long()` |
| 划分 | 前 1600 行训练，后 448 行测试；先固定数据再训练，不在每个 epoch 重造测试集 |
| 网络 | Linear(16,32) → ReLU → Linear(32,2)，输出 logits |
| 训练 | CrossEntropyLoss，Adam，lr=0.01，batch=64，20 epoch，固定 seed |
| 记录 | 每 epoch 的训练 loss、训练 accuracy、测试 accuracy；每个样本只计一次测试预测 |

测试集只用于最后检查和如实记录，不依据其结果不断挑网络/seed。可先每 epoch 记录测试曲线用于学习观察，但不要把反复据此改参数后的分数当无偏泛化评估。

## 300 分钟执行表

| 时间 | 任务 | 检查点 |
|---|---|---|
| 0–40 | 验证 Python/PyTorch CPU 环境，创建脚本和日志目录 | [环境手册](../guides/environment.md) 的 CPU 梯度检查通过 |
| 40–100 | 写数据生成、固定划分、MLP 与训练循环 | 单批量 forward/backward 正常、shape/dtype 正确 |
| 100–145 | 做 32 样本过拟合与有限差分梯度检查 | loss 确实下降、梯度可核对 |
| 145–205 | 运行完整训练，保存每 epoch 数据和 checkpoint | 一张 loss 图与完整 CSV |
| 205–255 | 重载模型验证；复跑一次；检查 CUDA/WSL 状态 | 重载输出一致；环境通过或阻塞明确 |
| 255–300 | 整理 README、实验记录、周报与自测 | 能从一条命令重跑，不只剩 notebook 输出 |

## 具体实现与检查

**步骤 1：先检查一个 batch。** 打印 `X.shape`、`y.shape`、dtype，确认 logits 为 batch×2、label 为整数类别。loss 必须是有限标量。调用 backward 后选一层打印梯度范数，确认存在且有限；step 前后复制同一参数并比较，确认更新确实发生。

**步骤 2：小数据过拟合。** 只取训练集前 32 个样本，从新初始化模型开始，整批训练最多 300 次更新，其他配置同上。不使用测试集。期望训练 accuracy 至少 95%；达不到先检查更新、标签和 loss，而不是增加模型规模。这是调试检查，不是泛化成果。

**步骤 3：有限差分。** 单独用 float64 建 `X:(2,2)`、`w:(2,1)`，定义 `L=((X@w)**2).mean()`。对 w 的每个元素用中心差分 `[L(w+ε)-L(w-ε)]/(2ε)`，ε=1e-6，与 autograd 对比。扰动时用 clone，不破坏要比较的计算图。最大绝对误差目标 <1e-5；失败先检查 dtype、扰动对象与是否错误累积 grad。

**步骤 4：正式训练。** 恢复完整训练数据并新初始化模型，写训练/评估两个函数。评估用 eval 与 no_grad；训练恢复 train。保存首末 epoch loss，画横轴 epoch、纵轴 loss 的曲线。合成任务通常容易学习，若测试 accuracy 低于 85%，先作为异常排查；不能通过挑 seed 或泄漏测试标签达到阈值。

**步骤 5：检查点。** 保存模型的 state_dict、配置和 seed。重新创建相同网络并加载，固定一批 32 个测试输入，在 eval/no_grad 下比较保存前后 logits，CPU `assert_close(rtol=1e-6, atol=1e-6)`。为未来可恢复训练，可选保存优化器状态，但不作为必做。

**步骤 6：重跑。** 同一设备、相同 seed 和配置重跑；比较曲线和结论，记录是否逐值一致。随机性或后端差异不自动代表 bug，但需说明。在 lab0 学到的三个故障中挑一个，保留最小错误样例和修正说明。

## 交付物

- `train_toy.py`：含 train/evaluate/save/load，不硬编码个人绝对路径。
- `check_grad.py`：有限差分与自动微分对照。
- `config.json`、环境版本、`epochs.csv`、`loss.png`、`model.pt`。
- `README.md`：实际运行命令、结果、一个 bug 的定位过程。
- [每周报告](../templates/weekly-report.md)：本周时间与 CUDA 环境状态。

## 验收清单

- [ ] 32 样本训练 accuracy ≥95%，或有可复现的错误定位材料并标“待补”。
- [ ] float64 梯度检查达到误差要求，解释清楚中心差分。
- [ ] 正式训练 loss 明显下降，测试结果与数据划分如实记录。
- [ ] 重载后固定输入 logits 通过一致性检查。
- [ ] 能口头解释 zero_grad、backward、step、train/eval、no_grad 五个动作。
- [ ] GPU 不可用不会阻塞本实验的 CPU 验收，但后续 GPU 环境状态没有被隐瞒。

## 常见故障与缩减方案

loss 不下降：先打印参数更新量，再检查标签生成与 loss 输入。NaN：先恢复默认 lr、检查输入和梯度的有限性。维度错误：只保留一批 4 个样本逐行打印 shape。超时：删去复杂封装和额外功能，保留小数据过拟合、梯度检查、正式训练和重载这四个核心检查。

资料回查：[张量](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) · [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) · [优化循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
