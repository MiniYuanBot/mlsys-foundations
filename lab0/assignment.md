# 基础练习：PyTorch 训练闭环

运行环境：CPU。以下命令均在项目根目录执行。

## 实现任务

已提供 `make_data`、固定 MLP、20 epoch 训练入口、checkpoint 保存/加载与独立测试。你只补：

1. `train_toy.train_step`：zero_grad → forward/loss → backward → step，返回 loss 和正确个数。
2. `train_toy.evaluate`：eval/no_grad、样本加权均值、最后不足 64 的 batch。
3. `check_grad.finite_difference`：float64 中心差分，不能直接调用 autograd 代替。

提示顺序：先读测试的输入输出 → 只跑一个 batch → 看 [PyTorch 优化循环](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。先打印参数更新量，再考虑调整代码，不更换默认网络/seed 追分。

## 实验步骤

固定 X=2048×16、seed42、前1600训练/后448测试，标签按课程公式；Linear16→32→2、ReLU、Adam lr0.01、batch64、20 epoch。测试集只作观察和如实记录，不据其挑 seed。

```bash
python -m unittest lab0.tests.Training -v
python -m lab0.check_grad
python -m unittest lab0.tests -v
python -m lab0.train_toy --out results/lab0/run1
python -m lab0.train_toy --out results/lab0/run2
python -m tools.plot results/lab0/run1/epochs.csv --x epoch --y train_loss --xlabel Epoch --ylabel Loss --title "lab0 CPU training" --out results/lab0/loss.png
```

前32训练样本≤300次更新达到≥95%；中心差分最大误差<1e-5；重载 logits 容差1e-6。正式测试准确率<85%应排查但不通过挑参数掩盖。完整两次曲线是否逐值一致都记录，不能只保留较好的一次。

交付：代码、两次 config/environment/command/log、epochs/raw/summary CSV、model.pt、loss 图、一个 bug 的最小复现、[报告](report.md) 和周报。给定的 checkpoint 功能仍需通过“训练后再重载”的真实检查。

增强（主线完成后最多一项）：连续40步与20步保存/恢复再20步；固定初始权重和 batch 顺序，恢复 Adam 状态，再做故意不恢复优化器的反例。预留45–60分钟，不挤掉复跑。
