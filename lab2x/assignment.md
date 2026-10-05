# 练习要求：残差与 RMSNorm 融合

运行环境：WSL2/CUDA/Triton。以下命令均在项目根目录执行。

## 实验任务

这是可选练习，建议完成 lab2 的归约与融合实验后进行。已有独立 eager 参考、GPU性质测试和统一 benchmark；`fused_add_rmsnorm.py` 仅给契约，kernel与wrapper由你组织，可回看lab2。

公式：z=x+residual；y=z×rsqrt(mean(z²,dim=-1)+1e-5)×gamma。输入contiguous FP32，x/r为[M,N]，gamma[N]，两个独立输出z/y，禁止原位改输入。归约除实际N，padding平方和贡献0。它不是减均值的LayerNorm。

## 运行

A eager；B triton4；C仅把num_warps改8。开发M∈{1,128}/N∈{128,513,1024}/seed42，54条。先按开发数据选择并写预测，再测留出(7,257),(7,1537)/seed2026，18条。

```bash
python -m unittest lab2x.tests -v
python -m lab2x.bench --implementation eager --out results/lab2x/dev-eager
python -m lab2x.bench --implementation triton4 --out results/lab2x/dev-triton4
python -m lab2x.bench --implementation triton8 --out results/lab2x/dev-triton8
# 先保存配置选择与预测，再做下列留出测量
python -m lab2x.bench --implementation eager --split heldout --out results/lab2x/heldout-eager
python -m lab2x.bench --implementation triton4 --split heldout --out results/lab2x/heldout-triton4
python -m lab2x.bench --implementation triton8 --split heldout --out results/lab2x/heldout-triton8
```

测试覆盖双输出误差、N1标量、抵消、gamma0、符号对称、输入不变。正确性起点1e-4/1e-5。两个输出都要计时并保留；不得少交z制造加速。逐形状延迟图用tools.plot起步，误差表注意output_errors的两个元素对应z/y。

交付72条原始记录、误差表、图、500–800字[报告](report.md)、来源/AI记录与[OJ差异/结果](../templates/oj-record.md)。预算内来不及全题适配就准确记录缺口。若选择TensorOJ Ascend路线，按[课程OJ手册](../../mlsys-research-course/guides/online-judge.md)替换整个X周并另建目录；本模板实现的是NVIDIA路线，不把它当NPU环境。
