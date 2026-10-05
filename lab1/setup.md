# lab1 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

复用lab0的torch环境。CPU可做正确性；主性能表必须在4060的WSL2 CUDA环境完成，参照 [全局setup第2–3节](../setup.md)。

```bash
python -m tools.doctor --require torch --out results/environment/lab1-cpu.json
python -m unittest lab1.tests.CPU -v
# 以下在CUDA环境运行
python -m tools.doctor --require cuda --out results/environment/lab1-cuda.json
python -m unittest lab1.tests.GPU -v
```

记录驱动、torch/wheel CUDA、桌面或笔记本4060、电源模式与后台负载；保持FP32矩阵精度一致。代码已使用highest精度设置，记录任何额外TF32设置。初次测量不加训练/autograd/dropout。

matplotlib只用于图；torch.profiler随PyTorch使用。若CUDA trace不可用，可交CPU诊断trace并明确限制，但GPU性能验收不因此通过。详见 [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

OOM先确认默认B4/S512/d64、无误保存所有迭代输出，再查后台占用；不要先扩大形状。不要把CPU fallback与CUDA数据合并。

