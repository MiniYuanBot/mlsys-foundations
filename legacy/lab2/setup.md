# lab2 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

前置lab1已完成测量工具检查。默认NVIDIA RTX4060、WSL2 Linux、torch兼容Triton；不走Windows原生Triton或MPS路线。按 [全局setup第4节](../setup.md) 操作。

```bash
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
python -m tools.doctor --require triton --out results/environment/lab2.json
```

随后自己从[官方向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)运行非block整倍数输入，保存正确性输出。doctor只验证导入/CUDA小计算，不验证JIT；向量加法是编译检查点。

记录PyTorch、Triton、驱动和GPU。保留已有兼容版本，错误时按该版本文档查API，不一口气升级所有包。显式计时20预热/100迭代，首次编译另记。

只支持2D contiguous FP32正尺寸输入；测试超出契约应明确拒绝。解释器可用于部分排错，但其耗时不计GPU分。30分钟整理最小例、90分钟停止盲试；CUDA不可用时完成公式/CPU参考/测试设计，正式54条仍标未执行。

