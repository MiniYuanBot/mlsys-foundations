# S02环境

复用lab2的WSL2 torch/Triton环境，不另装FlashAttention库也可做本地契约。先通过基础softmax和向量加法JIT smoke，再运行GPU测试。至少记录torch、Triton、驱动、GPU能力、精度设置。

```bash
python -m tools.doctor --require triton --out results/environment/attention.json
python -m unittest lab2.attention.tests -v
```

wrapper初始TODO故意失败。编译产物/寄存器观察按安装版本Triton和NVIDIA工具文档进行；不硬编码跨版本编译对象私有属性。TileLang/CS336完整上游lab另建环境并固定commit，按其当前依赖安装；不要污染本地torch/vLLM环境。8GB上保留小B/H与最大S1537，不扩大至论文模型规模。

