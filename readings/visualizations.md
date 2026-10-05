# 可视化资料

按当前课内题选一两项即可，不必把所有工具装一遍。先自己画出预测，再对照资料修正；在图旁记下输入、假设和证据来源。

下表的 A1、B2 指对应章节 A、B 课中的题号。交互演示帮助理解机制；论文图和教学动画表达各自的假设；性能时间线需要实际 trace 才能分析自己的程序。它们不能互相替代。

链接、正文及图注核验于 2026-10-05，未逐一运行网页控件或复现上游代码。网页失效时可按标题到作者仓库或官方文档查找。

## 张量、梯度与训练状态

配合 [lab0](../modules/lab0-pytorch.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [Stride Visualizer](https://ezyang.github.io/stride-visualizer/index.html) | 浏览器交互；形状、步长与存储地址 | A1：调整 stride，核对逻辑索引到存储单元的映射；不模拟缓存或 GPU 访存 |
| [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) | 作者博客；stride、storage、view 静态图 | A1：分清 view 与拷贝；内部类名较旧，当前源码另查 |
| [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | 官方计算图 | A2、B2：标局部导数、共享节点与梯度累加；依赖边不是执行时长 |
| [micrograd](https://github.com/karpathy/micrograd) | 作者 README 的 Tracing / Visualization，节点含数值与梯度 | A3：重画共享节点的反向传播；只读图即可，不要求安装 Graphviz |
| [TensorBoard 教程](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) | 官方截图；第 3 节模型图、第 5 节训练曲线 | B1：对照参数与 loss 轨迹定位故障；保留未平滑的逐步值，纸笔绘图即可 |

## Attention、资源与测量

配合 [lab1](../modules/lab1-measurement.md)，FlashAttention 图也用于 [lab2 B3](../modules/lab2-kernels.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | 浏览器交互；QKV、attention 行与 token 输出 | A1：追踪一个 token 的数据流，再换成本题的尺寸和 dtype 计算 |
| [LLM Visualization](https://bbycroft.net/llm)（[作者仓库](https://github.com/bbycroft/llm-viz)） | 浏览器 3D 模型结构，需 WebGL | A1 补充：定位矩阵、注意力头与逐 token 推理；空间布局不是 GPU 的线程或物理存储布局 |
| [FlashAttention](https://arxiv.org/pdf/2205.14135) | Figure 1 的 HBM/SRAM 数据流；Algorithm 1 | A2、lab2 B3：重画分块读写与归约状态；纸面资源模型与真实内存峰值分开记录 |
| [Nsight Systems 的开销与延迟](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/) | 官方静态 trace 截图，Figures 8–11 | B1：辨认 CPU 提交、排队、GPU 执行与同步，不能相加所有泳道时长 |
| [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | 采集与导出 trace 的代码，配合时间线查看器 | B2：核对实际内核与采样区间；需要自己的 trace，正式计时另在关闭 profiler 后进行 |
| [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/) | 显存时间线、OOM 截图及交互查看器入口 | B3：对照张量存活、缓存与跨迭代分配；只读示例也能做题，分配器视图不包含所有设备分配 |

## GPU 工作分配与归约

配合 [lab2](../modules/lab2-kernels.md) 和选修 [lab2x](../labs/lab2x-kernel-challenge.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [NVIDIA GEMM 指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) | 分块、算术强度、Tile/Wave Quantization 静态图 | lab1 A3、lab2 A1/A3：计算复用、tile 空位和 wave；不直接套用示例硬件峰值 |
| [CUDA 合并访存图解](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) | 对齐、错位与跨步访问图 | lab2 A2：画 lane→sector 连线，区分有效字节与覆盖的 sector |
| [Triton 矩阵乘](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | grouped 与 row-major ordering 图 | lab2 A3：解释块顺序与数据复用；逻辑顺序不保证硬件执行顺序 |
| [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | 逐行归约代码与 GB/s 对照曲线 | lab2 B1/B2、lab2x A2：重建 mask、临时张量和流量账本，区分有效带宽与实际总流量 |
| [Triton Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | 前向代码、反向分组归约图、性能曲线 | lab2x A1：比较归约组织；LayerNorm 减均值，RMSNorm 不减，反向图不能当作前向实现 |
| [NVIDIA 矩阵转置教程](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) | global/shared memory 布局与 bank conflict 图解 | lab2x A3：计算 32×32 与 32×33 的 bank 映射；历史测速不是本机结果 |
| [GPU Puzzles](https://github.com/srush/GPU-Puzzles) | 作者 Notebook 的线程、内存访问图；Shared、Sum、Axis Sum | lab2x B3：逐轮画归约与同步；执行可视化需 Numba/CUDA，仅读图或 CPU 模拟无需安装 |

## 推理服务与调度

配合 [lab3](../modules/lab3-serving.md) 和 [lab4](../modules/lab4-scheduling.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | 原论文与演讲的 iteration-level scheduling 示意 | lab3 A1：按迭代画加入、退出和空闲槽；固定轮时长只是教学假设 |
| [vLLM：PagedAttention 图解](https://vllm.ai/blog/2023-06-20-vllm) | 分页生成、共享页与 COW 逐步示意 | lab3 A2/B2：追踪页表、引用计数和碎片；当前引擎语义须再查固定版本源码 |
| [SGLang：RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) | Figure 4，九步静态 radix tree 图 | lab3 A3：重建边分裂和前缀复用；树上的命中 token 数不是 GPU 加速比 |
| [Sarathi-Serve 演讲](https://www.usenix.org/system/files/osdi24_slides-agrawal.pdf)（[论文](https://www.usenix.org/system/files/osdi24-agrawal.pdf)） | slides 第 10、14、17 页时间线；论文 Figure 1 实测与 Figure 2 示意 | lab4 A1、lab6 A3：画 chunk/decode 甘特图，重建测量点并区分实测与示意 |
| [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | Figure 6 的资源与 KV 传输图；§2.3 | lab4 A2/A3：为 PD 交接标依赖与带宽成本，再检查 chunk 的历史 KV 读取假设 |
| [Perfetto UI](https://perfetto.dev/docs/visualization/perfetto-ui) | 浏览器交互时间线；缩放、区间选择与事件详情 | lab4 B3：拆分发送、排队、执行、传输；没有 trace 时先画教学时间轴，相邻事件不自动构成因果 |

## 分布式训练与 RL 系统

配合 [lab5](../modules/lab5-directions.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [UltraScale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) | ZeRO 内存图与动画；附录 Ring AllReduce 动画 | A1/A2：逐步跟踪 chunk 归约与状态分片；实际通信算法和临时内存另查实现 |
| [NVIDIA Megatron 图解](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/) | Figures 4–6，GPipe、1F1B、交错流水线静态图 | A3：绘制依赖、bubble 与各 stage 激活存活；真实通信和不均衡会改变时间线 |
| [HybridFlow](https://arxiv.org/html/2409.19256v2) | Figures 1–2 的 RLHF 数据流与控制模型 | B1/B2：画 rollout、reward、更新和权重同步的 DAG；调度收益与训练质量分别验证 |

## 实验设计与证据组织

配合 [lab6](../modules/lab6-reproduction.md) 和 [lab7](../modules/lab7-proposal.md)。

| 资料 | 形式与定位 | 用它完成什么 |
|---|---|---|
| [置信区间交互图](https://rpsychologist.com/d3/ci/) | Kristoffer Magnusson 的浏览器模拟；样本量、区间宽度与覆盖 | lab6 A2、lab7 A2：区分效果大小、不确定性与长期覆盖；演示不检验本题的独立性 |
| [NIST 交互效应矩阵](https://itl.nist.gov/div898/handbook/pri/section5/pri594.htm) | 静态主效应/交互图及公式 | lab6 B1：画自己的分组均值线，对照相反方向的效应；原例 ±1 编码的斜率解释不能直接套用 |
| [Google Research：Reusable Holdout](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/) | 作者博客中的多重比较漫画与自适应选择说明 | lab7 A1：标出留出信息进入决策的位置；反复查看普通留出集不等于实现该算法 |
| [DVC 依赖图](https://doc.dvc.org/command-reference/dag) | 官方 ASCII 与 Mermaid DAG 示例 | lab7 B2：从输入、运行到图表标转换命令和失效范围；不用安装 DVC，依赖图也不是机制因果图 |

需要查论文原文时回到 [论文清单](papers.md)；需要完整上游作业时查 [开源实验](open-source-labs.md)。这些图示可以作为课内题参考，不增加额外必做实验。
