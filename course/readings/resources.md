---
title: 资料汇总
updated: 2026-10-06
tags: [resources, papers, visualization]
---

# 资料汇总

[← 课程大纲](../syllabus.md)

按讲义的顺序整理。每个主题分成三部分：**论文脉络**（从奠基工作到前沿，带着“它解决了上一篇的什么问题”去读）、**可视化/图解**（先看图建立直觉，再读论文）、**代码**（参考实现与教学项目）。

链接于 2026-10-05/06 用脚本逐个检查过可访问性，arXiv 的标题也逐一核对过。标注 *[浏览器]* 的链接会拦截脚本访问，用浏览器打开应该正常。

> [!tip] 怎么用这页
> 不要从头读到尾。每一讲的讲义末尾已经列了必读材料；这页用于**查阅**：想深入某个主题、找可视化资料或准备选题时再来翻。

---

## 苗旭鹏老师组的代表性工作

| 方向 | 工作 | 链接 |
|---|---|---|
| 超优化 | Mirage: A Multi-Level Superoptimizer for Tensor Programs（OSDI'25） | [arXiv](https://arxiv.org/abs/2405.05751) · [代码](https://github.com/mirage-project/mirage) |
| Megakernel | MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs | [arXiv](https://arxiv.org/abs/2512.22219) · [代码（mpk 分支）](https://github.com/mirage-project/mirage/tree/mpk) |
| 推测解码 | SpecInfer: tree-based speculative inference and verification（ASPLOS'24） | [arXiv](https://arxiv.org/abs/2305.09781) · [代码](https://github.com/flexflow/flexflow-serve) |
| 推理服务综述 | Towards Efficient Generative LLM Serving: A Survey from Algorithms to Systems | [arXiv](https://arxiv.org/abs/2312.15234) |
| 抢占式实例上的服务 | SpotServe（ASPLOS'24） | [arXiv](https://arxiv.org/abs/2311.15566) |
| 异构集群服务 | Helix（ASPLOS'25） | [arXiv](https://arxiv.org/abs/2406.01566) |
| KV cache 压缩 | PQCache（SIGMOD'25） | [arXiv](https://arxiv.org/abs/2407.12820) |
| 自动并行训练 | Galvatron（VLDB'23） | [arXiv](https://arxiv.org/abs/2211.13878) · [代码](https://github.com/PKU-DAIR/Hetu-Galvatron) |
| 分布式深度学习系统 | Hetu | [代码](https://github.com/PKU-DAIR/Hetu) |
| 论文清单 | Awesome-Machine-Learning-System-Papers | [GitHub](https://github.com/Hsword/Awesome-Machine-Learning-System-Papers) |

个人主页 hsword.github.io 目前无法访问（GitHub Pages 返回 404），可以看 [GitHub](https://github.com/hsword) 和 [alphaXiv 主页](https://www.alphaxiv.org/@xupeng-miao)。

---

## L0 张量、内存与 CPU 性能

**论文脉络**：Roofline（[CACM'09](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)） → Goto GEMM（[TOMS'08](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_final.pdf)） → [BLIS](https://github.com/flame/blis) → Halide（[项目页](https://halide-lang.org/)） → TVM（[OSDI'18](https://arxiv.org/abs/1802.04799)） → PyTorch 2（[ASPLOS'24](https://pytorch.org/assets/pytorch2-2.pdf)）

**可视化**：[Stride Visualizer](https://ezyang.github.io/stride-visualizer/index.html) · [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) · [JAX Scaling Book: Rooflines](https://jax-ml.github.io/scaling-book/roofline/) · [Algorithmica: Matmul](https://en.algorithmica.org/hpc/algorithms/matmul/) · [siboehm: Fast MMM on CPU](https://siboehm.com/articles/22/Fast-MMM-on-CPU) · [Making DL Go Brrrr](https://horace.io/brrr_intro.html)

**代码**：[BLISlab](https://github.com/flame/blislab) · [tinygrad](https://github.com/tinygrad/tinygrad) · [CMU 10-414 Needle](https://dlsyscourse.org/) · [Stanford CS149 asst1](https://github.com/stanford-cs149/asst1)（C++ 线程/SIMD，带 Mac 说明）

## L1 GPU 执行模型与 CUDA GEMM

**论文脉络**：CUTLASS/CuTe（[代码](https://github.com/NVIDIA/cutlass)） → Hopper TMA/WGMMA（[PyTorch 博客](https://pytorch.org/blog/hopper-tma-unit/)） → ThunderKittens（[ICLR'25](https://arxiv.org/abs/2410.20399)） → DeepGEMM（[代码](https://github.com/deepseek-ai/DeepGEMM)） → Blackwell TMEM（[Colfax 教程](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)）

**可视化**
- Simon Boehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)：本讲的核心读物，每一步都有图
- Aleksa Gordić, [Inside NVIDIA GPUs: anatomy of high-performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- [Modal GPU Glossary](https://modal.com/gpu-glossary)：图文版 GPU 术语表，可交互
- [JAX Scaling Book: GPUs](https://jax-ml.github.io/scaling-book/gpus/)
- Lei Mao, [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- Colfax: [WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) · [TMA](https://research.colfax-intl.com/tutorial-hopper-tma/)
- Horace He, [Strangely, matmuls run faster on predictable data](https://www.thonking.ai/p/strangely-matrix-multiplications)：功耗与降频

**代码与课程**：[MIT 6.S894 Accelerated Computing labs](https://accelerated-computing.academy/fall25/labs/)（RTX 4000 Ada，与 4060 同属一代） · [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) · [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)（GPL-3） · [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) · [Stanford CS149 asst3](https://github.com/stanford-cs149/asst3) · [LCPU 作业](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars) · [GPU MODE 讲座](https://github.com/gpu-mode/lectures)（[视频](https://www.youtube.com/@GPUMODE)） · [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## L2 Attention kernel

**论文脉络**
1. Online softmax（[2018](https://arxiv.org/abs/1805.02867)）：一遍扫描同时求出最大值和归一化因子
2. Self-attention does not need O(n²) memory（[2021](https://arxiv.org/abs/2112.05682)）
3. FlashAttention（[NeurIPS'22](https://arxiv.org/abs/2205.14135)）：IO 感知的分块，不物化 N×N 矩阵
4. FlashAttention-2（[ICLR'24](https://arxiv.org/abs/2307.08691)）：更好的并行划分，减少非 matmul 运算
5. FlashDecoding（[博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)） → FlashInfer（[MLSys'25](https://arxiv.org/abs/2501.01005)）：面向推理服务的 decode 与稀疏/分页 attention
6. FlashAttention-3（[NeurIPS'24](https://arxiv.org/abs/2407.08608)）：Hopper 上的 warp specialization 与 FP8
7. FlashAttention-4（[2026](https://arxiv.org/abs/2603.05451)）：Blackwell、TMEM、CuTe-DSL

旁支：稀疏 attention [NSA](https://arxiv.org/abs/2502.11089) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA) · 线性 attention [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) · 低比特 [SageAttention](https://github.com/thu-ml/SageAttention)

**可视化**：[FlashDecoding 动画](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) · [Tri Dao: FA3 博客](https://tridao.me/blog/2024/flash3/) · [Modal: 逆向 FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) · [UW CSE599m online softmax 推导笔记](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) · [FlexAttention](https://pytorch.org/blog/flexattention/) · [FlashInfer cascade](https://flashinfer.ai/2024/02/02/cascade-inference.html) · [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) · [bbycroft LLM 3D 可视化](https://bbycroft.net/llm)

**代码**：[flash-attention](https://github.com/Dao-AILab/flash-attention)（FA1–4） · [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)（约 100 行 CUDA） · [Triton 教程：fused attention](https://triton-lang.org/main/getting-started/tutorials/index.html) · [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)（CPU 解释器即可运行） · [Stanford CS336 作业 2](https://github.com/stanford-cs336/assignment2-systems)

## L3 深度学习编译器与超优化

**论文脉络**：TVM（[OSDI'18](https://arxiv.org/abs/1802.04799)） → TensorIR（[ASPLOS'23](https://arxiv.org/abs/2207.04296)） → Triton（[MAPL'19](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)） → PyTorch 2（[ASPLOS'24](https://pytorch.org/assets/pytorch2-2.pdf)） → TileLang（[2025](https://arxiv.org/abs/2504.17577)） → **Mirage**（[OSDI'25](https://arxiv.org/abs/2405.05751)） → Megakernel：[No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)、[TP megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)、**MPK**（[arXiv](https://arxiv.org/abs/2512.22219)）

**可视化**：[No Bubbles 的流水线图](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) · [Mirage 文档](https://mirage-project.readthedocs.io/) · Zhihao Jia, [Compiling LLMs into a MegaKernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) *[浏览器]*

**代码**：[Mirage](https://github.com/mirage-project/mirage) · [Megakernels](https://github.com/HazyResearch/Megakernels) · [TileLang](https://github.com/tile-ai/tilelang) · [depyf](https://github.com/thuml/depyf)（查看 torch.compile 生成的代码） · [MLC 课程](https://mlc.ai/)（陈天奇）

> 目前还没有 Mirage/MPK 的教学版精简实现，lab3 的 mini-Mirage 就是为了填补这个空白。

## L4 LLM 推理系统

**论文脉络**
1. Orca（[OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu)）：迭代级（continuous）batching
2. vLLM / PagedAttention（[SOSP'23](https://arxiv.org/abs/2309.06180)）：像操作系统分页一样管理 KV cache
3. SGLang / RadixAttention（[NeurIPS'24](https://arxiv.org/abs/2312.07104)）：基数树实现自动前缀复用
4. Sarathi-Serve（[OSDI'24](https://arxiv.org/abs/2403.02310)）：chunked prefill，避免 decode 被长 prefill 卡住
5. PD 分离：DistServe（[OSDI'24](https://arxiv.org/abs/2401.09670)）、Splitwise（[ISCA'24](https://arxiv.org/abs/2311.18677)）
6. Mooncake（[arXiv](https://arxiv.org/abs/2407.00079)）：以 KV cache 为中心的分离式架构
7. KV 压缩与卸载：[H2O](https://arxiv.org/abs/2306.14048) · [SnapKV](https://arxiv.org/abs/2404.14469) · [KVQuant](https://arxiv.org/abs/2401.18079) · [CacheGen](https://arxiv.org/abs/2310.07240) · [PQCache](https://arxiv.org/abs/2407.12820) · [LMCache](https://github.com/LMCache/LMCache) · [NanoFlow](https://arxiv.org/abs/2408.12757)

**可视化**
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)：本讲的核心读物（[vLLM 官方博客转载](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)）
- [vLLM 发布博客的 PagedAttention 动画](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching：[Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference) · [HuggingFace](https://huggingface.co/blog/continuous_batching) · [Lequn Chen](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/)
- [SGLang RadixAttention 博客](https://lmsys.org/blog/2024-01-17-sglang/) · [v0.4 零开销调度](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [DistServe 博客（goodput 图示）](https://hao-ai-lab.github.io/blogs/distserve/)

**代码**：[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（约 1200 行 Python，在 8 GB 的 4070 Laptop 上跑过） · [mini-sglang](https://github.com/sgl-project/mini-sglang) · [gpt-fast](https://github.com/meta-pytorch/gpt-fast) · Rust：[learning-lm-rs](https://github.com/InfiniTensor/learning-lm-rs)、[candle](https://github.com/huggingface/candle)、[vllm.rs](https://github.com/guoqingbao/vllm.rs) · C：[llama2.c](https://github.com/karpathy/llama2.c) · Mac：[tiny-llm](https://github.com/skyzh/tiny-llm)（MLX + C++/Metal）

## L5 推测解码

**论文脉络**：Speculative decoding（[ICML'23](https://arxiv.org/abs/2211.17192)） → **SpecInfer**（[ASPLOS'24](https://arxiv.org/abs/2305.09781)） → Medusa（[ICML'24](https://arxiv.org/abs/2401.10774)） → Lookahead（[ICML'24](https://arxiv.org/abs/2402.02057)） → EAGLE [1](https://arxiv.org/abs/2401.15077) / [2](https://arxiv.org/abs/2406.16858) / [3](https://arxiv.org/abs/2503.01840) → 长上下文推测 [MagicDec](https://github.com/Infini-AI-Lab/MagicDec)

**可视化**：[Lookahead decoding 动画](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) · [Medusa 项目页](https://sites.google.com/view/medusa-llm) · [EAGLE 项目页](https://sites.google.com/view/eagle-llm) · [Google: Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) · [HF assisted generation](https://huggingface.co/blog/assisted-generation) · [Jay Mody: 用 NumPy 实现 speculative sampling](https://jaykmody.com/blog/speculative-sampling/)

**代码**：[EAGLE](https://github.com/SafeAILab/EAGLE) · [Medusa](https://github.com/FasterDecoding/Medusa) · [LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) · [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)

## L6 集合通信与并行训练

**论文脉络**
1. Ring allreduce（[Gibiansky 的图解](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)） → [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. Megatron TP（[2019](https://arxiv.org/abs/1909.08053)） → PTD-P 与交错 1F1B（[SC'21](https://arxiv.org/abs/2104.04473)） → Sequence parallel（[2022](https://arxiv.org/abs/2205.05198)）
3. ZeRO（[SC'20](https://arxiv.org/abs/1910.02054)） → FSDP（[VLDB'23](https://arxiv.org/abs/2304.11277)）
4. Context parallel：[Ring Attention](https://arxiv.org/abs/2310.01889) · [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
5. 自动并行：**Galvatron**（[VLDB'23](https://arxiv.org/abs/2211.13878)）
6. MoE 与计算/通信重叠：[DeepEP](https://github.com/deepseek-ai/DeepEP) · [DualPipe](https://github.com/deepseek-ai/DualPipe) · FLUX（[arXiv](https://arxiv.org/abs/2406.06858)） · Comet（[MLSys'25](https://arxiv.org/abs/2502.19811)，代码在 [bytedance/flux](https://github.com/bytedance/flux)）
7. RL 训练系统：[InstructGPT](https://arxiv.org/abs/2203.02155) → OpenRLHF（[arXiv](https://arxiv.org/abs/2405.11143)） → HybridFlow/verl（[EuroSys'25](https://arxiv.org/abs/2409.19256)） → AReaL（[arXiv](https://arxiv.org/abs/2505.24298)） → [slime](https://github.com/THUDM/slime) → DARTS（[arXiv](https://arxiv.org/abs/2605.30859)，arXiv 页面注明 ICML 2026 录用）

**可视化**
- [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：交互式，图很多，**本讲首选**
- [JAX Scaling Book](https://jax-ml.github.io/scaling-book/)：[sharding](https://jax-ml.github.io/scaling-book/sharding/)、[training](https://jax-ml.github.io/scaling-book/training/) 两章
- Simon Boehm：[数据并行](https://siboehm.com/articles/22/data-parallel-training) · [流水线并行（bubble 图）](https://siboehm.com/articles/22/pipeline-parallel-training)
- [Lilian Weng: How to train really large models](https://lilianweng.github.io/posts/2021-09-25-train-large/) · [Insu Jang: 并行方式总览](https://insujang.github.io/2022-06-11/parallelism-in-distributed-deep-learning/)
- [slime 博客的架构图](https://lmsys.org/blog/2025-07-09-slime/) · [HF: RLHF/PPO 的 N 个实现细节](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)

**代码与课程**：[picotron](https://github.com/huggingface/picotron)（DP/TP/PP/CP，每种 < 300 行；[教程](https://github.com/huggingface/picotron_tutorial)） · [UCSD CSE 234 PA2](https://github.com/hao-ai-lab/cse234-w25-PA)（自己写 allreduce/alltoall 与 TP） · [nccl-tests](https://github.com/NVIDIA/nccl-tests) · [torchtitan](https://github.com/pytorch/torchtitan) · [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) · [verl](https://github.com/volcengine/verl) · [MIT 6.5840](https://pdos.csail.mit.edu/6.824/)（分布式系统 lab 的测试设计值得借鉴）

## 量化（选修）

**论文脉络**：FP8（[2022](https://arxiv.org/abs/2209.05433)） · LLM.int8()（[NeurIPS'22](https://arxiv.org/abs/2208.07339)） · GPTQ（[ICLR'23](https://arxiv.org/abs/2210.17323)） · SmoothQuant（[ICML'23](https://arxiv.org/abs/2211.10438)） · AWQ（[MLSys'24](https://arxiv.org/abs/2306.00978)） · Marlin（[PPoPP'25](https://arxiv.org/abs/2408.11743)） · QServe（[MLSys'25](https://arxiv.org/abs/2405.04532)）

**可视化**：Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) · Tim Dettmers, [LLM.int8() and emergent features](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)

**代码**：[llama2.c `runq.c`](https://github.com/karpathy/llama2.c)（Q8_0） · [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine) · [Marlin](https://github.com/IST-DASLab/marlin) · [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940)

---

## 通用资料

| 资料 | 说明 |
|---|---|
| [Stas Bekman: ML Engineering](https://github.com/stas00/ml-engineering) | 训练与推理工程实践大全 |
| [GPU MODE](https://github.com/gpu-mode/lectures) | 100 多期 kernel 与系统讲座：NCCL、ring attention、量化、CUTLASS、FlashInfer |
| [Umar Jamil](https://www.youtube.com/@umarjamilai) | 长篇从零实现视频，包括 Triton FlashAttention |
| [Chips and Cheese](https://chipsandcheese.com/) | CPU/GPU 微架构深度测评 |
| [LCPU AI Infra Seminars](https://infra.seminars.lcpu.dev/) | 北大 Linux 社的 AI Infra 课程：kernel、互联、服务、RL |
| [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Transformer 结构图解 |

## 在线评测与练习平台

| 平台 | 内容 | 适合 |
|---|---|---|
| [LeetGPU](https://leetgpu.com/) | CUDA/Triton/CuTe 等 kernel 题，免费 GPU | lab1、lab2 课后加练 |
| [GPU MODE reference-kernels](https://github.com/gpu-mode/reference-kernels) + [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) | 排行榜，可以免费提交到 A100/H100/B200 | 想在 Hopper 上验证时 |
| [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) / [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) / [Triton-Puzzles](https://github.com/srush/Triton-Puzzles) | 广播 → CUDA → Triton 的小谜题 | L0–L2 热身 |
