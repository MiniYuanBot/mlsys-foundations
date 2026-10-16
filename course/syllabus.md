---
title: MLSys 研究入门课程大纲
updated: 2026-10-06
tags: [syllabus]
---

# MLSys 研究入门：从张量运行时到推理系统

**目标**：16 周后，你能独立读懂并修改一个前沿 MLSys 系统的核心代码，复现一篇论文的关键机制，并提出一个有数据支撑的研究问题。

**节奏**：每周 2 次课，每次 3 小时，读讲义并做课内自测；另有约 5–7 小时做 lab。每讲两周。讲义分节，每读完一节就去做对应的 lab 部分。

**方向侧重**：编译器与超优化（Mirage/MPK）、LLM 推理服务、推测解码（SpecInfer）、分布式训练（Hetu/Galvatron）。这些都是 [苗旭鹏老师组](https://github.com/Hsword) 的主要研究方向。

**每个 lab 三级阶梯**：① 代码填空，熟悉架构 → ② 实现核心代码，达到性能目标 → ③ 精简复现一个前沿设计。实验代码见 [labs](../labs/README.md)。

---

## 课程总表

| 周    | 讲义                                                                          | Lab                                              | ③ 前沿复现目标                                                                     |
| ----- | ----------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 1–2   | [L0 张量、内存与 CPU 性能](lectures/L0-tensor-runtime.md)                     | [lab0 minitensor](../labs/lab0/README.md)（C++） | Goto GEMM 达到 BLAS 的 80%；tinygrad 式 lazy 融合                                  |
| 3–4   | [L1 GPU 执行模型与 CUDA GEMM](lectures/L1-cuda-gemm.md)                       | [lab1](../labs/lab1/README.md)（CUDA）           | Tensor Core `mma.sync` 三级流水线 HGEMM，达到 cuBLAS 的 60–80%                     |
| 5–6   | [L2 Attention：从 online softmax 到 FlashAttention](lectures/L2-attention.md) | [lab2](../labs/lab2/README.md)（Triton + CUDA）  | Triton FA2 与手写 `mma.sync` FA2 达到 SDPA 的 90%+；FlashDecoding split-KV         |
| 7–8   | [L3 深度学习编译器与超优化](lectures/L3-compilers.md)                         | [lab3](../labs/lab3/README.md)（C++ + NVRTC）    | mini-Mirage：改写 + 随机验证 + 代价搜索，自动发现 FlashAttention 式的重算          |
| 9–11  | [L4 LLM 推理系统](lectures/L4-serving.md)                                     | [lab4](../labs/lab4/README.md)（Rust + CUDA）    | 跑通 Qwen3-0.6B；PagedAttention + continuous batching + chunked prefill + 前缀缓存 |
| 12    | [L5 推测解码](lectures/L5-speculative.md)                                     | [lab5](../labs/lab5/README.md)（Rust + CUDA）    | 无损的拒绝采样；SpecInfer token tree + tree attention                              |
| 13–14 | [L6 集合通信与并行训练](lectures/L6-distributed.md)                           | [lab6](../labs/lab6/README.md)（C++ MPI）        | 手写 ring / recursive doubling allreduce + α-β 模型；DDP bucket 重叠 + ZeRO-1      |
| 15–16 | [L7 研究方法与研究项目](lectures/L7-research.md)                              | [lab7](../labs/lab7/README.md)（自选）           | 预注册 + 强基线 + 消融 + 负对照 + 留出集的完整研究循环；两页提案                   |

## 各讲要点

### L0 张量、内存与 CPU 性能（第 1–2 周）

张量 = 存储 + shape/stride/offset；视图与广播为什么不复制数据。Roofline 模型；缓存层次；GEMM 优化阶梯：循环顺序 → 分块 → 打包 + 寄存器微内核 → 多线程。Lazy 求值与算子融合，以及它通向 TVM、Triton、Mirage、megakernel 的脉络。

### L1 GPU 执行模型与 CUDA GEMM（第 3–4 周）

SIMT、warp、线程块、SM；global/shared memory 与寄存器；合并访存与 bank conflict；occupancy。GEMM 在 GPU 上的阶梯：naive → shared memory tiling → 寄存器分块 → 向量化 → warptiling → Tensor Core（`mma.sync`）。Hopper/Blackwell 的 TMA、WGMMA、TMEM。

### L2 Attention kernel（第 5–6 周）

Attention 的计算与访存；online softmax 推导；FlashAttention 1/2 的分块与重算；FlashDecoding 的 split-KV；paged attention；FlashAttention-3/4 与 FlashInfer。

### L3 深度学习编译器与超优化（第 7–8 周）

计算与调度分离（Halide/TVM）；以 tile 为单位编程（Triton/TileLang）；图捕获（torch.compile）；超优化（Mirage 的 μGraph 搜索与概率验证）；megakernel（MPK、Hazy Research）。

### L4 LLM 推理系统（第 9–11 周）

prefill 与 decode；KV cache 的显存账；Orca continuous batching；vLLM PagedAttention；SGLang RadixAttention；Sarathi-Serve chunked prefill；DistServe/Splitwise PD 分离；Mooncake；KV 压缩（PQCache 等）。

### L5 推测解码（第 12 周）

为什么 decode 受限于访存；拒绝采样为什么能保证输出分布不变；SpecInfer token tree 与 tree attention；Medusa、EAGLE 1/2/3、lookahead；系统层面的代价。

### L6 集合通信与并行训练（第 13–14 周）

α-β 通信模型；ring/tree allreduce；DDP 与 bucket；ZeRO/FSDP；Megatron TP/PP/SP；context parallel；MoE expert parallel；计算与通信重叠（FLUX/Comet）；自动并行（Galvatron）；RL 训练系统（verl、AReaL）。

### L7 研究方法与研究项目（第 15–16 周）

怎样读论文、找问题；强基线、预注册、消融、负对照、留出集；写 2 页设计评审与研究提案。

## 完成标准

- 每个 lab：`grade.md` 中的自动测试与性能目标全部达标，并提交报告（[模板](../labs/templates/report.md)）。
- 课程结束：一份研究提案。包括问题、最接近的三篇工作、强基线、你的初步数据、两周内可以完成的下一步。

## 配套资料

- [起步指南](guides/getting-started.md)、[环境准备](guides/environment.md)、[论文阅读方法](guides/reading.md)、[实验规范](guides/experiments.md)
- [资料汇总](readings/resources.md)：各讲的论文脉络、可视化资料、参考课程与开源实现
- [模板](templates/README.md)：阅读卡、实验记录、周报、提案
