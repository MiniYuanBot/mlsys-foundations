---
title: 从基础检查到AI Infrastructure系统实现与研究
updated: 2026-10-05
---

# 系统实现与研究主线

[课程索引](guides/course-map.md) · [论文路线](readings/papers.md) · [可视化资料](readings/visualizations.md) · [开源实验选用](readings/open-source-labs.md) · [配套代码](../mlsys-research-labs/README.md)

课程涵盖C++内存布局、CUDA kernel、流式attention、推理运行时和跨进程通信，最后完成一个研究实验。每部分提供基础练习、系统实现与研究问题。

## 学习层次

- **B 基础**：lab0–L07的基础测试与测量规范，用来证明先修能力。已经掌握的内容可直接通过检查与解释，不重复训练相同技能；缺证据仍需补齐。
- **S 系统**：每模块的主任务。提供工程骨架和独立参考，让你写关键算法、读生成代码/trace、修改真实引擎。结课需要完成对应的系统任务。
- **R 研究**：每模块提供问题卡，最终在L06选择**一个方向**完成机制主张→强基线→预注册→消融→负对照→留出→局限。其余R只需读懂问题，不要求全做。

建议16周×11h（每周约4h阅读/源码导读＋7h实验/分析/报告），共176h。系统任务8–14h可跨两周，基础练习的时间表只用于安排基础部分。第一次安装、编译和排错计入时间；需要补C++/CUDA先修或新增远端资源时可延至18周。若必须8周，只能选一条窄路线并减少广度，不能声称同样覆盖。

## 实现层次

| 层 | 使用的语言/工具 | 必须掌握的控制 |
|---|---|---|
| CPU张量后端 | C++17、CMake、CTest | stride/alias、循环/packing、编译优化、线程/分配成本 |
| GPU kernel | CUDA C++、cuBLAS、CUDA events | thread/block、global/shared memory、同步、越界、tile复用 |
| GPU DSL/编译器 | Triton；TileLang或Mirage选读 | program/tile、online归约、精度、生成代码、资源占用 |
| 推理运行时 | Python控制平面 + C++/CUDA kernel | 请求状态、KV页表/refcount、调度决策、执行路径与观测 |
| 原生推理引擎 | llama.cpp C++（R选项） | 推测解码accept/reject、draft开销、可应用源码patch |
| 分布式训练/通信 | torch.distributed/Gloo；C++ MPI；NCCL远端 | 真实进程、通信量、同步/overlap、加权梯度、状态分片 |
| 实验分析 | Python、CSV/JSON、profiler | 配对输入、单位/统计、版本、失败、噪声、可重建图 |

Python文件并不等于Python解释器在执行每个GPU操作，Triton也使用Python语法。反之，文件改成C++不自动成为系统研究：必须能解释执行路径、硬件成本与公平对照。

## 16周执行安排

| 周 | 重点 | 约7h实作的产出 |
|---|---|---|
| 1 | L00诊断＋原生后端 | B训练/梯度检查；S00布局/stride C++契约，报告缺的先修 |
| 2 | 测量与CUDA执行模型 | B01可靠计时/显存；S01 naive GEMM、cuBLAS、CPU oracle |
| 3 | CUDA共享内存与瓶颈 | S01 tiled GEMM、尾部/同步、强基线、tile消融与资源账本 |
| 4 | 融合与在线归约 | B02 softmax；推导online softmax，S02小形状正确性 |
| 5 | IO-aware attention | S02 streaming attention、显存/延迟/生成代码、留出形状 |
| 6 | 服务与KV管理 | B03固定模型/真实请求基线；S03页表/refcount/COW契约 |
| 7 | 实际引擎源码 | S03 vLLM调用链、固定源码版本、观测patch及开销检查 |
| 8 | 开放负载与SLO | B04固定矩阵；S04精确token/到达trace、send-lag审计 |
| 9 | 调度/缓存机制 | S04 cache工作负载、真实iteration trace、公平性/失败/负对照 |
| 10 | 分布式语义 | B05长尾模拟作为模型检查；S05真实2-rank Gloo梯度等价 |
| 11 | 通信与训练系统 | S05 C++ MPI ring与Gloo bucket；读DDP/HybridFlow指定部分 |
| 12 | 选择研究主张 | L06选kernel/compiler、serving或distributed/RL之一；固定上游commit/问题/预算 |
| 13 | 实施最小改动 | L06复现强基线与一个机制改动；编译/正确性/诊断证据 |
| 14 | 消融与负结果 | L06完成主矩阵、至少一个消融、负对照和噪声检查 |
| 15 | 留出与外部边界 | L07未参与调参的形状/负载/规模；必要时真实远端验证 |
| 16 | Artifact与提案 | 一命令重建、论文重合表、两周计划、10分钟答辩 |

lab2x融合RMSNorm仍为独立选修；它不能替代S02 streaming attention。若选CS336完整systems作业或CMU完整CUDA作业，按开源适配表替代相应S任务，不要求再做一份同质自编实验。

## S00：原生张量后端与布局（6–8h）

代码：[lab0/native](../mlsys-research-labs/lab0/native/README.md)。完成只读strided view与matmul、bounds/非连续契约；保留易读实现，再做一次循环/packing优化。对同精度A/B和BLAS强基线记录编译flag、线程、分配和拷贝范围。不要将参考double累加与student float的差异全部当成优化收益。

主线读Roofline和PyTorch 2指定部分；TVM的计算/调度分离与Mirage多层搜索留作R选读。基础课内题见 [lab0](modules/lab0-pytorch.md)；进入 S00 后再做下面的布局计算。

1. **布局转换何时值得做？** 教学测量中，一次 compact 拷贝耗时 `0.8 ms`，同一只读张量的每个消费算子在 strided 与 compact 布局下分别耗时 `0.35、0.15 ms`。画“直接消费”和“拷贝后复用”的数据流，求总耗时首次严格下降所需的消费次数，并画次数从 1 到 10 的成本曲线。若每次消费前源张量都被修改，重新计算并标出哪些缓存副本失效。产出成本式与一条真实测量时需要核对的假设，例如缓存、拷贝范围或输出契约。

R问题：固定实际模型shape，跨多个消费算子的compact/strided盈亏点是否稳定？

## S01：CUDA GEMM、资源账本与强基线（8–12h）

代码：[lab1/cuda](../mlsys-research-labs/lab1/cuda/README.md)。已给naive/CPU oracle/cuBLAS，学生补共享内存tiled GEMM，涵盖非整齐M/N/K、边界mask、barrier和race。CUDA events记录稳定执行，首调用/JIT/传输分别记；比较同精度模式。

主线读Goto GEMM和ThunderKittens指定部分，CUDA programming model作为实现查阅。必须交：thread/block与tile数据流图、正确性、三轮逐shape数据、编译/资源观察，以及一个tile变化为何有或无收益的解释。R问题：skinny或小batch GEMM中，启动/occupancy/访存哪个主导？先用可观察证据区分，不能仅凭TFLOPS命名瓶颈。

## S02：Streaming Attention与编译器（10–14h）

代码：[lab2/attention](../mlsys-research-labs/lab2/attention/README.md)。从online softmax推导，写分块QK/PV前向，避免全局S² scores；提供SDPA独立参考、causal/尾部/数值边界。显示显存随S变化、稳定延迟与误差，审查生成代码或资源统计。

主线读FlashAttention与Mirage指定部分；FA2/TileLang/FlashInfer为问题相关选读，说明它改变编译/运行时的哪层。FA3用于理解Hopper异步流水，4060不支持相关原artifact路径，不强制安装。R问题：同一算子不同schedule在未见shape的稳定性与编译成本；只改一个变量，保留变慢点。不能以单一shape调参作为新颖性证据。

## S03：KV页表与真实运行时（8–12h）

代码：[lab3/runtime](../mlsys-research-labs/lab3/runtime/README.md)。教学页池只是理解引用计数/COW/碎片的可测试模型；通过后必须跟踪固定版本vLLM的scheduler→KV manager→block pool→execution调用链，并做一处有开关的诊断patch。

读PagedAttention与SGLang；区分页分配、完整前缀块缓存、部分页COW和实际KV kernel。报告教学模型与所选版本的差异，不把所有机制都归为一篇论文。原型测试、真实引擎回归、patch开销和真实trace都是交付。R问题：缓存共享提高复用同时是否引入容量/回收/公平性问题？先核对当前引擎是否已解决。

## S04：精确负载、调度决策与缓存（8–12h）

代码：[lab4/trace](../mlsys-research-labs/lab4/trace/README.md)。适配上游benchmark客户端，而不是盲猜流式协议。存精确token IDs/到达时间与实际send-lag；固定单一chunk策略，比较prefix overlap0/80%×cache off/on×3seed=12轮。显式区分cold reset/warm cache，失败不丢。

主线读Orca与Sarathi-Serve，缓存分析复用SGLang阅读卡；增加scheduler iteration观测，做无patch/off/on开销对照。R选项：length-aware调度公平性、prefix污染、SLO与goodput边界。若选cache×chunk二维实验，作为另一个预注册研究矩阵；不能把同时变化的因素归因于单机制。服务优化必须含真实引擎patch或源码适配证据，不能只交参数表。

## S05：真实分布式训练与通信（8–12h）

代码：[lab5/distributed](../mlsys-research-labs/lab5/distributed/README.md)。两CPU进程Gloo是真实跨进程collective，验证同步后参数与单进程全batch oracle相同、world_size1与不等样本数的加权；不能用两个线程或单进程模型替代。再完成C++ MPI ring与bucket通信测量，理解数据移动与同步。

主线读DDP与HybridFlow指定部分；ZeRO（状态分片）、FLUX/Comet（计算通信重叠）、AReaL/DARTS（rollout与质量耦合）按研究问题选读，不要求全部实现。单4060不能证明多GPU扩展性；NCCL/TP/FSDP吞吐与真实overlap放到已批准的≥2GPU节点。CPU/MPI可给出语义和所测主机性能证据，不能外推GPU/NCCL性能；模拟只给假设推演。R问题：bucket大小与有效overlap、异步rollout staleness与质量约束。没有质量/收敛数据不能声称更快达到同等质量。

## lab6–lab7：研究循环（约35h实作，跨5周）

从三路线选一，先交2页设计评审，再编码：
- kernel/compiler：在线attention或融合算子，强baseline为同契约SDPA/cuBLAS/成熟kernel，研究shape/编译/资源边界。
- serving：缓存/调度patch，或**llama.cpp C++推测解码patch**；先做轻量ngram，再研究draft长度/接受率/总耗时，不能以接受率代替系统加速，EAGLE-3只读机制不要求在8GB训练drafter。
- distributed/RL：Gloo/MPI用于语义，真实多GPU通信/质量机制需对应资源；资源不足时选可本地完成的瓶颈调查，报告外部有效性待补。

lab6 budget512/2048与L07更长输入是**基础方法练习**，也可作为服务方向的基线。R至少包括：已核验原文主张/源码版本、最接近三工作、同工作量强baseline、可复现的最小改动、预注册主指标、三轮原始数据、一个消融、一个负对照、一个留出条件、失败/成本/质量边界和从原始数据重建图的命令。

样本少时给逐轮值与范围，不硬凑显著性。没有支持假设可以通过；无新颖性也可交有边界的机制报告，但不能称为创新算法。

## 完成度与资源

B/S/R每层各100分，不相加刷总分。S00–S05逐项通过；L06一条R形成完整证据链；L07完成留出/归档/提案。各层核心项必过且≥80。只做B称“基础测量训练完成”，不是课程主线完成；阅读一篇前沿论文不算R实验证据。

具体评分记录是S00–S07与R06达标；S06/R06复用同一研究项目材料，R06留出项在L07结束后补齐。其他章节R成绩为选修记录，不要求重复做八个研究项目。

课程要求的是研究步骤与系统能力，不保证复现论文大规模加速或产出论文。4060本地、CPU语义、多GPU实测、纯模拟四类证据明确分开。依赖由你配置，脚手架不自动下载上游或模型。上游版本/license/使用范围见 [开源lab手册](readings/open-source-labs.md)。
