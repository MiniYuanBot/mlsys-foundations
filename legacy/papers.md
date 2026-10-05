# 论文路线：从实现机制到研究证据

本路线于 **2026-10-05** 核验一手来源，配合 [系统任务与周计划](../syllabus.md) 和 [开源实验选用指南](open-source-labs.md)。学习预算为 **16 周 × 每周 11 小时 = 176 小时**，各模块周次以系统主线为准。每模块主线只安排 **1 份基础阅读 + 1 份进阶阅读**，读指定机制与关键实验；后面的拓展为选读，不累加为必修。lab7 的基础阅读是 artifact 规范，明确不把规范当作研究论文。

按系统主线，每周约 4 小时用于阅读/源码导读、7 小时用于实验/分析/报告。论文只读本页指定机制，不要求每篇全文精读；已经读过的来源可复用阅读卡。硬件专属 artifact、多机部署或完整 RL 训练不塞进这个预算。阅读时先解释机制，再看代码；遇到本机不支持的机制，完成边界分析和可运行的局部实验，不以安装大型软件栈替代学习。

每篇记录三种状态，不能互相替代：**R 阅读**（问题、机制、假设与限制）；**T 教学实现**（自己的小规模代码与独立测试）；**A 原 artifact 验证**（作者指定代码版本、硬件/数据和目标图表的复跑）。使用缩小模型或替换硬件得到的是「缩规模机制实验」，只有满足原实验条件时才能写原结果复现。论文发表并不表示本课程已实测其结论。

## lab0 · 第 1 周：语言边界、布局与测量

**基础：Roofline（2009）**。[作者机构存档](https://escholarship.org/uc/item/6vv2j84j)。必读运算强度、带宽上界、计算上界与性能限制的构造方法。回答：相同 FLOPs 为什么可因访问布局不同而速度不同？一次输入拷贝是否算进计时？本机测得带宽与厂商峰值分别支持什么结论？

**进阶：PyTorch 2（ASPLOS 2024）**。[作者发布的论文](https://pytorch.org/assets/pytorch2-2.pdf)。必读 Dynamo 图捕获、graph break、Inductor 后端与编译开销；不要求从零实现编译器。回答：Python、C++ 与 GPU kernel 在调用链中各做什么？`torch.compile` 的首轮开销与稳态耗时如何分开？

**连接实验。** 用同一数据完成 Python/NumPy 或 PyTorch 参考、C++ 连续/跨步访问实现；固定 dtype、布局、线程数及输入大小。生成 correctness、copy time、compute time 三类证据，可选用一个小算子观察 eager 与 compiled 的首次/稳态差异。CPU 可完成 C++ 布局部分；4060 可补 GPU trace。这里的估计 Roofline 不等于测得所有硬件上界，不能仅用 wall time 宣称已经定位了缓存瓶颈。

## lab1 · 第 2–3 周：CUDA GEMM 与硬件映射

**基础：Anatomy of High-Performance Matrix Multiplication（2008）**。[作者所在大学的论文副本](https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf)。必读分块、数据复用与层次存储的推理。这篇讨论 CPU GEMM；回答哪些原理能迁移到 GPU，哪些缓存/TLB/线程假设不能直接套用。

**进阶：ThunderKittens（2024 预印本，ICLR 2025）**。[论文](https://arxiv.org/abs/2410.20399)。必读 tile 抽象、异步数据移动和计算流水线设计。画出「读取 → shared memory → 寄存器/计算 → 写回」的依赖图，解释同步错误与资源占用的代价。

**连接实验。** 先核对提供的 naive CUDA GEMM，再补齐 shared-memory tiled GEMM；cuBLAS 为必做强基线，WMMA 为选修；检查非方阵、尾块、不同累加精度。至少改变一个 tile 参数并解释带宽、占用或同步的证据。4060 可做上述教学内核；当前 ThunderKittens 的重点是 Hopper/Blackwell，不能把其 H100 教程默认当作 Ada 可运行代码。H100 上的专用指令/吞吐结论只读或远端验证，具体参考代码见 [开源指南](open-source-labs.md)。

## lab2 · 第 4–5 周：IO-aware Attention 与编译器

**基础：FlashAttention（2022）**。[论文](https://arxiv.org/abs/2205.14135)。必读分块、online softmax、IO 复杂度和重计算，先手算两块输入的最大值/归一化累计更新。回答：为什么不显式物化整个注意力矩阵仍能得到精确算法？「精确」和浮点误差容忍有什么区别？

**进阶：Mirage: A Multi-Level Superoptimizer for Tensor Programs（OSDI 2025）**。[会议论文入口](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)。必读多层 µGraph、搜索空间、剪枝与等价性验证。回答：搜索耗时是否进入端到端成本？正确性验证究竟覆盖什么，何时不能只依赖随机数值测试？

**连接实验。** 从已有 softmax/RMSNorm 练习进入 blockwise attention，保留 dense/SDPA 参考；用不同序列长度验证峰值内存与耗时趋势。编译器方向的 R 选项只选择一小段 fused expression，对比手写 CUDA/Triton 或 TileLang 与编译器产物，记录 compile time 和 warm runtime。4060 的缩小 shape 能验证教学机制；不能据此复现 H100 大模型结果。Mirage 当前仓库也介绍后续 MPK，阅读 OSDI 2025 论文不等于已获取该论文原 artifact，复跑前须另行锁定对应历史版本。

## lab3 · 第 6–7 周：KV 页表、生命周期与运行时

**基础：Efficient Memory Management for LLM Serving with PagedAttention（2023）**。[论文](https://arxiv.org/abs/2309.06180)。必读内部/外部浪费、页表、共享、copy-on-write。回答：释放请求和驱逐缓存为何不是同一操作？共享后逻辑 token 总数与物理 KV 占用为何不同？

**进阶：SGLang（NeurIPS 2024）**。[论文](https://arxiv.org/abs/2312.07104)。必读 RadixAttention 的前缀复用、树结构与缓存生命周期。回答：prefix hit、活跃引用和可驱逐缓存分别是什么状态？工作负载的前缀重复为何改变收益，冷门请求是否也得到同等好处？

**连接实验。** 完成 [lab3/runtime](../../mlsys-research-labs/lab3/runtime/README.md) 的 CPU 元数据状态机、独立契约测试和连续预留容量对照，再沿实际 vLLM 调用链做一个可验证的小修改。必须提交源码 SHA、patch 与执行证据。CPU 测试不能证明 GPU KV kernel 正确；本练习允许部分尾页 COW，而教学参照 vLLM V1 的完整块 prefix cache 有不同语义，报告必须解释差异。4060 从小模型/短上下文开展真实生命周期实验；读 SGLang 不要求再安装第二套 serving 系统。

## lab4 · 第 8–9 周：连续批处理、调度与缓存

**基础：Orca（OSDI 2022）**。[会议论文入口](https://www.usenix.org/conference/osdi22/presentation/yu)。必读 iteration-level scheduling 与 selective batching，画出长短输出混合请求在 request-level 与 iteration-level 调度下的时间线。区分请求完成、下一批准入与 GPU 单步执行。

**进阶：Sarathi-Serve（2024）**。[论文](https://arxiv.org/abs/2403.02310)。必读 chunked prefill、prefill/decode 干扰、吞吐与时延的折中。回答：更小 token budget 会改善哪个指标，又可能伤害哪个指标？调度策略是否偷用了不可提前知道的输出长度？

**连接实验。** 在真实引擎上先固定 chunk 策略，比较 prefix overlap 与 cache 开关，保留到达 trace、输入/输出分布、缓存容量和预热规则；分别报告 TTFT、ITL、吞吐、联合 SLO goodput、失败/超时及冷热请求差异。之后才把 cache×chunk 的二维对照作为预注册研究扩展，不能把同时变化的因素归因于单机制。4060 可测小模型单卡机制和相对变化；不能复现论文多 GPU 大模型容量结论。缓存作为主要方向时复用 lab3 的 SGLang 阅读卡，不再重复全文阅读。

## lab5 · 第 10–11 周：分布式训练与 RL 系统

**基础：PyTorch Distributed: Experiences on Accelerating Data Parallel Training（2020）**。[论文](https://arxiv.org/abs/2006.15704)。必读梯度平均、bucket、梯度就绪顺序与算通重叠。回答：两个 rank 的平均梯度什么时候等价于单进程全 batch？不同 batch 大小、loss reduction 与梯度累积如何影响答案？

**进阶：HybridFlow（2024 预印本，EuroSys 2025）**。[论文](https://arxiv.org/abs/2409.19256)。必读 RLHF 数据流、不同模型组件的执行/并行布局和资源切换。画出 rollout、reward、actor update 之间的依赖，解释它为什么比单一训练 loop 多出系统问题。

**连接实验。** 先实现/验证两进程 CPU Gloo 数据并行，并与单进程全 batch oracle 比较；再完成C++ MPI ring与Gloo bucket/MLP对照；参数化通信成本或受控trace分析rollout是R拓展，不能替代真实collective任务。4060 可做小模型算子和单卡 profile，CPU多进程可验证分布式语义和所测host性能，不能外推GPU互连。NCCL 带宽、跨 GPU overlap 和完整 RL 收敛需多 GPU/更大预算；用模拟结果不能声称训练质量保持或分布式加速成立。远端选修见 FLUX/Comet 与 AReaL。

## lab6 · 第 12–14 周：研究问题、局部复现与负结果

**基础：Rigorous Benchmarking in Reasonable Time（ISMM 2013）**。[作者机构存档](https://kar.kent.ac.uk/33611/)。必读重复实验层级、变异来源与不确定性。回答：同一进程内重复 kernel 和重启进程/服务后的重复测量，是否是同一种独立样本？如何选择实验预算而不是只增加内层循环？

**进阶：EAGLE-3（2025）**。[论文](https://arxiv.org/abs/2503.01840)。必读草稿模型输入、验证过程、训练与推理成本边界，以及接受率和端到端速度的区别。选择一个可被否定的问题，例如「接受长度增加是否总能降低总 token 延迟」，写对照与失败标准。该阅读用于练习拆分论文主张，不要求所有同学训练 drafter。

**连接实验。** 从 lab1–5 选已具备基线的一条机制，预注册、先证明正确、做至少一个消融，再在新形状/负载上验证。可选 C++ 路线是在 llama.cpp 中阅读并修改 speculative execution，先做无第二模型的 ngram 方法，再按预算考虑 draft model；其原理基础为下方 speculative sampling 选读。用同一目标模型、精度与采样设置对照，计入 draft/verify/回滚成本。4060 可做小模型/短上下文实验；无法承担的 EAGLE-3 训练及大模型结果记为未验证。选择其他系统方向也可完成本模块，不必强行迁移到 speculative decoding。

## lab7 · 第 15–16 周：Artifact、审稿视角与 proposal

**基础规范：MLSys 2025 Artifact Evaluation**。[官方要求](https://mlsys.org/Conferences/2025/CallForAE)。阅读 availability、functionality、硬件/软件/数据依赖和小样本 smoke 的要求。该届页面说明所评徽章范围，不能把它误写成所有年份的永久规则。课程只借用证据组织方式，不自动授予会议认证。

**进阶：FlashInfer-Bench（2026-01 预印本）**。[论文](https://arxiv.org/abs/2601.00227)。必读 definition、workload、solution、evaluation 的分离与从 kernel 到 serving 的验证链。回答：固定公开样例上正确为何不足以排除测试集特化？kernel speedup 何时不能转化成端到端收益？

**连接实验。** 为你的 lab6 结果组织一份他人可审查的 artifact：环境清单、输入/代码 hash、最小运行命令、参考结果、错误容忍、原始日志、生成图表脚本、失败条件与资源预算。加入未参与调优的留出案例并形成两页研究 proposal。4060 可完成小型 artifact 与 held-out 验证；只审阅作者公开结果或只复制一张图都不能称为独立复现。

## 选读池：按研究问题选择，不另加必修周

| 方向与一手来源 | 选读后必须做的动作 | 本机边界 |
|---|---|---|
| [Triton（MAPL 2019）](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) | 对照 CUDA/Triton 的 tile 表达，区分原始论文语言与今天 Python DSL API | 4060 可做教学 kernel；不能用现代实现直接声称复现 2019 artifact |
| [FlashAttention-2（2023 / ICLR 2024）](https://arxiv.org/abs/2307.08691) | 用一个 shape 解释工作划分变化；记录相对 FA1 的机制差异 | Ada 上先核验当前实现支持与 dtype |
| [FlashAttention-3（2024）](https://arxiv.org/abs/2407.08608) | 画出异步数据移动/计算流水线；分解依赖硬件的部分 | H100/H800 路线，4060 不运行其 Hopper 专属机制 |
| [TileLang（2025）](https://arxiv.org/abs/2504.17577) | 同一 GEMM 做一种 DSL 实现，比较代码量、编译时间、数值与稳态耗时 | 可选 Ada 兼容路径和小 shape；性能须实测 |
| [FlashInfer（MLSys 2025）](https://arxiv.org/abs/2501.01005) | 阅读 KV 格式、plan/run、动态负载与 CUDA graph 约束；追踪页表如何成为 kernel 输入 | 先核对后端和硬件要求；单卡局部实验不能替代完整 artifact |
| [DistServe（2024）](https://arxiv.org/abs/2401.09670) / [Mooncake（2024）](https://arxiv.org/abs/2407.00079) | 建立 prefill/decode 拆分与 KV 传输成本表，设计一项能推翻收益假设的网络条件 | 单卡只能做局部模型；跨机性能需远端设备与真实链路 |
| [FLUX（2024）](https://arxiv.org/abs/2406.06858) / [Comet（2025）](https://arxiv.org/abs/2502.19811) | 读细粒度算通重叠；比较顺序执行、理想上界与实际 overlap 的关键路径 | CPU 模型只做推理；多 GPU 性能要实测 |
| [AReaL（2025）](https://arxiv.org/abs/2505.24298) | 区分吞吐、staleness、训练质量；检查异步 rollout 质量主张需要哪些证据 | 官方多 GPU 示例不保证 8GB 可运行 |
| [DARTS（2026-05-29 预印本）](https://arxiv.org/abs/2605.30859) | 区分只重新调度长尾与主动改变轨迹分布；比较吞吐时同时提出质量/分布检验 | 作者在 arXiv 注明 accepted to ICML 2026；课程不把此标注当作已跑通 artifact。短 trace 无法证明 RL 质量结论 |
| [Accelerating LLM Decoding with Speculative Sampling（2023）](https://arxiv.org/abs/2302.01318) | 手算接受/拒绝及修正分布，再分析 draft 与 verify 的盈亏平衡 | greedy 对照可看 token 一致；随机采样不能简单要求逐 token 同序列 |

编译方向的基础选读：[TVM（2018）](https://arxiv.org/abs/1802.04799)，关注计算与调度分离、图级与算子级优化，连接L00布局问题与L02编译方向。它不额外占用主线必读名额。

L06进阶阅读默认EAGLE-3；选择kernel/compiler或distributed/RL研究时，用本表中最接近问题的一篇进阶工作替换，阅读卡仍为一基础＋一进阶。

## 阅读卡的验收

每模块提交两张短卡，每张最多一页：**问题 → 基线 → 关键机制 → 一个可检验预测 → 对应代码/实验 → 已确认与未确认的证据**。必须写出「如果某个条件改变，论文结论可能失效在哪里」。引用准确论文版本与来源；把自己的推断标为推断。能讲清一条机制并完成局部实证，比列出十篇标题更有价值。
