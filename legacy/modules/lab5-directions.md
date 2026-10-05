---
title: lab5｜分布式、RL 系统与导师方向筛选
tags: [mlsys, module, distributed, rl]
weeks: [10, 11]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab5｜分布式、RL 系统与导师方向筛选

[索引](../guides/course-map.md) · [前一模块](lab4-scheduling.md) · [实验 lab5](../labs/lab5-tail.md) · [下一模块](lab6-reproduction.md)

**时间：3h＋3h＋5h。** 这一周扩展地图，不要求你在 4060 上训练大规模 RL 模型。需区分“知道问题所在”和“已具备研究该方向的资源”。

## A 课：并行与资源安排，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–35 | 从 [CS336](https://cs336.stanford.edu/spring2025/) 选 parallelism 讲义，区分 DP、TP、PP 与状态分片 | 表格：切分对象、通信时机、显存收益、主要代价 |
| 35–80 | 选读 [ZeRO](https://arxiv.org/abs/1910.02054) 的动机与三个分片阶段 | 对参数、梯度、优化器状态分别写“复制/分片” |
| 80–95 | 用假定 1B 参数的玩具账本算 FP16 权重与 FP32 Adam 两个状态的字节数 | 明确说明主权重/梯度/临时缓冲还需另算 |
| 95–105 | 休息 | — |
| 105–150 | 扫读 [Helix](https://arxiv.org/abs/2406.01566) 与 [FlexLLM](https://www.usenix.org/conference/nsdi26/presentation/oliaro)，每篇约 20 分钟 | 各一行：资源异构/共存问题、目标、需要的硬件证据 |
| 150–180 | 把自己的设备与候选研究资源匹配，划出能在单卡验证的片段 | 三项资源缺口与最小验证方案 |

### 课内题

前两题放入并行讲义与显存账本时段，第三题选做。以下计算都是明确假设下的模型，不要求多 GPU 或新环境。

1. **必做｜手推 ring 与不均匀样本的梯度。** 4 个 rank 按 `0→1→2→3→0` 成环，rank `r` 的局部平均梯度为 `(r+1)×[1,2,3,4]`，本地样本数依次为 `[1,2,3,4]`。先决定每个 rank 送入归约的向量及最后的除数，再把向量分成 4 个等长 chunk，画 3 步 reduce-scatter 和 3 步 all-gather；每步标 chunk 编号及已包含哪些 rank 的贡献。用集中式按样本加权求和自检结果，并与“直接平均四个局部均值”比较。再将完整向量大小换为 32 MiB，设每步启动成本 5 μs、单向有效带宽 20 GiB/s、各链路并行且无计算/重叠成本，计算每 rank 发送字节数和该向量的总通信时间，不计样本数标量的归约。参考 [UltraScale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/index.html) 的附录 “Ring AllReduce” 图解；真实 NCCL 可选其它算法，此模型不能证明其延迟。
2. **必做｜检查 ZeRO 的显存结论是否漏项。** 模型恰有 `10^9` 个参数，8 个数据并行 rank；BF16 参数与梯度各 2 B/参数，FP32 主权重 4 B/参数，Adam 两个状态合计 8 B/参数，不做 FP32 梯度累积。画 DP、ZeRO-1/2/3 的驻留状态分片图，计算每 rank 的状态字节数，并同时报告 GB 与 GiB。每卡容量 8 GiB，另有激活 2 GiB 和通信工作区 1 GiB；ZeRO-3 还需临时申请最大一层 `10^8` 参数的完整 BF16 缓冲，假设不复用本地分片。哪些方案仅从此账本看能装下？自检主权重不重复计入，未分片的激活不除以 8。对照 [UltraScale 的 ZeRO 内存图与逐步动画](https://nanotron-ultrascale-playbook.static.hf.space/index.html)；列出两个仍可能让实际运行 OOM 的账本外因素。
3. **选做｜画流水线并辨认 bubble 的分母。** 4 个 stage、4 个 microbatch，每次前向 1 ms、反向 2 ms，忽略通信与优化器；每 stage 同时只能做一项工作，所有前向全部完成后才允许开始反向。画 GPipe 的完整甘特图，标每项任务依赖、总历时和各 stage 空闲时间；分别求 `空闲设备时间/总可用设备时间` 与 `额外历时/理想无气泡历时`，说明二者为何不是同一个比例。然后照 [NVIDIA Megatron 作者博客 Figure 5](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/) 改画满足数据依赖且每 stage 不重叠执行的 1F1B 调度，逐 stage 比较峰值待反向激活份数。参考图是静态调度模型；节点间通信和 stage 不均衡会改变真实时间线。

## B 课：RL rollout 与研究方向比较，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–45 | 选读 [HybridFlow](https://arxiv.org/abs/2409.19256) 的工作流与系统结构，把 rollout、reward、训练/更新连起来 | 数据与模型状态流向图，标出同步点 |
| 45–80 | 选读 [DARTS（arXiv 正文，作者注明 ICML 2026 录用）](https://arxiv.org/abs/2605.30859) 的摘要、动机、核心图，关注 rollout 长尾问题 | 区分长度差异、资源等待、策略/分布改变的影响；不要求掌握全部算法 |
| 80–90 | 休息 | — |
| 90–115 | 扫读 [Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) 和 [SpecInfer](https://arxiv.org/abs/2305.09781) 的摘要/系统图 | kernel/compiler 与 speculative serving 各一个问题 |
| 115–145 | 对照 [导师论文列表](https://hsword.github.io/publications/)（[备用入口](https://cs.pku.edu.cn/info/1072/4600.htm)） 将论文放到三条路线：kernel/compiler、推理调度、RL 系统 | 每条最多两篇最接近论文，不按热度排序 |
| 145–180 | 为每条路线填写兴趣、可测问题、最近工作、两周可执行性、导师支持与硬件缺口 | [lab5](../labs/lab5-tail.md) 的方向筛选表草稿 |

### 课内题

前两题用于工作流绘图和方向比较；第三题选做。只讨论给定工作量的资源安排，不运行 RL 训练。

1. **必做｜从 RLHF 数据依赖求关键路径。** 给定同步一轮 PPO：actor rollout 100 ms；随后 reward、reference log-prob、critic value 各耗时 `20/30/25 ms`，使用三个独立资源，可同时开始；advantage 计算 5 ms，依赖三者；之后 actor update 60 ms 与 critic update 40 ms 在两组独立资源并行；actor 更新后需 10 ms 同步权重给下一轮 rollout，下一轮还必须等待 critic update 完成。参考 [HybridFlow Figure 1–2](https://arxiv.org/html/2409.19256v2) 的静态数据流图，画出带耗时与数据标签的 DAG，标明轨迹、旧 log-prob、reward、value、模型版本流向，求本轮到下一轮可开始的最短时间。再把三个准备任务放到同一资源串行运行并重算。自检关键路径上的任务无非法重叠；没有旧策略版本标记时，哪一项 PPO 数据对应关系无法核验？
2. **必做｜量化 rollout 长尾，而不偷换工作量。** 4 个请求同时到达，待生成长度分别为 `40/40/40/200` token；两台相同 worker，各同时只运行一个请求，以 20 token/s 顺序生成，无迁移、prefill 和调度成本。比较静态分组 `worker0=[40,40]、worker1=[40,200]` 与全局队列 `[200,40,40,40]` 的空闲即补位策略（假设预先知道长度，以长请求优先），同一时刻空闲时 worker0 先取任务。画甘特图，计算 makespan、总有效 token/s、以 `有效生成时间/(2×makespan)` 定义的利用率。然后把每条输出截断到最多 80 token，重算并报告丢弃 token 数。参考 HybridFlow 的 generation 阶段图理解同步边界；用一句可被实验推翻的陈述区分“同一工作量完成更快”和“更少工作更快”，还要说明实际在线系统如何取得长度估计；不能从本题推出训练质量不变。
3. **选做｜判断更快的 rollout 是否换来了更快的学习。** 两方案每轮总时长分别为 120 s 和 90 s，使用相同数量 GPU；同一固定评测集上，于第 `[0,10,20,30,40]` 轮记录的分数为 A=`[0.20,0.35,0.48,0.58,0.62]`、B=`[0.20,0.32,0.43,0.51,0.57]`，这里只给一个 seed。画“轮数–质量”和“累计墙钟时间–质量”两幅图，比较观测到的首次达到 0.55 的时间；只允许用评测点，不插值。再说明若目标提高到 0.60，应怎样记录 B 的结果，为什么不能外推胜负。设计最小补充记录：固定评测、至少三个 seed、长度分布、截断率及策略版本差。结合 [DARTS 原文](https://arxiv.org/abs/2605.30859) 的长尾问题检查自己的主张究竟涉及系统排程还是采样分布改变。

## 方向选择规则

兴趣可按 1–5 分自评，但“有无具体问题、有无可做实验、有无指导资源”用事实填写。不能因为一篇论文很热门就选题，也不能因为单卡做不了完整系统就立刻排除方向：先找可验证子问题，再问导师能否提供扩展资源。

从 kernel/compiler、推理调度、RL 系统三条路线中选择一条，在 lab6–7 继续研究同一个具体问题。结合导师建议确定范围，保留 lab6 的控制变量、复现范围与 lab7 的查重/提案标准；调整实验前把输入、baseline、指标和时间写清。

若做过 [lab2x 融合算子](../labs/lab2x-kernel-challenge.md)，将其留出结果与 OJ 接口差异带入 kernel/compiler 方向表。对昇腾有兴趣时，在 B 课的方向比较时段读 [TensorOJ 路线](../guides/online-judge.md)，用题面判断编程模型与资源需求；尚未实测的 NPU 性能不要计入自己的证据。跨硬件开发可作为这一路线的候选子问题，不要求临时开启第四条必做研究方向。

## 资料汇总

核心选读：[ZeRO](https://arxiv.org/abs/1910.02054)、[HybridFlow](https://arxiv.org/abs/2409.19256)、[DARTS](https://arxiv.org/abs/2605.30859)。方向扫读：[Helix](https://arxiv.org/abs/2406.01566)、[FlexLLM](https://www.usenix.org/conference/nsdi26/presentation/oliaro)、[Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)、[SpecInfer](https://arxiv.org/abs/2305.09781)。论文状态以官方会议或原文页面为准，导师列表只作检索入口。
