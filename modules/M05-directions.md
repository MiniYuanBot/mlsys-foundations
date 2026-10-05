---
title: M05｜分布式、RL 系统与导师方向筛选
tags: [mlsys, module, distributed, rl]
week: 6
status: not-started
---

# M05｜分布式、RL 系统与导师方向筛选

[索引](../index.md) · [前一模块](M04-scheduling.md) · [实验 L05](../labs/L05-tail.md) · [下一模块](M06-reproduction.md)

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

课内题：如果张量并行增加通信，为什么仍可能降低延迟？如果只在一个进程模拟两块 GPU，哪些结论一定不能声称？思考答案时用“瓶颈转移”而非“多卡必然更快”。

## B 课：RL rollout 与研究方向比较，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–45 | 选读 [HybridFlow](https://arxiv.org/abs/2409.19256) 的工作流与系统结构，把 rollout、reward、训练/更新连起来 | 数据与模型状态流向图，标出同步点 |
| 45–80 | 选读 [DARTS（arXiv 正文，作者注明 ICML 2026 录用）](https://arxiv.org/abs/2605.30859) 的摘要、动机、核心图，关注 rollout 长尾问题 | 区分长度差异、资源等待、策略/分布改变的影响；不要求掌握全部算法 |
| 80–90 | 休息 | — |
| 90–115 | 扫读 [Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) 和 [SpecInfer](https://arxiv.org/abs/2305.09781) 的摘要/系统图 | kernel/compiler 与 speculative serving 各一个问题 |
| 115–145 | 对照 [导师论文列表](https://hsword.github.io/publications/)（[备用入口](https://cs.pku.edu.cn/info/1072/4600.htm)） 将论文放到三条路线：kernel/compiler、推理调度、RL 系统 | 每条最多两篇最接近论文，不按热度排序 |
| 145–180 | 为每条路线填写兴趣、可测问题、最近工作、两周可执行性、导师支持与硬件缺口 | [L05](../labs/L05-tail.md) 的方向筛选表草稿 |

课内题：缩短 rollout 的总完成时间，是否就能保证训练达到相同质量所需时间更短？截断长输出或改变采样策略可能改变什么？你现有实验只能支持哪个层面的主张？

## 方向选择规则

兴趣可按 1–5 分自评，但“有无具体问题、有无可做实验、有无指导资源”用事实填写。不能因为一篇论文很热门就选题，也不能因为单卡做不了完整系统就立刻排除方向：先找可验证子问题，再问导师能否提供扩展资源。

默认仍沿推理调度完成后两周，保证一条完整训练链。如果老师已给出具体 kernel 或 RL 子课题，可以改选，但要保留 M06 的控制变量、复现范围与 M07 的查重/提案标准；改实验前把新输入、baseline、指标和时间写清。

若做过 [L02X 融合算子](../labs/L02X-kernel-challenge.md)，将其留出结果与 OJ 接口差异带入 kernel/compiler 方向表。对昇腾有兴趣时，在 B 课的方向比较时段读 [TensorOJ 路线](../05-online-judge.md)，用题面判断编程模型与资源需求；尚未实测的 NPU 性能不要计入自己的证据。跨硬件开发可作为这一路线的候选子问题，不要求临时开启第四条必做研究方向。

## 资料汇总

核心选读：[ZeRO](https://arxiv.org/abs/1910.02054)、[HybridFlow](https://arxiv.org/abs/2409.19256)、[DARTS](https://arxiv.org/abs/2605.30859)。方向扫读：[Helix](https://arxiv.org/abs/2406.01566)、[FlexLLM](https://www.usenix.org/conference/nsdi26/presentation/oliaro)、[Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)、[SpecInfer](https://arxiv.org/abs/2305.09781)。论文状态以官方会议或原文页面为准，导师列表只作检索入口。
