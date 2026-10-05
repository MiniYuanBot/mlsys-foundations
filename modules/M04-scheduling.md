---
title: M04｜调度、在线负载与延迟目标
tags: [mlsys, module, scheduling]
week: 5
status: not-started
---

# M04｜调度、在线负载与延迟目标

[索引](../index.md) · [前一模块](M03-serving.md) · [实验 L04](../labs/L04-workloads.md) · [下一模块](M05-directions.md)

**时间：3h＋3h＋5h。** 先修：L03 服务能运行。目标：理解 throughput、TTFT、TPOT 可能冲突，学会把工作负载视为实验设计的一部分。

## A 课：两篇论文的机制与假设，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 从 L03 结果出发，列出吞吐不错但用户体验差的两种可能 | 一张候选瓶颈表 |
| 20–70 | 精读 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 中 prefill/decode 干扰、chunked prefill 与调度机制的段落 | 画一个长 prompt 拆到数轮、与 decode 共存的时序 |
| 70–95 | 自设每轮 512 token 预算、4 个 decode 请求、一个 1024-token prompt，手算教学调度 | 记录每轮分配；说明这是简化模型而非 vLLM 的完整实现 |
| 95–105 | 休息 | — |
| 105–155 | 选读 [DistServe](https://arxiv.org/abs/2401.09670) 的问题、架构、goodput 定义及评估设置 | 写出解耦的收益来源，以及额外传输/放置成本 |
| 155–180 | 比较两者：同卡时间安排与跨资源放置各控制什么 | 两张论文卡的核心段落与三条待核实问题 |

课内题：为什么更小的 prefill chunk 可能改善 decode 间隔却损失某些吞吐？解耦 prefill/decode 为什么不是在单卡开两个进程就自然得到论文效果？回答必须考虑资源竞争、通信和负载。

## B 课：公平负载与 SLO，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–35 | 对照 [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 阅读 request-rate、burstiness、max-concurrency | 分别写出开放到达与限并发客户端的区别 |
| 35–75 | 设计短/长 prompt 两类，固定输出长度；根据 L03 容量初测选两个有限到达率 | 完整的 2×2 条件表和三个 seed |
| 75–95 | 用一个 10 请求的例题计算 SLO 满足率，加入一个失败请求 | 解释为何失败不能从分母消失 |
| 95–105 | 休息 | — |
| 105–145 | 拆解一张论文性能图：横轴、样本、平均/尾部、baseline、是否约束相同 | 重画图的空白坐标，说明自己需要哪些原始字段 |
| 145–180 | 写 L04 预期趋势与反例，选主指标；把分析流程写成脚本接口 | 运行计划和 `summarize.py` 的输入/输出约定 |

课内作业：为“长 prompt 导致 TTFT 增加”列出三个替代解释，并各给一个检查：纯 prefill 计算增加、服务排队变化、客户端问题。只观测到一个总延迟数字时，不足以断言某一解释。

## 本周实验与验收

[L04](../labs/L04-workloads.md) 的必做是两类输入长度×两个有限到达率，每条件三轮。混合负载和真实对话数据作为选做，不挤占可靠测量时间。保留每轮散点及失败数，不能只交一张光滑折线。

## 资料汇总

必读：[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[DistServe](https://arxiv.org/abs/2401.09670)、[vLLM benchmark](https://docs.vllm.ai/en/stable/cli/bench/serve/)。源码参考：[Sarathi-Serve 仓库](https://github.com/microsoft/sarathi-serve)，本周不要求运行原 artifact。
