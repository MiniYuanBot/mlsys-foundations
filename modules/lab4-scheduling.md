---
title: lab4｜调度、在线负载与延迟目标
tags: [mlsys, module, scheduling]
weeks: [8, 9]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab4｜调度、在线负载与延迟目标

[索引](../guides/course-map.md) · [前一模块](lab3-serving.md) · [实验 lab4](../labs/lab4-workloads.md) · [下一模块](lab5-directions.md)

**时间：3h＋3h＋5h。** 先修：lab3 服务能运行。目标：理解 throughput、TTFT、TPOT 可能冲突，学会把工作负载视为实验设计的一部分。

## A 课：两篇论文的机制与假设，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–20 | 从 lab3 结果出发，列出吞吐不错但用户体验差的两种可能 | 一张候选瓶颈表 |
| 20–70 | 精读 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 中 prefill/decode 干扰、chunked prefill 与调度机制的段落 | 画一个长 prompt 拆到数轮、与 decode 共存的时序 |
| 70–95 | 自设每轮 512 token 预算、4 个 decode 请求、一个 1024-token prompt，手算教学调度 | 记录每轮分配；说明这是简化模型而非 vLLM 的完整实现 |
| 95–105 | 休息 | — |
| 105–155 | 选读 [DistServe](https://arxiv.org/abs/2401.09670) 的问题、架构、goodput 定义及评估设置 | 写出解耦的收益来源，以及额外传输/放置成本 |
| 155–180 | 比较两者：同卡时间安排与跨资源放置各控制什么 | 两张论文卡的核心段落与三条待核实问题 |

### 课内题

用本课的调度手算和论文比较时段完成前两题；第三题选做。时间模型由题目给定，不代替实际 GPU 测量。

1. **必做｜把 chunk 大小换成延迟账本。** 4 个请求从 `t=0` 起持续 decode，各轮各产生 1 token，在本题区间内不结束；同一时刻到达一个 1024-token prompt。每轮 token 预算为 512，先分配这 4 个 decode token，再给 prompt 一个 chunk。比较 chunk 上限为 508 与 128 的两种策略，末块可较短。假设轮时长为 `T(p,d)=2+0.01p+0.05d ms`，本轮所有 token 均在轮末可见，新请求在最后一个 prefill chunk 结束时产生首 token。画到 prefill 完成的甘特图，逐轮列出 `p/d/T`，计算新请求 TTFT、旧请求最大 token 间隔和 prompt 处理速率。自检两种策略处理的 prompt token 总数一致，固定开销按轮数累加。参考 [Sarathi-Serve 演讲 slides 第 10、14、17 页](https://www.usenix.org/system/files/osdi24_slides-agrawal.pdf) 的静态时间线，指出题目的线性成本遗漏了哪种 mixed batching 现象。
2. **必做｜定位 PD 解耦的交接成本。** 模型有 24 层、8 个 KV heads、head dim 为 64，KV 用 BF16；prompt 为 2048 token。共置方案排队 12 ms、prefill 18 ms，prefill 完成即向客户端发送首 token，后续一步 decode 3 ms；解耦方案 prefill 排队 3 ms、执行 14 ms，首 token 随即直接发给客户端；随后传完整 prompt KV，传输启动开销 0.2 ms，完成后另一张空闲卡做一步 3 ms decode。忽略其它通信和重叠，分别取有效带宽 `12 GiB/s` 与 `1 GiB/s`。画两条含 KV 传输依赖边的时序，计算 KV 字节数、TTFT 与首个 token 间隔；判定 `TTFT≤25 ms`、首间隔 `≤12 ms` 是否同时成立。参考 [DistServe Figure 6](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) 的静态架构图。说明首 token 的发送位置为何改变指标解释，以及比较每 GPU goodput 还缺哪些信息。
3. **选做｜用历史 KV 读取构造 chunk 的反例。** 一个长度 `L=4096` 的 prompt 切成等长 chunk；比较 `C=1024/512/256`。简化为每个 chunk 都从 HBM 完整读取此前的 KV，当前 chunk 的写入和内部 attention 成本不计，每个历史 token 的 KV 为 48 KiB。画“chunk 编号 → 此前上下文长度”柱状图，推导总历史 KV 读取量并逐项求和核对。解释为什么它只是一种访存模型：跨 tile 复用、缓存和具体 kernel 都可能改变实际流量；不能把这一数值直接除以显存带宽来宣称真实 TTFT。可对照 DistServe §2.3 的 chunked-prefill 讨论，寻找需要 profiler 验证的假设。

## B 课：公平负载与 SLO，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–35 | 对照 [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 阅读 request-rate、burstiness、max-concurrency | 分别写出开放到达与限并发客户端的区别 |
| 35–75 | 设计短/长 prompt 两类，固定输出长度；根据 lab3 容量初测选两个有限到达率 | 完整的 2×2 条件表和三个 seed |
| 75–95 | 用一个 10 请求的例题计算 SLO 满足率，加入一个失败请求 | 解释为何失败不能从分母消失 |
| 95–105 | 休息 | — |
| 105–145 | 拆解一张论文性能图：横轴、样本、平均/尾部、baseline、是否约束相同 | 重画图的空白坐标，说明自己需要哪些原始字段 |
| 145–180 | 写 lab4 预期趋势与反例，选主指标；把分析流程写成脚本接口 | 运行计划和 `summarize.py` 的输入/输出约定 |

### 课内题

在负载设计、SLO 例题和分析接口时段完成前两题；第三题选做，不要求安装 trace 工具。

1. **必做｜检查客户端是否改变了到达过程。** 计划发送时刻为 `[0,10,20,30,40,50] ms`，请求 ID 依次为 A–F；每个请求一旦发出恰好 25 ms 后完成。比较无限并发客户端与最多允许 2 个未完成请求的客户端；后者按计划顺序保留待发送队列，一有空位立即发送，完成事件先于同一时刻的新发送处理。分别画计划到达、实际发送与完成的三条时间轴，算每个请求 send lag、最大在途数，以及 `t=50 ms` 前（含该时刻）的实际发送数。自检每个完成时刻等于自己的实际发送时刻加 25 ms。说明若只用“完成减实际发送”，会漏掉哪一段等待；把 `request-rate` 和 `max-concurrency` 同时固定为何仍可能产生不同负载。
2. **必做｜重建失败感知的 SLO 图。** 某固定 1 秒测量窗包含 6 个请求，全部终止于窗内。结果为 `A=(40,8,5)`、`B=(80,8,5)`、`C=(50,12,5)`、`D=(60,10,5)`、`E=(30,NA,1)`、`F=失败且无输出`；三元组是 `(TTFT ms, 请求级TPOT ms/token, 输出token数)`，成功条件为 `TTFT≤60 ms` 且多 token 请求 `TPOT≤10 ms/token`，单 token 请求只考察 TTFT。画 TTFT–TPOT 散点图和 SLO 边界，E 单独标注，F 保留在分母。计算成功完成率、SLO 满足率、goodput（满足 SLO 的请求/秒）与成功输出 token 吞吐；再计算误删 F 会怎样改变比例。自检图上的分类计数等于统计表；token 吞吐与请求 goodput 的单位不能混用。
3. **选做｜从同一条时间线提出三个可区分的解释。** 一条合成请求 trace 的事件为：计划发送 0、实际发送 12、服务接收 14、prefill 开始 44、结束 64、客户端首 token 70，单位均为 ms，假设时钟已对齐。画区间图并拆分客户端发送延迟、服务排队、prefill 执行和剩余传输/处理时间。针对“长 prompt 的 TTFT 更高”，分别提出计算量、排队、客户端拥塞三种解释，并为每种选一个新增事件或控制变量，写出什么观测会否定它。参考 [Perfetto UI 官方指南](https://perfetto.dev/docs/visualization/perfetto-ui) 的交互时间线与区间选择方式；可在浏览器查看工具，也可纸上作图。这里的时间戳是练习数据，界面上的相邻区间并不自行证明因果关系。

## 本周实验与验收

[lab4](../labs/lab4-workloads.md) 的必做是两类输入长度×两个有限到达率，每条件三轮。混合负载和真实对话数据作为选做，不挤占可靠测量时间。保留每轮散点及失败数，不能只交一张光滑折线。

## 资料汇总

必读：[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[DistServe](https://arxiv.org/abs/2401.09670)、[vLLM benchmark](https://docs.vllm.ai/en/stable/cli/bench/serve/)。源码参考：[Sarathi-Serve 仓库](https://github.com/microsoft/sarathi-serve)，本周不要求运行原 artifact。
