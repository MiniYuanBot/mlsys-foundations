---
title: lab3｜推理服务、连续批处理与 KV cache
tags: [mlsys, module, inference]
weeks: [6, 7]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab3｜推理服务、连续批处理与 KV cache

[索引](../guides/course-map.md) · [前一模块](lab2-kernels.md) · [实验 lab3](../labs/lab3-serving.md) · [下一模块](lab4-scheduling.md)

**时间：3h＋3h＋5h。** 先修：能够测量单个算子。核心转变：系统服务一组持续到来的请求，单个 kernel 的速度不是唯一目标。

## A 课：prefill、decode 与批处理，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–30 | 从 [CS336](https://cs336.stanford.edu/spring2025/) 找 inference 讲义；画 prompt 到逐 token 输出的流程 | 区分 prefill 与 decode 的输入、并行度与缓存 |
| 30–75 | 选读 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 的问题描述和 iteration-level scheduling | 将请求 A/B/C 的加入和结束画成 6 个迭代的表 |
| 75–95 | 手工比较静态 batch 等全部结束与每轮补入新请求的空闲位置 | 给同一组长度算有用 token 数与空闲槽，不把它当真实 GPU 吞吐 |
| 95–105 | 休息 | — |
| 105–150 | 读 [PagedAttention](https://arxiv.org/abs/2309.06180) 的动机、分页示意和系统设计关键段落 | 区分逻辑 token 位置、物理 KV block、内部/外部碎片 |
| 150–180 | 完成两张论文卡的精简版，比较 Orca 与 PagedAttention 各解决哪一层 | 一张“调度 vs 内存管理”的对照表 |

### 课内题

在本课的手算和论文卡时段完成前两题；第三题选做。图可手绘，计算可用已有的 Python 环境核对。

1. **必做｜重建连续批处理时间线。** A/B/C/D 在 `t=0` 按此顺序入队，均已完成 prefill，分别还需 `6/2/4/1` 个 decode 步；最多同时运行 2 个请求，每轮固定 10 ms，每个活跃请求产生 1 token，完成者在轮末离开。分别画出“同一批全部结束才装入下一批”和“下一轮立即按队列补位”的槽位图。计算各请求等待时间（入队至首次执行开始）、完成时间、总轮数和空闲槽位数；自检总有效槽位应等于总剩余 token 数。指出固定轮时长忽略了哪些 GPU 代价，因此不能把槽位利用率直接称为吞吐提升。参考 [Orca 论文](https://www.usenix.org/conference/osdi22/presentation/yu) 的 iteration-level scheduling 示意，核对新请求加入和旧请求退出的边界。
2. **必做｜逐事件追踪分页与写时复制。** 教学模型有 2 层、2 个 KV heads、head dim 为 8，K/V 均为 FP16；每页容纳 4 token，忽略对齐和元数据。A 已有 6 token，占用两页；随后依次执行 `fork B from A → A 追加 1 token → B 追加 3 token → 释放 A`。fork 共享完整前缀，包括未满尾页；写共享尾页前只复制有效 token，满页追加时另分配新页，refcount 为 0 即释放。画每一步的页表、物理页内容和引用计数，计算每 token/页字节数、峰值页数、尾页空位与复制字节数。自检每页引用计数等于指向它的页表项数。参考 [vLLM 作者图解](https://vllm.ai/blog/2023-06-20-vllm) 中分页生成和并行采样示意；这是论文式 COW 练习，不能据此断言当前引擎会缓存未满的前缀页。
3. **选做｜复现 RadixAttention 的分裂过程。** 空缓存依次收到 token ID 序列 `[1,2,3,4]`、`[1,2,5]`、`[1,2,3,6]`；模型、位置编码和 cache namespace 相同，容量充足，只统计 prompt KV，不含生成和 logits 重算。参考 [SGLang 作者博客 Figure 4](https://www.lmsys.org/blog/2024-01-17-sglang/) 的九步静态树图，画出每次插入后的压缩 radix tree，标出分裂边、命中前缀长度及新增 KV token 数。再构造一个“含相同 token 子串却不能复用前缀”的请求。自检树上每条根到叶路径能还原原始输入；说明这幅树图不能直接给出 GPU 内存占用或服务加速比。

## B 课：把模型变成可测服务，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–30 | 阅读 [环境手册](../guides/environment.md) vLLM 部分，准备独立环境并开始模型下载 | 版本文件与下载状态，安装问题及时计时 |
| 30–65 | 从 [模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 进入 config，读取层数、KV heads、head dim、dtype；计算每 token 理论 KV | `2 × 层数 × KV头数 × 头维度 × 每元素字节` 的账本 |
| 65–95 | 读 [服务 CLI](https://docs.vllm.ai/en/stable/cli/serve/) 的 max-model-len、max-num-seqs、token budget、prefix caching | 五个参数的中文说明与固定值 |
| 95–105 | 休息 | — |
| 105–150 | 读 [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 的长度、到达率、明细保存；结合 [指标规范](../guides/experiments.md) | 把 TTFT、TPOT、完成延迟、吞吐写成单位明确的字段表 |
| 150–180 | 为 lab3 写 smoke test 与容量初测计划；用一组虚构时间戳手算指标，明确标“例题” | 例题答案与实验的运行前记录 |

### 课内题

在指标表和 smoke test 设计时段完成前两题；第三题选做。全部输入为练习数据，无需启动服务。

1. **必做｜从事件重建延迟分布。** 客户端单调时钟单位为 ms：A 的发送/首 token/末 token 为 `0/100/160`，输出 4 token，逐 token 到达时刻为 `[100,110,130,160]`；B 为 `20/100/220`，输出 3 token，到达时刻为 `[100,120,220]`；C 为 `40/90/90`，输出 1 token；D 于 60 发送、260 超时，无 token。画四条请求时间线，计算成功请求的 TTFT、完成延迟、请求级 TPOT，以及把 A/B 的所有 token 间隔合并后的均值。说明两种 TPOT 聚合为何可能不同，C 的 TPOT 应如何记录，D 应进入哪些分母。自检每个多 token 请求的间隔总和等于末 token 减首 token；均值不能掩盖单个长停顿。
2. **必做｜计算 KV 容量上界。** 假设模型有 24 层、8 个 KV heads、head dim 为 64，KV 为 BF16，每页 16 token，KV pool 恰为 3 GiB。每请求最多保留 1536 个 prompt token 和 512 个生成 token，不共享、不驱逐；池内只计 K/V，`1 GiB=2^30 B`。画“单 token → 单页 → 单请求 → 并发请求”的账本，求最大整数并发；再算长度分别为 `17/31/32/33` 的四个请求实际占用页数及内部碎片率（分母为已分配页的总 token 容量）。自检逻辑 token 字节与物理页字节分开计算。结合 [vLLM 分页图解](https://vllm.ai/blog/2023-06-20-vllm)，列出真实设备上还要预留的三类显存；该上界不等于 `max-num-seqs` 的推荐值。
3. **选做｜追查指标实现，而非猜测定义。** 在你实验将固定的 vLLM tag/commit 中定位 `bench serve` 的请求结果字段与聚合函数；从 [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 进入文档或源码，不安装新版本。画“流式响应 → token/文本事件 → 单请求统计 → 跨请求统计”的数据流，附函数名和永久链接。用第 1 题数据核对 TPOT、ITL、失败和单 token 请求的处理，标出一次网络流式事件与一个 token 不必一一对应的位置。产出一张字段对照表，区分源码已经证实的行为与仍需 trace 才能判断的行为。

## 本周实验与验收

[lab3](../labs/lab3-serving.md) 使用同一模型完成服务闭环，保留三次容量初测。它建立的是可重复基线，不要求比较所有服务框架。第 7 周向导师展示模型/负载/指标表及一个实际遇到的问题，请其判断这条实验主线是否适合组内课题。

## 资料汇总

必读：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[serve](https://docs.vllm.ai/en/stable/cli/serve/)、[bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)。选读：[SGLang](https://arxiv.org/abs/2312.07104) 的摘要与系统图，仅用于扩展地图。
