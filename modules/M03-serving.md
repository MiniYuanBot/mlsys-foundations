---
title: M03｜推理服务、连续批处理与 KV cache
tags: [mlsys, module, inference]
week: 4
status: not-started
---

# M03｜推理服务、连续批处理与 KV cache

[索引](../index.md) · [前一模块](M02-kernels.md) · [实验 L03](../labs/L03-serving.md) · [下一模块](M04-scheduling.md)

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

课内题：连续批处理是否消除了所有排队？分页 KV 是否降低每个真实 token 本身所需的 KV 字节？共享前缀带来的收益与分页减少浪费是否同一机制？答案要说明条件。

## B 课：把模型变成可测服务，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–30 | 阅读 [环境手册](../01-environment.md) vLLM 部分，准备独立环境并开始模型下载 | 版本文件与下载状态，安装问题及时计时 |
| 30–65 | 从 [模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 进入 config，读取层数、KV heads、head dim、dtype；计算每 token 理论 KV | `2 × 层数 × KV头数 × 头维度 × 每元素字节` 的账本 |
| 65–95 | 读 [服务 CLI](https://docs.vllm.ai/en/stable/cli/serve/) 的 max-model-len、max-num-seqs、token budget、prefix caching | 五个参数的中文说明与固定值 |
| 95–105 | 休息 | — |
| 105–150 | 读 [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 的长度、到达率、明细保存；结合 [指标规范](../02-experiment-standard.md) | 把 TTFT、TPOT、完成延迟、吞吐写成单位明确的字段表 |
| 150–180 | 为 L03 写 smoke test 与容量初测计划；用一组虚构时间戳手算指标，明确标“例题” | 例题答案与实验的运行前记录 |

课内题：输出 64 token，首 token 在 0.10 秒、最后 token 在 0.73 秒时，请求级 TPOT 是多少？答案为 0.63/63=0.01 秒/token；不是 0.73/64。真实工具是否按相同时间戳定义，需要查实现。

## 本周实验与验收

[L03](../labs/L03-serving.md) 使用同一模型完成服务闭环，保留三次容量初测。它建立的是可重复基线，不要求比较所有服务框架。第 4 周向导师展示模型/负载/指标表及一个实际遇到的问题，请其判断这条实验主线是否适合组内课题。

## 资料汇总

必读：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[serve](https://docs.vllm.ai/en/stable/cli/serve/)、[bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)。选读：[SGLang](https://arxiv.org/abs/2312.07104) 的摘要与系统图，仅用于扩展地图。
