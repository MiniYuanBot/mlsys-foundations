---
title: M06｜从论文主张到机制复现
tags: [mlsys, module, reproduction]
week: 7
status: not-started
---

# M06｜从论文主张到机制复现

[索引](../index.md) · [前一模块](M05-directions.md) · [实验 L06](../labs/L06-reproduction.md) · [下一模块](M07-proposal.md)

**时间：3h＋3h＋5h。** 本周检验 chunked prefill 中 token budget 与性能权衡的关系。现代 vLLM 中的机制检验不等于重跑 Sarathi 原实现，二者必须写清。

## A 课：拆解一条主张，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–25 | 回看 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 与自己的论文卡 | 从论文中选一条关于 chunk/干扰/权衡的具体主张，定位小节或图 |
| 25–65 | 找到该主张的实验设置，逐项抄录模型、硬件、负载、baseline、指标 | “原文→本地”差异表，不能记不清就猜 |
| 65–95 | 浏览 [原项目代码](https://github.com/microsoft/sarathi-serve) 的 README 和配置入口 | 判断哪些原实验超出 8 GB 单卡；不在此时安装整套 artifact |
| 95–105 | 休息 | — |
| 105–145 | 对照 [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/) 确认 chunk 开关与 token budget 的语义 | 明确本地可检验的主张，以及不对应原实现的部分 |
| 145–180 | 在看新数据前填写实验记录，写支持与反驳条件 | L06 预注册草稿，记录时间或 Git commit |

课内题：把 budget 从 2048 改成 512，影响可能来自每轮 prefill 规模、更多迭代、调度细节或其他实现路径。怎样避免见到任何差异都说“证明论文正确”？至少写两个需要日志或进一步实验区分的解释。

## B 课：控制变量与可复查报告，180 分钟

| 时段 | 学习步骤与材料 | 当场产出 |
|---|---|---|
| 0–35 | 用 [实验规范](../02-experiment-standard.md) 审查 L04 的原始结果，检查种子、输出长度、失败数 | 修正清单；修正不覆盖旧数据 |
| 35–75 | 设计 2 个 budget×2 个到达率×3 轮，固定长 prompt 和全部其他设置 | 12 次运行矩阵、交错顺序和停止条件 |
| 75–95 | 用上一周数据设计图形：TTFT/TPOT 及输出吞吐分开 | 图的空白草稿和每条线代表什么 |
| 95–105 | 休息 | — |
| 105–145 | 填 [复现报告模板](../templates/reproduction-report.md) 的前半；列三项对照公平性风险 | 标明软件默认值变化、负载变化与硬件差异的风险 |
| 145–180 | 口头做一次 5 分钟实验答辩；自己提三条反驳，修正过大的主张 | 最终运行计划和待查日志字段 |

课内作业：将“我的设置比默认快了，所以是创新”改写为合格结论。答案应包含具体条件、指标和不确定性，并说明调参本身不能证明新颖性，现代框架可能已实现相关机制。

## 本周实验与验收

执行 [L06](../labs/L06-reproduction.md)。验收关注假设是否在看数据前明确、是否只改变声明的因素、是否保存所有结果，以及是否敢于写出不支持假设的证据。老师读报告时应能知道你完成了哪一种复现。

## 资料汇总

必读：[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)、[Sarathi 仓库](https://github.com/microsoft/sarathi-serve)、[vLLM serve](https://docs.vllm.ai/en/stable/cli/serve/)、[实验规范](../02-experiment-standard.md)。工具文档中的当前默认值不是原论文实验设置，二者分别记录。
