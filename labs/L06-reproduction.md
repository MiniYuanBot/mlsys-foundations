---
title: L06｜检验 prefill token budget 的性能权衡
tags: [mlsys, lab, reproduction]
week: 7
status: not-started
---

# L06｜检验 prefill token budget 的性能权衡

[索引](../index.md) · [对应课程 M06](../modules/M06-reproduction.md) · [前置 L04](L04-workloads.md)

> [!tip] AI 辅助增强
> 本周可选：增加一项可解释的内部观测（30–45 分钟）。具体步骤与验收见 [增强实验手册](../06-ai-assisted-labs.md)。原实验提前完成时使用剩余时间，未完成则顺延；不额外叠加到本周 5h。

## 实验定位

以 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 的 chunked prefill 动机为起点，在当前安装版本的 vLLM 中改变每轮 token budget，检验延迟与吞吐的权衡。它是**缩规模机制检验**；没有使用原论文的模型、机器、代码和负载时，不能写成“复现原论文全部结论/加速比”。

必读：[vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)、[benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)、[复现报告模板](../templates/reproduction-report.md)。所有测量复用已通过的脚本，本周不另写服务引擎。

## 运行前写下假设

示例假设：“在长输入和较高到达率下，将 budget 从 2048 降到 512 可能降低请求级 p95 TPOT，但可能增加 TTFT 或减少输出吞吐；较低到达率下差异可能较小。”这是待检验预测，结果可能不支持。

主指标固定为请求级 p95 TPOT；辅助指标为 p95 TTFT、输出吞吐、失败率与既定 SLO 满足率。TPOT 是请求内平均生成间隔的统计，不是最坏单 token 间隔；因此不能据此完整证明 decode stall 被消除。更细的 ITL/iteration trace 可以作为后续验证。

在看数据前写：什么趋势支持预测；什么趋势不支持；若三轮波动大于差异，结论应是证据不足。可用“差异约超过 10% 且大于典型轮间波动”判断是否值得继续研究，但不把它当统计显著性检验。

## 固定矩阵与公平基线

| 变量 | 设置 |
|---|---|
| baseline A | chunked prefill 开启，`max-num-batched-tokens=2048` |
| treatment B | chunked prefill 开启，`max-num-batched-tokens=512` |
| 负载 | 目标输入 1024，输出 64，ignore_eos 开启，随机长度范围按固定长度设置 |
| 到达率 | 沿用 L04 的 λ_low 与 λ_high，两个方案使用完全相同数值 |
| 重复 | 每条件 seed 11/22/33；2 budget×2 rate×3=12 次 |
| 请求数 | 沿用 L04 固定的 N；若必须改则整组改，并在运行前记录 |
| 固定项 | 模型/revision、dtype、max-model-len、max-num-seqs、eager、prefix cache、软件、设备、SLO |

本实验不关闭 chunked prefill，否则同时改变机制开关与 budget，无法用当前设计解释差异。若当前版本不允许某设置，先查日志与对应版本文档；不能静默换一个含义不同的参数。输入必须长于较小 budget，才更可能出现感兴趣的切分行为。

## 300 分钟执行表

| 时间 | 任务 |
|---|---|
| 0–30 | 完成预注册与“原论文→本实验”差异表 |
| 30–65 | 检查 A/B 两种服务能启动，分别预热和做小 smoke |
| 65–170 | 运行 12 个正式条件，保留 A/B 的启动日志 |
| 170–220 | 汇总、检查噪声与失败、画主图和权衡图 |
| 220–270 | 写 2–3 页等量的机制复现报告，核对主张范围 |
| 270–300 | 从原始文件重建主图，填写周报与下周问题 |

## 执行顺序

1. 基于 L03 的完整服务命令，只把 `--max-num-batched-tokens` 分别设为 2048、512。启动后核对生效配置、模型与缓存设置；同一时间只运行一个服务。
2. 每次切换服务，先让旧请求排空，停止自己启动的旧服务，确认显存释放到合理水平，再启动新服务。用相同形状的 8 个请求预热，预热不入正式结果。
3. 保存事先确定的区组顺序：seed=11 时 A 的 low/high 后 B 的 low/high；seed=22 时先 B 后 A；seed=33 时再 A 后 B。每个服务实例完成两个 rate，减少重启成本；记录先后次序以便检查漂移。
4. 每次 benchmark 完成立即检查成功数、实际长度与配置。失败运行保留，修复后另编号重跑，不能覆盖。
5. **重新测量 A。** L04 的旧 baseline 可以作历史参考，但不能直接当本周唯一对照，因为温度、电源、后台任务或环境可能已经变化。

## 结果分析

主图以到达率为横轴、p95 TPOT 为纵轴，A/B 分组展示三轮点；权衡图分别展示 TTFT 与输出吞吐。给出每组的中位数、范围，不只给一个百分比。SLO 沿用 L04 的阈值，不为 B 重新定一个更宽目标。

若 B 变快，检查是否实际输出变少、失败更多、缓存命中或后端变化。若 B 更慢，讨论小模型、小 batch、eager 模式或更多调度迭代是否改变瓶颈，但把这些标为解释候选。若差异很小，说明当前条件下未找到有力证据，不硬凑趋势。

日志至少证明配置生效；若没有每轮内部调度 trace，就只能从受控参数变化观察系统行为，不能声称已经直接测到了每一次 prefill/decode 干扰。提出一个额外观测能如何排除替代解释。

## 交付与验收

交付：预注册记录、原文定位与差异表、12 份原始结果、A/B 启动日志、命令矩阵、汇总/绘图脚本、主图、权衡图、[复现报告](../templates/reproduction-report.md)。

- [ ] 唯一主动改变的服务配置是 token budget；其他变化如实列出。
- [ ] 两个 rate、三个 seed、A/B 对照完整，失败没有被抹去。
- [ ] 说明请求级 TPOT 与 ITL 的区别，没有过度解释。
- [ ] 明确写“现代 vLLM 上的机制检验”，不冒充原 artifact 的完全重现。
- [ ] 列出至少两个尚存限制，以及一个最小后续检验。
- [ ] 不支持假设也可通过，只要证据与解释合格。

若结果噪声大，先检查当前设备和短时测量问题；本周预算内可以延长同配置测量，预算不足则作为下一步，不能不断更换参数直到出现想要的曲线。

资料回查：[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) · [原代码](https://github.com/microsoft/sarathi-serve) · [vLLM serve](https://docs.vllm.ai/en/stable/cli/serve/)
