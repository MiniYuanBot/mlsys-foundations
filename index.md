---
title: MLSys 科研入门与选题训练｜课程索引
tags: [mlsys, course, index]
version: 2.1
updated: 2026-10-05
---

# MLSys 科研入门与选题训练

**8 周 × 每周 11 小时：两节各 3 小时的课，加 5 小时实验。** 面向大三上、数理和计算机基础扎实、熟悉 Python 训练框架但尚不熟悉 PyTorch 的你。默认设备为 Mac mini 16 GB 与 Windows / RTX 4060 8 GB；申请时间线按 2027 年暑期规划。

本版加入 [在线算子 OJ](05-online-judge.md) 和 [AI 辅助实验增强](06-ai-assisted-labs.md)。第三周可执行增强版 softmax 实验；若希望完整做一个更难的融合算子，推荐插入 [X 挑战周](labs/L02X-kernel-challenge.md)，总周期为 9 周、每周仍为 11h。TensorOJ 的 Ascend C 入门可作为 X 周的另一条路线。

课程的阶段目标是：独立完成一次小规模、可复查的系统实验，读懂相关论文的机制与证据，形成值得向苗旭鹏老师请教的具体研究问题。八周结束的交付物是**实验仓库、论文地图、复现报告和两页研究提案**。论文接收、第一作者署名和推免资格需要后续工作及学校、课题组的相应决定，不能由完成课程直接推出。

## 从这里开始

1. 将整个 `mlsys-research-course` 文件夹解压，直接作为 Obsidian 仓库打开；已有仓库也可以复制整个文件夹进去。内部采用标准相对 Markdown 链接，支持 Obsidian 和普通 Markdown 编辑器，无需插件。
2. 先用 15 分钟读 [使用说明与周节奏](00-start.md)，再看 [设备与环境](01-environment.md) 中自己对应的路径。这 15 分钟计入 M00 第一节课。
3. 在课表中放入 A 课 3 小时、B 课 3 小时、实验 2＋3 小时四个时间块。周报、读论文、调试都已经放进这些时间，不另加隐藏作业。
4. 从 [M00：PyTorch 桥接](modules/M00-pytorch.md) 开始。本周只执行本周必做项，不提前安装所有框架。

## 八周课表

| 周 | 模块与两节课 | 5 小时实验 | 本周必须拿出的东西 |
|---|---|---|---|
| 1 | [M00 PyTorch](modules/M00-pytorch.md)：张量/自动微分；训练循环/调试 | [L00 小型分类器](labs/L00-pytorch.md) | 可重跑训练脚本、梯度检查、过拟合与检查点验证 |
| 2 | [M01 性能测量](modules/M01-measurement.md)：Transformer 资源账本；计时与 profiler | [L01 Attention 测量](labs/L01-attention.md) | 正确性检查、6 个配置的计时/显存表、一个 trace |
| 3 | [M02 GPU 内核](modules/M02-kernels.md)：访存/并行；Triton/融合与 OJ 入门 | [L02 Fused Softmax](labs/L02-softmax.md) | 能解释的内核、3 种实现对照；增强版加入性质检查、参数消融与留出形状 |
| 4 | [M03 推理服务](modules/M03-serving.md)：批处理；KV cache/分页 | [L03 vLLM 基线](labs/L03-serving.md) | 本地 0.5B 服务、原始请求结果、容量初测 |
| 5 | [M04 调度与负载](modules/M04-scheduling.md)：chunked prefill；负载与 SLO | [L04 负载矩阵](labs/L04-workloads.md) | 长度×到达率的 12 次测量及解释 |
| 6 | [M05 方向比较](modules/M05-directions.md)：分布式资源；RL rollout/导师方向 | [L05 长尾模拟与方向筛选](labs/L05-tail.md) | 可核算的模拟、三条候选方向的证据与资源表 |
| 7 | [M06 机制复现](modules/M06-reproduction.md)：拆论文实验；控制变量/替代解释 | [L06 token budget 实验](labs/L06-reproduction.md) | 注册过的假设、12 次对照、机制复现报告 |
| 8 | [M07 选题提案](modules/M07-proposal.md)：相关工作查重；研究计划/答辩 | [L07 留出条件检验](labs/L07-proposal.md) | 新条件下的证据、两页提案、10 分钟汇报提纲 |

**插入周（选修）：** M02 后做 [L02X 残差＋RMSNorm 融合](labs/L02X-kernel-challenge.md)，或 [TensorOJ Ascend C 路线](05-online-judge.md)。二选一，后续周次整体顺延。其他每周增强任务仅在原实验提前完成后使用剩余时间，不同时追加所有挑战。

## 全程查阅的手册

- [00 使用说明与周节奏](00-start.md)：遇到卡点怎么办、如何验收、怎样找老师反馈。
- [01 设备与环境](01-environment.md)：Mac、WSL2、4060 8 GB、vLLM 和北大 HPC。
- [02 实验规范](02-experiment-standard.md)：计时、显存、吞吐/延迟、重复、文件组织和结论边界。
- [03 论文阅读与 idea 方法](03-reading-method.md)：分层阅读、200 篇建议的执行方案、证据矩阵。
- [04 到 2027 年暑期的路径](04-research-roadmap.md)：阶段成果、导师交流、项目继续/转向条件。
- [05 在线算子 OJ](05-online-judge.md)：TensorOJ、Tensara、LeetGPU、GPU MODE 的定位、题目入口、平台边界与验收。
- [06 AI 辅助实验增强](06-ai-assisted-labs.md)：每周进阶任务、独立验证、时间安排与理解检查。
- [资料汇总](resources.md)：按课程用途排列的官方课程、论文、代码与平台文档。

## 模板

- [论文卡](templates/paper-card.md)
- [实验记录](templates/experiment-record.md)
- [每周汇报与求助](templates/weekly-report.md)
- [复现报告](templates/reproduction-report.md)
- [两页研究提案](templates/research-proposal.md)

## 进度看板

- [ ] M00 / L00：CPU 训练闭环，CUDA 环境状态已记录。
- [ ] M01 / L01：能区分计算、访存、同步和测量误差。
- [ ] M02 / L02：能说明一个融合内核为什么可能有效，也知道何时无效。
- [ ] 选修 X 周：完成融合算子或 Ascend C 入门，区分本地子集与完整 OJ 通过。
- [ ] M03 / L03：能让同一模型服务稳定运行并保存原始结果。
- [ ] M04 / L04：能设计负载，区分吞吐与延迟目标。
- [ ] M05 / L05：能按兴趣、证据、资源判断方向。
- [ ] M06 / L06：能控制变量检验一个文献中的机制。
- [ ] M07 / L07：能提出可证伪问题，并给出下一阶段的执行计划。

> [!important] 课程默认选择
> 单卡推理系统是训练主线，因为你的设备可以承载小规模实验，而且同一套实验资产能连续使用。它不是提前替你决定终身研究方向。第六周集中比较 kernel/compiler、推理调度与 RL 系统；最终选题必须结合导师反馈。

> [!note] 版本与实测范围
> 原课程资料核对日期为 2026-10-04，OJ 与增强实验资料补充核对于 2026-10-05。课程命令未在你的 RTX 4060 上运行；在线平台也未进行账号登录或代码提交测试。版本、硬件、平台权限与队列可能影响执行。每个涉及 GPU 的实验均提供检查点和失败记录要求，不预设你一定得到某个加速比。
