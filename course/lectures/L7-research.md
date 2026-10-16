---
title: L7 研究方法与研究项目
lab: lab7
weeks: 15-16
tags: [lecture, research-method, statistics, preregistration, reading, proposal]
---

# L7 研究方法与研究项目

[← 课程大纲](../syllabus.md) · 上一讲：[L6 集合通信与并行训练](L6-distributed.md) · 配套实验：[lab7](../../mlsys-research-labs/lab7/README.md)

> [!abstract] 这一讲要解决的三个问题
> 1. **好的 MLSys 研究问题从哪里来？** 怎样从“我测到一个奇怪的数”走到“一个可以发表的问题”？
> 2. **怎样让一个性能结论值得相信？** 强基线、预注册、置信区间、消融、负对照、留出集，各自防的是什么错误？
> 3. **怎样读论文、写提案、和导师沟通？** 怎样把 16 周的课程变成一份能拿去申请的研究材料？

前六讲教你“系统是怎样工作的”，这一讲教你“怎样产生关于系统的新知识”。它没有新的公式，但有一份**常见错误清单**：几乎每一条都来自真实论文的审稿意见。

| 节 | 内容 | 对应 lab |
|---|---|---|
| §1 | 问题从哪里来：缺口驱动的选题 | Part C1 |
| §2 | 什么样的结论可信：七种常见错误 | Part C2–C5 |
| §3 | 统计：中位数、bootstrap、最小效应 | Part A |
| §4 | 预注册与留出集 | Part B、C3、C5 |
| §5 | 怎样读一篇系统论文 | C2 |
| §6 | 提案、汇报与导师沟通 | C6 |
| §7 | 从课程到论文：时间线 | — |

---

## §1 问题从哪里来

### 1.1 缺口驱动

MLSys 的大多数好论文都遵循同一个模式：

```
1. measure something real              lab4: chunked prefill only gives 13%
2. find a gap between expectation      the paper (Sarathi-Serve) reports much more
   and reality
3. explain the gap with a model        step time = a*prefill + b*decode + ...; on a
                                       0.6B model the prefill share is small
4. design a mechanism that exploits    choose the chunk per step from the model
   the explanation
5. show it works where it should,      SLO met with lower TTFT; no gain when
   and NOT where it should not         prefill share is small (negative control)
```

这门课的每个 lab 都刻意留下了这样的缺口：lab3 的代价模型排错了顺序，lab4 的 chunked prefill 收益很小，lab5 的草稿模型太贵，lab6 的 bucket 划分很粗糙。[lab7/topics.md](../../mlsys-research-labs/lab7/topics.md) 把它们整理成了 9 个候选题目。

### 1.2 好问题的四个检验

| 检验 | 问自己 |
|---|---|
| **真实** | 这个问题我亲手测到过吗？有图吗？ |
| **重要** | 如果解决了，谁会受益？节省多少（时间、显存、钱）？ |
| **新** | 检索 30 分钟，有人做过吗？做过的话，在什么条件下？ |
| **可做** | 第一天能做什么实验？4060 能完成主要实验吗？ |

“新”不等于“没人想到过”。把一个已有的方法在新的条件下（小模型、消费级 GPU、新的负载）重新检验，找出它的适用边界，同样是有价值的贡献，也是学生最常见的第一篇论文的形态。

### 1.3 苗老师组的方向，与本课程的对应

| 组内方向（代表作） | 课程中的对应 | 候选题目 |
|---|---|---|
| 超优化与编译（Mirage、MPK） | L3 / lab3 | T1、T2 |
| 推测解码（SpecInfer） | L5 / lab5 | T6、T7 |
| LLM 推理服务（综述、SpotServe、Helix、PQCache） | L4 / lab4 | T4、T5 |
| 自动并行训练（Galvatron、Hetu） | L6 / lab6 | T8、T9 |

---

## §2 什么样的结论可信：七种常见错误

| 错误 | 例子 | 防御 |
|---|---|---|
| **弱基线** | 和自己写的、没优化过的版本比 | 和最强的现成实现比（cuBLAS、SDPA、vLLM），或者说明为什么比不了 |
| **挑选结果** | 跑了 10 次，报最好的那次 | 报中位数和置信区间；所有运行都保留 |
| **事后定标准** | 看到快了 3%，就说 3% 是有意义的 | 在看数据之前写下最小效应（预注册） |
| **调参调到测试集上** | 在报告的那几个形状上反复调参 | 留出集：从未在开发中看过的条件 |
| **把相关当因果** | 加了机制之后变快了，但其实是因为改动顺带减少了一次内存拷贝 | 消融：只去掉机制本身，看收益是否随之消失 |
| **方法在哪里都有效** | 宣称在所有负载下都有收益 | 负对照：在理论上不该有效的条件下，确认它确实无效 |
| **测量本身有偏** | 先跑基线，GPU 冷；后跑处理，GPU 热 | 交错运行各组（lab7 `run.py` 会打乱顺序），预热，记录环境 |

lab7 示例中的消融（一个 bucket，也就是没有重叠）在所有条件下都无效，这证明收益确实来自重叠，而不是 bucket 代码路径的其他副作用。

---

## §3 统计：中位数、bootstrap、最小效应

### 3.1 为什么用中位数

性能测量的噪声不是对称的：后台进程、频率下降、缓存冷启动只会让某一次**变慢**，不会让它变快。均值会被这些偶然的慢样本拉偏，中位数不会。

### 3.2 bootstrap 置信区间

有 n 次测量 x₁…xₙ。我们想知道：“如果再做一遍实验，中位数会落在哪个范围？”bootstrap 的思路是**用样本本身模拟“再做一遍”**：

```
repeat 10000 times:
    draw n values from {x1..xn} WITH replacement     e.g. x3 x1 x3 x5 x2
    compute the median of this draw
sort the 10000 medians; the 2.5% and 97.5% quantiles are the 95% CI
```

它不需要假设数据服从正态分布，这对偏斜的性能数据很重要。lab7 Part A 要你实现它，并测试覆盖率：模拟 300 次实验，95% 的置信区间应当约有 95% 的时候包含真值。

### 3.3 比值与判定

“快了多少”用处理 / 基线的中位数之比表示，它的置信区间要对两组**分别**重抽样。再与预注册的最小效应 δ 比较：

```
want "lower" by at least delta = 0.05  ->  target ratio 0.95

  CI entirely below 0.95          [0.886 ---- 0.944]  |          supported
  CI straddles 0.95                     [0.817 ------|-- 1.005]  inconclusive
  CI entirely above 0.95                             | [0.97 -- 1.02]  refuted
                                                    0.95
```

**inconclusive 不是失败**，它说明需要更多的重复，或者效应确实太小、噪声太大。在报告里如实写出“n=5 时不足以区分”，比强行宣称有效更可信。

### 3.4 需要多少次重复

粗略的经验：置信区间的宽度大约与 1/√n 成正比。lab7 示例中，n=5 时，dev 的置信区间宽约 0.19（0.817–1.005），要检测 5% 的效应，大约需要把区间缩到 0.05 以下，也就是 n ≈ 5 × (0.19/0.05)² ≈ 70 次。另一种做法是降低噪声：固定 GPU 频率、关闭后台程序、让每次运行更长。实践中通常两种办法一起用。

---

## §4 预注册与留出集

### 4.1 预注册

在看到数据**之前**，把以下内容写下来，并且“锁住”：

```json
{
  "hypothesis": "...",
  "metric": "step_ms", "direction": "lower", "min_effect": 0.05,
  "arms": {"baseline": ..., "treatment": ..., "ablation": ...},
  "dev": [...], "heldout": [...], "repeats": 5
}
```

lab7 的工具用 SHA-256 锁住这个文件：锁住之后若有改动，held-out 运行会被拒绝。如果你确实需要修改（比如发现指标定义错了），就新建一个 `-v2`，并写明原因。这样，审稿人（和你自己）都能看到你的标准是在什么时候定下的。

### 4.2 dev 与 held-out

```
dev conditions      look as often as you like; tune, debug, iterate
                         |
                    --lock--   (hypothesis, margin and held-out set are now frozen)
                         |
held-out conditions run ONCE; report whatever comes out
```

lab7 的示例正好说明了留出集的价值：dev 条件（P=4，batch 16）的结果是 inconclusive；held-out 条件中，P=4、batch 64 是 supported，而 P=2 完全没有效果。如果只看 dev，或者只挑 P=4、batch 64 来报告，都会得出误导性的结论。

---

## §5 怎样读一篇系统论文

### 5.1 三遍读法（Keshav）

| 遍 | 时间 | 读什么 | 得到什么 |
|---|---|---|---|
| 1 | 10 分钟 | 标题、摘要、引言、各节标题、图、结论 | 它解决什么问题？主要结论是什么？值不值得读第二遍？ |
| 2 | 1 小时 | 正文，跳过证明和实现细节；**仔细看每一张实验图** | 方法是什么？基线是什么？在什么条件下有效？ |
| 3 | 数小时 | 全部，尝试在脑中“重新实现” | 隐藏的假设是什么？哪里可以改进？ |

（S. Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)）

### 5.2 系统论文的阅读卡

读完第二遍后，填写一张卡片（[paper-card 模板](../templates/paper-card.md)），重点回答：

1. **瓶颈**：作者声称瓶颈在哪里？他们是怎样**测出来**的？
2. **机制**：用一张图画出方法。它省下的是计算、访存、通信还是等待？
3. **证据**：最强的那张实验图是哪张？基线是什么？硬件和负载是什么？
4. **边界**：在什么条件下方法会失效？作者讨论过吗？
5. **一个可检验的问题**：如果换成我的 4060 或者小模型，结论还成立吗？

### 5.3 苗老师建议的“读 200 篇”

一次读 200 篇确实不现实，但可以分层：

- **约 20 篇精读**：本课程各讲“演进脉络”表中的核心论文，读三遍并写阅读卡；
- **约 60 篇第二遍**：你选定方向内的论文，读到能讲出方法和基线；
- **约 120 篇第一遍**：顶会（OSDI、SOSP、MLSys、ASPLOS、EuroSys、NSDI）近两年的相关论文，只读标题、摘要和图。

最后这一层的目的不是理解，而是建立“这个领域有什么”的地图，以便在选题时快速判断“是否有人做过”。[苗老师的论文清单](https://github.com/Hsword/Awesome-Machine-Learning-System-Papers) 就是很好的地图。

---

## §6 提案、汇报与导师沟通

### 6.1 两页提案的结构

| 部分 | 篇幅 | 回答 |
|---|---|---|
| 问题与重要性 | 1 段 | 谁、在什么条件下、遇到什么瓶颈？ |
| 最接近的 3 项工作 | 1 张表 | 它们做了什么，我的不同在哪里？ |
| **已有证据** | **1 张图** | 我已经测到了什么？ |
| 方法 | 1 段 | 最小版本是什么？ |
| 未来两周的实验 | 1 段 | 基线、指标、消融、负对照、止损条件 |
| 风险与需要指导的地方 | 1 段 | 最想请老师判断的**一件事** |

模板见 [research-proposal.md](../templates/research-proposal.md)。

### 6.2 和导师沟通

- **带着数据去**：一张你测的图，胜过十分钟的口头描述；
- **带着一个具体的问题去**：“这个方向值不值得做？”不如“我测到 X，觉得原因是 Y，下一步打算做 Z。您看 Y 的解释对吗？”；
- **请导师判断他最擅长判断的事**：问题是否重要、是否已经被解决、应该去问谁。环境和 API 报错先自己解决，或者问组里的同学；
- **报告负面结果**：“我试了 A，没用，原因可能是 B”是有价值的进展，也说明你的判断力。

---

## §7 从课程到论文：时间线

详见 [研究路线](../guides/research-roadmap.md)。粗略的节奏：

| 时间 | 目标 | 可以拿出的证据 |
|---|---|---|
| 第 1–14 周 | 完成 lab0–lab6 | 7 份 lab 报告，每份都有自己测到的图 |
| 第 15–16 周 | lab7：一个小而完整的研究循环 | 设计评审、预注册、held-out 结果、两页提案 |
| 之后 2–3 周 | 带着提案和导师讨论，验证最危险的假设 | 一张关键图 + 最接近工作的对比 |
| 之后 2–3 个月 | 扩大实验：更多模型、硬件、负载；补齐消融 | 多条件的证据、局限说明 |
| 之后 | 与导师一起写作，选择合适的投稿时机 | 论文稿 |

**最重要的一点**：研究的进展不是“代码写了多少”，而是“不确定性减少了多少”。每周问自己：这周的实验排除了哪个解释？下周要排除哪个？

## 资料汇总

| 资料 | 用途 |
|---|---|
| S. Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | 三遍读法 |
| Gernot Heiser, [Systems Benchmarking Crimes](https://gernot-heiser.org/benchmarking-crimes.html) | 系统论文中常见的测量错误清单，与 §2 对照阅读 |
| [Hsword / Awesome-MLSys-Papers](https://github.com/Hsword/Awesome-Machine-Learning-System-Papers) | 苗老师组整理的论文地图 |
| Efron & Tibshirani, *An Introduction to the Bootstrap*（第 1–6 章） | §3 的理论 |
| [lab7/topics.md](../../mlsys-research-labs/lab7/topics.md) | 9 个基于前面 lab 实测缺口的候选题目 |
| [研究路线](../guides/research-roadmap.md)、[论文阅读方法](../guides/reading.md) | 时间规划与阅读方法 |

## 自测清单

- [ ] 能用“缺口驱动”的五步，描述一篇你读过的 MLSys 论文
- [ ] 能说出七种常见错误，并各举一个防御手段
- [ ] 能解释 bootstrap 置信区间的原理，以及为什么用中位数
- [ ] 能根据置信区间和最小效应给出 supported、refuted 或 inconclusive 的判定
- [ ] 能说明预注册和留出集分别防止了什么
- [ ] 能用三遍读法读一篇论文，并填好阅读卡
- [ ] 能写出一份两页提案，其中有你自己测到的一张图
