# Lab 7：研究项目 —— 从一个测到的缺口到一份能说服导师的提案

> 配套讲义：[L7 研究方法与研究项目](../../mlsys-research-course/lectures/L7-research.md)
> 预计用时：2 周 × 约 10 小时（可以延长）　·　语言：你选题所在 lab 的语言 + Python 实验工具　·　硬件：取决于题目

前面六个 lab 都有标准答案，这一个没有。目标是完成一个**小而完整**的研究循环：从前面 lab 中实际测到的一个缺口出发，提出一个可以被证伪的假设，在**看数据之前**写下判定规则，用强基线、消融、负对照和留出集检验它，最后写成一份两页提案，带着它去和导师讨论。

“小而完整”比“大而模糊”有价值得多。一个结论是“我的 bucket 策略在 P=4、batch 64 时快 9%，但在 P=2 时没有效果，原因是……”的项目，比一个“我想做一个更好的推理系统”的计划，更能说明你会做研究。

| 部分                   | 内容                                                                           | 难度梯度                   |
| ---------------------- | ------------------------------------------------------------------------------ | -------------------------- |
| **A** 实验统计工具     | 中位数的 bootstrap 置信区间、比值的 CI、按预注册的阈值做判定                   | ① 填空（Python，约 40 行） |
| **B** 跑通一个示例研究 | 用提供的预注册文件，在 lab6 上走完 dev → lock → held-out → report              | ② 熟悉方法                 |
| **C** 你的研究项目     | 选题 → 设计评审 → 预注册 → 实现 → 主实验、消融、负对照 → 留出验证 → 提案与汇报 | ③ 真正的研究               |

---

## Part A：实验统计工具（约 1.5 小时）

读讲义 §3。笔记本 GPU 上的测量很嘈杂：频率会随温度变化，后台进程会抢资源。所以每个条件都要重复多次，用**中位数**加 **bootstrap 置信区间**来描述，再拿置信区间和**事先定好的**最小效应比较，得出结论。

| TODO | 函数           | 内容                                                                        |
| ---- | -------------- | --------------------------------------------------------------------------- |
| S1   | `bootstrap_ci` | 有放回地重抽样，计算统计量，取分位数                                        |
| S2   | `ratio_ci`     | treatment / baseline 的中位数之比；两组分别重抽样                           |
| S3   | `decide`       | 置信区间完全越过阈值为 supported，完全没越过为 refuted，其余为 inconclusive |

```bash
python -m unittest lab7.tests.test_stats -v      # stdlib only: any Python 3.10+, no GPU needed
```

`test_coverage_is_roughly_nominal` 会模拟 300 次实验，检查你的 95% 置信区间是否真的约有 95% 的时候包含真值。

## Part B：跑通示例研究（约 1.5 小时，在 WSL 中）

`examples/bucket_overlap.prereg.json` 是一份完整的预注册：问题、假设、指标、方向、最小效应 5%、每组 5 次重复、三个 arm（基线、处理、一个消融），以及 dev 条件和 held-out 条件。

```bash
cmake --build build/lab6 -j                       # your lab6 must pass its tests
python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --phase dev
python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --phase heldout   # refuses: not locked
python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --lock --phase heldout
python -m lab7.exp.report results/lab7/bucket_overlap --markdown
```

参考实现的结果：

| 阶段                | 条件          | 处理 / 基线 [95% CI] | 判定                                       |
| ------------------- | ------------- | -------------------- | ------------------------------------------ |
| dev                 | P=4，batch 16 | 0.926 [0.817, 1.005] | inconclusive                               |
| held-out            | P=2，batch 16 | 1.020 [0.772, 1.162] | inconclusive（点估计完全没有效果）         |
| held-out            | P=4，batch 64 | 0.907 [0.886, 0.944] | **supported**                              |
| 消融（一个 bucket） | 三个条件      | 0.97–1.02            | 无效果，证明收益确实来自多个 bucket 的重叠 |

**这正是一个诚实的研究结论**：效果在 4 个 rank、较大 batch 时成立；在 2 个 rank 时消失，因为需要藏的通信本来就少；而 n=5 时，dev 条件下的噪声太大，无法下结论。

> 运行这个示例时，作者发现 lab6 的 `CommThread` 有一个罕见的崩溃：线程在成员变量构造完成之前就启动了，属于未定义行为。60 次运行里出现了 2 次。正是“失败的运行不丢弃、要记录下来”这条规则让它暴露了出来。

## Part C：你的研究项目（约 15 小时以上）

### C1 选题（第 1 天）

从 [topics.md](topics.md) 中选一个题目，或者按同样的格式提出自己的题目。选题的标准：

- 缺口是你**亲手测到**的，有图为证；
- 第一天就能做一个实验，检验问题的前提是否成立；
- 4060 加 CPU 就能完成主要实验（需要 HPC 的部分放到“下一步”）。

### C2 设计评审（第 2 天，1 页）

用 [design-review.md](design-review.md) 模板写一页设计评审，最好和同学或导师过一遍。重点写出**最接近的 3 项工作**，以及你的问题和它们的区别。检索 30 分钟以上。

### C3 预注册（第 3 天，写代码之前）

复制 `examples/bucket_overlap.prereg.json`，改成你的题目。必须包含：

- 一个基线 arm：**强基线**，不能是故意削弱过的版本；
- 一个处理 arm；
- 至少一个消融 arm：去掉方法中的一个组件，看效果是否随之消失；
- 至少一个负对照：在一个你的方法**不应该**有效的条件下，它确实没有效果（作为 arm，或者作为一个 dev 条件）；
- 至少两个 held-out 条件：在 dev 阶段从来不看。

然后提交到 git。lock 之后若要修改，只能另建一个 `-v2` 文件，并写明理由。

### C4 实现与 dev 迭代（第 4–8 天）

在 dev 条件上随便迭代、调参、改代码，这正是 dev 的用途。每次正式运行都用 `lab7.exp.run`，结果保留在 `results/lab7/` 下（不进 git，自己归档）。

### C5 锁定与留出（第 9 天）

```bash
python -m lab7.exp.run <your>.prereg.json --lock --phase heldout
python -m lab7.exp.report results/lab7/<name> --markdown
```

held-out 只跑一次。不管结果是 supported、refuted 还是 inconclusive，都如实写进报告。

### C6 提案与汇报（第 10 天）

- **两页提案**：用课程仓库的 [research-proposal.md](../../mlsys-research-course/templates/research-proposal.md) 模板；
- **10 分钟汇报**：问题（1 页）→ 我测到的缺口（1 张图）→ 方法（1 页）→ 结果，包括 held-out 和负面结果（1–2 页）→ 局限与下一步（1 页）→ 想请老师判断的一件事。

验收清单见 [grade.md](grade.md)。

## 文件

```
lab7/
+-- exp/stats_student.py   TODO(S1-S3)
+-- exp/prereg.py          provided: load, validate, lock (sha256), check before held-out
+-- exp/run.py             provided: run the matrix, interleave arms, keep failures, record provenance
+-- exp/report.py          provided: medians, CIs, ratios, verdicts (markdown for your report)
+-- examples/bucket_overlap.prereg.json
+-- topics.md              candidate projects grounded in lab1-lab6 measurements
+-- design-review.md       one-page design review template
+-- tests/test_stats.py
```
