---
title: 贯穿课程的实验规范
tags: [mlsys, methodology]
---

# 贯穿课程的实验规范

[返回大纲](../syllabus.md)

本页是跨 lab 的通用规范。各 lab 的目录结构、工具命令和归档要求见 `labs/` 中的 [实验记录](../../labs/docs/experiment-records.md)。

## 每次实验保留什么

课程笔记放在 `course/`，实验代码放在 `labs/`。可以把仓库根目录作为 Obsidian vault 打开。每次运行的目录至少包含 `config.json`（全部参数）、`command.txt`、`environment.txt`、`stdout.log`、`raw.json` 或 `raw.csv`、`summary.csv`。图从原始数据生成，不能手改数值。文件名包含实验、配置和重复编号，例如 `lab4-long-high-r2`。

配置须包括：日期、代码 commit、设备、OS/WSL、驱动、编译器与编译参数、Python/PyTorch/Triton/Rust 版本、CUDA 版本、模型与 tokenizer revision、dtype、输入/输出长度、seed、预热次数、重复编号、到达率、并发与批 token 限制、缓存设置。未用到的字段写 `not applicable`，不留让人猜测的空白。

## 先正确，再计时

张量实验先与简单参考实现比较。报告最大绝对误差以及相对误差；接近零时相对误差可能夸大。FP32 的教学检查起点可用 `rtol=1e-4, atol=1e-5`，必要时依据算子误差单独解释，不能为了通过不断放宽。FP16/BF16 另设合理容差，不混成同一正确性标准。

训练实验关注 loss、梯度、数据划分和保存加载。推理引擎实验关注成功数、失败数、实际生成长度和超时；“程序没有崩溃”不是所有实验都正确的充分条件。lab4、lab5 另有与 Hugging Face 参考 logits 对齐的检查，先通过它再谈速度。

## GPU 计时与显存

CUDA 异步执行。只在调用前后读时钟，往往只量到提交工作。课程用同步后的整段 wall time 或同一 CUDA stream 上的 CUDA events；报告具体采用哪种。下面是 Python 侧的计时片段，先定义待测无参函数 `fn`，例如 `fn = lambda: attention(q, k, v)`，并提前创建输入；它不是可以独立运行的完整脚本。参考 [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) 与 [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

```python
import time
import torch
for _ in range(20):
    out = fn()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    out = fn()
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - t0) * 1000 / 100
```

C++/CUDA/Rust 的 benchmark 同理：预热之后同步，再测一段固定次数的循环。输入分配在计时段外；若函数本身创建输出，保留这个成本并对所有实现一致。首次编译与稳定运行分开记录；不要把参考实现未预热、优化实现已预热的结果放在一起。独立做三轮，报告各轮数值、其中位数和范围。100 次内部迭代用来摊薄计时开销，不是 100 次独立实验。

测显存前同步，记下输入已驻留时的基线，再重置峰值统计。allocator 缓存、框架常驻内存与算子临时张量不同；尽量在新进程比较不同实现。自己写的引擎要分别记录权重、KV cache 和临时缓冲的显存，`nvidia-smi` 的总量不能当作活跃 KV token 的用量。

## 服务指标定义

lab4、lab5 的 benchmark 自己产生这些指标。令一个成功请求发送时间为 $t_0$，收到第一个输出 token 的时间为 $t_1$，收到最后一个输出 token 的时间为 $t_n$，实际输出 token 数为 $n$。

- TTFT：$t_1-t_0$，包含排队与 prefill，不等于纯 GPU prefill 时间。
- 端到端完成延迟：完成时刻减发送时刻。
- TPOT：常用请求级平均 $(t_n-t_1)/(n-1)$，仅在 $n>1$ 时定义。
- ITL：相邻输出 token 的时间间隔分布。
- 请求吞吐：成功完成请求数 / 测量时段长度。输出吞吐：实际完成的输出 token 数 / 相同测量时段长度。必须写清是否含排空阶段。
- SLO 满足率：本课程以**所有发出的测量请求数**作分母，满足预注册延迟目标且成功的请求作分子；失败不算满足。goodput 则是满足目标的请求数 / 测量时段长度。

不能把各请求 throughput 的算术平均当系统总吞吐。换一个工具或改了统计口径时，先读字段与单位，不直接把“总 token 吞吐”和你定义的“输出吞吐”互换。

## 到达过程和统计口径

一批请求尽快发出可用于压力和容量初测，不代表真实在线到达。有限请求率搭配 Poisson 到达用于主实验；额外限制并发会改变到达过程，要把客户端排队计入或单独报告，不能仍声称是同一个开放负载。

每种条件至少三轮，固定比较条件的 seed 集合，顺序交错。课程中每轮几十个请求只是探索性预算；p95 仅受少量尾部样本影响，需展示轮间波动。正式论文要按观察到的方差、持续时间与研究目标增加样本与重复，不能把本课样本量当作统计充分性的保证。课程不要求报告 p99。

结果太短、启动/排空占比很高或噪声很大时，先标注局限。不要将多轮拼成一个大样本来掩盖环境漂移。误差条至少注明是轮间范围；如果未计算置信区间就不要写“95% 置信区间”。lab7 提供 bootstrap 置信区间的实现，用它之前先读清楚它算的是什么。

## 在线评测与 AI 辅助实现

使用在线 OJ 时，额外保存题目 URL/版本、完整输入输出契约、提交 ID、代码版本、语言、硬件、计时范围与评分定义。平台若未提供逐形状耗时或重复测量，就不能把榜单总分写成独立性能证据；不同设备的 baseline 与优化版分别比较。完整记录要求见 [在线 OJ](online-judge.md)。

AI 辅助实现需要独立参考、至少一个新条件和一次解释/修改检查。避免参考代码与待测代码共用同一个可疑索引逻辑；合成数据验证分析器时明确标为 synthetic，不放入真实性能结果。

## 每周结论的四句话

1. 在什么硬件、软件和负载下，我观察到了什么？给出图或表的编号。
2. 哪个机制可能解释它？证据能排除哪些替代解释？
3. 哪些结论当前不能推出？例如单卡结果不能证明网络优化。
4. 下一次只改变哪个变量，就能区分剩下的解释？

课内可以用“超过 10% 且大于典型轮间波动”作为是否值得追加实验的启发式门槛，不能用它宣称统计显著或创新。负结果同样保留。模型缩小、工作负载合成、模拟环境均要写在图注中。

## 系统实现与研究证据

原生实验额外记录编译器与 flags、目标架构、链接库、源码 SHA、线程设置，以及传输、分配、packing 的计时范围。CUDA kernel 先检查尾块、输入不变性、边界与同步，再做 sanitizer 和资源分析；精度模式相同才能直接比较速度。提供的参考实现跑通只能证明入口可用，不能替代学生实现的正确性。

分布式实验使用真实进程，先证明梯度与单进程 oracle 等价，再测通信；异步 API 返回不能作为通信完成或重叠的证据。CPU host 性能与 GPU 互连性能分开报告。阅读或修改上游系统时，保留可应用的 patch、真实的源码路径和调用链，诊断开关要做关闭与开启的对照，正式计时关闭重型观测。

研究实验在结果前冻结主张、基线、矩阵、指标、消融、负对照和留出条件。报告最接近的三项工作、实际修改、成本与反例；没有加速不影响结论的价值，缺少正确性与公平对照则不行。重建图表的命令只读归档的 raw 数据，昂贵的硬件复跑另设显式入口。细则见 [课程主线](../syllabus.md) 与 [上游使用记录](../../labs/templates/upstream-record.md)。

## 参考资料

- [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [labs 实验记录规范](../../labs/docs/experiment-records.md)
- [实验记录模板](../templates/experiment-record.md)
