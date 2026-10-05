---
title: 贯穿课程的实验规范
tags: [mlsys, methodology]
---

# 贯穿课程的实验规范

[返回大纲](../syllabus.md)

## 每次实验保留什么

课程笔记与实验仓库分开。课程文件可以放 Obsidian；代码另建 Git 仓库。每次运行的目录至少包含 `config.json`（全部参数）、`command.txt`、`environment.txt`、`stdout.log`、`raw.json` 或 `raw.csv`、`summary.csv`。图从原始数据生成，不能手改数值。文件名包含实验、配置和重复编号，例如 `lab4-long-high-r2`。

配置须包括：日期、代码 commit、设备、OS/WSL、驱动、Python、PyTorch/vLLM/Triton 版本、模型/tokenizer revision、dtype、输入/输出长度、seed、预热次数、重复编号、到达率、并发/批 token 限制、缓存设置、是否 eager。未用到的字段写 `not applicable`，不留让人猜测的空白。

## 先正确，再计时

张量实验先与简单参考实现比较。报告最大绝对误差以及是否通过 `torch.testing.assert_close`；接近零时相对误差可能夸大。FP32 的教学检查起点可用 `rtol=1e-4, atol=1e-5`，必要时依据算子误差单独解释，不能为了通过不断放宽。FP16/BF16 另设合理容差，不混成同一正确性标准。

训练实验关注 loss、梯度、数据划分和保存加载。服务实验关注成功数、失败数、实际长度和超时；“HTTP 200”不是所有实验都正确的充分条件。

## GPU 计时与显存

CUDA 异步执行。只在 Python 中对一次函数调用前后读时钟，往往只量到提交工作。参考 [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) 与 [Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。课程用同步后的整段 wall time 或同一 CUDA stream 上的 CUDA events；报告具体采用哪种。下面是计时片段，先定义待测无参函数 `fn`，例如 `fn = lambda: attention(q, k, v)`，并提前创建输入；它不是可以独立运行的完整脚本。

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

输入分配在计时段外；若函数本身创建输出，保留这个成本并对所有实现一致。首次编译与稳定运行分开记录；不要把参考实现未预热、优化实现已预热的结果放在一起。独立做三轮，报告各轮数值、其中位数和范围。100 次内部迭代用来摊薄计时开销，不是 100 次独立实验。

测显存前同步，记下输入已驻留时的基线，再重置峰值统计。分别报告 `memory_allocated`、`memory_reserved` 和 `max_memory_allocated` 的口径。allocator 缓存、框架常驻内存与算子临时张量不同；尽量在新进程比较不同实现。vLLM 显存池预留后的 `nvidia-smi` 数字，不能当作实际活跃 KV token 使用量。

## 服务指标定义

令一个成功请求发送时间为 $t_0$，收到第一个输出 token 的时间为 $t_1$，收到最后一个输出 token 的时间为 $t_n$，实际输出 token 数为 $n$。

- TTFT：$t_1-t_0$，包含请求经过的排队、prefill 与客户端/传输等开销，不等于纯 GPU prefill 时间。
- 端到端完成延迟：以客户端观察到的完成时刻减发送时刻；具体工具是否包含最后协议开销，应查实现。
- TPOT：常用请求级平均 $(t_n-t_1)/(n-1)$，仅在 $n>1$ 时定义；按使用的工具实际时间口径记录。
- ITL：相邻输出 token 的时间间隔分布。流式协议可能一次返回多个 token，客户端 chunk 不是天然的一 token；使用工具的实际计算定义，不自行把 chunk 数当 token 数。
- 请求吞吐：成功完成请求数 / 测量时段长度。输出吞吐：实际完成的输出 token 数 / 相同测量时段长度。必须写清是否含排空阶段。
- SLO 满足率：本课程以**所有发出的测量请求数**作分母，满足预注册延迟目标且成功的请求作分子；失败不算满足。goodput 则是满足目标的请求数 / 测量时段长度。

[vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) 提供多种汇总与明细。保存 JSON 后先读字段与单位，不直接把工具的“总 token 吞吐”和你定义的“输出吞吐”互换。不能把各请求 throughput 的算术平均当系统总吞吐。

## 到达过程和统计口径

`request-rate=inf` 是一批请求尽快发出，可用于压力/容量初测，不代表真实在线到达。有限请求率搭配 Poisson 到达用于本课程主实验；客户端额外限并发会改变到达过程。若启用了限并发，要把客户端排队计入或单独报告，不能仍声称是同一个开放负载。

每种条件至少三轮，固定比较条件的 seed 集合，顺序交错。课程中每轮 120 个请求只是探索性预算；p95 仅受少量尾部样本影响，需展示轮间波动。正式论文要按观察到的方差、持续时间与研究目标增加样本/重复，不能把本课样本量当作统计充分性的保证。课程不要求报告 p99。

结果太短、启动/排空占比很高或噪声很大时，先标注局限。不要将多轮拼成一个大样本来掩盖环境漂移。误差条至少注明是轮间范围；如果未计算置信区间就不要写“95% 置信区间”。

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

原生实验额外记录编译器/flags、目标架构、链接库、源码SHA、线程设置、传输/分配/packing的计时范围。CUDA kernel先检查尾块、输入不变性、边界与同步，再做sanitizer和资源分析；精度模式相同才能直接比较速度。完整参考成功只能证明参考入口可用，不能替代学生实现正确性。

运行时实验保留可应用patch、真实import/二进制路径和调用链。给诊断开关做无patch/off/on对照，正式计时关闭重型观测；保留失败请求、调度决定和send-lag。分布式实验使用真实进程，先证明梯度与单进程oracle等价，再测通信；异步API返回不能作为通信完成或重叠的证据。CPU host性能与GPU互连性能分开报告。

研究实验在结果前冻结主张、基线、矩阵、指标、消融、负对照和留出条件。报告最接近三项工作、实际修改、成本与反例；没有加速不扣结论分，缺正确性与公平对照不能用阅读分弥补。重建图表的命令只读归档raw数据，昂贵的硬件复跑另设显式入口。细则见 [课程主线](../syllabus.md) 与 [上游使用记录](../../mlsys-research-labs/templates/upstream-record.md)。

## 参考资料

- [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [实验记录模板](../templates/experiment-record.md)
