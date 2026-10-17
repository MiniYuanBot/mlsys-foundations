# 实验记录与结果归档

依据 [课程实验规范](../../course/guides/experiments.md)。主实验先做正确性，再计时；保持工作量、设备、dtype、预热与计时边界相同。

## 每次运行

```text
results/lab1/explicit-r1/
  config.json
  command.txt
  argv.json
  environment.json
  environment.txt
  stdout.log
  raw.csv (or raw.json)
  summary.csv
  correctness.json / plots / other evidence
```

lab7 的 `exp.run` 会自动建立这样的目录；目录已存在就拒绝运行，重跑时另取目录名。其他命令可以通过 capture 运行：

```bash
python -m tools.capture --out results/your-new-run --config path/to/config.json -- your-command arguments
```

capture 保存命令、环境、输出和退出码；**不会从 stdout 猜造 raw/summary**。各 lab 的 bench 程序直接输出 CSV，重定向到 `raw.csv` 即可；需要汇总时自己核对并转换。`argv.json`/`executed_argv.json` 是精确参数数组，`command.txt` 为 POSIX 展示形式。

同时记录：设备型号、台式或笔记本、电源状态、OS/WSL/驱动、编译器与编译参数、Python/torch/Triton/Rust/CUDA 版本、commit 与是否 dirty、dtype、长度、seed、重复、预热、rate、并发与 token 预算、缓存设置、模型与 tokenizer commit。不适用写 not applicable；脚手架无法自动判断的硬件状态在报告补齐。没有 Git 时记录 unversioned，并保留源码快照；正式实验前建议首次 Git 提交。

环境信息用 `bash tools/check_env.sh --lab N` 和 `python -m tools.doctor` 导出，保存到 `results/environment/`。

## Kernel 测量

各 lab 的 bench 程序自带计时。lab0、lab1 的 GEMM bench 计时前会先和参考实现核对结果；其他 bench 不做检查，计时前先确认对应的测试已经通过。lab0、lab1 的 C++/CUDA bench 每个尺寸重复多轮，每轮输出一行；lab2 的 Python bench 预热 5 次，再用 CUDA event 计时 20 次，取中位数。改计时参数时，所有实现都要一起改。

一轮里的多次内部调用不算多个独立实验。怀疑有热漂移时，换一个新进程复测，并在报告里注明。profiler 单独运行，不和计时混在一起。SDPA 的自动后端要从 trace 里核实，不能直接称为 FlashAttention。CPU 和 CUDA 的数据不要画在同一条曲线上。

## 引擎与调度数据

lab4、lab5 的 `bin/bench` 每次运行输出一行汇总：吞吐、TTFT/TPOT/端到端延迟的 p50 和 p99，以及调度统计。这些是合成负载上的结果，报告里要写明负载参数（请求数、prompt 和输出长度、`--rate`）。TTFT 不等于纯 prefill 的 GPU 时间，TPOT 也不等于最坏情况下的 token 间隔。

`--rate 0` 会在开始时一次性发出全部请求，适合测容量；对比调度策略时用有限的到达率。同一时刻只跑一个 benchmark。换 bench 参数时同步更新 `config.json`，不要把不同 token 预算的轮次混在一张表里。失败的运行和重跑的目录也要保留。

## 绘图和归档

`tools.plot` 提供基础散点图，保留各轮数据点；在章节分析中补充分组中位数/范围、合适的标签，并按需关联指标数据。没有计算置信区间，就不要将误差线标为置信区间。

`results/` 被 Git 忽略是为了保护缓存和大原始文件，不是免交证据。每章完成后把选定小型原始记录、汇总、图和报告复制到你建立的 `evidence/labN/` 并提交；大文件另存归档介质，在 [归档清单](../portfolio/artifacts.md) 记录可访问位置、文件清单、SHA256 和重建命令。不要提交模型、环境、token 或私密请求。用 Python `hashlib` 或系统校验工具计算归档哈希。失败数据也要保留。

每周只问四句：观察到什么；可能机制是什么；证据不能说明什么；下一次只加哪项检查。来源与 AI 辅助用三行记录，不需要保存全部对话。
