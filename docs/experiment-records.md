# 实验记录与结果归档

依据 [课程实验规范](../../mlsys-research-course/guides/experiments.md)。主实验先做正确性，再计时；保持工作量、设备、dtype、预热与计时边界相同。

## 每次运行

```text
results/lab1/explicit-r1/
  config.json
  command.txt
  argv.json
  environment.json
  environment.txt
  stdout.log
  raw.csv (或 raw.json)
  summary.csv
  correctness.json / 图 / 其他证据
```

训练、张量 benchmark、模拟 runner 自动记录这些元信息和相应结果；目录已存在就拒绝运行，重跑另起名称。外部命令可用：

```bash
python -m tools.capture --out results/your-new-run --config path/to/config.json -- your-command arguments
```

capture 保存命令、环境、输出和退出码；**不会从 stdout 猜造 raw/summary**，真实服务 raw 由 vLLM 输出，normalized/summary 由你核对并转换。`argv.json`/`executed_argv.json` 是精确参数数组，`command.txt` 为 POSIX 展示形式。服务启动日志保存在独立 servers 目录，不能把启动耗时混进 benchmark。

同时记录：设备型号/桌面或笔记本/电源、OS/WSL/驱动、Python/torch/Triton/vLLM、commit 与是否 dirty、dtype、长度、seed、重复、预热、rate、并发/budget、缓存/eager、模型与 tokenizer commit。不适用写 not applicable；脚手架无法自动判断的硬件状态在报告补齐。没有 Git 时记录 unversioned，并保留源码快照；正式实验前建议首次 Git 提交。

## 张量测量

提供的计时器在输入创建后进行首调用（单列）、20 次预热、每轮 100 次 forward、前后 CUDA 同步，保留三轮。每个实现用单独命令/进程运行。同一 shape/seed 的张量生成顺序一致。显存独立采样：同步、输入驻留 baseline、reset peak、调用并保留输出、同步读取 absolute peak/increment/reserved。

100 次内部调用不是 100 个独立实验。三轮在同一进程中按轮测量，独立保留；若怀疑热漂移可增加新进程复测并注明。profiler 单独执行。SDPA 自动后端需从 trace 核实，不能直接称其为 FlashAttention。CPU 与 CUDA 不混成同一条曲线。

## 服务数据

原 JSON 原样保留。根据安装版本记录字段映射，再转成该 lab 规定的请求格式。成功/失败、实际长度、时段和分母需一一可核对。duration 应注明是否含排空阶段；统计采用已核实的工具口径，TTFT 不等于纯 prefill GPU 时间，TPOT 不等于最坏 ITL。

容量阶段允许 rate=inf，正式矩阵只用有限 rate、burstiness=1；不加客户端 max-concurrency。每次重启服务先等 readiness，再运行 8 个相同形状预热请求。客户端与服务端分终端，同一时刻只有一个服务、一个 benchmark；不要把 commands.txt 整体当 shell 脚本执行。

保存预注册、pilot、SLO、失败与重跑目录。raw 不足以还原请求级 SLO 时标待补，不能以工具平均值冒充。synthetic 文件只用于测试，不作实测证据。

## 绘图和归档

`tools.plot` 提供散点图起点，各轮点都保留；在章节分析中补充分组中位数/范围、合适标签和必要的指标 join。没有计算置信区间就不要这样命名误差线。lab5 标“模拟：独立等速槽位模型”。

`results/` 被 Git 忽略是为了保护缓存/大原始文件，不是免交证据。每章完成后把选定小型原始记录、汇总、图和报告复制到你建立的 `evidence/labN/` 并提交；大文件另存归档介质，在 [归档清单](../portfolio/artifacts.md) 记录可访问位置、文件清单、SHA256 和重建命令。不要提交模型、环境、token 或私密请求。用 Python `hashlib` 或系统校验工具计算归档哈希。失败数据也要保留。

每周只问四句：观察到什么；可能机制是什么；证据不能说明什么；下一次只加哪项检查。来源与 AI 辅助用三行记录，不需要保存全部对话。

