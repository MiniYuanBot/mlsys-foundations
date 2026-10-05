# 基础练习：长尾调度模拟

运行环境：CPU/标准库。以下命令均在项目根目录执行。

## 实现任务

本章可独立在CPU开展。提供固定输入生成、barrier参考、结果写出；你补heap式FCFS与summary，先手算再编码。Python标准库heapq足够，不需要模拟框架。

固定128任务、t0全到达、等速独立槽位rate64 token/s、K4/8。uniform全128；long-tail以0.8概率取整数64–128、0.2概率取512–1024，seed11/22/33，保存每组输入并对策略复用。random.Random的实现选择记录在脚本；不要每策略重采样。

## 运行与检查

```bash
python -m unittest lab5.tests.Schedulers -v
python -m unittest lab5.tests.Conservation -v
python -m lab5.tail_sim --out results/lab5/run1
python -m tools.plot results/lab5/run1/summary.csv --x K --y makespan --group kind strategy --xlabel Slots --ylabel "Makespan (s)" --title "Simulation: independent equal-speed slots" --out results/lab5/makespan.png
python -m tools.plot results/lab5/run1/summary.csv --x K --y utilization --group kind strategy --xlabel Slots --ylabel Utilization --title "Simulation: independent equal-speed slots" --out results/lab5/utilization.png
```

手算K2/p=[1,4,1,4]：barrier8、FCFS6。任务自身完成时间与barrier释放时间不同；返回finish应为自身完成。堆按(available_time,slot_id)打破并列，输出按原task_id顺序。

必须通过K1总和、等长整批利用率1、makespan下界max(sum(p)/K,max(p))，且不丢任务、无重复/槽位重叠。24条汇总=2分布×2K×3seed×2策略；uniform不同seed相同只是确定性检查。

交付lengths.csv、timeline/raw.csv、24条summary、两张模拟图、[方向表](directions.md)、[报告](report.md)和给导师三个问题。图和正文都注明独立等速槽位假设；不建模真实batch效率/带宽/通信/RL质量，不把模拟相对收益写成GPU/RL训练加速。

增强45–60分钟：仅FCFS每次分配占所选槽位δ秒，K4，δ0/0.01/0.1/0.5，三长尾seed共12条。busy useful与128δ分开，检查总占用≤1；这是教学敏感性分析，不是实测调度开销或公平真实成本模型。
