# lab7 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

继承lab6全部环境、lab4的design/SLO；本章只改变目标输入长度，不重新标定有利rate、不升级框架。

```bash
python -m tools.doctor --require vllm --out results/environment/lab7.json
python -m unittest lab6.tests lab7.tests -v
```

默认输入1536+输出64≤context2048；降级context1536时输入1408，context1024时lab6应已768、本章896。design已含long_input与heldout_input，核对实际token长度，不仅看命令。无法满足“比上周长且不超限”时先改写问题并记录原因。

每次重启服务以本章新长度8请求预热。沿用A/B区组、request-rate、N、SLO和prefix/eager等控制；过载保留为边界数据。任何降低rate的后续实验另存目录并两方案一起改。

文献、提案与汇报可在任意设备编辑；结果图从原始数据生成。没有真实留出测量时明确标未完成，不以合成图或推测替代。

