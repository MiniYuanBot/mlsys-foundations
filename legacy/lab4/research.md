# lab4｜R问题卡：缓存或长度感知调度的SLO边界

选择prefix污染或长度偏斜之一，固定强baseline/公平性口径，做一个policy patch。除goodput还记录短长请求分组延迟、饥饿/失败、客户端send-lag；低复用负对照与新负载留出必须保留。不能同时换多个开关后归因于一个机制。

阅读链：Sarathi-Serve/SGLang→DistServe/Mooncake。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab4 --tier research --assessment lab4/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

