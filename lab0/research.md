# lab0｜R问题卡：布局转换的摊销边界

固定同一真实模型算子与同精度计算，比较strided直接访问 vs compact一次后由1/2/4个后续算子复用。预先固定开发shape，观察总latency/拷贝时间/额外峰值内存；holdout用非方阵或不同复用次数。单个算子局部变快不等于端到端获益。

阅读链：TVM→Mirage/TileLang。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab0 --tier research --assessment lab0/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

