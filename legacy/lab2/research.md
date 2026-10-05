# lab2｜R问题卡：attention分块选择对陌生形状是否稳定

从S02固定精度/causal契约出发，比较显式、SDPA、streaming强基线，选择一个tile或work partition变体；指标含误差/峰值内存/稳定延迟/编译成本。留出S383/1537或另一个预先未参与选择的shape。TileLang/Mirage对照是可选，不同时装所有compiler。

阅读链：FlashAttention/FA2→TileLang/Mirage/FlashInfer。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab2 --tier research --assessment lab2/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

