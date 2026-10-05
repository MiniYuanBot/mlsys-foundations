# lab3｜R问题卡：共享前缀的容量、回收与公平性

从真实引擎调用链定位一个可测问题，设计cache容量或回收规则变体。固定精确token前缀、到达trace、cold/warm协议；报告reuse、实际KV压力、TTFT/吞吐/失败，不能将教学页表模拟当真实kernel速度。查现有block reuse/eviction逻辑后再声称差异。

阅读链：PagedAttention→SGLang/FlashInfer。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab3 --tier research --assessment lab3/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

