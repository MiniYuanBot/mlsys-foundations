# lab5｜R问题卡：通信重叠或rollout异步的成立条件

从真实Gloo/MPI语义开始，选择一个bucket/消息大小实验；有≥2GPU权限再用真实NCCL/训练trace检验overlap，不能用CPU延迟预测GPU扩展。若选RL，先读HybridFlow/AReaL/DARTS，明确policy staleness/采样分布/同等质量约束；没有真实日志和质量验证则仅写可测的局部问题。

阅读链：ZeRO→FLUX/Comet；HybridFlow→AReaL/DARTS。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab5 --tier research --assessment lab5/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

