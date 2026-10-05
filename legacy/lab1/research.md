# lab1｜R问题卡：小shape GEMM的瓶颈转换

从cuBLAS和自写同精度kernel出发，固定M/N/K开发网格，单独改变tile/循环布局；以ptxas资源、有效带宽/计算量和trace区分launch/访存/occupancy解释。留出skinny与不整齐shape，不要求胜cuBLAS。加入FP16 Tensor Core必须另声明精度契约与对照。

阅读链：FlashAttention-2→FA3/ThunderKittens。原文链接、必读部分、硬件边界见 [论文路线](../../mlsys-research-course/readings/papers.md)，上游lab与源码见 [复用清单](../../mlsys-research-course/readings/open-source-labs.md)。先核实最接近工作覆盖了什么，再把假设写入 [预注册](../templates/preregister.md)。

这是一条可选择的研究路线，额外约12–24h，推荐放进L06/L07的固定研究时段，不叠加到每章S层预算。交付：原文主张→强baseline→一个改动→主矩阵三轮→消融/负对照→留出→代价/限制→可重建artifact。符号模拟、CPU语义、单GPU与多GPU四种证据分别标记。

R层100分见research.rubric.json，复制research.assessment.example.json填证据后运行 `python -m tools.grade lab1 --tier research --assessment lab1/research.assessment.json`。只读论文或写计划不能标R完成。最终主线以L06选定研究项目与L07结课包验收。

