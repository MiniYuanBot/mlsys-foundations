# S02验收

系统层100分：正确性30；streaming实现/资源推导25；强基线、显存和tile消融25；留出与编译产物解释20。按lab2/systems.rubric.json执行。全局NxN中间张量、缺非整齐边界、把TF32/FP16变化混成同精度提升，均不能通过关键项。

FP32朴素教学kernel不胜SDPA不扣性能分；没有实际测量、没有误差/内存证据会扣分。R层阅读FA3/TileLang/Mirage并不等于在对应硬件完成artifact。

