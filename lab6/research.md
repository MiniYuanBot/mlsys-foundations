# lab6：研究项目

B层原budget实验保留作为方法检查；S层要求可应用的系统改动，R层要求完整可证伪实验。推荐跨第12–14周，约21h实作+12h阅读；**只选择一个方向**，不要同周同时实现三个系统。

## 三个可执行起点

| 路线 | 输入/基线 | 最小改动与真实层次 | 研究检验 | 不能声称 |
|---|---|---|---|---|
| kernel/compiler | S02精确attention契约；SDPA/自己可读版本 | CUDA/Triton tile或分工变体；固定精度 | 开发×3轮、单变量消融、未见S/D、编译与显存成本 | 小shape赢一次=新kernel算法 |
| serving runtime | S03/S04固定token trace；未改引擎 | vLLM调度/缓存一处policy patch，或llama.cpp C++ speculation patch | cold/warm、公平性/失败、trace与logging开销负对照 | 参数调优=发明调度算法 |
| distributed/RL | 单机oracle、现有collective/同步pipeline | bucket/overlap/任务分配改动；资源足够才做真实多GPU | 通信量、同步/overlap、同等训练工作/质量、负对照 | CPU双rank=多GPU加速；更短轨迹=同质训练更快 |

默认可本地推进serving分支；想落实第二个C++系统任务，选 [llama.cpp speculation源码](https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/speculative-simple.cpp) 与 [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)。这些是导航路径，执行前固定实际commit与LICENSE，不保证master与旧命令兼容。

## C++ speculation具体方案（可替代vLLM研究项目）

1. 读[经典speculative sampling](https://arxiv.org/abs/2302.01318)与[EAGLE-3](https://arxiv.org/abs/2503.01840)的验证/接受机制。手算一个draft被拒绝的例子，区分精确采样语义与greedy一致性检查。
2. 独立clone固定llama.cpp commit，按其README构建CPU或CUDA；保存转换工具commit、原模型revision、GGUF哈希、量化设置。先用0.5B小模型与上游支持的ngram方案，减少第二模型显存压力。
3. 修改C++中draft候选长度选择或增加开关化诊断，保存可应用diff与上游测试；先原engine baseline再改动。不要把任意自编接受规则当论文等价采样。
4. 开发32个固定提示、greedy/固定seed/output上限64，重复三轮；比较关闭speculation、上游固定配置、你的变体。输入、EOS和实际输出相同或明确按工作量归一。记录总完成时段、目标模型调用、draft耗时、接受/拒绝数、输出token IDs及差异。
5. 一种消融：固定draft长度 vs 你的自适应规则；一种负对照：低复用/低接受率提示；一种留出：新的提示集合/长度。序列输出一致只证明已测greedy条件，不证明所有随机采样分布等价。
6. 正式性能不混入昂贵logging。若只减少target调用但总时延变慢，保存并分析draft开销/同步/小batch。

本仓库不附完整engine答案，任务是对真实上游代码完成最小改动。修改点在所选commit确认后填入 [artifact计划](artifact-plan.json) 和 [upstream记录](../templates/upstream-record.md)。

## 预注册与验收

先写主张、原文具体位置、改变什么、何种结果否定、强baseline和预算。把真实配置与命令注册到run_matrix，每轮不覆盖；三轮不能只保留最好。最少一组消融和一项负对照是必做，不把R变成“多跑几个参数”。

S层先验收patch正确/基线/观测/成本，R层再验收证据链、留出、质量边界与最接近工作。用 `python -m tools.grade lab6 --tier systems` / `--tier research` 查看所需证据。人工分不会自动验证真实数据，需要逐项核对。主报告继续用reproduction-report.md，并把字段改为所选路线，旧budget专用条件不强加给kernel/MPI项目。

结课时S06和R06均需达标，可引用同一项目证据；R06中的留出项在L07结束后补齐。每个方向的扩展阅读可替换papers.md中L06的默认EAGLE-3进阶阅读，基础实验方法阅读仍保留，不把无关论文硬加到kernel或通信项目。
