# lab7：留出、可复查artifact与后续科研计划

原更长输入实验是B层的serving示范；S/R层必须继承L06**实际选择的研究路线**。不强制kernel或distributed项目再跑无关vLLM矩阵。

## 未参与选择的条件

- kernel/compiler：新shape/stride或固定精度下的新批量；不能同时换dtype并仍叫shape外推。
- serving/speculation：新的提示集合、prefix偏斜或输入长度；沿用旧SLO/rate或预先声明新问题；保持目标模型/精度。
- distributed/RL：world_size/消息大小/任务分布；若换硬件记录拓扑与驱动。单机rank增多只说明该主机进程场景，不能冒充跨机扩展。

在看结果前登记条件、预测和放弃规则；三轮、失败/超时/正确性检查照旧。不支持假设或反转是有效结果，不能重调后继续把同一条件称为留出。

## Artifact验收

1. `upstream-record.md`：源码commit、许可证、修改路径、可应用diff、依赖与编译器/硬件。
2. `artifact-plan.json`：主张/原文定位、baseline、指标、主矩阵/消融/负对照/留出、seed与预算。
3. 原始数据清单与SHA256、失败数据、脚本入口、图/表生成命令。fresh checkout从命令重建汇总与图；昂贵GPU实验可作为另一显式入口，不在“绘图”命令暗中重新跑。
4. 10分钟说明：代码真正改了哪里；机制证据是什么；哪种结果无法解释；工作量/质量是否公平；最接近三工作覆盖了多少；下一轮只增加哪项证据。
5. 1200–1800字提案不以“我实现了已有算法”当创新。写清六个两周任务，最危险假设优先，资源申请指向具体缺失证据。

对应 `python -m tools.grade lab7 --tier systems` 和 `--tier research`。结课至少完整完成L06一条R，并在本章完成留出/归档/提案；各章R题不是全做要求。可复查和新颖性分开判断。

