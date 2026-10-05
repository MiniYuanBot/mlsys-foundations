# lab6：研究实验

选择一个系统问题，完成最小代码改动，用基线、消融与负对照检验自己的解释。

## 内容

- [research.md](research.md)：内核与编译器、推理服务、分布式训练三条研究路线。
- [artifact-plan.json](artifact-plan.json)：主张、配置、实验矩阵与预算记录。
- [reproduction-report.md](reproduction-report.md)：研究报告；[systems.md](systems.md)：系统任务的验收要求。
- [assignment.md](assignment.md)、`plan.py`、`tests.py`：prefill token budget 方法练习。
- [setup.md](setup.md)、[grade.md](grade.md)：方法练习的环境与评分。

## 使用顺序

1. 从 `research.md` 选择一条路线，确定论文、上游代码和可检验的问题。
2. 填写实验计划，先验证基线，再实施代码改动。
3. 完成正确性、消融和负对照，整理原始结果与报告。
4. 将未参与配置选择的条件留给 [lab7](../lab7/README.md)。

只需深入完成一条研究路线。Token budget 是方法练习；所选项目的环境、命令和验收以 `research.md`、`systems.md` 为准。
