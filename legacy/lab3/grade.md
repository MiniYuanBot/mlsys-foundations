# lab3 自评分（100 分）

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

依据 [课程 lab3-serving.md](../../mlsys-research-course/labs/lab3-serving.md)；规则见 [全局测试与评分](../docs/grading.md)。不按加速比奖励分数，负结果也可满分。

| ID | 类型 | 分值 | 必过线 | 验收依据 |
|---|---|---:|---:|---|
| metrics | 自动 | 30 | 30 | `lab3.tests.Metrics`：手算合成请求、单位、TPOT 未定义、失败/吞吐分母 |
| adapter | 人工自评 | 20 | 20 | **真实字段适配**：保存安装版本、原字段/单位/数组对齐映射（10）；将 smoke 原 JSON 转成逐请求记录，人工核对成功/失败/实际长度（10）。 |
| capacity | 人工自评 | 35 | 35 | **真实服务基线**：服务日志、帮助、固定模型/tokenizer commit（10）；8 预热/20 smoke/3×120 全成功且实际输出 64（15）；实际输入长度分布、C 三轮值/中位数/范围和重算入口（10）。 |
| analysis | 人工自评 | 15 | 5 | **解释与可复查性**：解释容量代理与在线容量的区别（5）；真实命令、日志、summary（5）；环境限制、来源与周报（5）。 |

自动组内全部检查通过才得该组分数；skip、异常、零测试不算通过。人工项按表内子项累加，证据不齐时不给相应分数。所有必过线达到且总分 ≥80 才输出 `self-assessed-pass`；80–89 基础完成，90–100 证据较完整，<80 待补。自动测试只证明覆盖的行为，不能代替真实实验或口头理解。

从项目根目录运行：

```bash
python -m unittest lab3.tests -v
python -m tools.grade lab3
```

复制 `assessment.example.json` 为本章 `assessment.json`，补分数、逐项核对说明和相对项目根目录的非空证据文件路径，再运行：

```bash
python -m tools.grade lab3 --assessment lab3/assessment.json
```

第一次运行出现未实现失败或环境阻塞是预期；不要改测试来获得分数。机器检查证据存在，但**不能判断论文、日志或图是否真实/合格**，人工分会单独标记。单纯拿本仓库的空报告模板作为 evidence 不符合自评规则。报告写入 `results/grades/lab3.json`，包含每组测试详细日志。

选做增强/OJ 状态单独记录在报告中，不加入主线 100 分，也不成为下一章的隐含门槛。

