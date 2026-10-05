# 测试与分层自评分

## 层次

1. `python -m unittest discover -s tests -v`：提供的基础设施/目录/评分器自检，不能代表学生已完成。
2. `python -m unittest labN.tests -v`：原B基础练习。
3. S各自入口：CMake/CTest、真实GPU/Gloo/MPI或指定契约测试，见子目录setup。系统层既检查教学实现，也要求真实引擎/硬件证据。
4. R研究证据人工审查：原文主张、强baseline、消融/负对照、留出、原始数据和artifact。

本页评分命令从仓库根目录运行；子目录setup中显式的`cd`按其说明执行。unittest使用标准库；PyTorch/Triton/编译器/MPI按setup由你安装。skip、零测试、异常、TODO不给自动分；GPU/native桥接读取实际构建程序，不mock成通过。真实Gloo测试会启动多个进程，有超时清理。

## 评分命令

```bash
python -m tools.grade lab1 --tier base
python -m tools.grade lab1 --tier systems
python -m tools.grade lab6 --tier research
```

默认base向后兼容，输出results/grades/labN.json；systems/research分别写labN-systems.json、labN-research.json。默认基础分不能替代系统/研究分。lab2x是独立选修，只有原rubric，用默认base入口评价，标题中的base不表示它是主线基础必做。

每层100分，自动组全过才得整组分；人工分按criteria分值范围和grade细则。≥80并达到全部required项minimum才显示self-assessed-pass，退出0；待补1、自动检测到环境阻塞2。人工项无法自动识别机器资源，自己注明blocked和缺少的真实证据。正确不快或负结果不扣性能门槛分。

## 人工证据

复制对应 `assessment.example.json`、`systems.assessment.example.json` 或 `research.assessment.example.json`，填为status=reviewed、真实score、非空evidence文件路径、逐项note。路径相对仓库根目录，不得越界。再运行：

```bash
python -m tools.grade lab1 --tier systems --assessment lab1/systems.assessment.json
```

评分器检查分值与文件存在，但**不能验证内容真实性、代码独立性或实验质量**。空模板、CPU结果、synthetic或上游原始成绩不得冒充自己的GPU/patch实验；人工分会单列。缺核心实现不能靠读论文补分。

## 结课门槛

S00–S05完成；L06至少一条R完整证据链；L07完成路线一致的留出、可重建artifact、相关工作与提案。B先修缺项先补。各章R题是方向选择，不要求每条都做；分数不相加平均，避免用容易的练习掩盖关键能力缺项。

机器记录对应：`lab0..lab7 --tier systems`均达标，另加`lab6 --tier research`达标。S06和R06评同一项目的实现与研究证据，可复用同一份raw/报告；R06的留出分在L07结束后补齐，不要求第14周提前完成。其余research成绩用于选修自评，不重复增加结课任务。

每章口头/修改检查10分钟：解释契约、布局/精度/同步/统计，预测新条件再验证。可以扩展测试，不能删失败条件、放宽容差或把NotImplementedError改skip获取通过。
