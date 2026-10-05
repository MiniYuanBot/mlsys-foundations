# 基础练习：负载、延迟与 SLO

运行环境：CPU 分析 + WSL2/CUDA 服务。以下命令均在项目根目录执行。

## 实现任务

先有通过的lab3真实基线及calibration.json。复用其adapter/metrics/服务配置；补 `analyze.slo_metrics` 与 `plan.build_matrix`。测试能检查分母与矩阵，真实SLO仍需请求级明细。

## 先冻结，再测量

从真实C计算low=0.3C/high=0.7C，不能把示例C=4当测量。默认每轮N120；pilot太短时可统一N=max(120,ceil(20×high))，先上限600；若整组>100分钟可整组N60并标p95很粗。N在正式实验前固定，后续两章沿用。

```bash
python -m unittest lab4.tests -v
python -m lab4.plan --calibration results/lab3/calibration.json --num-prompts 120 --out results/lab4/design.json
python -m lab3.commands --config lab3/service.json --lab lab4 --design results/lab4/design.json --pilot --out results/lab4/pilot-plan
```

如服务配置已降上下文，创建design时加匹配的`--max-model-len 1536`或1024；long/heldout长度会按课程适配，并与service.json核对。

先预热，再执行pilot-plan中的长输入、low、30请求。用lab3.adapter保存其normalized.json，然后：

```bash
python -m lab4.analyze results/lab4/pilot-plan/runs/lab4-pilot/normalized.json --freeze-slo --out results/lab4/slo.json
python -m lab3.commands --config lab3/service.json --lab lab4 --design results/lab4/design.json --out results/lab4/plan
```

将预注册、design、slo、run_matrix一起保存/提交，再逐行跑plan命令。每seed内短low/长high/短high/长low，下个seed逆序。固定128与long两长度、64输出、burstiness1、有限rate、不开客户端max-concurrency，共12次；同seed不自动意味着实际到达时间完全相同。

## 分析

每轮先保留失败/实际长度/时段，再转换成normalized。下面是**单轮**示例，把生成矩阵中的run_id代入；完整汇总给analyze传入全部12个normalized文件：

```bash
python -m lab4.analyze results/lab4/pilot-plan/runs/lab4-pilot/normalized.json --slo results/lab4/slo.json --out results/lab4/pilot-summary.csv
```

pilot-summary只能调试，不能充当正式12轮。你需要写本章分析脚本，将summary按run_id与run_matrix做一对一join，断言无缺失/重复，然后作两图：p95 TTFT按长度与rate分组，输出吞吐和SLO分别用明确纵轴。每轮点和范围都保留。

联合满足率=成功且TTFT/TPOT都满足的数/N；goodput=满足数/duration。没有TPOT的请求不满足联合SLO；失败仍在分母。阈值不能看完正式结果后放宽。更高rate不保证更高成功吞吐。

交付预注册、design/run_matrix/SLO、12次raw/normalized/命令、分析脚本/图、600–900字[报告](report.md)。增强30–45分钟：输出正确分母N与错误分母成功数的对照，零失败两者相同也应记录。
