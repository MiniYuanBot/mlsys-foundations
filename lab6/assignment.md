# 方法练习：Prefill token budget

运行环境：WSL2/CUDA 服务。以下命令均在项目根目录执行。

## 实验任务

复用lab3服务/adapter/metrics、lab4的SLO分析，不另写服务引擎。你在 `plan.build_matrix` 实现12次区组计划，在[预注册](../templates/preregister.md)中自行提出可反驳假设，完成[机制报告](reproduction-report.md)。

参考[课程题面](../../mlsys-research-course/labs/lab6-reproduction.md)与[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)。本章叫现代vLLM上的缩规模机制检验，不能声称重现原论文所有硬件/模型/加速比。

## 冻结矩阵

A budget2048、B budget512；两者chunked prefill都开。long输入默认1024（必须>512）、输出64，沿用lab4 **相同数值** low/high、N、SLO和其他服务配置。seed11先A low/high后B low/high；seed22先B后A；seed33先A后B。重新测A，不能只用上周历史结果作对照。

```bash
python -m unittest lab6.tests -v
python -m lab3.commands --config lab3/service.json --lab lab6 --design results/lab4/design.json --out results/lab6/plan
```

检查commands.txt和矩阵后先保存预注册，再运行。每次换budget：旧请求排空→停止自己启动的旧服务→确认释放→启动相应budget→核对日志→8请求同形状预热→两个rate。生成文件给出两种服务命令和warmup样例，**每次重启把日志输出目录改为新restart编号**。不要同时启动A/B，也不要直接把整个commands.txt交给shell。

## 分析与报告

主指标请求级p95 TPOT；辅助p95 TTFT/输出吞吐/失败率/原SLO。用lab3.adapter逐轮转换、lab4.analyze汇总，自己与run_matrix一对一join并画A/B三轮点、中位数/范围。主图TPOT对rate，权衡图TTFT与输出吞吐。出现失败/实际长度变化时先审计工作量。

不支持假设也可通过。波动大于差异应写证据不足；10%仅是决定追加测量的启发式门槛，不是统计显著性。TPOT是请求内平均间隔，不能证明最坏ITL或每次stall已消除；缺少内部trace时只描述受控参数改变的系统现象。

交付预注册、原文定位/差异表、12份原始结果与A/B启动日志、命令矩阵、分析脚本/图、2–3页等量机制报告，至少两个限制与一个后续最小检验。

增强30–45分钟：查安装版本实际metrics字段，选一个内部观测，A/B各一短诊断轮、记录采样时间/频率/开销。不存在字段就交观测设计，不编造指标，不混入正式数据。
