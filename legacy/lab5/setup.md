# lab5 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

Python3.10+标准库即可运行调度器/测试/评分，不需要PyTorch、CUDA、vLLM或RL框架。作图才需要matplotlib。

可直接用已有合适环境；希望独立时自行创建：
```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python -m tools.doctor --require none --out results/environment/lab5.json
python -m unittest lab5.tests -v
```

Windows创建/激活语法见根setup；若不作图无需安装任何包。初始NotImplementedError是待补FCFS/summary，不是环境报错。

使用给定seed11/22/33一次生成并保存输入，对两策略重放。rate64是教学单位转换，不来自GPU。报告Python版本，数据生成使用独立random.Random，不受其他随机调用干扰。图注必须标明模拟；不把本章结果用来完成真实GPU服务验收。

