# lab6 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

沿用lab3 vLLM环境、模型/分词器snapshot，以及lab4冻结的design.json和slo.json。无需安装新依赖或服务引擎；不要覆盖torch环境。

```bash
python -m tools.doctor --require vllm --out results/environment/lab6.json
python -m unittest lab6.tests -v
```

按安装版本帮助核对max-num-batched-tokens支持2048与512，二者chunked prefill都开启；context与长输入>512须合法。若某设置不允许，记录版本/日志，先查文档和改写实验，不静默换机制。

同一时刻只运行一个服务；每次A/B切换旧请求排空后停止自己启动的服务、确认显存，再启动/核对新配置，8请求预热。分别保存A/B启动日志和每次restart编号。生成器的两条server样例不表示它会自动安全切换服务。

重新测当周A，保持rate/N/SLO/设备/电源/权重不变。如果环境必须升级或降级，整组重测并说明与前章不可直接配对的差异。无GPU时可完成预注册/矩阵/报告结构，真实结果仍待补。

