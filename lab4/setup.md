# lab4 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

前置lab3真实容量基线全成功、固定64输出、实际输入核对及C可重算。沿用同一 `.venv-vllm`、模型snapshot、设备、服务配置，不安装新框架。

```bash
python -m tools.doctor --require vllm --out results/environment/lab4.json
python -m unittest lab3.tests lab4.tests -v
```

CPU标准库足以开发SLO与矩阵；matplotlib用于作图。真实12轮需要GPU服务。保留lab3/service.json和calibration.json，按 [任务说明](assignment.md) 生成design.json；核对数值rate、统一N和上下文。

pilot先预热；length/output异常或任何失败时先修复，不冻结无效SLO。检查bench支持burstiness1和有限rate；不开客户端max-concurrency。prefix caching继续关闭，不能混入未声明的缓存收益。

若环境/模型/容量配置改变，应重做lab3标定并重新预注册整组实验，不能悄悄沿用旧C。没有请求级明细时先补采集，不能用汇总均值计算满足率。

