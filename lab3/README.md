# lab3：推理服务与 KV 缓存

建立可核对的推理服务基线，再实现分页 KV 缓存模型，并修改真实服务引擎中的观测代码。

## 内容

- [assignment.md](assignment.md)：启动服务、适配请求数据与测量容量。
- `commands.py`、`service.example.json`：服务配置与命令生成。
- `adapter.py`、`metrics.py`、`calibrate.py`、[schema.md](schema.md)：数据转换、指标与容量校准。
- [runtime/](runtime/README.md)：KV 页表、引用计数、写时复制（COW）与 vLLM 源码任务。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：缓存管理拓展。
- [setup.md](setup.md)、[grade.md](grade.md)、[report.md](report.md)：环境、基础评分与报告。

## 使用顺序

1. 用合成数据检查适配器与指标，确认每个字段的单位和含义。
2. 启动服务，核对小规模请求，再测量容量基线。
3. 完成 `runtime/` 的缓存实现与测试，随后在固定的 vLLM 源码上添加诊断。

命令生成器只写出文本，需按说明逐条运行。CPU 缓存模型用于检查机制，服务容量与延迟必须来自真实运行。
