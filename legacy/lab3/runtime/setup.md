# 本节环境由你配置

## CPU 状态机

Python 3.10+ 标准库即可，不依赖 PyTorch、pytest、模型或 GPU。从 labs 仓库根目录运行：

```bash
python --version
python -m unittest lab3.runtime.tests.TestProvidedInfrastructure -v
python -m lab3.runtime.run --baseline-only
python -m unittest lab3.runtime.tests -v
```

最后一条在未完成学生 TODO 时应非零退出。所有数据只有少量整数，测试通常秒级完成；它们不测显存与速度。

## 引擎源码实践

先遵循 [lab3 setup](../setup.md) 与[全局 setup](../../setup.md)，在 WSL2/Linux 中单独维护 vLLM 环境，避免覆盖 kernel 学习环境。教学阅读采用 v0.10.2；纯 Python 改动可参考该版本官方 [Python-only build / GPU installation](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html)。读完安装要求后，由你选择匹配的 wheel、Python、CUDA 与驱动，不把最新 wheel 与旧源码混装。

以下是你手动执行的源码获取步骤，本项目不会自动下载或安装：

```bash
mkdir -p external
git clone --depth 1 --branch v0.10.2 https://github.com/vllm-project/vllm.git external/vllm-lab3
git -C external/vllm-lab3 rev-parse HEAD
```

确认 labs 根目录 `.gitignore` 已忽略 `external/` 后再操作；外部项目本身的许可证和 Git 历史保留在它的目录中。若选不同版本，记录真实 SHA 并更新报告的源码路径。更换版本后重新检查 tests、server CLI 与事件结构，不假定主分支的 API 始终存在。

安装完成后，确认导入的代码就是你修改的 checkout：

```bash
python -c "import inspect; from vllm.v1.core.block_pool import BlockPool; print(inspect.getfile(BlockPool))"
```

调用链与 patch 编写可以离线完成；执行测试需要完整的引擎依赖。有些上游单元测试不启动 GPU，但仍可能 import GPU 构建相关模块，不能据此承诺纯 CPU 环境可运行整个 vLLM 测试集。选你修改处所在版本的实际测试，报告收集到的测试名称和 skip 原因。

## RTX 4060 8GB 的预算

真实引擎从已有的小模型（约 0.5B–1.5B）、context 512–1024、并发 1–4 开始，先做单请求 smoke。FP16 权重粗算约 `2 × 参数量` 字节，但这不是完整显存占用。实际还需 KV、工作区和 graph 等空间；这些配置是起点，不是可运行保证。先根据模型配置计算 KV 预算，再看实际启动日志和峰值显存。避免同时启动两个服务占满 8GB。

资源受限时缩短输入与输出、减少并发，在保持对照相同的条件下完成三种 workload；若仍受阻，保留日志，完成 CPU 语义和源码阅读，将真实执行项标为未完成。大模型、多 GPU 与 KV 跨机传输留给选修远端实验，并单列硬件和费用预算；本节不要求你申请付费资源。

无需为本状态机安装任何软件。仓库作者只验证了标准库脚手架与初始 TODO 的失败行为；真实引擎安装、GPU 行为及 patch 留给实验执行时验证。
