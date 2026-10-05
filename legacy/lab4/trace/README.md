# 请求重放与调度观测

保存可重放的请求负载，比较前缀缓存与调度行为，并用引擎事件解释结果。

主要文件：

- `workload.py`：符号前缀与到达时间生成器。
- `replay.py`：待完成的 tokenizer 适配与请求重放。
- `audit.py`：待完成的发送延迟审计。
- `tests.py`：负载与重放契约测试。

建议顺序：

1. 按 [setup.md](setup.md) 准备引擎和客户端。
2. 阅读 [assignment.md](assignment.md)，完成适配、审计与测试。
3. 固定 token trace，运行缓存对照和调度诊断。
4. 按 [grade.md](grade.md) 整理结果。

生成器中的符号 ID 不是模型 token，也不代表缓存命中。真实延迟和吞吐必须由实际引擎测得，并记录客户端发送延迟。
