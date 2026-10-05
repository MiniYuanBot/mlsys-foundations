# 系统核心自评分：100 分

本表评价 lab3 的 runtime 部分，与旧的服务数据适配测试分别记录。不能因为旧测试通过就把本节标成完成；同样，本节 CPU 测试通过也不能替代真实引擎集成。没有统一硬件速度阈值。

| 项目 | 分值 | 可审查证据 |
|---|---:|---|
| 分配、追加、释放、内存守恒 | 15 | 对应契约测试与逐事件快照 |
| 共享引用、部分尾页 COW、完整尾页不复制 | 15 | 对应测试；手绘页表；两条请求内容隔离 |
| 原子失败、空输入、重复/不存在 ID | 15 | 错误前后 snapshot 完全相等；不修改测试期望 |
| 连续预留与分页资源对照 | 15 | 三条同容量 trace、页大小敏感性、准确区分逻辑/物理占用 |
| 实际 vLLM 调用链与教学模型差异 | 10 | 固定 SHA、路径/函数名、两条完整生命周期图 |
| 实际 vLLM 小修改及回归验证 | 20 | patch、开关行为、真实导入路径、测试与三类 workload 日志；仅有 diff 此项最多 5 分 |
| 论文机制与研究实验规范 | 10 | PagedAttention/SGLang 阅读卡、可证伪问题、控制变量、原始证据与局限 |

**完成门槛：≥80 分，且全部 `TestPagedKVContract` 通过，真实引擎 patch 已执行验证。** 关键项失败时标为「未完成」，不能用阅读或拓展加分补掉。环境阻塞不是通过，给已完成证据的分数并明确缺项。

运行：

```bash
python -m unittest lab3.runtime.tests -v
python -m lab3.runtime.run
```

初始预计 2 个基础设施测试通过、10 个学生契约测试因 TODO 报错；总命令非零退出。不允许把 `NotImplementedError` 改成 skip 或吞掉异常来得到绿色结果。检查 runner 逐步状态正确后再保存 JSONL；runner 成功本身不能代替测试。

拓展单独评级：**E0** 仅阅读；**E1** 有预注册、实现及正确性；**E2** 有公平对照和真实实验；**E3** 有留出条件、负结果分析及可重放产物。E0–E3 不叠加到基础 100 分，避免硬件资源替代核心能力。

建议在 lab3 报告增加：`runtime_score`、`contract_test_status`、`engine_patch_status`、`evidence_kind`、`extension_level`。本表已接入 `python -m tools.grade lab3 --tier systems`；默认base仍只评服务基础练习。真实引擎patch必过项由你按证据核验。
