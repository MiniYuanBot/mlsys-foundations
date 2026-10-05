# 阅读、开源实验与两周研究问题

链接于 2026-10-05 核验。这里是机制复现路线；论文中的完整集群规模和最终加速比不作为单卡作业承诺。本目录教学代码原创，未复制下面工程的实现。复用代码时固定实际 commit，保留 LICENSE/NOTICE 和修改说明。

| 阅读 / 源码 | 学什么 | 本实验连接与资源边界 |
|---|---|---|
| [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)、[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) | process group、collective、梯度 ready 与 DDP | 把手写同步与 DDP 的 bucket 时机对应；CPU 真实 Gloo 起步 |
| [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems)；[tests/test_ddp.py](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_ddp.py)、[tests/test_sharded_optimizer.py](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_sharded_optimizer.py) | 从 correctness adapter 到 distributed optimizer | 主线完成后独立做 DDP/optimizer sharding 部分；MIT；核验时 main 为 Spring 2026，使用2025材料应固定对应历史版本 |
| [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)；[src/all_reduce.cu](https://github.com/NVIDIA/nccl-tests/blob/master/src/all_reduce.cu)、[src/common.cu](https://github.com/NVIDIA/nccl-tests/blob/master/src/common.cu) | CUDA/NCCL buffer、计时、正确性与带宽 | BSD-3-Clause；多 GPU 的真实 baseline，不是本目录 CPU 实验的替代命名 |
| [FLUX (2024)](https://arxiv.org/abs/2406.06858)；[gemm_rs benchmark](https://github.com/bytedance/flux/blob/main/test/python/gemm_rs/test_gemm_rs.py) | 细粒度算通融合与 dependent computation | 对照 coarse-grained overlap；[工程](https://github.com/bytedance/flux) Apache-2.0。真实通信隐藏验证要求多 GPU 与对应 NVSHMEM/CUDA 环境 |
| [Comet (2025)](https://arxiv.org/abs/2502.19811)；[MoE benchmark](https://github.com/bytedance/flux/tree/main/test/python/moe_ag_scatter) | MoE 通信依赖、细粒度重排、负载分配 | 读懂 dispatch/GEMM/combine；小型路由模型可 CPU 验证 token 守恒，多 GPU 才验证通信重叠 |
| [Zero Bubble Pipeline Parallelism (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/d5a8e37f38a08c68162452dcba89ae9c-Paper-Conference.pdf)；[工程](https://github.com/sail-sg/zero-bubble-pipeline-parallelism) | 分离 input-gradient 与 weight-gradient，排流水线空泡 | 可先 CPU 检查 DAG/依赖；实测吞吐与显存需多 GPU，模拟空泡减少不是训练加速 |
| [HybridFlow (2024 / EuroSys 2025)](https://arxiv.org/abs/2409.19256)；[verl GRPO examples](https://github.com/verl-project/verl/tree/main/examples/grpo_trainer) | rollout 与训练数据流、controller、模型重新分片 | verl Apache-2.0；结合上级长尾热身追踪阶段耗时，不把RL算法loss当系统实验核心 |
| [AReaL (2025)](https://arxiv.org/abs/2505.24298)；[gsm8k_rl.py](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)、[gsm8k_grpo.yaml](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml) | 异步 rollout、policy staleness 与吞吐/质量取舍 | Apache-2.0；当前示例配置为8 GPU、rollout/training各4的后端布局，不应承诺4060直接运行 |

## 选一个 R 实验，避免四条一起铺开

1. **Collective 分桶（CPU可做，2–4 GPU升级）**：固定全局数据、总 payload 和线程数，改变 bucket size；基线逐参数同步/单大 bucket/DDP，消融只改分桶、只改发起时机。CPU报告同机结果；升级后测 NCCL 通信尾部和训练步长。问题：较小 bucket 的重叠收益是否被 launch latency 抵消，拐点随层大小如何变化？
2. **Shard 与算通依赖（CPU语义，2–4 GPU性能）**：从 CS336 sharded optimizer 测试起步，画 DP/ZeRO stage1 的状态所有权，确保参数更新等价；记录 optimizer state bytes、peak memory、all-gather/reduce-scatter payload。随后阅读 FLUX，选择一个 GEMM+reduce-scatter 配置做源码追踪/微基准。不能用1 rank FSDP 声称 shard 的 scaling 效果。
3. **MoE 不均衡（CPU守恒，真实多 GPU网络）**：固定路由 trace，比较均匀/热点 expert 分布，核对每 token dispatch/combine exactly once、权重加和；在有硬件时扩展 all-to-all + grouped GEMM，以 Comet 为机制阅读。指标是最慢 expert 的尾部、通信/计算时间线、端到端延迟，不能只报平均 tokens/s。
4. **RL rollout 长尾（单卡trace，至少两组GPU异步复现）**：先用一个能装入显存的小模型采集实际生成长度与阶段时间；把 trace 放进上级调度器只验证假设，结果标为 trace-driven model。真正 verl/AReaL 实验记录 rollout 吞吐、trainer idle time、weight-version lag、reward/验证集指标；同步/异步比较同时控制采样配置与资源。单卡交替训练生成可以学数据流，不能冒称实现分离式并行带来的加速。

## 两周模板

- 第1–2天：写一个可证伪问题，固定源码 SHA、环境与硬件，选一个小 workload，读论文设计/评价部分并追踪一个源码入口。
- 第3–4天：通过 correctness，建立同硬件同工作量 baseline；输出原始日志和第一张时间线。
- 第5–7天：只改一个机制，做3次独立重复；若无收益，检查是否变成其他瓶颈，保留负结果。
- 第8–10天：扫一个主要维度，做至少两个消融，核对通信字节/峰值显存/有效工作量是否保持可比。
- 第11–12天：在第二 workload 或第二 shape 验证；给出预先写好的证伪条件与结论边界。
- 第13–14天：交付4页mini-paper（问题、设计、实验、限制）、可复现命令、固定配置、raw data、图和待导师讨论的3个问题。

例：不是“复现FLUX”，而是“在固定双 GPU 拓扑和 MLP shape 下，bucket 从 1 MiB 减到 64 KiB 是否降低 exposed communication tail；若 launch/packing 成本增长超过被隐藏通信，则机制不成立”。没有多 GPU 时将问题改成“同机 CPU Gloo bucket 的启动成本拐点”，在标题里明确资源范围。
