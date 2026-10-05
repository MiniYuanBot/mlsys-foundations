# 梯度同步与通信重叠实验

[配置](setup.md) · [测试与评分](grade.md) · [研究拓展](research.md) · [上级 lab5](../README.md)

预计 **8–12 小时**。线性模型用于手算和定位同步错误；主要任务是实现 collective、读取时间线、测量通信开销。`../tail_sim.py` 是单独的调度基础练习。

| 阶段 | 时间 | 已给框架 | 你要完成的系统工作 |
|---|---:|---|---|
| B：正确性 | 2 h | 确定性数据、SGD、全 batch oracle、torchrun、多步日志 | `student.py` 中的梯度同步；先等分，再正确处理 5/4 样本分片 |
| S1：host collective | 2–3 h | C++17/MPI 可编译入口、内置 MPI 对照、CMake/CTest | `cpp/ring_allreduce.cpp` 的 ring reduce-scatter + all-gather；说明每步 chunk 所有权 |
| S2：系统观测 | 2–3 h | rank 独立 trace、同步前后梯度、PID/参数日志 | 定位真实 collective，统计调用次数/字节数，写 bucket benchmark，并对照框架 DDP |
| S3：实验解释 | 2–4 h | 全 batch oracle 与下方测量协议 | 控制变量、重复测量、负结果、通信占比；形成一个两周问题 |

## B：Gloo 梯度同步

在项目根目录、自己配置的环境中运行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 --module lab5.distributed.train --samples 8 --steps 5 --out results/lab5/gloo-equal
python -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 --module lab5.distributed.train --samples 9 --steps 5 --trace --out results/lab5/gloo-uneven
python -m unittest lab5.distributed.tests -v
```

初次执行会因为核心 TODO 失败。两个 rank 分别持有 `indices = rank, rank+world_size, ...`，用本地均值损失反传，然后在 optimizer 更新前同步。提供的 oracle 使用相同全局数据、初值、学习率进行单进程全 batch 更新，每一步都要一致。

先推导 `g = sum_r(n_r * g_r) / sum_r(n_r)`。等样本数时才可以把各 rank 的均值梯度相加后除以 world size。不要在错误位置增加 `barrier` 来掩盖问题；barrier 不会替你规约梯度。所有 rank 必须按相同顺序调用 collective。空分片不属于本次接口契约，入口会拒绝 `samples < world_size`。

交付 `rank0.json`、`rank1.json`、`oracle.json`；指出同步前梯度为何不同、更新后参数为何相同。`--trace` 写出每个 rank 的 Chrome trace JSON；用本地 trace viewer 检查 `gradient_sync` 范围内的 `c10d`/`gloo` collective 事件（事件名依 PyTorch 版本变化），不能只提交自己加的同名标注。

每次运行使用新的 `--out` 目录，入口拒绝覆盖已有 rank/oracle 文件。fixture 的 `elapsed_seconds` 包含 Python 逐步记录和可能的 profiler 开销，只作诊断；后面的性能实验需要独立的稳态 benchmark，不能直接把这个时间换算成训练吞吐。

## S1：C++ ring all-reduce

`cpp/ring_allreduce.cpp` 中固定 rank r 的输入为 `10*r+i`，长度 `2*world_size`。2 rank 输入 `[0,1,2,3]` 与 `[10,11,12,13]`，输出应为 `[10,12,14,16]`；3 rank 输出 `[30,33,36,39,42,45]`。框架用 MPI_Allreduce 生成独立对照，但学生 ring 只能用 point-to-point MPI_Sendrecv 完成两阶段算法。

手画每一步 chunk 流动，然后写代码、测试 1/2/3 rank。count 不可整除 rank 数、不同 dtype、零长度输入是后续扩展，不应先把实现范围扩大到无法调试。MPI 练习涉及 buffer 生命周期、邻接关系、消息 tag、同步顺序和死锁；不是 CUDA kernel。CPU 同机 MPI/Gloo 测量也不能证明跨节点网络或 NCCL 性能。

## S2：通信测量

先保留全部正确性测试，再新增自己的 `bucket_bench.py`。测量张量总字节数 `S ∈ {4 KiB, 1 MiB, 16 MiB}`、bucket 个数 `K ∈ {1,4,16}`、rank 数 `{1,2}`，dtype 固定 float32。同一 S 改变 K 时总通信 payload 必须不变，分别报告打包/解包、collective 与端到端耗时。每个配置至少 10 次 warm-up、30 次测量，重复启动 3 次；测量前同步，结束计时之前等待通信完成。主计时关闭 profiler，另跑短 trace。CPU 线程数固定为 1 并记录 CPU、内存、OS、PyTorch 与 Gloo 版本信息。

继续做一个小 MLP，比较逐参数同步、合并 bucket、PyTorch DDP 三条路径。固定相同全局 batch 与优化器语义，记录每 rank 的步时间并使用 `max(rank_time)` 表示最慢 rank 的步长。不能从进程启动时间推导 collective latency，也不能用两参数线性模型判断大模型 scaling。

理论 ring all-reduce 每 rank 发送 `2*(p-1)/p*S` 字节（不含协议头、样本计数与本地拷贝）。把理论、API payload 和实际传输层流量分列；只观察到 API payload 时不要把它写成网卡实测字节。Gloo 后端可选择不同算法，不能把你的 MPI ring 公式当作 Gloo 内部实现证明。对照 [NCCL 的 algbw/busbw 定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。

## S3：通信与计算重叠

当前 fixture 的全部反向在同步之前完成，按设计没有计算/通信重叠。先以这个串行时间线为 baseline；在 MLP 中用梯度 ready hook 或 DDP bucket 发起异步通信，并在 optimizer step 前等候。检查一轮已完成通信后才可复用 buffer，尤其注意 bucket 视图与 in-place 更新。`async_op=True` 本身不是重叠证据。

提交串行与异步的 trace；若 trace 没有暴露后端完成区间，就不能精确报告 overlap ratio，只能给端到端结果与观测局限。若能观察实际区间，报告计算区间与通信区间交集的并集时长、暴露的通信尾部，以及 `T_serial`/`T_overlap`。Python launch 区间交叠不足以证明 GPU kernel 交叠。CPU 未加速或因争用而变慢也是有效结论。

## 工具与运行范围

| 工具 | 本章学习内容 | 运行范围 |
|---|---|---|
| Python / PyTorch distributed | autograd 边界、process group、梯度语义、DDP bucket | CPU 两进程即可 |
| Gloo | CPU 张量的真实 collective 与进程同步 | 同机可验证语义，不代表机间网络 |
| C++17 / MPI | 手动 buffer、ring 的数据流、死锁排查 | CPU + MPI；独立于 PyTorch |
| CUDA C++ / NCCL | device buffer、stream/event、GPU collective 与实际拓扑 | 至少两块真实 GPU 才能验证跨 GPU 通信 |
| Nsight Systems | CUDA/NCCL/NVTX 时间线与重叠证据 | 对照已安装版本支持范围，见 setup |

单 RTX 4060 可做 CUDA kernel、GPU 显存/算子 profile；本章 distributed 主线可全用 CPU。单卡启动多个 rank 不等价于多 GPU NCCL，不能用它报告 tensor parallel scaling、NVLink、跨 GPU overlap。真正的多 GPU 扩展见 [配置](setup.md)，论文与两周计划见 [research.md](research.md)。
