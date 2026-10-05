# KV 分页与引擎实验

预计 **8–12 小时**。先实现可检查的资源管理，再将分配、共享和释放的生命周期映射到真实引擎。

已提供数据结构、状态检查、独立契约测试、事件 runner 和连续预留容量参考。需要完成 `allocator.py` 中四个 TODO。Python 在这里承担调度与内存管理的控制平面；token ID 只是 CPU 上可手算的占位数据，**没有实现 GPU KV 存储、attention kernel 或生产级 serving runtime**。

## A. 基础实现：页表、引用计数和原子失败（3–4 小时）

读 [PagedAttention 论文](https://arxiv.org/abs/2309.06180) 中内存浪费、分页、共享与 copy-on-write 的设计。画出 3 个物理页、每页 4 token 时，两条请求的逻辑页到物理页映射。然后依次实现：

1. `allocate`：按当前 token 数分配页，创建页表；空序列不分配页。
2. `release`：释放引用，最后一个引用消失才归还页；重复 release 必须拒绝。
3. `fork_prefix`：新请求共享已计算前缀，维护引用计数。
4. `append`：未满且独占的尾页可写；未满且共享的尾页先复制；已满共享页无需复制。若容量不足，所有页、token、引用计数和序列状态必须完全不变。

所有操作是同步的；此阶段不讨论并发锁。`snapshot()`、`read()`、`audit()` 已提供，便于逐步定位状态破坏。测试依据预先写定的 token 与页数检查，不把学生实现自身作为 oracle。

```bash
python -m unittest lab3.runtime.tests -v
python -m lab3.runtime.run
```

初始 TODO 必须报错。`TestProvidedInfrastructure` 通过只说明脚手架可运行；它不代表 allocator 已完成。

手算锚点：每页 4 token，a 持有 `[1,2,3,4,5]`，b 共享 a 前缀。此时物理 token 为 5、逻辑 token 为 10。b 追加 6 后，应有 3 个活跃页、物理 token 为 7、逻辑 token 为 11；a 的内容不变。解释差值来自哪里，并说明页表如何变化。

## B. 资源实验：分页与连续预留（1–2 小时）

`reference.py` 提供完整的连续预留**容量模型**，每条请求为 `max_length` 私有预留空间。它不模拟内存地址、外部碎片、计算时间或 GPU 性能。比较时两种策略使用相同 token-slot 总容量、相同请求序列和相同最大长度；分别统计成功准入数、活跃页数、物理/逻辑 token、页内浪费以及未使用的预留空间。

准备 3 条短 trace：短输出混合长输出、所有请求共享前缀、完全不共享前缀。每条至少包括一次 release 后再分配。将每个事件的页表和计数写成 JSONL，检查全程内存守恒。改变页大小 2/4/8，解释页内浪费与页表开销的取舍；不要把 token slot 直接当作字节。

计算真实模型 KV 字节时，显式列出层数、KV head 数、head dimension、dtype 字节数，基础密集 attention 的每 token KV 大小为 `2 × layers × kv_heads × head_dim × dtype_bytes`；若模型使用 MLA、滑动窗口或混合层，先修正假设。权重、激活、工作区与 CUDA graph 内存另计。

## C. 真实引擎源码与小修改（3–4 小时，必做）

源码阅读使用 **vLLM v0.10.2**。以下路径已于 2026-10-05 核验。实际运行可以用已适配本机的另一版本，但须提交 commit SHA、路径映射和差异说明，不能把不同版本结果混为一组。

| 路径 | 阅读时回答的问题 |
|---|---|
| [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/sched/scheduler.py) | `schedule()` 如何给运行中/等待中的请求分配 token budget？内存不足后如何抢占与回队？ |
| [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/kv_cache_manager.py) | `get_computed_blocks()`、`allocate_slots()`、`free()` 如何连接命中、分配、释放？ |
| [`vllm/v1/core/single_type_kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/single_type_kv_cache_manager.py) | 一种 attention 类型的页需求怎样转换为块池操作？ |
| [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/block_pool.py) | `get_new_blocks()`、`touch()`、`free_blocks()` 与可驱逐缓存是什么关系？ |

沿调用点补上 coordinator 等中间层，提交带文件路径和函数名的调用链图，覆盖「请求进入 → prefix 命中 → allocate → schedule → 结束释放」和「容量不足 → 抢占/重算」两条路径。

**必须写一个真实引擎 patch**：首选在块池/调度生命周期增加默认关闭的、可开关的结构化诊断，记录事件、请求关联（若该层可得）、分配块数、空闲块数及抢占。先查现有 KV events/metrics，说明为什么已有事件不足；也可以直接修补一个现有事件缺失或写一个带回归测试的诊断检查。不要为完成作业重复实现已有完整监控功能。

交付 `upstream-commit.txt`、`engine.patch`、改动处回归测试和运行日志。至少运行「重复前缀」「不共享前缀」「受限 KV 容量」三种小 workload，确认事件符合调用链。诊断开关关闭时应保留原行为；开关开启的计时开销要单独报告。只写 diff 未执行，不算集成验证完成。没有可用引擎环境时，源码分析可先交，执行项保留未完成状态。

**必须比较差异**：本练习允许共享未满尾页并用 COW，且 refcount 归零即清空；v0.10.2 的 prefix cache 以完整块建立缓存，并可把 refcount=0 的缓存块保留为可驱逐候选，还有 null block、不同 attention 类型、异步工作等处理。因此这里的 `fork_prefix` 不是对 V1 prefix cache API 的逐行复刻，测试也不能证明 V1 的正确性。[块池源码依据](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/block_pool.py)

## D. 论文驱动拓展（二选一，另需 4–8 小时）

**缓存偏斜与公平性。** 阅读 [SGLang / RadixAttention（NeurIPS 2024）](https://arxiv.org/abs/2312.07104)，对照 [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)。构建热点前缀和冷门前缀两组请求，固定到达 trace、cache 容量、token 总量，对比均匀与偏斜访问。报告两组各自的命中率、排队、TTFT 与 goodput，避免只报总体均值。本教学状态机缺少到达时间和计算成本模型，真实延迟需由引擎测量；CPU trace 只能讨论命中/容量。

**缓存淘汰。** 在新实验文件中扩展「活跃引用/已缓存可淘汰/真正空闲」三态，至少比较两个策略。保留相同容量、相同序列和相同预热条件，禁止驱逐 refcount>0 的页。先建立测试再对接引擎；负结果也可获得满分。阅读 [Mooncake（2024 预印本）](https://arxiv.org/abs/2407.00079)，解释单卡页池与跨设备 KV 传输的共同点及不同成本。多 GPU/网络系统只能作为后续远端实验，不能用本 CPU 状态机声称复现整篇论文。

所有拓展先写一段可被否定的假设、对照组、主要指标和失败标准。完成后用一个新 seed/新 prefix 分布作留出验证。

## 提交物

按 [grade.md](grade.md) 自评；在 lab3 报告中链接本节实现、契约测试、三条 trace、资源分析、真实引擎调用链、patch 和执行证据。拓展单独标为「已运行 / 设计完成 / 环境阻塞」，禁止用设计文档替代实测结论。

本目录教学代码未复制第三方实现。vLLM 和 SGLang 作为源码阅读/外部练习依赖；对应仓库 [vLLM LICENSE](https://github.com/vllm-project/vllm/blob/v0.10.2/LICENSE) 与 [SGLang LICENSE](https://github.com/sgl-project/sglang/blob/main/LICENSE) 均为 Apache-2.0，若后续复制代码，保留相应版权/许可证及修改说明。
