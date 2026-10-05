# 开源实验选用指南

本页于 **2026-10-05** 核验所列一手仓库/课程入口；它为 [16 周论文与实验路线](papers.md) 提供现成脚手架、测试和真实系统源码。主线以本地 labs 的 README/setup/grade 为交付入口，每模块只选必要的一项外部依赖，不要求把下面所有项目安装一遍。

## 如何采用现成材料

区分三个角色：**学生版 lab** 可以接入题目、接口与测试；**官方完成示例** 用来阅读、验证基线或在自己尝试后比较；**生产/论文系统** 用来追踪调用链、修改一个机制并验证。第三方代码跑通不自动等于自己的任务完成。优先链接或在独立 checkout 中操作，本课程本次没有复制这些项目的完整实现。

开始前，在实验目录写一份 `upstream-manifest.json`，至少记录 repo URL、**完整 commit SHA**、许可证路径、使用的文件、上游版本、自己的 patch、依赖锁定文件和核验日期。文中 `main/master` 链接是查找入口，**不是可重放的版本锁定**。你手动获取后立即 `git rev-parse HEAD`，将结果写入 manifest，再把证据链接改为该 SHA 的 permalink；若实验期间更新版本，作为新的实验条件重新记录。本文没有把未下载的主分支伪装成已固定/已测试的 commit。

用独立目录和环境，例如 `external/cs336-systems/`、`external/vllm-lab3/`、`external/llama.cpp/`，并确保 labs 根 `.gitignore` 忽略 `external/`、构建产物、权重和虚拟环境。安装步骤由你按各 lab 的 setup 与上游文档执行；kernels、vLLM、分布式/RL 不共用一个无限叠加依赖的环境。保留许可证、版权、NOTICE 和修改说明；外部模型与数据许可另外核验，代码许可不会自动覆盖它们。

## lab0–lab1：C++/CUDA、布局与 GEMM

| 来源与具体路径 | 复用范围与作业动作 | 许可证 / 版本要求 |
|---|---|---|
| [CMU LLM Systems HW1](https://github.com/llmsystem/llmsys_hw1)，[`src/combine.cu`](https://github.com/llmsystem/llmsys_hw1/blob/main/src/combine.cu)、[`minitorch/cuda_kernel_ops.py`](https://github.com/llmsystem/llmsys_hw1/blob/main/minitorch/cuda_kernel_ops.py) | 学生版 CUDA map/zip/reduce/GEMM 接口与 TODO 可作为外部练习；选择一项完成，再提交 correctness 与访问布局分析。Python wrapper 是绑定层，核心任务在 CUDA。不要把其他人的已完成作业作为本人的 kernel | [MIT](https://raw.githubusercontent.com/llmsystem/llmsys_hw1/main/LICENSE)；固定所选 main SHA，按其依赖单独建环境 |
| [NVIDIA CUDA Samples WMMA 示例](https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu) | 完成自己的 tiled GEMM 后，阅读官方完成示例，比较 fragment、shared memory 与累加精度；裁剪一个最小可解释实验。样例不是学生答案提交物 | 文件头 BSD-3-Clause；路径锚定 **v12.8**，再记录解析出的 commit。其他版本路径可能改变 |
| [CUTLASS Ampere TF32 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu) | 选修库基线：读类型/布局配置，解释 TF32 与 FP32 数值、运算量和吞吐比较条件 | 文件头 BSD-3-Clause；固定 SHA。不要把不同精度速度比写成同精度优化 |
| [GPU MODE lecture_004](https://github.com/gpu-mode/lectures/tree/main/lecture_004)、[lecture_008](https://github.com/gpu-mode/lectures/tree/main/lecture_008) | 用内存架构与 profiling 材料回答本地 trace 的一个瓶颈问题，补充 profiler 观察清单；它不是额外必修 lab | [Apache-2.0](https://github.com/gpu-mode/lectures/blob/main/LICENSE)；若执行代码，固定 SHA 与工具版本 |

4060 可做自写 CUDA、shared-memory tiling 和兼容的 Tensor Core 路径；具体 shape/工具版本仍需本机验证。[ThunderKittens educational_h100](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/gemm/educational_h100) 为 [MIT](https://github.com/HazyResearch/ThunderKittens/blob/main/LICENSE) 的**H100 阅读/远端选修**，不要作为 Ada 必过项。

## lab2：Attention、Triton 与 compiler

**直接可采用的学生版：Stanford CS336 Assignment 2 Systems。** 当前 [README](https://github.com/stanford-cs336/assignment2-systems) 标为 **Spring 2026**；不要沿用其旧版本年份。已核验 [handout `cs336_assignment2_systems.pdf`](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)、[`tests/adapters.py`](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/adapters.py)、[`tests/test_attention.py`](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_attention.py)，[MIT](https://github.com/stanford-cs336/assignment2-systems/blob/main/LICENSE)。选择 attention/benchmarking 对应题段，把自己的实现接到 adapter，保持测试期望不变；保留 upstream SHA、handout 版本和测试输出。上游 `uv run`/提交脚本可能自动安装依赖，先阅读脚本，再由你在独立环境中运行。

官方完成参考为 Triton [`python/tutorials/02-fused-softmax.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py) 与 [`06-fused-attention.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)，[MIT](https://raw.githubusercontent.com/triton-lang/triton/main/LICENSE)。先做自己的 online-softmax/attention 版本，再用官方实现作为外部基线；指出自己的新增代码、边界处理与实验设计。完整复制教程只能算阅读，不能算独立实现。冻结 Triton checkout SHA，同时记录实际安装的 Triton/PyTorch/CUDA 版本，避免代码与 wheel API 不匹配。

进阶可选 TileLang [`examples/gemm/example_gemm.py`](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)，[MIT](https://github.com/tile-ai/tilelang/blob/main/LICENSE)。同一个表达式在 CUDA/Triton/TileLang 中最多选两个对比，记录编译、正确性、稳态时间与适用 shape。Mirage 的 [OSDI 2025 论文入口](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) 可用于 compiler 设计阅读；[仓库](https://github.com/mirage-project/mirage) 当前也描述 MPK 后续工作，因此未经历史版本核对，不将当前 README 的运行命令认作 2025 论文 artifact。

FlashAttention [官方实现](https://github.com/Dao-AILab/flash-attention) 为 [BSD-3-Clause](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/LICENSE)。FA3 的 Hopper 路线与普通 attention correctness 要分开：本机未运行的专用 kernel 不能列为已通过。

## lab3–lab4：真实 KV 与调度控制平面

vLLM 采用 **v0.10.2 作为源码阅读参照**，不要求为了阅读而覆盖已有环境。许可证为 [Apache-2.0](https://github.com/vllm-project/vllm/blob/v0.10.2/LICENSE)。已核验路径：

- [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/sched/scheduler.py)：追踪请求状态、token budget、抢占与恢复。
- [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/kv_cache_manager.py)、[`single_type_kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/single_type_kv_cache_manager.py)、[`block_pool.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/block_pool.py)：追踪命中、分配、引用与可驱逐页。
- [`vllm/benchmarks/serve.py`](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/benchmarks/serve.py)：研究到达过程和结果字段；该版本旧 `benchmarks/benchmark_serving.py` 已是转发入口，不能凭旧路径猜命令语义。

先完成本地 [runtime 状态机](../../mlsys-research-labs/lab3/runtime/README.md)，再提交实际引擎 patch 与对应回归证据。可做有必要的生命周期诊断、已有事件缺失修补或受控策略修改；只改 launch 参数不满足「改运行时」交付。记录真实 import 文件路径，避免编辑 checkout 却测试了另一个安装包。若用其他版本运行，提交路径/机制映射；基线与改动版必须使用同一个基础 SHA 和依赖环境。

缓存进阶参考 SGLang [`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)，[Apache-2.0](https://github.com/sgl-project/sglang/blob/main/LICENSE)。冻结 SHA，选择命中/引用/驱逐的一条调用链，做 uniform 与 skewed prefix 的公平对照；SGLang 作为第二系统是选修，不强制再安装一套重型 serving 环境。Python 调度控制代码仍然是 MLSys 系统代码，评价标准是机制、正确性与实验，而非单凭文件扩展名。

## lab5：分布式语义、通信与 RL

| 来源与具体路径 | 可直接采用的任务 | 许可证 / 资源范围 |
|---|---|---|
| CS336 [`tests/test_ddp.py`](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_ddp.py)、[`tests/test_sharded_optimizer.py`](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_sharded_optimizer.py)、[`tests/test_fsdp.py`](https://github.com/stanford-cs336/assignment2-systems/blob/main/tests/test_fsdp.py) 及前述 adapters | 主线选 DDP：对接自己的两进程实现，检查全 batch oracle；分片 optimizer/FSDP 是进阶选项。测试所需 backend 以锁定版本为准 | MIT；DDP 有 CPU Gloo 路径。CPU 结果证明语义，不证明 NCCL 性能 |
| NCCL Tests [`src/all_reduce.cu`](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/all_reduce.cu)、[`doc/PERFORMANCE.md`](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md) | 远端有至少两张 GPU 时，扫描消息大小并区分 algbw/busbw；记录拓扑、NCCL/CUDA 和进程布局 | [BSD-3-Clause](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/LICENSE.txt)；固定 SHA `b4d5beebca8a76cf01335f724d154b9b9d394d96`。单 GPU不能验证互连性能 |
| FLUX [`test/python/gemm_rs/test_gemm_rs.py`](https://github.com/bytedance/flux/blob/main/test/python/gemm_rs/test_gemm_rs.py)、[`test/python/moe_ag_scatter`](https://github.com/bytedance/flux/tree/main/test/python/moe_ag_scatter) | 选读/远端：对照 GEMM+通信串行与融合，解释关键路径；连接 FLUX/Comet 论文 | [Apache-2.0](https://github.com/bytedance/flux/blob/main/LICENSE)；固定 SHA，需核对 NVSHMEM 与多 GPU要求，不是4060主线必装 |
| verl [`examples/grpo_trainer`](https://github.com/verl-project/verl/tree/main/examples/grpo_trainer)、[`examples/profile`](https://github.com/verl-project/verl/tree/main/examples/profile) | 主线先读配置与依赖图，标出 rollout、reward、actor update；有资源再复跑一个官方配置并标注缩规模变化 | [Apache-2.0](https://github.com/verl-project/verl/blob/main/LICENSE)；独立 RL 环境、固定 SHA；不假定任意脚本单卡可跑 |
| AReaL [`examples/math/gsm8k_rl.py`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_rl.py)、[`gsm8k_grpo.yaml`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_grpo.yaml) | 异步 RL 选修：读 rollout/actor 分配，设计吞吐与质量双重验证 | [Apache-2.0](https://github.com/areal-project/AReaL/blob/main/LICENSE)；已核验示例为8 GPU配置。不能把改一个模型名解释为4060适配完成 |

CPU Gloo、时间线模拟、单卡测量与真实多 GPU 实验要分别标记。对于 trace 模拟，提交成本模型与假设；对于实际 overlap，提交 profiler 证据，并把「通信发起异步」与「确实隐藏了关键路径」区分开。

## lab6：可选的 C++ speculative decoding 路线

[llama.cpp](https://github.com/ggml-org/llama.cpp) 的 [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) 源码可作为实际 C++ runtime 实践，已核验 [`examples/speculative-simple/speculative-simple.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/examples/speculative-simple/speculative-simple.cpp)、[`common/speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)、[`docs/speculative.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)。这是完成示例与真实 runtime，任务是读懂验证/接受/回滚路径并写自己的小修改，而不是把示例复制进 labs 改名提交。

在独立 CMake build 目录中固定 SHA，先跑 baseline，再做一个长度控制或可审查诊断 patch。优先 ngram 方法以避免额外 drafter 内存；记录 target、tokenizer、量化、上下文和采样设置。正确性先于速度：greedy 先做输出一致性检查，随机采样的分布正确性需另设统计检验。草稿更长但端到端更慢是有效的研究结果。不同 commit 的 CLI 与 speculative 实现可变化，命令以所固定版本的帮助和文档为准。

## lab7：把实验做成可重放 artifact

采用 [MLSys 2025 AE](https://mlsys.org/Conferences/2025/CallForAE) 的 artifact 组织方法；它是官方说明，不是可以整体复制并重新许可的课程代码。提交最小依赖、smoke 命令、完整实验入口、资源预算与从日志生成图表的步骤。不要将会议徽章要求转写为自己已经获得认证。

进阶参考 FlashInfer-Bench [`flashinfer_bench/`](https://github.com/flashinfer-ai/flashinfer-bench/tree/main/flashinfer_bench) 与 [`examples/fi_gqa_e2e_example.py`](https://github.com/flashinfer-ai/flashinfer-bench/blob/main/examples/fi_gqa_e2e_example.py)，代码许可为 [Apache-2.0（文件名 `LICENCE`）](https://github.com/flashinfer-ai/flashinfer-bench/blob/main/LICENCE)。冻结 SHA，先阅读定义/工作负载/实现/评估的分离方式，把自己的实验整理成同样可审查的结构；无需把整个框架嵌入课程。运行官方数据集前另查数据许可与容量，仅选一个适配本机的 workload，并保留 correctness 失败和性能失败记录。

## 课程规则与验收边界

### 替代任务如何接入本仓库评分

外部学生版可替代同一核心实现，不需要把算法重新写一遍；本仓库的接口契约、强基线与S层必过项仍保留。先固定上游版本，再写很薄的适配层，不能把官方完成参考接成自己的实现。

| 外部任务 | 本仓库接入点 | 仍需保留的验收 |
|---|---|---|
| CMU自己完成的CUDA GEMM | `lab1/cuda/student.cu`调用自己的上游kernel；CMake增加所需源文件 | 共享内存实现、FP32契约、非整齐形状、cuBLAS对照及`--test`；仅完成map/zip不能替代GEMM |
| CS336自己完成的attention | `lab2/attention/attention.py`适配shape/dtype与causal前向接口 | 本地独立正确性、无全局S²中间结果、稳定计时/显存及留出；上游更完整的反向任务为选修 |
| CS336自己完成的DDP | 复用自己的同步逻辑到`lab5/distributed/student.py`，必要时保留单独上游测试入口 | 实际1/2 rank、非均分样本的全batch等价、trace/bucket实验；C++ MPI仍是独立必做任务 |

如果上游契约不兼容，先在报告列出差异，再做最小适配并保留原测试结果；不能修改本地测试期望或删除评分项来制造通过。外部性能结果的硬件与精度不能直接混入本地表格。

1. 先读外部课的 handout、README 与学习/提交规则。许可证允许复用代码不等于允许把答案作为课程独立作业提交；自学记录也要区分借用、修改与原创。本文不要求联系上游、提交作业或发布 patch。
2. 保留独立测试和参考期望；可以修复经证据确认的上游 bug，但必须展示失败最小例与修复理由。不要通过跳过测试、吞掉 TODO 或调宽容忍把未完成标成通过。
3. 完成示例可作基线，不替代核心 TODO；从零部分集中在已经有正确性契约的一个机制，避免整个系统无指导重写。
4. 论文阅读、教学实现、原 artifact 验证分别评分；缩小工作负载和硬件变更必须披露。能运行并不意味着已复现论文性能。
5. [DLSys 官方课程站](https://dlsyscourse.org/) 可作补充课程入口；本次未建立一个已核验且许可明确的独立 HW3 GitHub 复用入口，因此不把猜测仓库放进安装或评分依赖。

本页完成的是来源、范围与使用步骤的配置指导；依赖安装、模型下载、CUDA 构建与外部 artifact 运行由你在实际实验时执行，逐项保存结果。
