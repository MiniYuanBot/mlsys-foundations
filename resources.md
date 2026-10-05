---
title: 资料汇总与阅读优先级
tags: [mlsys, resources]
updated: 2026-10-05
---

# 资料汇总与阅读优先级

[返回索引](index.md)

原课程资料核对日期：2026-10-04；在线 OJ 与增强实验资料补充日期：2026-10-05。下面主要使用官方课程、项目文档、论文原文和会议页面。课程中的“必读”均指模块指定部分，不要求把整套课程、整篇长文档或整本教材读完。网页与软件版本会更新，实验应固定自己安装的版本及模型 revision。

## A. 课程和训练材料

| 资料 | 使用位置与范围 | 优先级 |
|---|---|---|
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | M00：张量、模型、数据、优化、保存加载 | 必做选读 |
| [Tensor 教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | M00：shape/dtype/device、索引与广播 | 必做 |
| [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | M00/L00：梯度记录与累积 | 必做 |
| [优化循环教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | M00/L00：训练与评估循环 | 必做 |
| [Stanford CS336，2025](https://cs336.stanford.edu/spring2025/) | M01–M05：Lecture 2、3、GPU/kernel、parallelism、inference 的指定内容 | 选读，不整课重修 |
| [CMU Deep Learning Systems](https://dlsyscourse.org/) | 自动微分/系统基础出现明确缺口时补读 | 可选 |
| [LCPU AI Infra 课程](https://infra.seminars.lcpu.dev/) | GPU、Triton、GEMM 入门补充 | M02 选读 |
| [LCPU 日程](https://infra.seminars.lcpu.dev/schedule) | 定位 Session 1 / Session 2 | 导航 |
| [LCPU Session 2.1 GEMM PDF](https://infra.seminars.lcpu.dev/slides/session0201.pdf) | M02：朴素计算、数据复用、shared-memory tiling | 必做选读 |
| [LCPU 训练仓库](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars) | 参考作业与项目结构 | 可选，不全做 |
| [LCPU Assignment 01](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment01/README.md) | M02：kernel 与解释器相关说明 | 定位查阅 |
| [Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | M02/L02：program、offset、mask | 必做 |
| [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | M02/L02：稳定 softmax、融合与测量 | 必做 |

你提供的两个 LCPU 链接均已纳入课程。其 CUDA/GEMM 深度训练很有价值，但完整做完会挤占本课程的论文与服务实验时间，因此只取与当前目标直接相关的部分。

## B. 测量与软件文档

| 资料 | 用途 |
|---|---|
| [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) | 按实际 OS/计算平台生成安装命令 |
| [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | L01/L02 计时与预热 |
| [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | L01 查看算子/内核及形状 |
| [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | L01 causal、dropout 和实现语义 |
| [SDPA 后端上下文](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) | 正确性排查时控制后端 |
| [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) | Windows 驱动、WSL2 与安装边界 |
| [vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) | Linux/硬件/二进制依赖要求 |
| [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/) | 服务参数、缓存与调度开关 |
| [vLLM bench serve CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) | 到达率、长度、结果保存、延迟分位数 |
| [vLLM benchmark 实现文档](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve/) | 需要核实指标字段/生成负载的实现时使用 |
| [Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 默认模型、config、tokenizer 与 revision |

## C. 核心论文与阅读任务

| 论文 | 模块/深度 | 带着什么问题读 |
|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | M01/M02，关键部分精读 | 为什么 IO 与中间张量会成为瓶颈？ |
| [Orca，OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | M03，关键部分精读 | iteration-level scheduling 改变了什么？ |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | M03，关键部分精读 | KV 分页解决哪类浪费，代价是什么？ |
| [Sarathi-Serve，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | M04/M06，精读并做机制检验 | chunk 大小如何影响 prefill/decode 权衡？ |
| [DistServe](https://arxiv.org/abs/2401.09670) | M04，关键部分精读 | 解耦需要哪些通信与资源条件？ |
| [ZeRO](https://arxiv.org/abs/1910.02054) | M05，机制选读 | 状态分片减少了什么，增加了什么？ |
| [HybridFlow](https://arxiv.org/abs/2409.19256) | M05，关键部分选读 | RL 数据流和状态安排为何复杂？ |
| [DARTS](https://arxiv.org/abs/2605.30859) | M05，前沿选读；arXiv 页面注明 ICML 2026 录用 | rollout 长尾与采样/质量约束如何关联？ |

本课程所谓 6–8 篇关键部分精读，允许根据时间在 ZeRO、HybridFlow、DARTS 中选择不同深度；不要将扫过摘要也记为精读。原文发布日期、修订与会议状态以链接页面为准。

## D. 扩展地图与导师方向

| 资料 | 如何使用 |
|---|---|
| [Efficient LLM Inference Survey](https://arxiv.org/abs/2312.15234) | 第一遍建立术语地图，关键主张回到原论文 |
| [SGLang](https://arxiv.org/abs/2312.07104) | M03 可选扫读：结构化程序与服务栈 |
| [Helix](https://arxiv.org/abs/2406.01566) | M05 扫读：异构资源上的推理安排 |
| [FlexLLM，NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/oliaro) | M05 扫读：服务与微调共存的系统设计 |
| [Mirage，OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) | M05 扫读：张量程序/内核优化方向 |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | M05 扫读：speculative inference 的系统问题 |
| [苗旭鹏老师论文列表](https://hsword.github.io/publications/) | 核对研究方向、寻找后续工作；核对时直接访问不稳定，见下方备用入口 |
| [北大计算机学院教师页](https://cs.pku.edu.cn/info/1072/4600.htm) | 官方教师资料与个人主页入口；论文列表不可访问时从此导航 |
| [MLSys 官网](https://mlsys.org/) | 进入对应年份论文与议程；检索时记录年份 |
| [OSDI 2026 论文目录](https://www.usenix.org/conference/osdi26/technical-sessions) | M07 查近作与已解决问题 |
| [NSDI 2026 论文目录](https://www.usenix.org/conference/nsdi26/technical-sessions) | M07 查近作与部署/网络约束 |
| [ACM Digital Library](https://dl.acm.org/) | 按 SOSP/ASPLOS 与关键词追踪论文原文 |

到 M07 时，通过引用/后续工作补读约 5 篇最相关论文，可把分层阅读地图扩展到约 20 篇。随后与导师商量如何逐步扩大到 200 篇，而不占用全部实现时间。

## E. 北京大学 HPC

| 文档 | 必须确认的事情 |
|---|---|
| [平台首页](https://hpc.pku.edu.cn/) | 官方入口和最新通知 |
| [开户说明](https://hpc.pku.edu.cn/ug/account/) | 科研/教学账号条件与项目负责人 |
| [成员账号](https://hpc.pku.edu.cn/ug/account/add_user/) | 导师/管理员如何添加成员 |
| [未名二号](https://hpc.pku.edu.cn/ug/cluster/wm2/) | 节点概况；具体 GPU 型号另确认 |
| [Slurm 使用](https://hpc.pku.edu.cn/ug/guide/slurm/) | 在计算节点执行，不在登录节点跑计算 |
| [sbatch](https://hpc.pku.edu.cn/ug/guide/slurm/sbatch/) | 提交短作业、日志与资源请求 |
| [缴费说明](https://hpc.pku.edu.cn/ug/account/recharge/) | 计费归属与实际账号安排，本文不预设单价 |

## F. 在线 OJ 与 AI 辅助增强实验

具体选择、时间预算、题目差异和验收见 [在线 OJ 手册](05-online-judge.md)、[每周增强实验](06-ai-assisted-labs.md)、[融合算子挑战周](labs/L02X-kernel-challenge.md)。每次只选一个平台，不要求全部注册或付费。

| 官方资源 | 课程用途 |
|---|---|
| [TensorOJ 首页](https://tensoroj.cn/cann/home) / [Vector 题单](https://tensoroj.cn/cann/public/vector) | Ascend C / 昇腾拓展；本次核对公开题库，未实测提交 |
| [TensorOJ Relu](https://tensoroj.cn/cann/public/vector/relu) / [Add](https://tensoroj.cn/cann/public/vector/add) | X 周 Ascend 路线二选一；完整题面比 FP32 教学子集更广 |
| [TensorOJ Softmax](https://tensoroj.cn/cann/public/vector/softmax) / [AddRmsNorm](https://tensoroj.cn/cann/public/vector/addrmsnorm) | 后续任意轴归约与融合挑战 |
| [Ascend C 编程范式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0035.html) | 分块、搬运、计算的入门；示例版本与运行版本需区分 |
| [Tensara 题库](https://tensara.org/problems) / [项目说明](https://github.com/tensara/tensara) / [CLI](https://tensara.org/cli) | 默认 Triton 在线补充；按题目与实际设备核对 |
| [Tensara Log Softmax](https://tensara.org/problems/log-softmax) / [Softmax](https://tensara.org/problems/softmax) | 分别为逐行 log-softmax 与指定维度 softmax；不能共用未经适配的入口 |
| [LeetGPU 题库](https://leetgpu.com/challenges) / [资料页](https://leetgpu.com/resources) | 替代平台；按题名定位归约、RMSNorm 与融合题 |
| [GPU MODE Popcorn CLI](https://github.com/gpu-mode/popcorn-cli) / [PMPP_v2](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2) | 后续更深入的真实硬件 kernel 训练 |
| [Triton LayerNorm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | L02X 前向归约参考；RMSNorm 公式不同 |
| [PyTorch 通用 checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) | L00 增强：模型＋优化器恢复 |
| [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/) | L06 增强：先核对本机版本实际暴露的指标 |

## 访问与版本说明

已检查课程内部链接。外部资源中，导师论文列表可被检索到，但本次直接访问曾返回错误；ACM 入口也可能限制自动访问。若暂时打不开，使用上表的北大官方教师页、论文原文链接与会议目录继续学习，不把单个导航页当作课程前置条件。API 文档可能跳转到新的分类路径，以页面提供的跳转和安装版本文档为准。

TensorOJ 的动态首页在浏览器读取时未成功，后通过其公开页面接口核对了题单、题面及 CANN 版本；LeetGPU 部分题库信息来自公开搜索索引。平台硬件配额、费用、排队、账号权限和实际评测未验证，不将网页可访问等同于已能运行实验。平台不可用时按 OJ 手册回到本地路径。

## 资料使用检查

- [ ] 模块中需要的链接已经直接嵌在相应步骤，本页只是总索引。
- [ ] 阅读原文时记录图/小节，不只记录“已读”。
- [ ] 使用代码时注明来源，区分复用与自己实现的部分。
- [ ] 用在线文档跑实验前保存当前版本帮助，不依赖未来可能变化的默认值。
- [ ] 每条研究结论可以追溯到自己的数据或明确论文位置。
