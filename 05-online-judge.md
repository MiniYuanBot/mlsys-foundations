---
title: 在线算子 OJ｜平台选择与练习路线
tags: [mlsys, oj, kernel, ascend]
updated: 2026-10-05
---

# 在线算子 OJ：平台选择与练习路线

[返回索引](index.md) · [AI 辅助实验增强](06-ai-assisted-labs.md) · [融合算子挑战](labs/L02X-kernel-challenge.md)

在线评测适合给算子实现增加外部正确性检查、远端硬件测量和陌生输入。课程优先选择与现有工具相容的 NVIDIA/Triton 平台；TensorOJ 用来探索 Ascend C / 昇腾。先选一个平台打通提交，不同时维护四套环境。

## 核对范围与推荐顺序

核对日期：2026-10-05。本次读取了公开题面、项目说明及 TensorOJ 首页所用的公开题库接口；**没有登录、提交代码、购买额度或实测评测服务**。下面的推荐依据是课程匹配程度，账号可用设备、额度、价格、排队时间与 AI 使用规则需在首次使用时核实。

| 平台与官方入口 | 已核对的能力 | 在本课程中的安排 | 使用前检查 |
|---|---|---|---|
| [Tensara](https://tensara.org/) / [官方项目说明](https://github.com/tensara/tensara) | 支持 CUDA、Triton 等；真实 GPU 评测、按 GPU 排榜；有具体题面、测试规模和 CLI | **默认 Triton 补充平台**：M02 完成一次小算子提交；X 周做归约/融合拓展 | 当前题目的语言、GPU、额度；不同题目接口并不统一 |
| [LeetGPU](https://leetgpu.com/) / [题库](https://leetgpu.com/challenges) | 浏览器 GPU 编程；主页列出 CUDA、Triton、PyTorch 等；题库含 Reduction、Softmax、RMS Normalization、融合残差归一化 | Tensara 不便使用时的替代；按题名选择一题 | 具体题目的语言支持、免费/Pro 边界、测试限制；本次部分页面依赖脚本渲染 |
| [TensorOJ / CANNJudge](https://tensoroj.cn/cann/home) / [Vector 题单](https://tensoroj.cn/cann/public/vector) | 公开题库含 Add、Relu、Softmax、AddRmsNorm；所查题面要求 Ascend C、昇腾 NPU，题目元数据标注 CANN 9.0.0 | **跨硬件选修**：M05 比较方向，或 X 周专做一次 Ascend C 入门 | 实际 NPU 型号、评测方式、版本、提交权限和额度未实测；不把 RTX 4060 当作昇腾运行环境 |
| [GPU MODE](https://www.gpumode.com/) / [Popcorn CLI](https://github.com/gpu-mode/popcorn-cli) | 官方 CLI 支持提交到评测平台；README 指向面向初学者的 PMPP_v2 题集 | L02X 后再选；适合更深入的 kernel 训练 | 先选当前开放题集，检查设备和比赛规则；不要求参加有截止日期的竞赛 |

选择建议：先用 Tensara 的小题确认能运行 Triton；若账号/网络不便，换 LeetGPU。两者都不通时，在 4060 上运行课程自测，记录“OJ 未完成”，继续学习。只有明确想探索昇腾时，才为 TensorOJ 单独安排学习时间。

OJ 评测环境通常围绕单个算子设计。L03–L07 的 vLLM 常驻服务、到达过程、请求延迟与原始日志仍需自己的机器或获批服务器；通过一个 attention 题不能代替服务实验。

## 题目路线：每个阶段只取一项

| 时机 | 练习与入口 | 要学会什么 | 时间与完成边界 |
|---|---|---|---|
| M02 B 课 | [Tensara 题库](https://tensara.org/problems) 的 Vector Addition，或 LeetGPU 同名题 | 根据提交模板适配入口、mask、dtype、设备；读一次真实评测反馈 | 替换 B 课原 135–160 分钟的作业导航，25 分钟；排队/登录未完成就记阻塞 |
| L02 增强 | 本地行 softmax 的参数消融；外部迁移可看 [Tensara Log Softmax](https://tensara.org/problems/log-softmax) | 稳定归约与形状变化；log-softmax 需要改变公式，不能直接提交 softmax 输出 | 消融在本周 5h 内，完整迁移到 OJ 放入 X 周；不以刷榜为验收 |
| X 周 NVIDIA 路线 | [L02X 融合残差＋RMSNorm](labs/L02X-kernel-challenge.md)；LeetGPU 题库搜索 Fused Residual Add and RMS Norm | 独立参考实现、融合、FP32 累加、误差与性能权衡 | 5h 实验，另有两节准备课；先按本课程小规模规格实现，OJ 扩展另计 |
| M05 / X 周 Ascend 路线 | [TensorOJ Relu](https://tensoroj.cn/cann/public/vector/relu) → [Add](https://tensoroj.cn/cann/public/vector/add)，二选一 | 读懂题面、数据搬运、分块与尾部处理；识别广播/多 dtype 增加的复杂度 | M05 只做题面比较；完整实现选 X 周，先跑受限子集，不把它报作全题通过 |
| 后续 kernel 方向 | [Tensara Softmax](https://tensara.org/problems/softmax)、[TensorOJ Softmax](https://tensoroj.cn/cann/public/vector/softmax)、[AddRmsNorm](https://tensoroj.cn/cann/public/vector/addrmsnorm) | 任意归约维度、多 dtype、融合或多核分块 | 在已有可测 baseline 后选一题，估时 5–10h；不是第三周的额外必做 |

上述题名和链接是本次核对结果，不依赖题库显示的行号。题目版本变更时以当前题面和模板为准，保留自己使用的版本记录。

## TensorOJ 的具体使用方案

这是额外编程模型的学习。先读 [Ascend C 官方编程范式入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0035.html)，理解 host 的分块信息怎样传到 device，以及搬入、计算、搬出如何组织。该链接是 8.3 版概念材料；实际 API 必须切到平台指定版本，不能把旧样例原样当作 9.0 可运行代码。

建议先选 Relu。公式简单，便于把精力用在 Ascend C 工具链；但完整题面含多 dtype、动态形状、特殊值，**并不是一个只需改几行的玩具题**。Add 还要求广播，只有同形向量加法通过时不能宣称完成 Add。

如果选择 X 周的 Ascend 路线，两节课各 180 分钟：

- A 课：30 分钟记录题目契约与评测信息，60 分钟读官方分块/搬运范式，休息 10 分钟，50 分钟追踪题目模板的数据流，30 分钟设计本地 CPU 参考与边界用例。
- B 课：45 分钟适配平台模板，45 分钟尝试最小 FP32 输入，休息 10 分钟，50 分钟定位错误与处理尾部，30 分钟固定实验矩阵。若账号或运行入口仍阻塞，保留材料并切回 NVIDIA 路线，勿盲目安装完整 CANN。

随后 5h 实验按 30/90/60/60/30/30 分钟分别完成：冻结规格、实现基线、边界检查、一个分块参数对照、解释结果、整理报告。先用 FP32、正负混合有限数、元素数 {1,31,32,33,255,256,257,4097} 做本地参考；远端若仅允许完整题单，保留它返回的全部结果，不假定能指定这些自定义用例。

交付：`contract.md`、带来源的源码、CPU 参考、评测 ID/日志、题目与 CANN 版本、错误记录、改前改后的对照。完整 OJ 通过须满足全题契约；只实现 FP32 子集时，本课程可以验收“Ascend 入门实验”，OJ 状态仍为“部分完成”。没有 NPU 实测就标为“仅参考/实现草稿”。

TensorOJ 的公开入口与题面已经核对；没有核实具体 NPU 型号、是否支持自定义测试、排队与账号额度。不要根据网页有编辑器或题库就写“随时免费可用”。平台题面若有文字、公式和参考实现不一致的地方，保留最小反例，先查实际 golden/说明，不让 AI 自行猜一个版本。

## 每次第一次提交的 30 分钟检查

1. **0–10 分钟：读契约。** 记输入 shape、dtype、layout、stride、归约轴、epsilon、输出、是否允许库调用、是否允许原位修改。保存题目 URL、日期、版本和入口签名。
2. **10–20 分钟：跑最小正确实现。** 按平台模板提交简单实现，先核对正确性及 GPU/NPU 信息。确认平台是否区分 Run 与正式 Submit；遵守当前题集的 AI/代码公开规则。
3. **20–30 分钟：保存结果。** 保留提交 ID、源码 commit、语言、设备、每个用例结果/耗时和评分定义。未提供的字段写“平台未提供”。不要反推未公开的隐藏测试或将输出硬编码。

这 30 分钟计入所选课程时间块；若 M02 只有 25 分钟，先完成前两步，记录留待 X 周补的内容。平台阻塞时及时转回本地，不用登录调试耗掉整个实验周。

## 同名算子的差异必须写清

- L02 是二维、contiguous、最后一维、FP32 的 softmax。Tensara Softmax 的公开题面要求任意形状和指定维度，直接搬过去并不满足契约；其中某些输入明显大于本课程的本地矩阵。
- Tensara Log Softmax 是逐行归约，但结果应是对数概率。默认公开测试列宽可到 8192；先用本地小形状验证，再讨论远端规模。
- TensorOJ Softmax 的题面还包含多 dtype、正负维度索引等要求；平台的完整规格与教学子集分别记录。
- RMSNorm 是否有可学习权重、epsilon 的位置、残差是否输出及输出 dtype，都不能只靠题名判断。L02X 专门给出一份本地契约。

## OJ 成绩怎样进入实验报告

平台判定只证明它所运行的测试通过。课程还要看你能否解释、构造边界输入、预测修改的结果。同一设备上保存 baseline 和优化版本的三次评测；若额度不足三次，可报告单次探索性结果，但不宣称稳定提速。

保留逐形状耗时，不只贴榜单总分。若平台只给总分，就把它单列为“OJ 分数”，用本地逐形状实验支撑机制分析。不同卡型分别画图；不能用远端 H100 优化版除以本地 4060 baseline 得出算法加速比。编译是否计时、测试是否包括数据搬运、评分是算术/几何平均，都需按实际说明记录。

平台无法提供自定义矩阵、原始耗时或 profiler 时，其结果只能补充现有报告，不能补全 L01/L02 缺失的测量项。模拟器/解释器的正确性通过与真机性能通过分别记录。

## 资料汇总与核对依据

- [TensorOJ 首页](https://tensoroj.cn/cann/home)、[Vector 题单](https://tensoroj.cn/cann/public/vector)、[公开题库数据](https://tensoroj.cn/cann/api/problems/contest/6a9644c4bf41025d606e16b6)：本次通过网页公开读取接口核对题名、题面与版本，未运行评测。
- [Ascend C 官方编程范式入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0035.html)：分块与搬运的概念材料，运行时匹配版本。
- [Tensara 项目说明](https://github.com/tensara/tensara)、[题库](https://tensara.org/problems)、[CLI](https://tensara.org/cli)：能力与使用入口。
- [LeetGPU 主页](https://leetgpu.com/)、[题库](https://leetgpu.com/challenges)、[资料页](https://leetgpu.com/resources)：语言与公开题名；本次部分内容来自搜索索引。
- [GPU MODE Popcorn CLI](https://github.com/gpu-mode/popcorn-cli)、[PMPP_v2 题集](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2)：后续深入练习。
