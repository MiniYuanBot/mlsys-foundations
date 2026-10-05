---
title: lab3｜建立可重复的单卡 vLLM 服务基线
tags: [mlsys, lab, serving]
weeks: [6, 7]
status: not-started
---

本页为基础学习材料，系统任务与周次见 [课程安排](../syllabus.md)。

# lab3｜建立可重复的单卡 vLLM 服务基线

[索引](../guides/course-map.md) · [对应课程 lab3](../modules/lab3-serving.md) · [环境手册](../guides/environment.md)

> [!tip] AI 辅助增强
> 本周可选：用可手算的反例检查指标脚本（30 分钟）。具体步骤与验收见 [增强实验手册](../guides/ai-assisted-labs.md)。原实验提前完成时使用剩余时间，未完成则顺延；不额外叠加到本周 5h。

## 目标与约束

在 4060 8 GB 上运行一个 0.5B 模型服务，确认请求长度、错误与延迟能被记录，为后续实验建立同一基线。模型：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。不要求下载更大的模型、不要求量化，也不将生成文本质量当作本实验指标。

需要：[vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)、[bench CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)。命令对应核对时的文档；实际运行前必须保存安装版本的 `--help`，尤其核对随机长度参数的语义。

## 300 分钟执行表

| 时间 | 任务 |
|---|---|
| 0–40 | 检查独立环境、驱动、模型文件与版本；完成环境表 |
| 40–80 | 启动服务，检查实际配置和一个普通请求 |
| 80–130 | 跑 8 个预热请求与 20 个测量请求，读懂 JSON 字段 |
| 130–210 | 完成三轮容量初测，核对成功数和实际输出长度 |
| 210–270 | 写结果转换/汇总小脚本，计算容量估计与延迟摘要 |
| 270–300 | 整理 README、日志、限制与周报 |

## 步骤 1：启动与 smoke test

在项目根目录创建 `results/lab3`。使用 [环境手册](../guides/environment.md) 的完整启动命令，保留服务端 stdout/stderr；一个终端运行服务，另一个终端运行客户端。关键固定项为 FP16、上下文 2048、max-num-seqs=4、token budget=2048、chunked prefill 开启、prefix caching 关闭、eager 开启、显存比例 0.70。

先检查服务：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain a GPU in one sentence:","max_tokens":32,"temperature":0}'
```

这个请求只检查服务和协议；可能提前 EOS，不能拿它证明 32 token 的固定工作量。若失败，先查服务端启动是否完成、模型名/端口是否一致，再查显存。不要直接开始 120 请求的测试。

## 步骤 2：固定长度基准

以下是三轮容量初测中第一轮的命令。先将请求数改成 8 运行一次预热，文件名标 `warmup`；再以 20 个请求做 smoke 测量，确认所有字段后才运行正式 120 请求。预热结果不混入正式统计。

```bash
vllm bench serve \
  --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 --endpoint /v1/completions \
  --dataset-name random --random-input-len 1024 \
  --random-output-len 64 --random-range-ratio 0 \
  --num-prompts 120 --request-rate inf --ignore-eos \
  --seed 11 --save-result --save-detailed \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95 \
  --result-dir results/lab3 --result-filename capacity-r1.json
```

后两轮 seed 为 22、33，文件名为 r2、r3。不要并行运行三个 benchmark，也不要在前一轮请求未排空时开始下一轮。每次模型重启后重新预热。`inf` 在这里只用于容量初测，后续有限到达率实验另行设置。

当前文档中的 `random-range-ratio=0` 表示固定目标长度；旧版本语义可能不同。请核对帮助和**实际输入长度分布**。随机 token 解码/重编码等路径可能改变实际长度，发现偏差先查数据生成实现，保存目标与实际长度。不能仅凭命令参数声称每个输入恰好 1024 token。

## 步骤 3：理解原始字段

先用 Python 打印结果 JSON 的顶层键、数组长度及一个请求的数据。不同版本字段名可能变，建立你自己的映射表：原字段名、单位、是否请求级、目标 CSV 字段。以对应版本实现为准，不能靠字段名猜秒还是毫秒。

汇总表至少有：`run_id,seed,num_requested,num_success,num_failed,duration_s,total_output_tokens,request_throughput,output_throughput,p50_ttft_ms,p95_ttft_ms,p50_tpot_ms,p95_tpot_ms`。保留原 JSON，不用转换后的 CSV 代替原始数据。逐请求长度和失败明细也要保留，工具未提供的字段标“未提供”，不要补造。

用简单例题验证脚本的单位：两请求 TTFT 为 0.1 与 0.3 秒时，毫秒表示分别为 100 与 300；100 个成功请求在 20 秒内完成，请求吞吐为 5 req/s。真实指标定义见 [实验规范](../guides/experiments.md)。

## 步骤 4：容量估计

令三轮成功请求吞吐的中位数为 C（req/s），保存到 `calibration.json`，同时保存三轮值、范围与负载长度。这个 C 是当前有限批次、固定长度、软件配置下的**粗略容量代理**，不是硬件最大吞吐或稳定在线容量的证明。

观察三轮是否差异很大、是否有失败、输入输出长度是否符合设定。若失败数不为零，先完成稳定基线，不拿失败运行的高吞吐作为容量。不要把下一周短输入的容量也直接认作 C；C 只作为统一选择测试到达率的参照。

## 交付与验收

交付：服务配置和启动日志、安装版本帮助、模型 revision、预热/smoke/三轮原始 JSON、字段映射、汇总脚本、`calibration.json`、README。

- [ ] 基准正式三轮所有请求成功；存在失败则定位后重跑并保留旧日志。
- [ ] 固定输出实际为 64 token；不符合时先检查 ignore_eos 与工具计数。
- [ ] 实际输入长度分布已核对，未掩盖偏差。
- [ ] C 可从原始 JSON 重算，TTFT/TPOT 单位正确。
- [ ] 所有轮次使用同一模型、配置、设备与缓存设置。

OOM 按 [降规模顺序](../guides/environment.md) 处理；降级后全部基准重做，后面课程沿用新配置。安装/下载阻塞超过本周预算时，保留环境报告并将本周延长，不用理论吞吐代替实测。

资料回查：[bench CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) · [serve CLI](https://docs.vllm.ai/en/stable/cli/serve/) · [模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
