# 基础练习：vLLM 服务与容量测量

运行环境：CPU 分析 + WSL2/CUDA 服务。以下命令均在项目根目录执行。

## 实现任务

先完成 [setup](setup.md)。已有命令生成器（只生成文本）、环境/日志采集、请求规范、合成反例与容量汇总入口。你补两块：

- `adapter.normalize`：检查**实际安装版本**的原始JSON、单位与数组对齐，转换[请求规范](schema.md)。这部分没有假定永久不变的字段名。
- `metrics.summarize`：成功/失败、实际输出、吞吐、p50/p95 TTFT/TPOT。percentile工具已给出。先手算fixture再编码。

```bash
python -m unittest lab3.tests -v
python -m lab3.metrics lab3/fixtures/synthetic.json --out results/lab3/synthetic-summary.csv
```

合成题不是服务成绩。真实adapter需额外用20请求smoke人工核对；映射证据是评分项。

## 服务运行与测量

复制并填好service.json后生成：
```bash
python -m lab3.commands --config lab3/service.json --out results/lab3/plan
```

按 `results/lab3/plan/commands.txt` 操作，**不要整体执行文件**。两个WSL终端，server命令在A运行，B等启动完成后：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Explain a GPU in one sentence:","max_tokens":32,"temperature":0}'
```

普通请求允许EOS，只证明协议可用；不证明固定长度工作量。随后逐行执行生成的L03-warmup（8）、lab3-smoke（20），核对后执行capacity-r1/r2/r3（各120，seed11/22/33）。不重复执行额外通用warmup也可以：L03矩阵自身包含首次warmup；重启后另补8请求预热。前一轮排空后才能下一轮。

默认目标1024输入/64输出、ignore_eos、rate=inf、random-range-ratio0。以实际长度为准；输出不全为64或有失败先修复并另编号重跑，保留原失败目录。

完成adapter并复制填写mapping.json，然后每轮转换：
```bash
python -m lab3.adapter --raw results/lab3/plan/runs/lab3-capacity-r1/raw.json --mapping lab3/mapping.json --out results/lab3/plan/runs/lab3-capacity-r1/normalized.json
# 对r2/r3分别替换文件名运行同样的命令；smoke也要核对
python -m lab3.metrics results/lab3/plan/runs/lab3-capacity-r1/normalized.json results/lab3/plan/runs/lab3-capacity-r2/normalized.json results/lab3/plan/runs/lab3-capacity-r3/normalized.json --out results/lab3/summary.csv
python -m lab3.calibrate results/lab3/plan/runs/lab3-capacity-r1/normalized.json results/lab3/plan/runs/lab3-capacity-r2/normalized.json results/lab3/plan/runs/lab3-capacity-r3/normalized.json --out results/lab3/calibration.json
```

C为三轮成功吞吐中位数，只是固定负载/有限批次容量代理。核对三轮模型/配置/设备/实际输入一致，校准函数不代替此人工审计。TTFT含排队/网络，不等于prefill计算。失败/缺失请求不得消失。

交付原始JSON、逐请求规范、映射、帮助、启动日志、revision、全部命令、summary/C、[报告](report.md)。增强30分钟：在给定四请求题外自己增加一个反例并核对，不把synthetic混进真实图。
