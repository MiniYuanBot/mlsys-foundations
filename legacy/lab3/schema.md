# 服务结果中间格式

vLLM 原生 JSON 随版本变化，`adapter.normalize` 是你需要完成的版本适配练习。保留原文件；先读实际 smoke 输出、对应版本实现和 `--help`，不要按字段名猜单位。

规范示例见 [synthetic.json](fixtures/synthetic.json)，仅用于单元测试。真实文件必须 `synthetic: false`：

- 顶层：`run_id`、`duration_s>0`、`num_requested`、`requests`。
- 每请求：唯一 `request_id`，布尔 `success`，实际 `input_tokens`/`output_tokens`，`ttft_s`、`tpot_s`、`error`。
- 成功请求必须保存实际长度和 TTFT；输出数 >1 才有 TPOT。输出数 =1 时 `tpot_s=null`，不能填零加入均值/分位数。
- 失败请求仍占一行，error 写真实原因，工具未提供时写“not provided by tool”；缺失计时写 null，不能伪造成零延迟。
- 所有发出请求必须可核对；部分成功数组与全请求数组不能直接 zip。缺少逐请求数据时先修采集或补日志，未恢复前不声称 SLO 验收通过。
- duration 记录为工具测量时段，并在映射说明中注明启动/排空边界。
- 本仓库 percentile 使用线性插值；小样本 p95 是探索性结果。
- 输出吞吐用成功完成输出 token 的总和除时段；失败若有部分输出，原始文件保留，该课程口径不计入完成输出吞吐。

字段映射填 [mapping.example.json](mapping.example.json) 的副本，补来源版本/文件、字段、单位、请求对齐与 TPOT 定义。如工具只给总延迟而没有已核实的 token 时刻口径，不要随意用 (E2EL-TTFT)/(n-1) 代替。允许增添 request-level 源字段以便审计。

lab3 合成题的手算：4 次请求/2 秒，成功 3、失败 1、输出总数 7，成功吞吐 1.5 req/s，输出吞吐 3.5 token/s；TPOT 仅 0.1/0.3 秒两个值。lab4 在 TTFT≤300ms 且 TPOT≤200ms 时只有第一个满足：满足率 1/4，goodput 0.5 req/s。

