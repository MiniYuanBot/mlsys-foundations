# 请求重放与调度观测实验

预计 8–12 小时。完成基础测量后，沿引擎调用链加入可开关的观测，保存真实请求 trace 并进行对照实验。

已给符号prefix与Poisson到达时间生成器、replay/materialize接口和审计测试；你补真实tokenizer/API适配、非阻塞绝对时间到达、timeout/drain、raw事件保存与send-lag审计。可直接适配固定commit的vLLM/SGLang benchmark客户端，保留来源与补丁。协议部分可复用现有客户端。

任务：
1. 先沿lab3阅读的scheduler→KV manager调用链选择一处调度决策，增加有开关的iteration trace：timestamp、request id、scheduled token count、waiting/running、配置；实际字段以固定源码为准。
2. 保存同一trace供A/B重放。prefix overlap=0/80%，cache off/on；固定chunk/budget、dtype、N、到达时间、实际输入输出长度。每条件seed11/22/33，共12轮，**先在每个cache条件定义cold reset与warmup协议**。
3. 实现materialize：保存精确token IDs，验证共享前缀相同且长度不漂移。符号prefix_key不能直接当真实cache命中。
4. 比较无patch、patch但logging off、logging on三个短诊断条件的开销，正式计时用最小观测或logging off，trace只用于机制诊断。
5. 为一个排队/缓存现象提出候选机制：命中减少prefill工作？KV压力改变？客户端排队？用iteration trace和失败长度审计排除至少一个替代解释。

```bash
python -m lab4.trace.workload --seed 11 --rate 1.2 --shared 0.8 --out results/lab4/trace11.json
python -m unittest lab4.trace.tests -v
```
命令中的 rate=1.2 仅作示例，实际使用基础实验测得的低负载 rate，统一固定 N=120。生成器不发送请求；学生replay写好后自行增加CLI并在报告记录准确入口。还需完成实际请求重放与引擎诊断。

阅读 [SGLang](https://arxiv.org/abs/2312.07104) 的RadixAttention和cache-aware调度，再回看 [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)。研究拓展任选一个：prefix流行度偏斜/缓存污染下的SLO；或长度感知调度中的公平性与饥饿。报告中应区分局部机制验证与完整系统复现。

交付：上游repo+commit、调用链、可应用diff、编译/单元测试、tokenized trace、12轮raw、send-lag表、patch开销控制、逐轮SLO/失败/公平性、至少一张机制trace。环境复用 lab3，见 [setup.md](setup.md)；评分见 [grade.md](grade.md) 和 `../systems.rubric.json`。
