# lab3 环境：独立服务环境与固定模型

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

先按 [全局setup第5节](../setup.md) 自行安装 `.venv-vllm`，与 `.venv-torch` 分离。WSL2/Linux、4060 8GB是教学默认；先运行：

```bash
python -m tools.doctor --require vllm --out results/environment/lab3.json
vllm serve --help > results/environment/vllm-serve-help.txt
vllm bench serve --help > results/environment/vllm-bench-help.txt
```

逐项核对serve的eager/chunked-prefill/prefix-caching/budget，以及bench的save-detailed、ignore-eos、random-range-ratio、burstiness/percentile选项；用安装版本帮助/实现确认单位和语义。不支持某项时先查对应文档，不能静默删开关。

## 固定模型与tokenizer

打开 [Qwen官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，在Files/history选择实际使用的**完整40位commit**。下载方式参考 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。在vLLM环境已有hf命令时先检查 `hf download --help`；缺失时查已安装huggingface_hub版本的入口，不必升级整个服务栈。

下面是需替换占位符的命令示意，由你执行；本仓库没有下载模型：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision YOUR_FULL_40_CHARACTER_COMMIT
```

保留命令、日期、最终snapshot路径和commit。使用下载完成的不可变snapshot，模型与tokenizer都指向该路径；不要指向会改变的main别名。拷贝模板：

```bash
cp lab3/service.example.json lab3/service.json
cp lab3/mapping.example.json lab3/mapping.json
```

在service.json填写model_path/tokenizer_path为**WSL绝对路径**，model_revision/tokenizer_revision为实际完整commit。生成器用本地snapshot路径和served-model-name固定客户端模型名，避免bench重新从未锁定的main加载tokenizer。snapshot不要移除/就地覆盖；跨机器重新填写路径但保持commit。

默认配置：half、context2048、max-num-seqs4、gpu-memory0.70、budget2048、eager=true、chunked=true、prefix=false；生成器只监听127.0.0.1。不要开放公网端口。

## 启动验收

```bash
python -m lab3.commands --config lab3/service.json --out results/lab3/plan
```

在两个WSL终端激活**同一vLLM环境**。A手动运行commands.txt中的server行；B等readiness，按 [任务说明](assignment.md) 执行curl检查、8次预热和20次smoke请求，再决定是否开始120请求。不要用doctor成功代替模型启动成功。

OOM按根setup的统一降级顺序处理，必要时同时改long_input=768/context1024，保存新配置并重做全组C。不得只缩小某个表现差的条件。不能启动时CPU侧指标练习可继续，真实服务分保持待补/阻塞。

