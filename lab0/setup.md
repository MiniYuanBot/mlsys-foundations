# lab0 环境

基础练习的环境与评分见本目录；系统任务见 [systems.md](systems.md)，研究拓展见 [research.md](research.md)。

CPU即可；不下载数据集、不需要CUDA。本章前置为 [全局setup第3节](../setup.md) 的PyTorch环境，作图用requirements-analysis.txt。

1. 在选定的Mac、Windows原生或WSL环境激活 `.venv-torch`。
2. 在根目录确认解释器并检查：
```bash
python -c "import sys, torch; print(sys.executable, torch.__version__)"
python -m tools.doctor --require torch --out results/environment/lab0.json
```
3. ready表示CPU梯度有限；随后 `python -m unittest lab0.tests -v` 会因TODO失败，补实现后应通过。
4. 不装vLLM/Triton也可完成本章。CUDA未就绪在周报记录，不扣CPU实验的GPU证据分（本章没有此项）。
5. 保存freeze与安装命令。跨设备重跑分别记录，不比较Mac与4060速度。

排错先用4样本检查shape/dtype/loss，确认更新量，再做32样本过拟合。长时间环境阻塞用[周报](../templates/weekly-report.md)记录。

