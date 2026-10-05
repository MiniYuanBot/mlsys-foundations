# 梯度同步与通信重叠

实现多进程梯度同步和 C++ ring all-reduce，测量通信成本并检查计算与通信是否重叠。

主要文件：

- `student.py`：待完成的梯度同步。
- `train.py`：Gloo 训练入口、单进程参考与运行记录。
- `tests.py`：等分和不等分数据的多进程测试。
- `cpp/`：MPI ring 实现入口及构建配置。
- [research.md](research.md)：论文阅读与研究拓展。

建议顺序：

1. 按 [setup.md](setup.md) 准备 PyTorch 和 MPI。
2. 按 [assignment.md](assignment.md) 完成 Gloo 同步与 MPI ring。
3. 通过测试，再测量 bucket、DDP 和通信重叠。
4. 按 [grade.md](grade.md) 自评，选择研究拓展。

主线可在 CPU 上完成。训练入口的计时包含记录开销；性能实验需另写稳态 benchmark。同机 CPU 或单卡多进程结果不能证明多 GPU、跨节点性能。
