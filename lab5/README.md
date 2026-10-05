# lab5：分布式训练与调度

用长尾调度模拟练习建模，再运行真实多进程训练，学习梯度同步、集合通信和计算重叠。

## 内容

- [assignment.md](assignment.md)：长尾任务的 barrier 与 FCFS 模拟。
- `tail_sim.py`、`tests.py`：模拟器与守恒检查；[directions.md](directions.md)：研究方向记录。
- [distributed/](distributed/README.md)：Gloo 多进程训练、独立参考与 C++ MPI ring。
- [systems.md](systems.md)：系统任务的验收要求；[research.md](research.md)：通信与训练系统拓展。
- [setup.md](setup.md)、[grade.md](grade.md)、[report.md](report.md)：环境、基础评分与报告。

## 使用顺序

1. 手算小规模调度例子，完成模拟器并检查其假设。
2. 进入 `distributed/`，完成梯度同步，与单进程参考逐步比较。
3. 实现 C++ MPI ring，分析通信量、消息大小与同步边界。
4. 按设备条件继续研究 bucket、重叠或多 GPU 扩展。

调度模拟、CPU 多进程与多 GPU 测量代表不同条件，结果须分别记录。真实多 GPU 实验是扩展任务。
