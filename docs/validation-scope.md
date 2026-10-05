# 框架验证范围

验证日期：2026-10-05。此页描述提供的脚手架，不是学生lab完成证明；没有生成或宣称任何GPU加速结果。

## 本次实际检查

| 检查 | 结果与含义 |
|---|---|
| `python -m unittest discover -s tests -v` | 11项通过：数据/命令/证据规范、评分状态与层次隔离、B/S/R rubric及模板对应、两仓库本地链接、Python语法 |
| `python -m unittest lab3.runtime.tests.TestProvidedInfrastructure -v` | 2项通过：页池初始状态/几何约束、连续预留容量参考；不代表学生allocator完成 |
| L00原生C++ | 用现有G++以`-std=c++17 -O2 -Wall -Wextra -pedantic`编译`main.cpp student.cpp`成功；`--reference-test`退出0，`--test`因S00-1 TODO退出1，符合起点预期 |
| S/R评分入口 | 8章×2层全部执行，均0分且未通过；S01/S02/S05缺构建程序或PyTorch而标环境阻塞，S00/S03/S04核心TODO失败，S06/S07及R人工证据待补 |
| 课程文档diff | `git diff --check`通过；检查课程与实验文档的格式 |

初始评分摘要位于`results/validation/scaffold-grades.json`，逐项日志位于`results/grades/labN-systems.json`及`labN-research.json`；这些是本机制作时的检查记录，被.gitignore忽略，不是可提交的学生成绩。

## 未实测的部分

- PyTorch训练与真实Gloo进程；本机没有PyTorch，因此未声称分布式正确性通过。
- CUDA/CMake工具链组合、cuBLAS/GPU计时、sanitizer/profiler、Triton编译与streaming attention。
- C++ MPI构建/运行，NCCL、多GPU或跨节点性能。
- vLLM源码构建、模型启动、调度patch、精确token trace的实际服务回放。
- llama.cpp、第三方课程作业及论文完整artifact。

已为这些部分提供独立oracle/测试入口和手动环境指导；源码/契约审阅不能代替目标环境验证。核心TODO留给学习者，初次学生测试失败是设计的一部分；禁止将TODO改skip或用参考入口冒充自己的通过结果。

制作使用现有Python3.13.1和已有G++，没有安装依赖、修改PATH/驱动、下载模型或clone上游。测试临时文件使用本仓库`results/.tmp`。跨平台编译/GPU执行须按各章setup自行完成并保存版本与结果。

## 重验

从仓库根目录执行：

```bash
python -m unittest discover -s tests -v
python -m unittest lab3.runtime.tests.TestProvidedInfrastructure -v
python -m tools.grade lab0 --tier systems
python -m tools.grade lab5 --tier systems
python -m tools.grade lab6 --tier research
```

后面三条在初始状态应非零退出；环境阻塞、TODO失败、缺人工证据均不得分。先按setup完成环境smoke，再完成学生实现，最后进行对应硬件测试/公平测量与证据自评。
