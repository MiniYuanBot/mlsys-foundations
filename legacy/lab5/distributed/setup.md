# 手动配置与执行指导

本目录不安装 PyTorch、MPI、CUDA、NCCL、Nsight，也不自动 clone 第三方工程。以下命令由你在选好的环境执行。默认 Linux / WSL2 Ubuntu；原生 Windows 的 distributed 能力随 PyTorch 构建而不同，优先用 Linux 路径统一实验。

## CPU / Gloo

1. 按项目根目录 `setup.md` 建独立虚拟环境，选择 Python 3.10+ 和支持该 Python 的 CPU PyTorch；本实验不需要模型与数据下载。
2. 到 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择 OS、CPU wheel 与对应命令，自行安装；不要混用另一个环境的 `torchrun`。
3. 检查以下输出，两项 distributed/Gloo 应为 `True`，并把 PyTorch 版本写入报告：

```bash
python -c "import torch; import torch.distributed as d; print(torch.__version__, d.is_available(), d.is_gloo_available())"
python -m torch.distributed.run --help
```

本地 rendezvous 需要 localhost 端口可用。遇到防火墙或主机名解析错误时，先复现 world_size=1，再按 [torch.distributed 官方说明](https://docs.pytorch.org/docs/stable/distributed.html) 检查 network interface；不要随意关闭防火墙。所有训练入口应从仓库根目录执行。单个 collective timeout 为 20 秒，测试外层为每次启动 75 秒；慢速首次导入可调整 tests.py 的 timeout 并记录理由，不能因超时把失败标为通过。

## C++17 / MPI

在你自己的 Linux / WSL2 环境中准备 C++17 编译器、CMake ≥3.18 和一种 MPI 实现（Open MPI 或 MPICH，开发 headers 与运行时必须匹配）。由你通过所用发行版包管理器安装，先确认 `mpicxx --version`、`mpiexec --version`、`cmake --version`。不要为了这个 CPU 实验安装 CUDA。

```bash
cmake -S lab5/distributed/cpp -B build/lab5-mpi -DCMAKE_BUILD_TYPE=Release
cmake --build build/lab5-mpi --parallel
ctest --test-dir build/lab5-mpi --output-on-failure
mpiexec -n 1 build/lab5-mpi/ring_allreduce
mpiexec -n 2 build/lab5-mpi/ring_allreduce --reference
mpiexec -n 2 build/lab5-mpi/ring_allreduce
mpiexec -n 3 build/lab5-mpi/ring_allreduce
```

`--reference` 只确认 MPI 对照能运行，不计入学生实现通过；核心 TODO 会 `MPI_Abort` 以便全部 rank 退出。CTest 每个测试 20 秒超时。若机器只有两个 CPU 逻辑核，第三个进程需要按所选 MPI 实现的文档配置 oversubscribe，记录此事实，勿把它当 scaling 数据。

## 可选多 GPU：原仓库 NCCL benchmark

需至少两块受 NCCL 支持的 NVIDIA GPU、Linux、兼容的 driver/CUDA/NCCL 开发库。先按官方 README 核对该 commit 的构建要求。下面固定到核验过的 [nccl-tests 2.20.0 commit](https://github.com/NVIDIA/nccl-tests/commit/b4d5beebca8a76cf01335f724d154b9b9d394d96)，**未在本机实测**；如实验室的软件栈不兼容，选兼容 commit 并在报告记录实际 SHA 和理由。

```bash
git clone https://github.com/NVIDIA/nccl-tests.git external/nccl-tests
git -C external/nccl-tests checkout --detach b4d5beebca8a76cf01335f724d154b9b9d394d96
git -C external/nccl-tests rev-parse HEAD
make -C external/nccl-tests CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
nvidia-smi topo -m
external/nccl-tests/build/all_reduce_perf -b 8 -e 64M -f 2 -g 2
```

保持 external clone 独立；不把它整体复制进课程仓库。上面 `-g 2` 是单进程多 GPU；跨节点要按官方说明 `make MPI=1 ...` 并使用 MPI launcher。保留完整 stdout、correctness 错误计数、GPU 型号/互联拓扑、版本、命令、SHA；无数据校验通过不能进入性能比较。记录原始输出的时间单位再画图，不能把不同版本输出的时间列默认为同一单位。

分别做无 profiler 的 3 次重复计时，以及带 profiler 的短诊断运行。安装了支持这些选项的 Nsight Systems 后可用：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output=results/lab5/nccl-trace external/nccl-tests/build/all_reduce_perf -b 1M -e 1M -g 2
```

NCCL 的专用 trace 选项依 Nsight/NCCL 版本变化；以 [Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 与本机 `nsys profile --help` 为准。只有 Nsight trace 不能替代未插桩计时。单 GPU 机器到这里应标记 `resource-blocked: multi-GPU`，保留 CPU 主线成绩。
