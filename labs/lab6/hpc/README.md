# 可选：在北大 HPC 的多 GPU 节点上测 NCCL

> ⚠️ **未验证**：作者没有 HPC 账号，下面的脚本没有实际运行过。分区名、模块名、CUDA 版本都要按 [北大高性能计算平台](https://hpc.pku.edu.cn/) 的文档修改。第一次运行请先申请一个短的交互作业（`salloc`）逐步调试。

本机只有一张 GPU，NCCL 的多卡通信只能在 HPC 上测。目标是把 lab6 的 α-β 模型与 ring 算法的结论，搬到 NVLink 或 PCIe 互联的真实 GPU 上验证。

## 1. nccl-tests：标准的 GPU 集合通信基准

```bash
# on a login node
git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
module load cuda/12.x nccl openmpi        # names differ per cluster: `module avail`
make MPI=1 MPI_HOME=$(dirname $(dirname $(which mpicc))) CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME -j
```

提交 `nccl_allreduce.sbatch`（在同一目录）：

```bash
sbatch lab6/hpc/nccl_allreduce.sbatch
```

输出中有 `algbw` 和 `busbw` 两列，其定义与 lab6 的 `bench_coll` 相同。

## 2. 要回答的问题

1. 同一台机器上 2、4、8 张卡的 busbw 是多少？与 `nvidia-smi topo -m` 显示的拓扑（NVLink、PCIe、跨 NUMA 节点）对得上吗？
2. 用 `NCCL_ALGO=Ring` 和 `NCCL_ALGO=Tree` 分别运行，比较二者的交叉点；再与你在 CPU 上测得的 ring 和 recursive doubling 的交叉点对比。
3. 用 1 MB 以下消息的耗时拟合 α，与 CPU 上的 α 相比，差了几个数量级？这对 GPU 训练中的 bucket 大小意味着什么？（PyTorch DDP 默认的 bucket 是 25 MB。）
4. 跨节点（`--nodes=2`）时 busbw 下降了多少？这就是为什么张量并行通常限制在节点之内。

## 3. 进一步：用 NCCL 写 DDP

把 lab6 Part C 的 `CommThread` 换成在一个单独的 CUDA stream 上调用 `ncclAllReduce`，把 MLP 换成 cuBLAS 上的实现，这正是 PyTorch DDP 的结构。也可以直接用 [picotron](https://github.com/huggingface/picotron) 或 [torchtitan](https://github.com/pytorch/torchtitan)，在多卡上复现 DP、TP、PP 的吞吐对比。
