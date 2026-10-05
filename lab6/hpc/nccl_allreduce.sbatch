#!/bin/bash
# UNVERIFIED template: adjust partition, account, modules and paths for PKU HPC.
#SBATCH --job-name=nccl-allreduce
#SBATCH --partition=gpu            # <- your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8        # one MPI rank per GPU
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=nccl-%j.out

module load cuda nccl openmpi      # <- `module avail` to find the right names

NCCL_TESTS=${NCCL_TESTS:-$HOME/nccl-tests}
nvidia-smi topo -m

for g in 2 4 8; do
  for algo in Ring Tree; do
    echo "=== GPUs=$g NCCL_ALGO=$algo"
    NCCL_ALGO=$algo srun --ntasks=$g --gpus=$g \
      $NCCL_TESTS/build/all_reduce_perf -b 8 -e 256M -f 4 -g 1 -c 0 -n 20
  done
done
