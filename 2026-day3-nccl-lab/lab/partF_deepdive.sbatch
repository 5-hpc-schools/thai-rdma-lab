#!/bin/bash
# PART F -- Turn on the FULL NCCL log and watch it set itself up.
# NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=ALL prints everything NCCL does.
# One node, 4 GPUs. The output is long (~7000 lines) -- that is the point.
#SBATCH -J nccl_F
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:08:00
#SBATCH -o partF_%j.out

module load cuda/12.6
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH
BIN=/project/tn999992-rdma/day3-nccl-lab/build/nccl-tests-mpi/build/all_reduce_perf

# THE two magic switches. SUBSYS=ALL = show every part of NCCL.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# small size: we only care about the SETUP log, not speed.
srun $BIN -b 8 -e 1M -f 4 -g 4 2>&1
echo DONE
