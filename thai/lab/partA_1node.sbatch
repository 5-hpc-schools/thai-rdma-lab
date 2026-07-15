#!/bin/bash
# PART A -- One node, 4 GPUs.  The GPUs talk over NVLink (inside the node).
# This needs NO network, so it starts fast and always works.
#SBATCH -J nccl_A
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o partA_%j.out

module load cuda/12.6
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH

# Show what NCCL does. Look for "via P2P" / "NVL" -> that is NVLink.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

BIN=/project/tn999992-rdma/day3-nccl-lab/build/nccl-tests-mpi/build/all_reduce_perf

# -g 4  = use 4 GPUs.   -b/-e = message size from 8 bytes to 512 MB.
srun $BIN -b 8 -e 512M -f 2 -g 4
