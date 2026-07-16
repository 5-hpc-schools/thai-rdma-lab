#!/bin/bash
# PART B -- Two nodes.  Now the GPUs on different nodes must talk over the
# Slingshot network.  NCCL uses the "cxi" provider = RDMA (GPUDirect).
# Look in the log for:   Selected provider is cxi   and   via NET/OFI ... GDRDMA
#
# NOTE the --network line: it asks Slurm for a Slingshot network channel (VNI).
# Without it, the network cannot start.
#SBATCH -J nccl_B
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH --network=job_vni,single_node_vni
#SBATCH -o partB_%j.out

module load cuda/12.6
module load aws-ofi-nccl/1.6.0+cuda12.6
source /project/tn999992-rdma/day3-nccl-lab/scripts/env_slingshot.sh

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

BIN=/project/tn999992-rdma/day3-nccl-lab/build/nccl-tests-mpi/build/all_reduce_perf

# 2 nodes x 1 GPU each = 2 ranks, so the data must cross the network.
srun -N2 --ntasks=2 --ntasks-per-node=1 $BIN -b 8 -e 256M -f 2 -g 1
