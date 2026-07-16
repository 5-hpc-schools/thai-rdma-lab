#!/bin/bash
# PART C -- Same two nodes, but we FORCE NCCL to use plain TCP instead of RDMA
# (NCCL_NET=Socket).  Same wires, but now the kernel copies every byte.
# Compare the bus bandwidth here with Part B: RDMA should be much faster.
#SBATCH -J nccl_C
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -o partC_%j.out

module load cuda/12.6
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH

# Force the slow path: plain TCP sockets over the Slingshot host NIC.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

BIN=/project/tn999992-rdma/day3-nccl-lab/build/nccl-tests-mpi/build/all_reduce_perf

srun -N2 --ntasks=2 --ntasks-per-node=1 $BIN -b 8 -e 256M -f 2 -g 1
