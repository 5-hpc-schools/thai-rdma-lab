#!/bin/bash
# PART D -- Look INSIDE NCCL.  One node, 4 GPUs (NVLink).
# Three small experiments: (1) protocols, (2) ring vs tree, (3) the topology map.
#SBATCH -J nccl_D
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:12:00
#SBATCH -o partD_%j.out

module load cuda/12.6
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH
BIN=/project/tn999992-rdma/day3-nccl-lab/build/nccl-tests-mpi/build/all_reduce_perf
pick(){ awk '/^ *(1048576|67108864) +[0-9]+ +float/{printf "  size=%-9s busbw=%s GB/s\n",$1,$12}'; }

echo "############ (1) PROTOCOLS : Simple vs LL vs LL128 ############"
for P in Simple LL LL128; do
  echo "-- NCCL_PROTO=$P --"
  NCCL_ALGO=Ring NCCL_PROTO=$P srun $BIN -b 1M -e 64M -f 8 -g 4 2>/dev/null | pick
done

echo; echo "############ (2) ALGORITHM : Ring vs Tree ############"
for A in Ring Tree; do
  echo "-- NCCL_ALGO=$A --"
  NCCL_ALGO=$A srun $BIN -b 1M -e 64M -f 8 -g 4 2>/dev/null | pick
done

echo; echo "############ (3) TOPOLOGY : what NCCL discovered ############"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH srun $BIN -b 8 -e 8 -g 4 2>&1 \
 | grep -E "NVL\[|nChannels|Ring [0-9]|Trees|NVLS|Setting affinity" | head -12
echo DONE
