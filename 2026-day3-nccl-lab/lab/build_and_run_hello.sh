#!/bin/bash
# PART E helper -- build your nccl_hello.cu and run it on 4 GPUs.
# Usage (inside a GPU job, e.g. `srun ... bash build_and_run_hello.sh`):
#   or:  salloc -p gpu -N1 --gpus-per-node=4 -t 00:10:00
#        srun bash build_and_run_hello.sh
set -e
module load cuda/12.6 >/dev/null 2>&1 || true
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl
export LD_LIBRARY_PATH=$SDKNCCL/lib:$LD_LIBRARY_PATH
SRC=${1:-nccl_hello.cu}
echo "building $SRC ..."
nvcc -arch=sm_80 -I$SDKNCCL/include "$SRC" -o nccl_hello -L$SDKNCCL/lib -lnccl
echo "running ..."
./nccl_hello
