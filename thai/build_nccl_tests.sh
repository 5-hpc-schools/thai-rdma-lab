#!/bin/bash
# Build the NCCL benchmark (nccl-tests) that ACTUALLY WORKS on LANTA.
#
# WHY this exact recipe (learned the hard way, 2026-07-15):
#   * LANTA's module `nccl/2.18.1-1+cuda11.0` is a CUDA-11 build. Mixing it with
#     a CUDA-12 nccl-tests crashes. Use the CUDA-12 NCCL 2.18.5 that ships inside
#     the NVHPC SDK instead (comm_libs/12.6/nccl) -> everything is CUDA 12.6.
#   * The CURRENT nccl-tests (v2.19.6+) is INCOMPATIBLE with NCCL 2.18.x: even a
#     1-GPU all_reduce dies with "invalid argument" / "illegal memory access".
#     Use the 2023-era nccl-tests commit e98ef24 (matches the NCCL 2.18 era).
#   * Build with MPI=1 (cray-mpich) so multi-node runs can bootstrap.
#
# Result: build/nccl-tests-mpi/build/all_reduce_perf  (+ all_gather_perf, etc.)
set -e
STG=${1:-/project/tn999992-rdma/day3-nccl-lab}
mkdir -p $STG/build && cd $STG/build

module load cuda/12.6                      # CUDA 12.6 (nvcc, runtime)
SDKNCCL=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl   # NCCL 2.18.5+cuda12

if [ ! -d nccl-tests-mpi ]; then
  git clone https://github.com/NVIDIA/nccl-tests.git nccl-tests-mpi
  cd nccl-tests-mpi
  git checkout -q e98ef24bc03bef33054c3bc690ce622576c803b6   # 2023-05, matches NCCL 2.18
  cd ..
fi
cd nccl-tests-mpi
make clean >/dev/null 2>&1 || true
make MPI=1 MPI_HOME=$CRAY_MPICH_DIR NCCL_HOME=$SDKNCCL CUDA_HOME=$CUDA_HOME -j8
echo "OK: $(ls -la build/all_reduce_perf)"
