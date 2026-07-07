#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -J mpi-bind
#SBATCH -A ltXXXXXX
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module reset
module load cpeCray/25.03
module list
./scripts/record_context.sh

make CXX=CC

echo "== Allocation and rank placement =="
srun ./build/hello_environment

for bind in cores sockets none; do
    echo
    echo "============================================================"
    echo "CPU binding: ${bind}"
    echo "============================================================"

    echo "== Memory bandwidth =="
    srun --cpu-bind="${bind}" ./build/memory_stream --mib-per-rank 64 --iters 8

    echo "== Collective latency/bandwidth =="
    srun --cpu-bind="${bind}" ./build/allreduce_benchmark --max-bytes 1M --iters 100 --warmup 10
done

for distribution in block:block cyclic:cyclic; do
    echo
    echo "============================================================"
    echo "Task distribution: ${distribution}"
    echo "============================================================"

    srun --distribution="${distribution}" --cpu-bind=cores ./build/halo_exchange --points-per-rank 500000 --steps 50 --warmup 5
done
