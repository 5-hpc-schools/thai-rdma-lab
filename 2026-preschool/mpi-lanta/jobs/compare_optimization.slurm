#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -J mpi-opt
#SBATCH -A ltXXXXXX
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module reset
module load cpeCray/25.03
module list
./scripts/record_context.sh

for opt in O0 O2 O3 Ofast; do
    build_dir="build/opt-${opt}"

    echo
    echo "============================================================"
    echo "Optimization: -${opt}"
    echo "============================================================"

    make CC=cc BUILD_DIR="${build_dir}" CFLAGS="-${opt} -std=c11 -Wall -Wextra -pedantic"

    echo "== Compute-bound sensitivity =="
    srun "${build_dir}/pi_mpi" --steps 100000000

    echo "== Memory-bandwidth sensitivity =="
    srun "${build_dir}/memory_stream" --mib-per-rank 64 --iters 8
done
