#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J mpi-tools
#SBATCH -A ltXXXXXX
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

mkdir -p logs

toolchains=(${TOOLCHAINS:-"cpeCray/25.03 cpeGNU/25.03 cpeIntel/25.03"})

for toolchain in "${toolchains[@]}"; do
    safe_toolchain="${toolchain//\//-}"
    build_dir="build/${safe_toolchain}"

    echo
    echo "============================================================"
    echo "Toolchain: ${toolchain}"
    echo "============================================================"

    module purge
    if ! module load "${toolchain}"; then
        echo "Skipping ${toolchain}; module could not be loaded."
        continue
    fi
    module list
    ./scripts/record_context.sh

    make CC=cc BUILD_DIR="${build_dir}" CFLAGS="-O3 -std=c11 -Wall -Wextra -pedantic"

    echo "== Rank placement for ${toolchain} =="
    srun "${build_dir}/hello_mpi"

    echo "== Compute kernel for ${toolchain} =="
    srun "${build_dir}/pi_mpi" --steps 100000000

    echo "== Memory kernel for ${toolchain} =="
    srun "${build_dir}/memory_stream" --mib-per-rank 64 --iters 8

    echo "== Collective kernel for ${toolchain} =="
    srun "${build_dir}/collective_cost" --max-bytes 1M --iters 50 --warmup 5
done
