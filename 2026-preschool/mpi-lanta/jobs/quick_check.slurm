#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -J mpi-quick
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

echo "== Environment and rank placement =="
srun ./build/hello_environment

echo "== Compute-bound example =="
srun ./build/compute_pi --steps 50000000

echo "== Memory bandwidth example =="
srun ./build/memory_stream --mib-per-rank 64 --iters 5

echo "== Point-to-point communication example =="
srun -n 2 --ntasks-per-node=2 ./build/ping_pong --max-bytes 1M --iters 500 --warmup 50

echo "== Collective communication example =="
srun ./build/allreduce_benchmark --max-bytes 1M --iters 50 --warmup 5

echo "== Neighbor communication example =="
srun ./build/halo_exchange --points-per-rank 500000 --steps 20 --warmup 3
