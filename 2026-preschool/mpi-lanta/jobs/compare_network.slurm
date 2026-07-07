#!/bin/bash
#SBATCH -p compute-devel
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -J mpi-net
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

echo
echo "== Ping-pong: two ranks on one node =="
srun -N 1 -n 2 --ntasks-per-node=2 --cpu-bind=cores ./build/ping_pong --max-bytes 8M --iters 1000 --warmup 100

echo
echo "== Ping-pong: two ranks across two nodes =="
srun -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=cores ./build/ping_pong --max-bytes 8M --iters 1000 --warmup 100

echo
echo "== Halo exchange: eight ranks on one node =="
srun -N 1 -n 8 --ntasks-per-node=8 --cpu-bind=cores ./build/halo_exchange --points-per-rank 500000 --steps 50 --warmup 5

echo
echo "== Halo exchange: eight ranks across two nodes =="
srun -N 2 -n 8 --ntasks-per-node=4 --cpu-bind=cores ./build/halo_exchange --points-per-rank 500000 --steps 50 --warmup 5

echo
echo "== Allreduce: eight ranks on one node =="
srun -N 1 -n 8 --ntasks-per-node=8 --cpu-bind=cores ./build/allreduce_benchmark --max-bytes 8M --iters 100 --warmup 10

echo
echo "== Allreduce: eight ranks across two nodes =="
srun -N 2 -n 8 --ntasks-per-node=4 --cpu-bind=cores ./build/allreduce_benchmark --max-bytes 8M --iters 100 --warmup 10
