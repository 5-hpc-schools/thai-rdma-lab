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

make CC=cc

echo "== Slide Unit 2: MPI hello and rank placement =="
srun ./build/hello_mpi

echo "== Hands-on Exercise 1: rank-dependent control flow =="
srun ./build/rank_control

echo "== Slide Unit 2: cyclic work distribution =="
srun ./build/work_distribution --items 24 --mode cyclic

echo "== Slide Unit 2: block-balanced work distribution =="
srun ./build/work_distribution --items 24 --mode block

echo "== Hands-on Exercise 2: serial Pi baseline =="
srun -n 1 ./build/pi_serial --steps 20000000

echo "== Hands-on Exercise 2: MPI Pi with MPI_Reduce =="
srun ./build/pi_mpi --steps 50000000

echo "== Slide polynomial example: serial baseline =="
srun -n 1 ./build/poly_serial --input input/poly.dat

echo "== Slide polynomial example: MPI_Bcast + MPI_Reduce =="
srun ./build/poly_mpi --input input/poly.dat

echo "== Point-to-point communication example =="
srun -n 2 --ntasks-per-node=2 ./build/ping_pong --max-bytes 1M --iters 500 --warmup 50

echo "== Collective communication example =="
srun ./build/collective_cost --max-bytes 1M --iters 50 --warmup 5

echo "== Neighbor communication example =="
srun ./build/halo_exchange --points-per-rank 500000 --steps 20 --warmup 3
