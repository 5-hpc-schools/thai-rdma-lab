# MPI in C on LANTA HPC

This module accompanies the Mohr HPC/MPI slides in `~/slides` and adapts the hands-on flow to LANTA. The code is intentionally C, not C++, so learners practice the same pointer, buffer, datatype, and explicit communication habits they will need for the RDMA training next week.

Start with [LANTA workflow notes](LANTA_WORKFLOW.md) if this is your first time using the command line, transferring files, loading modules, or submitting Slurm jobs on LANTA.

## Slide Alignment

| Slide deck | Tutorial focus |
|---|---|
| `ASEAN2026-0-mohr-parallel101.pdf` | parallel terminology, performance, strong/weak scaling, distributed memory |
| `ASEAN2026-1-mohr-mpi101.pdf` | SPMD, ranks, communicators, MPI messages, collectives, work distribution, polynomial example |
| `ASEAN2026-3-mohr-handson.pdf` | cluster access, job scripts, MPI hello, Pi exercise |

## Exercise Sequence

| Step | Program | Slide topic | What learners practice |
|---|---|---|---|
| 1 | `hello_mpi` | MPI basic routines | `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, rank placement |
| 2 | `rank_control` | SPMD control flow | rank 0 does one thing, all ranks do another |
| 3 | `work_distribution` | cyclic and block-balanced work | mapping loop iterations to ranks |
| 4 | `pi_serial` | serial baseline | one-process midpoint integration |
| 5 | `pi_mpi` | MPI collective reduction | cyclic work plus `MPI_Reduce` |
| 6 | `poly_serial` | polynomial example | file input and serial search |
| 7 | `poly_mpi` | broadcast and reduce | rank 0 input, `MPI_Bcast`, distributed search, `MPI_Reduce` |
| 8 | `ping_pong` | MPI messages | explicit buffers, counts, datatypes, latency, bandwidth |
| 9 | `collective_cost` | collectives | `MPI_Allreduce` payload-size cost |
| 10 | `halo_exchange` | ghost-cell communication | neighbor exchange with nonblocking point-to-point MPI |
| 11 | `memory_stream` | node-level performance | rank density, memory bandwidth, CPU binding |

The first seven programs are the core slide companion. The last four are short bridges toward RDMA: they make learners look directly at buffers, message sizes, rank placement, memory movement, and nearest-neighbor data exchange.

## LANTA Notes

These instructions follow the ThaiSC LANTA public documentation checked on 2026-07-07:

- LANTA is an HPE Cray EX cluster. ThaiSC recommends the HPE Cray Programming Environment for compatibility with the HPE Slingshot network.
- Use Cray compiler wrappers for native Cray MPICH builds: `cc` for C, `CC` for C++, and `ftn` for Fortran.
- This tutorial uses `cc` and C11.
- Launch MPI programs with `srun` inside Slurm jobs, not `mpirun`, `mpiexec`, or `aprun`.
- Current documented CPE toolchains include `cpeCray/25.03`, `cpeGNU/25.03`, and `cpeIntel/25.03`.
- Since the LANTA Slurm 24.05 upgrade, `#SBATCH --cpus-per-task` is inherited by `srun` by default. In hybrid jobs, pure-MPI preprocessing steps may still need `srun -c1`.

Sources:

- ThaiSC LANTA quick start guide: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/447021064/Quick+Start+Guide>
- ThaiSC LANTA file transfers: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/259883051/File+transfers>
- ThaiSC LANTA software management system: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/267091990/Software+management+system>
- ThaiSC LANTA software installation guideline: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/383582209/Software+installation+guideline>
- ThaiSC LANTA June 2025 user guide update: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/814219273>

## Build on LANTA

```bash
cd 2026-preschool/mpi-lanta
module reset
module load cpeCray/25.03
./scripts/record_context.sh
make
```

The Makefile defaults to the Cray C wrapper `cc`. To build with another MPI C wrapper on a non-LANTA system:

```bash
make CC=mpicc
```

## Run a Quick Check

Edit `jobs/quick_check.slurm` and replace `ltXXXXXX` with your LANTA project account. Then submit:

```bash
cd 2026-preschool/mpi-lanta
sbatch jobs/quick_check.slurm
```

For short class exercises, the scripts use `compute-devel`. Change the partition to `compute` for normal production runs if your account or queue policy requires it.

## Manual Slide Companion Commands

Inside an allocation or job script:

```bash
srun -n 4 ./build/hello_mpi
srun -n 4 ./build/rank_control
srun -n 4 ./build/work_distribution --items 16 --mode cyclic
srun -n 4 ./build/work_distribution --items 16 --mode block
srun -n 1 ./build/pi_serial --steps 10000000
srun -n 4 ./build/pi_mpi --steps 10000000
srun -n 1 ./build/poly_serial --input input/poly.dat
srun -n 4 ./build/poly_mpi --input input/poly.dat
```

RDMA-prep probes:

```bash
srun -n 2 ./build/ping_pong --max-bytes 8M
srun -n 8 ./build/collective_cost --max-bytes 8M
srun -n 8 ./build/halo_exchange --points-per-rank 500000
```

## Comparison Jobs

Each script builds the examples and prints enough environment information to make the results reproducible.

```bash
sbatch jobs/quick_check.slurm
sbatch jobs/compare_toolchains.slurm
sbatch jobs/compare_optimization.slurm
sbatch jobs/compare_network.slurm
sbatch jobs/compare_binding.slurm
```

Before running, replace `ltXXXXXX` in each script or submit with an account override if permitted:

```bash
sbatch -A ltYOURPROJECT jobs/compare_network.slurm
```

## Interpreting Results

Look for patterns, not single numbers.

- If `pi_mpi` changes significantly between `-O0`, `-O2`, and `-O3`, students are seeing compiler optimization effects.
- If `memory_stream` slows down as ranks per node increase, students are seeing memory bandwidth contention.
- If `ping_pong` latency is higher across two nodes than within one node, students are seeing network latency.
- If `collective_cost` grows nonlinearly with rank count or message size, students are seeing collective communication cost.
- If `halo_exchange` changes when the same total ranks are placed on one node versus two nodes, students are seeing how rank layout affects neighbor communication.

Record the loaded modules, Slurm allocation, rank layout, and benchmark output together. Performance results without the environment are hard to explain later.
