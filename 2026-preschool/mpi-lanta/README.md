# MPI Training on LANTA HPC

This module teaches how to compile and run C++ MPI programs on LANTA, then measure how environment choices affect performance.

Start with [LANTA workflow notes](LANTA_WORKFLOW.md) if this is your first time using the command line, transferring files, loading modules, or submitting Slurm jobs on LANTA.

## What Students Compare

| Example | Main lesson | Environment knobs |
|---|---|---|
| `hello_environment` | Inspect rank placement and loaded runtime environment | Slurm rank layout, loaded modules, CPU binding |
| `compute_pi` | Compute-bound scaling | Compiler toolchain, optimization level, number of ranks |
| `memory_stream` | Per-rank memory bandwidth | Ranks per node, CPU binding, NUMA placement |
| `ping_pong` | Point-to-point latency and bandwidth | Same-node vs cross-node placement, message size |
| `allreduce_benchmark` | Collective communication cost | Rank count, node count, MPI collective implementation |
| `halo_exchange` | Neighbor communication pattern | Rank mapping, node count, communication/computation ratio |

## LANTA Notes

These instructions follow the ThaiSC LANTA public documentation checked on 2026-07-07:

- LANTA is an HPE Cray EX cluster. ThaiSC recommends the HPE Cray Programming Environment for compatibility with the HPE Slingshot network.
- Use Cray compiler wrappers for native Cray MPICH builds: `cc` for C, `CC` for C++, and `ftn` for Fortran.
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

The Makefile defaults to the Cray C++ wrapper `CC`. To build with another MPI wrapper on a non-LANTA system:

```bash
make CXX=mpicxx
```

## Run a Quick Check

Edit `jobs/quick_check.slurm` and replace `ltXXXXXX` with your LANTA project account. Then submit:

```bash
cd 2026-preschool/mpi-lanta
sbatch jobs/quick_check.slurm
```

For short class exercises, the scripts use `compute-devel`. Change the partition to `compute` for normal production runs if your account or queue policy requires it.

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

- If `compute_pi` changes significantly between `-O0`, `-O2`, and `-O3`, students are seeing compiler optimization effects.
- If `memory_stream` slows down as ranks per node increase, students are seeing memory bandwidth contention.
- If `ping_pong` latency is higher across two nodes than within one node, students are seeing network latency.
- If `allreduce_benchmark` grows nonlinearly with rank count or message size, students are seeing collective communication cost.
- If `halo_exchange` changes when the same total ranks are placed on one node versus two nodes, students are seeing how rank layout affects nearest-neighbor communication.

Record the loaded modules, Slurm allocation, rank layout, and benchmark output together. Performance results without the environment are hard to explain later.
