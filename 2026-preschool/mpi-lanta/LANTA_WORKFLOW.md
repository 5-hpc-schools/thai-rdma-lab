# LANTA Workflow Notes

Use this as the operational checklist around the MPI examples. It covers the shell workflow, file transfer, modules, job environment expectations, and what to record so benchmark results can be reproduced.

## Basic Command-Line Workflow

Log in to the LANTA frontend node with SSH:

```bash
ssh <lanta-username>@lanta.nstda.or.th
```

LANTA login requires your LANTA username, password, and two-factor authentication code. The ThaiSC account used for the website is not the same as the LANTA account used for SSH.

After login, check where you are and what project storage is available:

```bash
pwd
hostname
myquota
sbalance
sinfo
```

Use the project directory for training runs and shared class materials. Home is convenient for small personal files, but project storage is the better default for builds, job output, and datasets:

```bash
cd /project/ltXXXXXX-YYYY
git clone https://github.com/5-hpc-schools/thai-rdma-lab.git
cd thai-rdma-lab/2026-preschool/mpi-lanta
```

Typical edit-build-submit loop:

```bash
module reset
module load cpeCray/25.03
make clean
make
sbatch -A ltXXXXXX jobs/quick_check.slurm
myqueue
tail -f logs/mpi-quick-<jobid>.out
```

Do not run Slurm scripts with `bash jobs/name.slurm` or `./jobs/name.slurm`. That runs the script on the login node. Use `sbatch`.

## File Transfer

Use the transfer node for moving data:

```bash
scp local-file <lanta-username>@transfer.lanta.nstda.or.th:/project/ltXXXXXX-YYYY/
scp -r local-directory <lanta-username>@transfer.lanta.nstda.or.th:/project/ltXXXXXX-YYYY/
```

For large files or transfers that take more than a few minutes, always use `transfer.lanta.nstda.or.th`, not the frontend login host. ThaiSC may terminate large transfers on the frontend if they affect other users.

For repeated synchronization, avoid `rsync -a` because it preserves permissions that may not match LANTA project-directory group ownership. Use:

```bash
rsync -rvz ./local-directory/ <lanta-username>@transfer.lanta.nstda.or.th:/project/ltXXXXXX-YYYY/local-directory/
```

For code, prefer Git when possible:

```bash
git pull --ff-only
git status --short
git log --oneline -1
```

## Modules and Builds

LANTA uses Lmod modules. Useful commands:

```bash
module avail
module overview
module spider cpeCray
module reset
module purge
module load cpeCray/25.03
module list
```

For MPI programs on LANTA, use the Cray wrappers from the active programming environment:

```bash
cc --version
cc --cray-print-opts=all
make CC=cc
```

Keep the build and run environments consistent. If you compile with `cpeCray/25.03`, load the same toolchain in the Slurm job before running the executable. Do not hide module loads in `~/.bashrc`; put them in the job script so the run is auditable.

## Job Environment Expectations

Slurm controls compute-node execution. A job script has two layers:

- `#SBATCH` lines request the allocation: account, partition, nodes, ranks, cores, time, output files.
- `srun` launches processes inside that allocation.

Important expectations:

- `sbatch jobs/name.slurm` submits the job to Slurm.
- If a job requests more than one node, the shell part of the script runs only on the first allocated node. Use `srun` for work that must run across all ranks or nodes.
- Use `srun` for MPI executables on LANTA. Do not use `mpirun`, `mpiexec`, or `aprun` for these examples.
- Since LANTA's Slurm 24.05 update, `#SBATCH --cpus-per-task` is passed to `srun` by default. In hybrid jobs, a pure-MPI helper step may need `srun -c1`.
- `SLURM_JOB_ID`, `SLURM_JOB_NUM_NODES`, `SLURM_NTASKS`, `SLURM_TASKS_PER_NODE`, and `SLURM_CPUS_PER_TASK` are available inside the job and are useful for logs.
- The training scripts write output to `logs/`. Keep stdout, stderr, module lists, and benchmark outputs together.

Useful commands:

```bash
sbatch -A ltXXXXXX jobs/quick_check.slurm
myqueue
squeue -u "$USER"
scancel <jobid>
scontrol show job <jobid>
```

## Reproducible Command Notes

For every benchmark result, record enough context that another student can repeat it:

```bash
date -Is
pwd
hostname
git status --short
git log --oneline -1
module list
cc --version
make print-env CC=cc
scontrol show job "$SLURM_JOB_ID"
```

The Slurm scripts in `jobs/` call `scripts/record_context.sh` after loading modules. The benchmark programs also print selected Slurm, module, compiler, and rank-placement details. Keep those lines with the timing output.

When comparing environments, change one variable at a time:

- toolchain: `cpeCray/25.03` vs `cpeGNU/25.03` vs `cpeIntel/25.03`
- optimization: `-O0` vs `-O2` vs `-O3` vs `-Ofast`
- placement: same node vs multiple nodes
- binding: `--cpu-bind=cores` vs `--cpu-bind=sockets` vs `--cpu-bind=none`
- rank density: fewer ranks per node vs more ranks per node

Save the command and result together:

```bash
mkdir -p logs/manual
{
  echo "# command: srun -N 2 -n 2 --ntasks-per-node=1 ./build/ping_pong --max-bytes 8M"
  ./scripts/record_context.sh
  srun -N 2 -n 2 --ntasks-per-node=1 ./build/ping_pong --max-bytes 8M
} | tee "logs/manual/pingpong-$(date +%Y%m%d-%H%M%S).log"
```

## Sources Checked

These notes were reviewed against ThaiSC LANTA public documentation on 2026-07-07:

- Quick start: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/447021064/Quick+Start+Guide>
- File transfers: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/259883051/File+transfers>
- Software management system: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/267091990/Software+management+system>
- Software installation guideline: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/383582209/Software+installation+guideline>
- June 2025 update notice: <https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/814219273>
