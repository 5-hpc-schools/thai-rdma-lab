# NCCL Lab — Command Cheat Sheet

Keep this next to you. These are all the commands you need.

## 1. Log in and copy the lab (do this once)

```bash
ssh <your-username>@lanta.nstda.or.th      # enter your password + 6-digit code
cp -r /project/tn999992-rdma/day3-nccl-lab/lab  ~/nccl-lab
cd ~/nccl-lab
```

## 2. Run a part (Parts A, B, C, D)

```bash
sbatch partA_1node.sbatch      # send the job to the GPUs
squeue -u $USER                # is it running? (PD = waiting, R = running, empty = done)
cat partA_*.out                # read the answer
```

Replace `partA` with `partB`, `partC`, or `partD` for the other parts.

## 3. Part E — write and run your own code

```bash
# edit the file, fill the 2 TODO lines:
nano nccl_hello.cu             # (or vim)
# get 4 GPUs for 10 minutes, then build + run:
salloc -p gpu -N1 --gpus-per-node=4 -t 00:10:00
srun bash build_and_run_hello.sh
exit                           # give the GPUs back when done
```

## Reading `squeue`

| you see | it means |
|---------|----------|
| `PD` (Resources) | waiting in line for free nodes — be patient |
| `R` | running now |
| (nothing)        | finished — read the `.out` file |

## If something goes wrong

- Job stuck `PD` a long time? A 1-node job (A, D) starts fast; a 2-node job (B, C) can wait.
- A 2-node job printed the header then stopped? Just `scancel <jobid>` and `sbatch` it again.
- Compile error in Part E? Read the message — it names the exact line. The answer is in
  `nccl_hello_solution.cu` if you are stuck.
