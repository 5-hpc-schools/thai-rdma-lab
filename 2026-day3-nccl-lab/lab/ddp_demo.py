#!/usr/bin/env python3
# ddp_demo.py -- a tiny, COMPLETE PyTorch DDP program.
# It trains one step of a toy model on every GPU. During loss.backward(), DDP
# all-reduces the gradients over NCCL -> NVLink (in a node) and Slingshot RDMA (between nodes).
#
# Launch (Slurm gives each task its rank via env vars):
#   NCCL_DEBUG=INFO srun -N2 --ntasks=8 --ntasks-per-node=4 \
#       --network=job_vni,single_node_vni python ddp_demo.py
#
# Watch the NCCL log: you will see the SAME "Selected provider is cxi" / "via NET/OFI"
# lines as Part B. That proves your Python all-reduce rode Slingshot RDMA.

import os, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1) Slurm sets these env vars for each task. torch.distributed reads them.
rank       = int(os.environ.get("SLURM_PROCID", 0))
world_size = int(os.environ.get("SLURM_NTASKS", 1))
local_rank = int(os.environ.get("SLURM_LOCALID", 0))
os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
os.environ.setdefault("MASTER_PORT", "29500")

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)   # <-- use NCCL = RDMA
if rank == 0:
    print(f"[setup] world_size={world_size}  backend=nccl  device={torch.cuda.get_device_name()}", flush=True)

# 2) A toy model, wrapped in DDP. DDP will all-reduce gradients for us.
model = nn.Linear(1024, 1024).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
opt   = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Each GPU gets DIFFERENT data (that is the whole point of data-parallel training).
x = torch.randn(64, 1024, device=local_rank) * (rank + 1)
y = torch.randn(64, 1024, device=local_rank)

for step in range(3):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()          # <-- DDP fires NCCL all-reduce on the gradients right here
    opt.step()               # <-- every GPU applies the SAME averaged gradient
    if rank == 0:
        print(f"[step {step}] loss={loss.item():.4f}", flush=True)

# 4) Prove all GPUs are in sync: all-reduce a number by hand (this IS ncclAllReduce from Python).
t = torch.tensor([float(rank)], device=local_rank)
dist.all_reduce(t, op=dist.ReduceOp.SUM)     # every GPU should get 0+1+...+(world_size-1)
if rank == 0:
    expected = world_size * (world_size - 1) // 2
    print(f"[all_reduce] sum of all ranks = {int(t.item())} (expected {expected}) "
          f"{'OK' if int(t.item()) == expected else 'WRONG'}", flush=True)

dist.destroy_process_group()
