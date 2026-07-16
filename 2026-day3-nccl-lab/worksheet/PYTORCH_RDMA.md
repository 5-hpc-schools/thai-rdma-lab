# From your Python code down to RDMA — and PyTorch DDP

Someone asked: *"I write Python. How does my code actually use RDMA? Do I call RDMA myself?"*

Short answer: **almost never directly.** You call **PyTorch**; PyTorch calls **NCCL** (the thing
you studied all day); NCCL drives the **Slingshot card = RDMA**. This page shows the whole path,
then how to write and run real multi-GPU Python.

---

## 1 · The layer cake — who calls whom

```
   YOUR PYTHON             loss.backward()  /  dist.all_reduce(x)
        │
        ▼
   torch.distributed        PyTorch's "talk to other GPUs" layer   (backend="nccl")
        │
        ▼
   NCCL                     builds rings/trees, picks Ring/Simple   ← ALL of Parts A–F
        │
        ▼
   aws-ofi-nccl             the plugin that maps NCCL → libfabric   ("Selected provider is cxi")
        │
        ▼
   libfabric (cxi)          the Slingshot driver
        │
        ▼
   Slingshot NIC            the card does the RDMA transfer         ← the network card reads GPU memory
```

So when your Python does `loss.backward()`, deep down the **NCCL proxy** (Field Guide §8) hands
chunks to libfabric, and the Slingshot card RDMA's them to the other nodes. **You never wrote a
single line of RDMA.** That is the point — the hard part is done for you.

💬 *Why is it good that you cannot see RDMA from Python?* (You get RDMA speed with zero RDMA code.
The libraries hide 20 years of network engineering behind one function call.)

---

## 2 · torch.distributed — the 5 lines that start everything

Every multi-GPU PyTorch program begins the same way:

```python
import torch, torch.distributed as dist

dist.init_process_group(backend="nccl")     # <-- "use NCCL" = use Slingshot RDMA between nodes
rank       = dist.get_rank()                 # my seat number (0,1,2,...)  -> NCCL "rank"
world_size = dist.get_world_size()           # how many GPUs total         -> NCCL "nranks"
torch.cuda.set_device(rank % torch.cuda.device_count())
```

- `backend="nccl"` is the whole trick: it tells PyTorch to use the NCCL you studied. On LANTA that
  means the data path is **Slingshot cxi RDMA** (between nodes) and **NVLink** (inside a node).
- `rank` and `world_size` are the **exact same words** from the NCCL log (`rank 2 nranks 4`).

Now you can call collectives directly — the same zoo as Field Guide §12:

```python
x = torch.ones(1_000_000, device="cuda")
dist.all_reduce(x, op=dist.ReduceOp.SUM)     # <-- this is ncclAllReduce, from Python!
# every GPU now holds the element-wise sum. Ran over NVLink / Slingshot RDMA.
```

`dist.all_reduce(x)` **is** the `ncclAllReduce(...)` you wrote by hand in Part E — just one layer up.

| Python (torch.distributed) | NCCL C call (Part E) |
|----------------------------|----------------------|
| `dist.all_reduce(x)`       | `ncclAllReduce(...)` |
| `dist.broadcast(x, src=0)` | `ncclBroadcast(...)` |
| `dist.all_gather(list, x)` | `ncclAllGather(...)` |
| `dist.reduce_scatter(...)` | `ncclReduceScatter(...)` |

---

## 3 · DDP — how real training uses all this without you asking

You almost never call `all_reduce` yourself in training. You wrap your model in **DDP**
(`DistributedDataParallel`) and it does the collectives **automatically**:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().cuda()
model = DDP(model, device_ids=[torch.cuda.current_device()])   # <-- the magic wrapper

for batch in data:                 # each GPU gets a DIFFERENT slice of data
    loss = model(batch).loss
    loss.backward()                # <-- DDP fires all-reduce on the gradients HERE, by itself
    optimizer.step()               # every GPU now has the SAME averaged gradients -> same update
```

**What DDP does under the hood during `backward()`:**

1. Each GPU computes gradients for its own data slice (they differ).
2. DDP groups gradients into **buckets** (~25 MB each) and, as each bucket fills, fires an
   **all-reduce (sum)** on it via NCCL — overlapping with the rest of the backward pass.
3. It divides by `world_size` → the **average** gradient.
4. Every GPU now holds identical averaged gradients, so `optimizer.step()` keeps all models in sync.

That is literally: **train on N GPUs = same as 1 GPU, because all-reduce keeps them identical.**
The all-reduce you measured today runs **millions of times** in a real training job — which is why
its speed (RDMA vs TCP, §Part B/C) decides how much your expensive GPUs cost you.

💬 *DDP does an all-reduce of gradients every single step. If that all-reduce were 5× slower (TCP
instead of RDMA), what happens to a 30-day training run?* (Communication becomes the bottleneck;
the GPUs idle waiting for gradients; the run can take much longer and cost much more. This is the
whole business case for RDMA.)

---

## 4 · "But I really want to call RDMA directly from Python"

Sometimes you do (research, custom systems). Your options, from easy to raw:

| you want… | use | notes |
|-----------|-----|-------|
| GPU collectives (99% of AI) | **PyTorch `dist`** / `cupy.cuda.nccl` | NCCL under the hood = RDMA. Best choice. |
| general CPU/GPU messaging over RDMA | **mpi4py** (`from mpi4py import MPI`) | MPI runs over the same fabric (cray-mpich/OFI). |
| direct one-sided RDMA (put/get) | **UCX-Py** (`import ucp`) | thin Python layer over UCX; real RDMA verbs/put/get. |
| raw InfiniBand verbs | **pyverbs** | lowest level; you manage queues yourself. Rarely needed. |

Key idea: **there is no "import rdma".** RDMA is a *capability of the hardware*, reached through a
library (NCCL, UCX, MPI, or verbs). You pick the library that fits, and RDMA happens inside it.
On LANTA the GPU path is NCCL→libfabric→cxi; the CPU path would be MPI/UCX→libfabric→cxi.

💬 *We spent all day on NCCL, not on raw verbs. Was that a shortcut or the real thing?* (The real
thing — real AI clusters use NCCL/MPI, not hand-written verbs. Knowing the layer below (what you
saw in the logs) is what makes you able to debug it.)

---

## 5 · Run it yourself (a real DDP job on LANTA)

`ddp_demo.py` (in the `lab/` folder) is a tiny, complete DDP program. Launch it across GPUs and —
here is the payoff — turn on `NCCL_DEBUG=INFO` and you will see the **exact same** `Selected
provider is cxi` / `via NET/OFI` lines from Part B. Your **Python** all-reduce rode Slingshot RDMA.

```bash
# (needs a python with torch; on LANTA we used: source pyenv/bin/activate)
NCCL_DEBUG=INFO srun -N2 --ntasks=8 --ntasks-per-node=4 \
    --network=job_vni,single_node_vni python ddp_demo.py
```

**We ran it. Real output (verified on LANTA, 2026-07-16):**

```
[setup] world_size=4  backend=nccl  device=NVIDIA A100-SXM4-40GB
[step 0] loss=1.3341
[step 1] loss=1.3338
[step 2] loss=1.3335
[all_reduce] sum of all ranks = 6 (expected 6) OK
```

And on **2 nodes**, the NCCL log printed — from a pure Python program — the exact same lines as
Part B, including full GPUDirect RDMA:

```
NET/OFI Selected provider is cxi, fabric is cxi (found 1 nics)   <- Python chose Slingshot
NET/OFI : GPU Direct RDMA Enabled for HCA 0 'cxi0'
Channel 00/0 : 3[3] -> 4[0] [send] via NET/OFI/0/GDRDMA          <- gradients cross via GPUDirect RDMA
```

(The near GPU got `GDRDMA`; a far GPU showed `distance 7 > 6` and fell back to host-staged — the
same NIC-distance detail from Field Guide §9. All real, all captured in `logs/pytorch_ddp_2node_cxi.txt`.)

That is the whole journey: **your Python `loss.backward()` → DDP → all-reduce → NCCL → cxi → RDMA
across Slingshot → back to your Python, with averaged gradients.** You now understand every layer,
top to bottom — and you have seen each one with your own eyes in the logs.
