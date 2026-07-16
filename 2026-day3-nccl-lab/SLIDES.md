# Make GPUs Talk
## Day 3 — NCCL, RDMA, and how AI trains on many GPUs

use ← → arrows (or click) to move

---

## Why are we here?

- A **GPU** is a very fast calculator for AI.
- One GPU is **too small** for a big AI model.
- So we use **many GPUs** together.
- **The problem:** they must **share their numbers** — fast.

> Slow sharing = GPUs wait = wasted money. Today: **how fast can GPUs share?**

---

## The job: all-reduce

Four students each added part of a list:

```
   A = 10      B = 20
   C = 30      D = 40
```

Everyone needs the **total = 100**. So they talk and add.

**all-reduce** = everyone adds, everyone gets the sum. It is the #1 job in AI training.

---

## The number we watch: busbw

- **busbw** = bus bandwidth, in **GB/s**.
- Think of a water pipe: fatter pipe = more water per second.
- **Bigger busbw = faster = better.** That is all you need.

---

## How to run (only 4 commands)

| command | meaning |
|---------|---------|
| `sbatch job.sbatch` | send job to the queue |
| `squeue -u $USER` | PD = waiting, R = running, empty = done |
| `cat name_*.out` | read the answer |
| `scancel <id>` | cancel |

---

# Part A
## 4 GPUs in ONE box — NVLink

---

## Inside one box: NVLink

```
   ┌──────── one node (one box) ────────┐
   │  GPU0 ══ GPU1 ══ GPU2 ══ GPU3       │
   │     ══ NVLink (very fast) ══        │
   └────────────────────────────────────┘
```

Like 4 friends at the **same table**, passing papers by hand. No network.

---

## Part A result

- busbw ≈ **215 GB/s** — HUGE.
- Log shows `type NVL` = NVLink was used.

> Why so fast? A **direct wire** on the board. No cable, no copies.

---

# Part B
## 2 boxes — the Slingshot network (RDMA)

---

## Between boxes: the network

```
   ┌─ node 1 ─┐     Slingshot     ┌─ node 2 ─┐
   │  GPU ●────┼═════ cable ═══════┼────● GPU │
   └──────────┘                   └──────────┘
```

Friends in **different buildings** now — they must **send mail**.

**RDMA** = the network card reads GPU memory **directly** (no CPU copy).
Like a courier who walks in and grabs the box.

---

## ✋ Guess first

Part A (NVLink) was **~200 GB/s**.

The network in Part B — **faster or slower?**

*(write your guess, then we run it)*

---

## Part B result

- busbw ≈ **9 GB/s**.
- Log proof:

```
NET/OFI Selected provider is cxi   (found 1 nics)
Channel 00 : 0 -> 1 [send] via NET/OFI/0
```

> Slower than NVLink — a long cable can't beat an on-board wire. But it is real RDMA.

---

# Part C
## Same cable, but SLOW on purpose (TCP)

---

## TCP: the slow way

- Turn RDMA **off** → plain **TCP** (`NCCL_NET=Socket`).
- Now the **CPU copies every byte**.
- Like a post office that **re-copies** your box at every desk.

---

## The headline result 🏆

| path | busbw |
|------|-------|
| Part B — RDMA | **~9 GB/s** |
| Part C — TCP | **~1.8 GB/s** |

## RDMA is ~**5× faster** than TCP — on the SAME cable.

> This 5× is the whole reason AI clusters buy RDMA networks.

---

# Part D
## 🔍 Look inside NCCL

---

## Protocols: how to pack the data

- **LL** = a postcard. Tiny, instant. Best for **small**.
- **LL128** = a padded envelope. Middle.
- **Simple** = a big truck. Best for **big**.

---

## Protocols — the crossover

| protocol | at 1 MB | at 64 MB |
|----------|---------|----------|
| Simple | ~21 | **~169** ✅ |
| LL | **~42** ✅ | ~71 |
| LL128 | ~31 | ~123 |

**Postcard wins small. Truck wins big.** No single best — so NCCL switches for you.

---

## Shape: Ring vs Tree

```
   Ring 🔄            Tree 🌳
   0 → 1             (0+1) ┐
   ↑    ↓                  ├─ sum
   3 ← 2             (2+3) ┘
```

- **Ring** — pass around a circle. Best for **big**.
- **Tree** — add like a tournament. Best for **small**.

Big data: **Ring ~102** vs **Tree ~77** GB/s.

---

## The machine map (topology)

```
=== System : maxBw 80.0  totalBw 240.0 ===
GPU/3000  + NVL[80.0] - GPU/81000
          + NVL[80.0] - GPU/c1000
          + NVL[80.0] - GPU/41000
```

Every GPU has an **80 GB/s NVLink** to every other = all-to-all (an NVSwitch).

---

## NVLS — could the switch do the math?

```
NVLS multicast support is not available
```

**NVLS** = the NVSwitch chip **adds the numbers itself**, in the wires — no GPU has to receive-add-resend.
Very fast, but needs newer GPUs (H100). On these A100s it is off → NCCL uses Ring/Tree instead.

---

# Part E
## ✍️ Now YOU write NCCL code

---

## Fill 2 lines

```c
ncclCommInitAll(comm, nGPU, devs);              // make the team
ncclAllReduce(send, recv, N, ncclFloat,
              ncclSum, comm[g], stream[g]);      // add + share
```

4 GPUs start with 0,1,2,3 → after all-reduce each holds **6**.

---

## It works ✅

```
GPU0 after all-reduce: 6 6 6 ...
RESULT OK
```

> That `ncclAllReduce` is the **exact call** that trains real AI on thousands of GPUs.

---

## all-reduce is one of a family

| collective | what it does |
|------------|--------------|
| **AllReduce** | everyone adds, everyone gets the total |
| **Broadcast** | one GPU's data → copied to all |
| **Reduce** | everyone adds → only rank 0 keeps it |
| **AllGather** | everyone collects everyone's piece |
| **ReduceScatter** | add, then split the result across GPUs |

Training an AI = **AllReduce the gradients**, millions of times.

---

# Part F
## 🔬 The first 100 milliseconds

Before it moves ONE byte, NCCL does 5 things.

---

## First: how to read a log line

```
x1001c4s7b0n1:2678741:2679205 [2] NCCL INFO Selected provider is cxi
 └── host ──┘ └pid─┘ └tid─┘ [GPU]  └──── message ────┘
```

- **`[2]`** = which GPU printed it. Follow **one** GPU's story in the mess.
- **`NCCL INFO`** = normal. **`NCCL WARN`** = stop and read it.

---

## Step 1 · Handshake 🤝

```
Bootstrap : Using hsn0:10.150.1.8
comm rank 2 nranks 4 ... commId 0x51677f56... - Init START
```

- Everyone gets the **same team secret** (`commId`).
- They swap addresses over a slow "hello" network.

---

## Step 2 · Draw the map 🗺️

```
=== System : maxBw 80.0 totalBw 240.0 ===
+ NVL[80.0] - GPU/81000
```

NCCL probes every GPU, link, and CPU → the map it plans on.

---

## Step 3 · Plan routes 🧭

```
nChannels 12
Ring 00 : 2 -> 3 -> 0
Tree 12 : -1 -> 0 -> 1
```

12 rings + 12 trees — many parallel lanes.

---

## Step 4 · Open the roads 🚧

```
Channel 00 : 1 -> 2 via P2P/direct pointer/read
Connected all rings
```

Pick a transport per link: **P2P** (NVLink) or **NET/OFI** (Slingshot RDMA).

---

## Step 5 · The cheat-sheet 📋

```
Algorithm | Tree | Ring | ...
AllReduce | 10.4µs/29 | ... | 28.8µs/160 |
             latency / bandwidth
```

NCCL predicts every option's speed → picks the best per size. **This is why Part D happened.**

---

## The proxy: the GPU's CPU helper

The GPU is great at math, **bad at waiting**. So each GPU gets a CPU helper thread — the **proxy**.

- GPU: "send chunk 5" → then keeps doing **math** on chunk 6
- Proxy: pushes chunk 5 to the NIC / Slingshot, reports back "done"

This overlap = the GPU **never stops**. (The proxy is who actually touches RDMA.)

---

## Ring all-reduce, the trick

Split data into N parts. Two phases, each N−1 steps:

- **reduce-scatter** — pass & add; each GPU ends owning one full sum.
- **all-gather** — circulate the finished parts to everyone.

Each GPU moves only **~2× the data**, no matter how many GPUs → busbw stays high.

---

# From Python to RDMA
## and PyTorch DDP

---

## The layer cake

```
 your Python   loss.backward()
     │
 torch.distributed   (backend="nccl")
     │
 NCCL     ← everything you saw today
     │
 aws-ofi-nccl → libfabric → cxi
     │
 Slingshot NIC   = RDMA
```

You never write RDMA. You call PyTorch; RDMA happens below.

---

## DDP: automatic all-reduce

```python
model = DDP(model)          # wrap it
loss.backward()             # DDP all-reduces the gradients HERE
optimizer.step()            # every GPU: same averaged gradient
```

Training on N GPUs = same as 1 GPU, because all-reduce keeps them identical.

---

## "Can I call RDMA directly from Python?"

There is no `import rdma`. RDMA is a hardware power you reach through a library:

| you want | use |
|----------|-----|
| GPU collectives (AI) | **PyTorch `dist`** (= NCCL) |
| CPU messaging | **mpi4py** |
| one-sided put/get | **UCX-Py** (`ucp`) |
| raw verbs | **pyverbs** |

99% of AI = **PyTorch → NCCL → RDMA**. You already used it today.

---

## We ran it — Python used RDMA ✅

From a pure **Python** DDP program on 2 nodes, the NCCL log printed:

```
NET/OFI Selected provider is cxi (found 1 nics)
Channel 00 : 3 -> 4 [send] via NET/OFI/0/GDRDMA
```

**GDRDMA** = the Slingshot card read GPU memory directly. Your `loss.backward()` rode RDMA.

---

## When it breaks — grep this 🔧

| symptom | first thing to check |
|---------|----------------------|
| **hang** | did every rank reach `Init COMPLETE`? |
| **slow** | is it picking Ring/Simple for big messages? |
| **`Device or resource busy`** | a network / transport setup issue — not your math |
| **wrong answer** | a bug in your code, not NCCL |

Reading the log **is** the skill.

---

## What you learned 🎓

- GPUs share numbers with **all-reduce**.
- Inside a box: **NVLink** (~200 GB/s). Between boxes: **Slingshot RDMA**.
- **RDMA ≈ 5× faster than TCP** — same cable.
- NCCL picks **protocol** (LL/Simple) and **shape** (Ring/Tree) for you.
- You wrote `ncclAllReduce`, read a real NCCL log, and traced Python → RDMA.

---

# Thank you 👏
## You can now read a raw NCCL log — a rare, real skill.

Guides: `LAB.html` · `FIELDGUIDE.html` · `PYTORCH_RDMA.html`
