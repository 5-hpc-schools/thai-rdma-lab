# Day 3 Lab — Make GPUs Talk with NCCL

## First, the big idea (read this — it is short)

A **GPU** is a very fast calculator. We use GPUs to train AI, like ChatGPT.

One GPU is too small for a big AI. So we use **many GPUs at the same time**. But there is a
problem: each GPU only does **part** of the work. To finish, the GPUs must **share their
numbers** with each other. If they share slowly, the GPUs just sit and wait. Slow sharing =
slow training = a lot of wasted money.

So today's one big question is: **how fast can GPUs share numbers?**

### A picture in your head

Think of 4 students. Each one added up part of a long list:

```
   Student A = 10      Student B = 20
   Student C = 30      Student D = 40
```

They ALL need the grand total (10 + 20 + 30 + 40 = **100**). So they must talk and add.
This "everyone adds, everyone ends up with the total" job has a name: **all-reduce**.

All-reduce is the #1 thing GPUs do when they train AI. Today you will run it and **measure
how fast it is**.

### The one number we watch: **busbw**

We measure speed in **GB/s** = gigabytes per second. Think of water in a pipe: a fat pipe
moves more water per second. `busbw` (bus bandwidth) is our "how fat is the pipe" number.

**Bigger busbw = faster = better.** That is all you need to remember.

---

## How to use the cluster (only 4 commands)

The GPUs live in a big computer called **LANTA**. You do not touch the GPUs directly. You
write a small request file and **send it to a waiting line**. A manager (called Slurm) runs
it when GPUs are free, and saves the answer in a file.

| command | what it does |
|---------|--------------|
| `sbatch job.sbatch` | send your job to the line |
| `squeue -u $USER`   | check your job: `PD` = waiting, `R` = running, empty = done |
| `cat name_*.out`    | read the answer file |
| `scancel <id>`      | cancel a job (if it gets stuck) |

**Do this once to get the lab files:**

```bash
cp -r /project/tn999992-rdma/day3-nccl-lab/lab  ~/nccl-lab
cd ~/nccl-lab
```

---

## Part A — 4 GPUs in ONE box (NVLink) 🏠

Inside one LANTA box (a "node") there are **4 GPUs**. They are joined by a private,
super-fast bridge called **NVLink**. No network here — it is like 4 friends sitting at the
**same table**, passing papers hand to hand.

```
        ┌─────────── one node (one box) ───────────┐
        │   GPU0 ══ GPU1 ══ GPU2 ══ GPU3            │
        │     ╚══════╩═ NVLink (very fast) ═╩═══╝   │
        └──────────────────────────────────────────┘
```

**Run it:**

```bash
sbatch partA_1node.sbatch
squeue -u $USER          # wait until your job disappears from the list
cat partA_*.out
```

**How to read the answer.** You will see a table. Each row is a different message size (small
at the top, big at the bottom). Look at the column named **`busbw`** on the far right, and
read the **last row** (the biggest message):

```
       size     ...   busbw
  536870912     ...  215.54     <- ~215 GB/s. HUGE.
```

**Now read the log.** The lines that start with `NCCL INFO` are NCCL telling you what it did.
Find a line with the word **`NVL`** (for example `type NVL/PIX`). `NVL` = NVLink. It proves
the GPUs used the fast table-bridge, not a network.

- ✏️ busbw at 512 MB = __________ GB/s  (we got ~**215**)

> 💡 **Question A:** Why is NVLink so fast? (Hint: the GPUs are on the same board, joined by a
> direct wire. No cable, no network, nobody copies the data in the middle.)

---

## Part B — 2 boxes, over the network (Slingshot RDMA) 🌍

Now your GPUs are in **two different boxes**. GPU on box 1 must send numbers to a GPU on box 2.
They are far apart, joined by a network cable. LANTA's network is called **Slingshot**.

It is like your 4 friends are now in **different buildings**. They cannot pass papers by hand.
They must **send mail**.

```
   ┌── node 1 ──┐        Slingshot        ┌── node 2 ──┐
   │  GPU  ●─────┼════════ network ════════┼─────● GPU  │
   └────────────┘         cable            └────────────┘
```

Here is the clever part. NCCL uses **RDMA**. RDMA means the **network card reads GPU memory
directly** and sends it — the CPU never touches the data.

> **Analogy:** RDMA is a courier who walks straight into your room, picks up the box, and
> carries it to your friend's room. Fast, no middle-men.

> ✋ **Guess first!** Part A (NVLink) was ~200 GB/s. Do you think the network in Part B will
> be **faster** or **slower**? Write your guess: ________. Then run it and see.

**Run it:**

```bash
sbatch partB_multinode_rdma.sbatch
squeue -u $USER          # a 2-box job may WAIT in line. Be patient.
cat partB_*.out
```

**Find these lines in the log** — this is the important proof:

```
NET/OFI Selected provider is cxi   (found 1 nics)     <- NCCL chose the Slingshot network
NCCL INFO Using network OFI                            <- it uses the RDMA plugin
Channel 00/0 : 0[0] -> 1[0] [send] via NET/OFI/0       <- data crosses the network card
```

- ✏️ busbw at 256 MB = __________ GB/s  (we got ~**9**)

> 💡 **Question B:** It is much slower than Part A (~9 vs ~200 GB/s). Is that surprising? A
> cable between two buildings can never be as fast as a wire on the same board — even with
> RDMA. This is why we keep GPUs close together when we can.

---

## Part C — the same 2 boxes, but SLOW on purpose (TCP) 🐌

Same two boxes. Same cable. But now we **turn RDMA OFF** and force the old, slow way: plain
**TCP** (`NCCL_NET=Socket`). With TCP, the **CPU copies every single byte**. Let's measure how
much speed we lose.

> **Analogy:** TCP is like sending your box through a post office where a clerk **re-copies**
> the contents at every desk. Same road, but so much slower.

**Run it:**

```bash
sbatch partC_multinode_tcp.sbatch
squeue -u $USER
cat partC_*.out
```

- ✏️ busbw at 256 MB (TCP) = __________ GB/s  (we got ~**1.8**)

**Now the key result. Compare Part B and Part C:**

- Part B (RDMA) ÷ Part C (TCP) = __________ times faster  (we got about **5×**)

> 💡 **Question C:** Same cable, but RDMA is ~5× faster than TCP. Why? (Answer: TCP makes the
> CPU copy every byte; RDMA lets the network card grab GPU memory directly — no copies. That
> is the whole point of RDMA, and the reason expensive AI clusters use it.)

---

## Part D — 🔍 Open the box: look inside NCCL

NCCL is not magic. It **looks at your machine, makes a plan**, and picks how to send data.
One run does three small experiments (one node, 4 GPUs):

```bash
sbatch partD_explore.sbatch
cat partD_*.out
```

### D1 · Protocols — how NCCL packs the data

To send numbers, NCCL can pack them 3 ways. Think of shipping a gift:

- **LL** = a small postcard. Tiny, leaves instantly. Best for **small** messages.
- **LL128** = a padded envelope. A middle choice, made for NVLink.
- **Simple** = a big truck. Slow to load, but carries a LOT. Best for **big** messages.

Our real numbers (4×A100), busbw in GB/s:

| protocol | at 1 MB (small) | at 64 MB (big) |
|----------|-----------------|----------------|
| Simple   | ~21             | ~**169** ✅ |
| LL       | ~**42** ✅       | ~71 |
| LL128    | ~31             | ~123 |

See the swap? **Postcard (LL) wins for small. Truck (Simple) wins for big.**

> 💡 **Question D1:** There is no single "best" way — it depends on the size. This is exactly
> why NCCL changes the protocol for you. You normally never set it by hand.

### D2 · Shape — Ring or Tree?

How do the GPUs pass data around?

- **Ring** 🔄 — pass around a circle: A→B→C→D→A. Great for **big** messages.
  ```
        GPU0 → GPU1
          ↑        ↓
        GPU3 ← GPU2
  ```
- **Tree** 🌳 — add up like a sports tournament: (A+B) then (C+D) then join. Great for **small**.

At big sizes we measured **Ring ~102** vs **Tree ~77** GB/s (average). Write yours:

- ✏️ Ring = ______ GB/s     Tree = ______ GB/s

### D3 · The map NCCL drew

In part (3) of the output, NCCL prints what it **discovered** about the machine. Look for:

```
+ NVL[80.0] - GPU/...     <- every GPU reaches every other GPU by NVLink at 80 GB/s
nChannels 12              <- NCCL opened 12 parallel "lanes" at the same time!
NVLS ... not available    <- one fancy feature (in-network add) is off here
```

> 💡 **Question D3:** All 4 GPUs are connected to all others (an "NVSwitch"). Why does that
> make it easy for NCCL to build a fast Ring? (Because every hop is the same fast 80 GB/s —
> no slow step in the circle.)

---

## Part E — ✍️ Now YOU write NCCL code

You have watched NCCL. Now you write the key line yourself. It is only **2 lines**.

Open `nccl_hello.cu`. Four GPUs each start with a number equal to their id: 0, 1, 2, 3.
After **all-reduce (sum)**, every GPU must hold 0+1+2+3 = **6**.

There are **2 spots marked `TODO`**. Each one already shows you exactly what to type — just
copy it into place. (Yes, we give you the answer. The goal is to *see* the calls, not to
struggle.)

**Fill the 2 lines, then build and run:**

```bash
nano nccl_hello.cu                          # fill the two TODO lines (or use vim)
salloc -p gpu -N1 --gpus-per-node=4 -t 00:10:00   # borrow 4 GPUs for 10 minutes
srun bash build_and_run_hello.sh            # this compiles your file and runs it
exit                                        # give the GPUs back
```

If your 2 lines are correct, you will see:

```
GPU0 after all-reduce: 6 6 6 ...  (expect 6)
RESULT OK
```

🎉 That `ncclAllReduce(...)` you just wrote is the **exact call** that trains real AI models
on thousands of GPUs. You have now done the real thing.

> 💡 **Question E:** In your `ncclAllReduce(...)` line there is `ncclSum`. What if you change it
> to `ncclMax`? (Try it! Every GPU would end up with the **biggest** value = 3, not the sum.)

---

## Part F — 🔬 Deep dive: the first 100 milliseconds

Before NCCL moves **one single number**, it does a LOT of secret work. It says hello, draws a
map of your machine, plans routes, opens roads, and writes a cheat-sheet. Let's turn on the
**full log** and watch every step. This is exactly how experts debug real AI clusters.

Two magic switches do it: `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=ALL`.

```bash
sbatch partF_deepdive.sbatch
cat partF_*.out | less        # long! ~7000 lines. Press q to quit less.
```

Below is the real story, in 5 steps. Your log will look almost the same.

### Step 1 · 🤝 Handshake (bootstrap)

The 4 GPUs must find each other and agree they are one team. NCCL gives them all the **same
secret number** (`commId`) and a "phone line" to swap addresses.

```
Bootstrap : Using bond0:10.168.4.6<0>
NET/Socket : Using [0]bond0 [1]hsn0 [2]hsn1
comm 0x3c75f60 rank 2 nranks 4 cudaDev 2 busId 81000 commId 0xad5f74829cf1c8d3 - Init START
...
comm 0x3c75f60 rank 2 ...                              commId 0xad5f74829cf1c8d3 - Init COMPLETE
```

- **`commId 0xad5f...`** — the team's secret id. **All 4 ranks show the SAME commId.** That is
  how they know they belong together.
- **`bond0`** — a slow, ordinary network used JUST to say hello and swap addresses (once).
  ⚠️ This is **not** the road for real data — that comes in Step 4.
- **`rank 2 ... cudaDev 2 busId 81000`** — rank 2 is GPU 2, sitting at PCI address 81000.

### Step 2 · 🗺️ Draw the map (topology)

Each GPU walks the hardware and writes down what it sees: every GPU, every link, every CPU.

```
=== System : maxBw 80.0 totalBw 240.0 ===
+ PCI[24.0] - GPU/3000 (0)
              + NVL[80.0] - GPU/81000
              + NVL[80.0] - GPU/C1000
              + NVL[80.0] - GPU/41000
```

- **`maxBw 80.0`** — the fastest single link is 80 GB/s (one NVLink).
- **`totalBw 240.0`** — all links added up = 240 GB/s of sharing power.
- **`+ NVL[80.0] - GPU/81000`** — this GPU has an 80 GB/s NVLink to GPU 81000. Every GPU has one
  to every other GPU → an **all-to-all** map (an NVSwitch). This map is what NCCL plans on.

### Step 3 · 🧭 Plan the routes (graph search)

NCCL now searches for the best **rings** and **trees** on that map.

```
Pattern 4, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX
Ring 00 : 2 -> 3 -> 0
Ring 01 : 1 -> 3 -> 2
Tree 12 : -1 -> 0 -> 1/-1/-1
```

- **`nChannels 12`** — NCCL opened **12 rings** (12 parallel lanes) to use all the NVLinks at once.
- **`Ring 00 : 2 -> 3 -> 0`** — one ring's pass-order: rank 2 sends to 3, 3 sends to 0, ...
- **`Tree 12 : -1 -> 0 -> 1`** — a tree: rank 0 has no parent (`-1`) and one child (rank 1).
- **`type NVL`** — all these routes ride NVLink.

### Step 4 · 🚧 Open the roads (transport)

For each link in each ring, NCCL picks **how** to actually move bytes, and connects them.

```
P2P Chunksize set to 524288
Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer/read
...
Connected all rings
Connected all trees
```

- **`via P2P/direct pointer/read`** — GPU 1 **reads GPU 2's memory directly** over NVLink. No
  CPU, no copy. (On 2 nodes this line instead says **`via NET/OFI`** = over Slingshot RDMA.)
- **`Chunksize 524288`** — it moves data in 512 KB pieces.
- **`Connected all rings / trees`** — every road is now open. Setup is almost done.

### Step 5 · 📋 Write the cheat-sheet (tuning)

Finally NCCL builds a cost table for **every** way to send, so at run time it can instantly pick
the fastest one for each message size.

```
Algorithm | Tree | Ring | CollNetDirect | CollNetChain | NVLS | NVLSTree
Protocol  | LL | LL128 | Simple | LL | LL128 | Simple | ...
threadThresholds 8/8/64 | 32/8/64 | 512 | 512
24 coll channels, 0 nvls channels, 32 p2p channels, 8 p2p channels per peer
```

- The big table is Algorithm (Ring/Tree/…) × Protocol (LL/LL128/Simple) — the same knobs you
  played with in Part D. **This table is why NCCL auto-chose the best one for you.**
- **`24 coll channels, 32 p2p channels`** — the final number of lanes it will use.

Now — and only now — the real work runs:

```
AllReduce: opCount 0 sendbuff 0x.. recvbuff 0x.. count 262144 datatype 7 op 0 [nranks=4]
```

### 🔦 Scavenger hunt (find these in YOUR partF log)

```bash
grep "commId"          partF_*.out | head        # 1. your team secret
grep "=== System"      partF_*.out               # 2. your machine's totalBw
grep -c "Ring 0"       partF_*.out               # 3. how many rings?
grep "via P2P"         partF_*.out | head -1      # 4. which GPU reads which?
grep "coll channels"   partF_*.out | head -1      # 5. how many channels at the end?
```

> 💡 **Question F:** The handshake used the slow `bond0` network, but the data uses fast NVLink
> (or Slingshot RDMA). Why use two different networks? (Hint: saying "hello" is tiny and rare;
> moving AI data is huge and constant. Use a cheap line for hello, a highway for the data.)

---

## What you learned today

- GPUs must **share numbers** to train AI. The job is called **all-reduce**.
- Inside one box: GPUs use **NVLink** — very fast (~200 GB/s).
- Between boxes: GPUs use the **network** (Slingshot on LANTA).
- **RDMA** lets the network card move GPU data with no CPU copy → about **5× faster than TCP**.
- NCCL is smart: it picks a **protocol** (LL / Simple) and a **shape** (Ring / Tree) by size,
  and opens many **channels** at once.
- Before moving one byte, NCCL **shakes hands, maps the machine, plans rings/trees, opens the
  roads, and writes a tuning cheat-sheet** — you saw all of it in the full log (Part F).
- You wrote **`ncclAllReduce`** with your own hands. That is how real AI is trained. Well done. 👏
