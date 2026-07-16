# NCCL Field Guide — read the log like a pro

This is the **big picture** companion to the lab. In Part F you turned on the full NCCL log.
Here we explain **every part of it**, from zero. You do not need any background. Each topic has:
a plain explanation, a **real line from the log**, and a **💬 talk-about-it** question.

Turn on the full log any time with:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

---

## 0 · How to read ONE log line

Every line looks like this:

```
x1001c4s7b0n1:2678741:2679205 [2] NCCL INFO Selected provider is cxi
└──── host ────┘ └─pid─┘ └─tid─┘ └GPU┘ └tag┘ └───── message ─────┘
```

- **host** = which box (node) printed it.
- **pid / tid** = which process / thread.
- **`[2]`** = which **GPU** on that box (GPU 2).
- **`NCCL INFO`** = a normal message. (`NCCL WARN` = a warning. Read those carefully.)

💬 *Why does NCCL print the GPU number in every line?* (Hint: 4–16 GPUs all print at once; the
number lets you follow ONE GPU's story in the mess.)

---

## 1 · Rank, device, communicator — the 3 words you must know

- **GPU / device** = one physical card.
- **rank** = a GPU's **seat number** in the team (0, 1, 2, …). Rank 0 is the "leader".
- **communicator (`comm`)** = the **team object**. It holds the plan (rings, trees, buffers).

Real line:

```
comm 0x3c75f60 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 81000 commId 0xad5f74829cf1c8d3 - Init START
```

- `rank 2` of `nranks 4` → seat 2 of a 4-GPU team.
- `cudaDev 2` → it is physical GPU 2.
- `busId 81000` → the GPU's hardware address on the board.
- `commId 0xad5f…` → the **team secret** (see §2). Every teammate shows the same one.

💬 *Rank and cudaDev are both "2" here — will they always match?* (No. On many nodes, rank 6
might be GPU 2 of node 1. Rank = seat in the team; cudaDev = card in one box.)

---

## 2 · The handshake (bootstrap) — how strangers become a team

At the start, the GPUs do **not** know each other. They have never met. Bootstrap is the
"exchange phone numbers" step. It happens in 4 moves:

1. **Make a ticket.** Rank 0 calls `ncclGetUniqueId`. This ticket holds rank 0's address and a
   random secret (the `commId`).
2. **Share the ticket.** The ticket is copied to every rank (by MPI, or a file). Now everyone
   knows how to reach rank 0.
3. **Call the leader.** Every rank opens a plain TCP socket to rank 0. They line up into a
   **bootstrap ring** (0→1→2→3→0).
4. **Swap all addresses.** Going around the ring, every rank learns every other rank's address.
   Now the team is formed.

Real lines:

```
Bootstrap : Using bond0:10.168.4.6<0>
NET/Socket : Using [0]bond0 [1]hsn0 [2]hsn1
comm 0x3c75f60 rank 2 ... commId 0xad5f74829cf1c8d3 - Init START
comm 0x3c75f60 rank 2 ... commId 0xad5f74829cf1c8d3 - Init COMPLETE
```

- **`bond0`** = a plain, slow office network. Bootstrap uses it because saying "hello" is tiny.
  ⚠️ The real AI data will NOT go here — it uses NVLink or Slingshot RDMA (see §7).
  (On a **2-node** run the hello uses `hsn0` instead — real line:
  `Bootstrap : Using hsn0:10.150.1.8<0>`, commId `0x51677f56986aa0d3` shared by both nodes.)
- **`commId`** = the team secret. If two different teams accidentally used the same one, they
  would try to join → chaos. That is why it is random.
- **`Init START … Init COMPLETE`** = the whole setup (§2–§11) happens between these two lines.

💬 *Why use a slow network (`bond0`) just to say hello, instead of the fast one?* (The fast NICs
are busy and precious; a handshake is a few bytes, once. Do not waste the highway on a postcard.)

💬 *What breaks if step 2 fails and one rank never gets the ticket?* (That rank never joins; the
others wait for it forever → the classic "NCCL hang". This is why hangs often mean a bootstrap /
network-setup problem, not a math problem.)

---

## 3 · The machine map (topology)

Before planning, each GPU **explores the hardware** and writes a map: every GPU, every link,
every CPU, and how fast each connection is.

```
=== System : maxBw 80.0 totalBw 240.0 ===
+ PCI[24.0] - GPU/3000 (0)
              + NVL[80.0] - GPU/81000
              + NVL[80.0] - GPU/C1000
              + NVL[80.0] - GPU/41000
```

The words in `[]` are **link types**, fastest to slowest:

| tag | meaning | speed here |
|-----|---------|-----------|
| `NVL` | NVLink (GPU↔GPU direct wire) | 80 GB/s |
| `PIX` | through one PCI switch | fast |
| `PXB` | through several PCI switches | medium |
| `PHB` | through the CPU's PCI host bridge | slower |
| `SYS` | across CPUs (across NUMA / the whole box) | slowest |

- **`maxBw 80.0`** = the single fastest link (one NVLink).
- **`totalBw 240.0`** = all of a GPU's links added up = its total sharing power.
- Every GPU has `NVL[80.0]` to every other GPU → **all-to-all** (an NVSwitch). No slow hops.

💬 *If one GPU could only reach another through `SYS` (across CPUs), how would that change NCCL's
plan?* (It would avoid that slow hop, maybe use fewer channels, and be slower overall. The map
decides everything.)

### 3b · NCCL can DUMP the map as XML

You can ask NCCL to save the exact map it drew. We did — see **`logs/topo_4gpu.xml`** in the lab
folder. Turn it on yourself with:

```bash
export NCCL_TOPO_DUMP_FILE=$PWD/topo.xml     # the machine map
export NCCL_GRAPH_DUMP_FILE=$PWD/graph.xml   # the rings it planned
```

**`topo_4gpu.xml`** — the machine, as a tree of CPUs → PCI → GPUs/NICs:

```xml
<cpu numaid="3" arch="x86_64" vendor="AuthenticAMD">
  <pci busid="0000:03:00.0" link_speed="16.0 GT/s PCIe" link_width="16">
    <gpu dev="0" sm="80" rank="0" gdr="1">
      <nvlink target="0000:81:00.0" count="4" tclass="0x030200"/>   <!-- 4 NVLink lanes to GPU 2 -->
      <nvlink target="0000:c1:00.0" count="4"/>                     <!-- 4 lanes to GPU 3 -->
      <nvlink target="0000:41:00.0" count="4"/>                     <!-- 4 lanes to GPU 1 -->
    </gpu>
  </pci>
  <nic><net name="hsn1" speed="200000" gdr="0"/></nic>              <!-- 200 Gb Slingshot NIC -->
  <nic><net name="bond0" speed="1000" gdr="0"/></nic>              <!-- 1 Gb hello network -->
</cpu>
```

Read it like this:
- **`<gpu dev="0" sm="80">`** = GPU 0, an A100 (compute capability sm_80).
- **`<nvlink target=... count="4">`** = 4 NVLink lanes to that GPU. 4 lanes × ~20 GB/s = the
  **80 GB/s** you saw. Every GPU lists 3 such neighbors → all-to-all.
- **`<net name="hsn1" speed="200000">`** = a Slingshot NIC, 200000 Mb = 200 Gb/s.
- **`gdr="1"` on the GPU** = this GPU *can* do GPUDirect RDMA; **`gdr="0"` on bond0** = that plain
  NIC cannot.

**`graph_4gpu.xml`** — the rings NCCL planned, as a list of channels:

```xml
<graph id="0" pattern="4" nchannels="12" speedintra="20" typeintra="NVL">
  <channel><gpu dev="0"/><gpu dev="1"/><gpu dev="2"/><gpu dev="3"/></channel>  <!-- ring: 0→1→2→3 -->
  <channel><gpu dev="0"/><gpu dev="1"/><gpu dev="3"/><gpu dev="2"/></channel>  <!-- ring: 0→1→3→2 -->
  ...
</graph>
```

- **`nchannels="12"`** = 12 rings. Each `<channel>` is one ring's GPU order.
- **`speedintra="20" typeintra="NVL"`** = each hop is a 20 GB/s NVLink lane.

💬 *These two files are just text. Why is it powerful that NCCL will hand you its whole plan?*
(You can see exactly what it thinks your machine looks like. If a cable is bad or a GPU is
missing, the XML shows it — this is how you prove a hardware problem to the admins.)

---

## 4 · Rings and Trees (the routes)

NCCL builds two kinds of routes and keeps **many copies** of each (called channels, §5).

**Ring** — pass data around a circle. Real lines (rank's view of ring 00 on 4 GPUs):

```
Ring 00 : 0 -> 1 -> 2
Ring 01 : 0 -> 1 -> 3
Pattern 4, crossNic 0, nChannels 12, bw 20.0/20.0, type NVL/PIX
```

- NCCL built **12 rings** (`nChannels 12`), each a slightly different order, to use every link.
- A ring is great for **big** messages: each GPU sends and receives at the same time, so the
  speed equals the link speed no matter how many GPUs.

**Tree** — add up like a tournament. Real line:

```
Trees [0] 1/-1/-1->0->-1 [8] 3/-1/-1->0->1 ...
```

Read `child1/child2/.../->me->parent`. `-1` means "none". So `1/-1/-1->0->-1` = rank 0 has one
child (rank 1) and no parent (it is the top). A tree is great for **small** messages because the
answer reaches everyone in `log(N)` hops (few steps), not `N` hops.

💬 *For 1024 GPUs, a ring passes data 1023 times; a tree only ~10 layers. So why not always use
the tree?* (For BIG messages the ring keeps every link 100% busy = more total bandwidth. Tree
wins on latency for small; ring wins on bandwidth for big. Hence §11.)

### 4b · Ring all-reduce, step by step (the actual algorithm)

This is the clever trick that makes ring the king for big data. Watch 4 GPUs add their arrays.

**Setup.** Each GPU cuts its array into **4 equal parts** (one part per GPU). GPU0's parts are
called `A0 A1 A2 A3`, GPU1's are `B0 B1 B2 B3`, and so on. We want every part summed:
part 0 must become `A0+B0+C0+D0`, everywhere.

**Phase 1 — reduce-scatter (N−1 = 3 steps).** Each GPU sends ONE part to its right neighbor,
who **adds** it. The part being sent shifts each step, so no link is ever idle.

```
start:     GPU0[A0 A1 A2 A3]  GPU1[B0 B1 B2 B3]  GPU2[C0 C1 C2 C3]  GPU3[D0 D1 D2 D3]

step 1  (each sends one part → right, neighbor adds):
           GPU0 sends A3→GPU1 ... after adding:
           GPU1 has B3+A3 ,  GPU2 has C0+B0 ,  GPU3 has D1+C1 ,  GPU0 has A2+D2

step 2  (send the part you just grew):
           now e.g. GPU2 holds C0+B0+A0 , GPU3 holds D1+C1+B1 , ...

step 3:
           each GPU now OWNS one fully-summed part:
           GPU0→ part2 = A2+B2+C2+D2 ,  GPU1→ part3 ,  GPU2→ part0 ,  GPU3→ part1
```

After 3 steps, each GPU holds **one** complete sum (its "owned" part). Nobody has the whole
answer yet — but every part exists, fully summed, on exactly one GPU.

**Phase 2 — all-gather (N−1 = 3 steps).** Now just **circulate** those finished parts around the
ring so everyone ends up with all 4.

```
step 1: each GPU sends its owned finished part → right (no adding now, just copy)
step 2: pass it along again
step 3: done — every GPU has [part0 part1 part2 part3], all fully summed = the all-reduce result
```

**Why this is genius (the math):** every GPU sent `2(N−1)` parts total, each part = `1/N` of the
array. So each GPU moves about **`2 × array size`** of data — **no matter how many GPUs!** That is
why ring bus-bandwidth stays near the link speed whether you have 4 GPUs or 4000. (This is what the
`busbw` number in Parts A/B measures.)

💬 *A tree finishes in ~`log(N)` steps but a ring takes `2(N−1)`. Yet ring is faster for big data.
How can more steps be faster?* (Each ring step moves only `1/N` of the data and ALL links work at
once. The tree moves big blocks but leaves most links idle. Big data → keep every wire busy → ring.)

### 4c · Tree all-reduce (the low-latency cousin)

For **small** messages the cost is the number of hops, not bandwidth. A tree needs far fewer hops:

```
Reduce UP (leaves → root):        Broadcast DOWN (root → leaves):
        0                                   0
       / \           add on the way up     / \      copy on the way down
      1   2          → root 0 has the      1   2
     /                total sum            /
    3                                     3
```

Reduce up ≈ `log(N)` steps, broadcast down ≈ `log(N)` steps → about `2·log(N)` total. For 1024
GPUs that is ~20 hops instead of ~2046. Tiny messages fly. That is the `Tree` you saw in Part D.

**One real detail:** NCCL does not use ONE tree — it uses a **double binary tree**. A single tree
leaves half the links idle (a parent only talks downward). NCCL overlaps **two** trees, flipped,
so every link carries traffic in both directions at once. That is why NCCL's tree gets good
bandwidth *and* good latency. (You saw `Trees [0]…[23]` in the log = many trees, not one.)

### 4d · Other algorithms you will meet

- **Recursive halving / doubling** — another all-reduce shape for power-of-two GPU counts. Step 1:
  pair up (0↔1, 2↔3) and exchange half the data. Step 2: pair at distance 2 (0↔2) and exchange a
  quarter. After `log(N)` steps everyone has the sum. Fewer steps than a ring, more than a tree — a
  middle ground some libraries prefer.
- **All-to-all** — every GPU sends a **different** piece to every other GPU (not the same data to
  all). This is the heavy pattern in **Mixture-of-Experts** models and some parallel FFTs. It moves
  the most data of any collective, so the network matters most here.
- **In-network reduction (SHARP / NVLS)** — the switch adds the numbers itself (§11). On InfiniBand
  this is called **SHARP**; on NVLink it is **NVLS**. Same idea: do the math in the wires, not the GPUs.

💬 *Ring, tree, recursive-doubling, in-network — four ways to do the SAME all-reduce. Who decides
which one your training uses?* (NCCL's cost model, §10 — it predicts all of them for your machine
and message size, then picks the winner. You never choose.)

---

## 5 · Channels and CUDA threads

A **channel** is one worker lane: a piece of the GPU (a block of CUDA threads on one SM) that
drives one ring or tree. More channels = more links used at once = more speed.

```
24 coll channels, 0 nvls channels, 32 p2p channels, 8 p2p channels per peer
Max NThreads | 512 | 640 | 512 |   (LL | LL128 | Simple)
```

- **`24 coll channels`** = 24 lanes for collectives (all-reduce, etc.).
- **`32 p2p channels`** = lanes for point-to-point send/recv.
- **`Max NThreads 512/640`** = how many CUDA threads run each lane (depends on protocol).

💬 *Why not use 1000 channels for maximum speed?* (Each channel eats an SM and memory. Past a
point you run out of GPU and it gets slower. NCCL picks a number that fits the machine.)

---

## 6 · Transports: P2P, SHM, NET (how bytes actually move)

For each link, NCCL picks the best way to move bytes:

| transport | what it is | when |
|-----------|-----------|------|
| **P2P** | one GPU reads/writes another GPU's memory directly | GPUs in the same box (NVLink/PCIe) |
| **SHM** | copy through shared host (CPU) memory | GPUs in one box with no direct link |
| **NET** | send over the network (OFI/cxi = Slingshot RDMA) | GPUs on **different** boxes |

Real lines:

```
Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer/read       (same box: NVLink)
Allocated 4194660 bytes of shared memory in /dev/shm/nccl-1ivDVe   (SHM segment)
Channel 00/0 : 0[0] -> 1[0] [send] via NET/OFI/0              (different boxes: Slingshot)
```

- **`via P2P/direct pointer/read`** = GPU 1 literally reads GPU 2's memory. No copy, no CPU.
- **`via NET/OFI`** = it goes to the network card and across Slingshot (your Part B).

💬 *Same-box P2P was ~200 GB/s; cross-box NET was ~9 GB/s. Both are "direct". Why the 20× gap?*
(NVLink is a short wire on the board; the network is a long cable to another rack. Distance and
the extra network card cost speed — even with RDMA.)

---

## 7 · Buffers and chunking

NCCL does not send your whole message at once. It slices it into **chunks** and pipelines them
through pre-made **buffers** (staging areas).

```
Allocated shareable buffer 0x7f73bc000000 size 6291456 ipcDesc 0x...
P2P Chunksize set to 524288
```

- **`size 6291456`** = a 6 MB buffer (from `NCCL_BUFFSIZE`). `ipcDesc` = a handle so other GPUs
  can see this buffer.
- **`Chunksize 524288`** = 512 KB pieces. NCCL sends piece 1 while computing piece 2 → a pipeline,
  so links never sit idle.

💬 *Why chunk at all — why not send the whole 256 MB in one go?* (Pipelining: while chunk 1 is
flying across the wire, the GPU already works on chunk 2. Overlap = no waiting = faster.)

---

## 8 · The proxy (the background helper) — in detail

Here is the problem. A GPU is a math monster, but it is **bad at waiting**. Talking to a network
card means: post a send, wait for the card to say "done", handle errors, retry. If the GPU did
that, its thousands of math cores would sit frozen while one of them babysits a slow NIC. Huge waste.

So NCCL starts a **proxy**: a small **CPU helper thread, one per GPU**. Division of labour:

```
   GPU (compute)                         CPU proxy thread
   ───────────────                       ─────────────────
   run the ring/tree kernel      ──▶     "here is chunk 5, please send it"
   keep doing MATH on chunk 6            post chunk 5 to the NIC / shared memory
   ...never stops...             ◀──     "chunk 5 arrived, here is incoming chunk 4"
```

Real lines from the log:

```
Connection to proxy localRank 0 -> connection 0x7f7458000d20     <- GPU wires itself to its proxy
ncclPollProxyResponse Received new opId=0x7f74c3663ab0            <- proxy reports a step is done
```

- The GPU and proxy talk through a tiny mailbox (a FIFO queue). The GPU drops "please move this
  chunk" notes; the proxy does the slow network work and drops back "done" notes.
- This is why NCCL **overlaps** communication and computation: while the proxy pushes chunk 5
  across Slingshot, the GPU is already reducing chunk 6. Nobody waits (§7 chunking makes this work).
- On the network path, the proxy is also who calls libfabric/cxi — so **the proxy is the thing
  that actually touches RDMA** (remember this for the Python section: PyTorch → NCCL → proxy → cxi).

💬 *The proxy runs on the CPU, which is much slower than the GPU. Why doesn't that slow everything
down?* (It only does bookkeeping and hands work to the NIC — it never does the heavy math. As long
as it keeps up with the chunk pipeline, the GPU never notices it is there.)

---

## 9 · GDR — GPUDirect RDMA (and when it quietly turns OFF)

Normally, to send GPU data over the network you copy twice: GPU → CPU memory → network card.
**GDR** removes the middle step: the **network card reads GPU memory directly**. (Think of a
**courier who walks straight into your room, picks up the box, and carries it out** — instead of
a clerk who photocopies it at every desk. That courier is GDR.)

Real lines from the **2-node** run:

```
NET/OFI : GPU Direct RDMA Enabled for HCA 0 'cxi0'                  <- the Slingshot card CAN do GDR
GPU Direct RDMA Disabled for GPU 3000 / HCA 0 (distance 7 > 6)      <- ...but NOT for this GPU
```

Read carefully — there are two different lines and they seem to disagree. They do not:

- The cxi card **supports** GDR (`Enabled for HCA cxi0`).
- **But** for our GPU (busId 3000) NCCL **turned it off**, because that GPU and the cxi0 card are
  **too far apart on the board** (`distance 7 > 6`, the level we allowed with `NCCL_NET_GDR_LEVEL`).
  So on our run, data still took the long way: GPU → host memory → card.

Honest takeaway: our Part B (~9 GB/s) used **RDMA** (kernel-bypass network, no CPU in the loop for
the transfer) but **not full GPUDirect** for that GPU — because we pinned everyone to one NIC
(`cxi0`) that is far from some GPUs. If each GPU could use its own **nearby** NIC (cxi0 or cxi1),
GDR would switch on and the number would climb. That is the real, measurable cost of the cxi1
work-around (see the lab README). On the **TCP** path (Part C) GDR is always off.

💬 *We turned RDMA ON and still only got 9 GB/s. This log line explains why. What is it, and what
would we change to go faster?* (GDR was disabled by distance; giving each GPU its nearest NIC —
fixing the cxi1 issue — would enable GDR and raise the bandwidth.)

---

## 10 · The tuning cost model — NCCL's secret cheat-sheet ⭐

This is the most important table in the whole log. Before running, NCCL **predicts** the speed of
every way to send, and saves it. Then per message size it instantly picks the best.

```
    Algorithm   |            Tree                |            Ring                |
    Protocol    |   LL   |  LL128  |  Simple |   LL   |  LL128  |  Simple |
    AllReduce   | 10.4/29| 21.5/86 | 24.0/110| 10.2/40| 25.4/147| 28.8/160|
                   ↑ latency(µs) / bandwidth(GB/s)
```

Read each cell as **latency (µs) / bandwidth (GB/s)** — start-up delay, and top speed.

The three **protocols** are three ways to pack the data — think of shipping a gift: **LL** = a
postcard (tiny, leaves instantly, best for *small* messages), **LL128** = a padded envelope (a
middle choice tuned for NVLink), **Simple** = a big truck (slow to load but carries a LOT, best
for *big* messages). The table below is how NCCL decides which one to send.

- **Small message** → start-up delay wins. Cheapest is **Tree-LL (10.4 µs)** or **Ring-LL
  (10.2 µs)**. → NCCL picks **LL**.
- **Big message** → top speed wins. Fastest is **Ring-Simple (160 GB/s)**. → NCCL picks
  **Ring + Simple**.

**That single table is the reason** Part D showed LL winning small and Simple winning big — you
were watching this cost model in action.

```
threadThresholds 8/8/64 | 32/8/64 | 512 | 512
```

= the message sizes where NCCL flips from one protocol to the next.

💬 *NCCL measured these numbers for YOUR machine at startup. Why not just hard-code "always use
Ring-Simple"?* (Every machine is different — NVLink vs PCIe vs network change the numbers. Measuring
makes NCCL fast everywhere without you tuning anything.)

---

## 11 · NVLS — NVLink SHARP (the feature you do NOT have)

```
NVLS multicast support is not available on dev 0
```

**NVLS** lets the **NVSwitch chip itself add the numbers** as they pass through — the network does
the math, not the GPUs. It is very fast, but needs newer hardware (Hopper H100 + NVSwitch v3).
On these A100s it is off, so NCCL uses Ring/Tree instead.

💬 *If the switch could add numbers itself, why is that faster than a ring?* (No GPU has to receive,
add, and re-send. The data is summed "in flight". Fewer trips = less time.)

---

## 12 · The collectives zoo

All-reduce is one of a family. Same machinery, different pattern:

| collective | what it does | picture |
|------------|-------------|---------|
| **AllReduce** | everyone adds, everyone gets the total | today's lab |
| **Broadcast** | one GPU's data copied to all | teacher hands out a sheet |
| **Reduce** | everyone adds, only rank 0 keeps it | collect homework to teacher |
| **AllGather** | everyone collects everyone's piece | each gets the full class list |
| **ReduceScatter** | add, then split the result across GPUs | add, then each keeps a slice |
| **Send/Recv** | one GPU to one GPU | passing one note |

The **operation** can be `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin`. The **datatype** can be
`ncclFloat`, `ncclHalf` (fp16), `ncclBfloat16`, `ncclInt`, … (AI uses fp16/bf16 to go faster.)

💬 *Training an AI averages the gradients from all GPUs. Which collective is that — AllReduce with
Sum then divide, or Broadcast?* (AllReduce-Sum, then divide by N = the average. This runs millions
of times during training.)

---

## 13 · Useful knobs (environment variables)

You set these before running. The safe ones to explore:

| variable | what it does |
|----------|-------------|
| `NCCL_DEBUG=INFO` | print the setup log (what this guide reads) |
| `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING` | show only some parts of the log |
| `NCCL_ALGO=Ring` / `Tree` | force the shape (normally: don't) |
| `NCCL_PROTO=Simple` / `LL` / `LL128` | force the protocol (normally: don't) |
| `NCCL_MAX_NCHANNELS=4` | limit the number of lanes |
| `NCCL_NET=Socket` | force slow TCP (Part C) |
| `NCCL_P2P_DISABLE=1` | turn off direct GPU↔GPU (forces SHM — slower, for testing) |

💬 *`NCCL_ALGO` and `NCCL_PROTO` exist, but experts say "leave them unset". Why give a knob you
should not touch?* (For 99% of cases NCCL's cost model beats a human. The knobs are for rare
debugging or a weird machine — not for daily use.)

---

## 14 · Debug like a pro — where to look when it breaks

| symptom | first thing to check in the log |
|---------|--------------------------------|
| **Hang** (nothing happens) | did every rank reach `Init COMPLETE`? A missing rank = stuck bootstrap (§2). |
| **Slow** | look at §10 — is it picking Simple/Ring for big messages? Is `nChannels` low? |
| **`Device or resource busy` / NET error** | a transport/network setup problem (§6, §9), not your math. |
| **Wrong answer / `#wrong` not 0** | a data or datatype bug in your code, not NCCL. |
| **Crash right at start** | version mismatch (NCCL vs test vs CUDA) — check the `NCCL version` line. |

💬 *A friend says "my 8-GPU training hangs at the start". Using this guide, what ONE thing do you
ask them to grep first?* (`grep "Init COMPLETE" — how many? If not 8, one rank never joined =
bootstrap/network problem.)

---

## You made it 🎓

You can now open a raw NCCL log — thousands of lines that look like noise — and **read the story**:
who joined the team, what the machine looks like, which routes were planned, how bytes move, and
why NCCL chose what it chose. That is a real, rare skill. Very few people can do it. Now you can.
