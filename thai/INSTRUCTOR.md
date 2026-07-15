# Instructor guide — Day-3 NCCL lab (LANTA)

Not for students. Run-of-show, answer key, expected numbers, and how to survive the queue.

## How to distribute
- **Student handout**: `worksheet/LAB.html` + `worksheet/CHEATSHEET.html` (self-contained; email /
  LMS / print / share link). Students read on their laptop, type into their LANTA ssh terminal.
- **Code + binary**: already on LANTA at `/project/tn999992-rdma/day3-nccl-lab/` (group `tn999992`,
  world-readable, prebuilt binary). Students only `cp -r .../lab ~/nccl-lab`. No building needed
  (except their own Part E, which the helper script compiles).
- Nothing else to install. `smoke.sh` re-checks the whole lab if you want to confirm before class.

## The ONE logistics decision: multi-node queue (Parts B, C)
80 students × a 2-node job = 160 nodes; they will NOT all start at once. Pick one:
1. **Instructor-led (recommended):** you run Part B and Part C live on the projector (pre-grab
   `salloc -p gpu -N2 --network=job_vni,single_node_vni -t 2:00:00` before class). Students do
   A, D, E themselves (1-node, start fast). Best for time-boxed sessions.
2. **Everyone submits:** fine if you have a reservation. Tell them to `sbatch partB` early and
   move on to Part D/E while it queues, then come back to read the `.out`.
Parts A, D, E are 1-node and schedule quickly — always student-run.

## Run-of-show (half day, ~3 h)
| time | what |
|------|------|
| 0:00 | 10-min intro: what is all-reduce, NVLink vs network, RDMA vs TCP (a few slides) |
| 0:10 | Part A (student) — NVLink, ~200 GB/s. Everyone gets a number. |
| 0:35 | Part B + C — RDMA vs TCP. Instructor-led demo OR submit-and-wait. |
| 1:15 | Part D (student) — Look inside NCCL: protocols, ring/tree, topology. |
| 2:00 | Part E (student) — write ncclAllReduce. The payoff: they run real NCCL code. |
| 2:30 | Part F (optional, for the curious) — full NCCL_DEBUG log, the 5-step boot sequence + scavenger hunt. |
| 2:45 | wrap: recap the 5× RDMA number and the protocol crossover. |

## Expected numbers (4×A100 nodes, verified 2026-07-15)
- Part A (NVLink, 512 MB): busbw ~**215 GB/s** in-place; log `type NVL/PIX`.
- Part B (2-node cxi RDMA, 256 MB): ~**9.3 GB/s**; log `Selected provider is cxi (found 1 nics)`.
- Part C (2-node TCP, 256 MB): ~**1.75 GB/s**. → **RDMA ≈ 5× TCP** (the headline).
- Part D protocols @1 MB: LL ~42, LL128 ~31, Simple ~21. @64 MB: Simple ~169, LL128 ~123, LL ~71.
  (crossover: LL wins small, Simple wins big.) Ring vs Tree @64 MB: 169 vs 132.
- Part E: prints `RESULT OK` (all-reduce of 0+1+2+3 = 6).

## Answer key (worksheet questions)
- **A**: busbw is huge (~200 GB/s) because NVLink is a direct GPU-to-GPU wire (no CPU, no network).
- **B (guess)**: the network is *slower* than NVLink (~9 vs ~200 GB/s) — a cable across nodes can't
  match an on-board link. Still, it is real RDMA.
- **C**: RDMA vs TCP on the same cable — RDMA lets the NIC read GPU memory directly (kernel bypass,
  no copies); TCP makes the CPU copy every byte. That's the ~5× gap.
- **D1**: no single best protocol — LL is low-latency (small msgs), Simple is full-bandwidth (big).
- **D2**: Ring for big messages here.
- **D3**: all-to-all NVLink (NVSwitch) means any ring order has equal 80 GB/s hops → easy, fast ring.
- **E**: changing `ncclSum`→`ncclMax` makes every GPU hold the maximum (3) instead of the sum (6).
- **F**: two networks because a handshake is tiny and rare (cheap `bond0`/TCP is fine) while AI
  data is huge and constant (needs the NVLink / Slingshot-RDMA highway). Scavenger-hunt answers
  on our nodes: totalBw 240 GB/s, ~12 rings, `via P2P/direct pointer/read`, 24 coll channels.

## Known issue to mention honestly
Only **one** of the two Slingshot NICs (cxi0) is usable for us right now; the 2nd (cxi1) returns
"Device or resource busy" — a LANTA site config matter (see README 干货 #5), not a student error.
One NIC (9.3 GB/s) is plenty for the lesson. If a Part B/C job hangs, cancel and resubmit.

## Reset / re-verify before class
```bash
ssh thai
bash /project/tn999992-rdma/day3-nccl-lab/smoke.sh   # re-runs assertions against latest results
# to regenerate results: sbatch the lab/part{A,B,C}.sbatch from results/ dir first
```
