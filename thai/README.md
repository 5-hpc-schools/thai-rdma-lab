# Day-3 NCCL-over-Slingshot Lab (LANTA / ThaiSC)

Hands-on lab: run GPU collective communication (all-reduce) on LANTA's A100 nodes, see the
Slingshot (cxi) RDMA path, and compare it against TCP. Built for ~80 Thai undergrads (weak
English → all student text is simple English). Students `ssh` to LANTA directly and `sbatch`.

- Student worksheet: [`worksheet/LAB.md`](worksheet/LAB.md)
- Runnable scripts: [`lab/`](lab/) + [`env_slingshot.sh`](env_slingshot.sh) + [`build_nccl_tests.sh`](build_nccl_tests.sh)
- On LANTA everything is staged under `/project/tn999992-rdma/day3-nccl-lab/`
  (group `tn999992` readable; students `cp -r .../lab ~/nccl-lab`).

Cluster facts: LANTA = HPE Cray EX, `switch/hpe_slingshot`, GPU nodes = 4× A100-SXM4-40GB,
Slingshot-11 200Gb 2-NIC (cxi0/cxi1). NCCL module is 2.18.x; no InfiniBand verbs (so Day-1/2
verbs/UCX callbacks do NOT transfer here — deferred).

---

## Promotion ledger

| stage | date | evidence |
|-------|------|----------|
| sandbox | 2026-07-15 | scripts written |
| **incubating** | 2026-07-15 | **All three parts verified end-to-end with the delivered `lab/*.sbatch` scripts (jobs 5978001/2/3):**<br>• Part A intra-node 4×A100 NVLink: **215 GB/s** peak busbw, `type NVL/PIX`.<br>• Part B inter-node 2-node **Slingshot cxi RDMA: 9.3 GB/s** to 256 MB, log `Selected provider is cxi (found 1 nics)` + `via NET/OFI`.<br>• Part C inter-node TCP (`NCCL_NET=Socket`): **1.75 GB/s**.<br>• **RDMA / TCP = 5.3×** — the teaching contrast holds.<br>• Part D explore: protocols LL **42** / LL128 **31** / Simple **21** GB/s @1MB; Ring **102** vs Tree **77** avg; topology `NVL[80.0]` all-to-all + `nChannels 12` (jobs 5978068/5978140).<br>• Part E fill-in-the-blank `ncclAllReduce`: builds + `RESULT OK` (job 5978083). |
| graduated | — | needs a class dry-run (80-student scale) + a rendered `~/handson/` copy |

---

## 干货 (only the surprises — things that cost hours)

1. **The prior probe's "NCCL works on Slingshot" was FALSE.** The old `.thaisc-probe` logs
   show `Selected provider is cxi` and `Connected all rings`, but **no run ever produced a
   single busbw data row** — every one died at the first collective. "Connected all rings"
   is only the *control plane*; it does not mean data moved. Never trust init logs as success.

2. **The current nccl-tests (v2.19.6+) is silently broken with NCCL 2.18.x.** Even a **1-GPU**
   all-reduce dies: `all_reduce.cu:50 invalid argument` / `common.cu:502 illegal memory access`.
   Bare CUDA works, a hand-written 1-GPU `ncclAllReduce` works — only nccl-tests fails. Fix:
   check out the **2023-era commit `e98ef24`** (matches the NCCL 2.18 era). This alone unblocked
   intra-node. The failure looks like a fabric/GPU bug but is purely a version mismatch.

3. **CUDA versions must match.** LANTA's module `nccl/2.18.1-1+cuda11.0` is a CUDA-11 build;
   mixing it with a CUDA-12 nccl-tests corrupts the context. Use the CUDA-12 NCCL 2.18.5 inside
   the NVHPC SDK: `/opt/nvidia/hpc_sdk/.../24.11/comm_libs/12.6/nccl`.

4. **Multi-node Slingshot needs a VNI, and Slurm does NOT give it automatically.** You must add
   `#SBATCH --network=job_vni,single_node_vni`. Without it, cxi cannot open a data domain:
   `fi_domain: Function not implemented` (single rank) or `Device or resource busy` on the first
   4-byte handshake (multi rank). With it, `SLINGSHOT_VNIS`/`SLINGSHOT_DEVICES` appear and the
   handshake succeeds. Put network options at the **job** level, not on `srun` (an `srun
   --network=...` can override the job VNI).

5. **On LANTA you must pin to ONE cxi NIC (`FI_CXI_DEVICE_NAME=cxi0`).** This is a LANTA
   site issue, confirmed against the authoritative same-architecture reference:
   - CSCS Alps (also HPE Cray EX + Slingshot) documents the *intended* config as BOTH NICs
     (`SLINGSHOT_DEVICES=cxi0,cxi1`, `NCCL_CROSS_NIC=1`, `NCCL_NET="AWS Libfabric"`,
     `NCCL_NCHANNELS_PER_NET_PEER=4`) — https://docs.cscs.ch/software/communication/nccl/
   - Running **that exact CSCS recipe on LANTA still fails**: `Device or resource busy` on the
     first 4-byte handshake, on both `dev:0` and `dev:1`, for 1-rank and 4-rank/node (job 5978054).
   - With `FI_CXI_DEVICE_NAME=cxi0` + `NCCL_CROSS_NIC=0` → `cxi (found 1 nics)` and the full
     8 B→256 MB sweep completes at **9.3 GB/s** (job 5978002).
   So the two-NIC path is broken **on LANTA specifically** (cxi1 VNI/service, most likely) — a
   real ThaiSC support ticket, not a user error. One NIC is plenty for the lab; Part C (TCP) is a
   guaranteed fallback.

### Cross-check against public sources (2026-07-15)
- VNI requirement is documented by HPE: *Control HPE Slingshot Network Resources Using Slurm*
  ("Slurm allocates a block of VNIs to each job"; missing VNI → multi-node launch failures).
- nccl-tests vs NCCL-2.18 crashes ("illegal memory access"/"invalid argument", multi-GPU) are a
  known class in NVIDIA/nccl issues; the `e98ef24` pin is the empirical fix here.
- Fabric env matches NERSC Perlmutter and CSCS Alps recommendations (same HPE Cray EX + cxi).

6. **Fabric env that matters** (HPE `ccl_env.sh` + NERSC/CSCS Perlmutter guidance, same arch):
   `FI_CXI_RX_MATCH_MODE=software`, `FI_CXI_DEFAULT_TX_SIZE=16384`, the three
   `FI_CXI_RDZV_{GET_MIN,THRESHOLD,EAGER_SIZE}=0` (stops hangs), `NCCL_PROTO=^LL128`.

7. **Killed jobs can leave cxi busy** — timeout-killing hung NCCL jobs is a bad habit here; use
   `timeout` inside the step sparingly and let jobs end cleanly.

8. **Queue reality**: 1-node GPU jobs usually start fast; 2+-node jobs can sit in `(Resources)`
   for many minutes at busy times. For 80 students, plan intra-node (Part A) as the per-student
   hands-on and multi-node (Part B/C) as submit-and-wait or instructor-led.

---

## Recipe (for an agent re-building this from zero)

**必验前提 (must hold, else the lab is wrong):**
- GPU node has 4× A100 and `cxi_stat` shows `Link state: up` on cxi0.
- `module load cuda/12.6` gives nvcc; NVHPC SDK NCCL exists at `comm_libs/12.6/nccl` (2.18.5).
- `sbatch --network=job_vni` is accepted (switch/hpe_slingshot).

**本质断言 (what "works" means):**
1. Build: `build_nccl_tests.sh` → `all_reduce_perf` exists (nccl-tests `e98ef24` + SDK NCCL + MPI).
2. Part A: `sbatch lab/partA_1node.sbatch` → output has ≥1 numeric busbw row and EXIT 0;
   log contains `via P2P` (NVLink). Peak busbw is O(100 GB/s).
3. Part C: `sbatch lab/partC_multinode_tcp.sbatch` → ≥1 busbw row, EXIT 0, `NCCL_NET=Socket`.
4. Part B: `sbatch lab/partB_multinode_rdma.sbatch` → log contains `Selected provider is cxi`
   and `GDRDMA`; small messages complete. (Large-message completion = graduation target.)

`smoke.sh` checks assertions 1–3 by parsing real job output.

## Layout
```
thai/
  README.md                 this file
  build_nccl_tests.sh       the ONE build recipe
  env_slingshot.sh          multi-node cxi env (sourced by Part B)
  lab/
    partA_1node.sbatch      intra-node NVLink  (verified 215 GB/s)
    partB_multinode_rdma.sbatch   inter-node cxi RDMA  (verified 9.3 GB/s)
    partC_multinode_tcp.sbatch    inter-node TCP  (verified 1.75 GB/s)
    partD_explore.sbatch    protocols (Simple/LL/LL128) + Ring vs Tree + topology map  (verified)
    nccl_hello.cu           fill-in-the-blank ncclAllReduce (student writes 2 lines)
    nccl_hello_solution.cu  answer key (verified RESULT OK)
    build_and_run_hello.sh  compile + run helper for Part E
  worksheet/LAB.md          simple-English student guide (Parts A–E)
  smoke.sh                  machine-checkable assertions
```
