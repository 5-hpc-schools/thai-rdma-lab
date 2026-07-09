# คู่มือฉบับสั้น: Roofline Analysis บน LANTA HPC — ระดับ Node และ Multi-node

> เป้าหมาย: ให้ผู้เรียนรัน benchmark สั้น ๆ บน LANTA, เก็บผลเป็น CSV, ดาวน์โหลดผลไปเปิดใน Google Colab และสร้างกราฟ Roofline เพื่ออ่านว่า kernel มีแนวโน้ม **memory-bound**, **compute-bound**, หรือเริ่มติด **communication-bound** เมื่อขยายหลาย node

คู่มือนี้ออกแบบให้เป็น **copy-paste only** สำหรับเวลาสอนสั้น ประมาณ 60–90 นาที  
คำสั่งทุกช่วงให้คัดลอกไปวางได้เลย โดยไม่ต้องเขียนไฟล์เอง

---

## 0. ภาพรวมที่ผู้เรียนควรรู้ก่อนรัน

Roofline model ใช้ดูความสัมพันธ์ระหว่าง

```text
Performance = FLOP/s
Operational intensity = FLOP / byte
Attainable performance <= min(Peak FLOP/s, Memory bandwidth x Operational intensity)
```

อ่านกราฟง่าย ๆ:

- จุดอยู่บนเส้นเอียงด้านซ้าย → งานมักติด **memory bandwidth**
- จุดอยู่ใกล้เส้นนอนด้านบน → งานมักติด **compute peak**
- จุดอยู่ไกลใต้หลังคา → อาจมีปัญหาอื่น เช่น vectorization, BLAS/thread binding, cache, NUMA, kernel launch overhead, หรือ communication
- สำหรับ multi-node อย่าเอา roofline ของ node เดียวไปคูณจำนวน node แล้วสรุปทันที เพราะ multi-node เพิ่มคอขวดใหม่คือ **network / communication**

ในคลาสนี้เราจะทำ 3 แบบ:

1. **Single-node local roofline**: วัด vector update และ matrix multiplication บน 1 node
2. **GPU roofline แบบสั้น**: วัด PyTorch vector update และ matrix multiplication บน 1 GPU ถ้ามี GPU allocation
3. **Multi-node practical roofline**: รัน benchmark เดิมบนหลาย node เพื่อดู aggregate local roofline และ optional MPI Allreduce bandwidth เพื่อเริ่มคิดเรื่อง communication roofline

> หมายเหตุ: benchmark ชุดนี้เป็น teaching-grade ใช้ Python/NumPy/PyTorch เพื่อให้รันเร็วและเข้าใจ workflow ได้ทันที ไม่ใช่ผลระดับ publication-grade หากต้องการผลจริงควรใช้ STREAM, HPL/HPCG, LIKWID, Nsight Compute, Intel Advisor, PAPI หรือ hardware counters เพิ่มเติม

---

## 1. Login และสร้างพื้นที่ทำงานบน LANTA

รันจากเครื่องของผู้เรียน:

```bash
ssh YOUR_LANTA_USERNAME@lanta.nstda.or.th
```

เมื่อเข้า LANTA แล้ว คัดลอกทั้งหมดนี้:

```bash
mkdir -p ~/roofline-training
cd ~/roofline-training
export LANTA_ACCOUNT=tn999992
pwd
```

ถ้าใช้ account/project อื่น ให้แก้บรรทัดนี้ก่อนรันงาน:

```bash
export LANTA_ACCOUNT=YOUR_PROJECT_ID
```

---

## 2. สำรวจ environment, modules, conda/mamba และ package ที่มีอยู่

สร้างไฟล์สำรวจระบบ:

```bash
cat > 00_env_browse.sh <<'BASH'
#!/bin/bash
set -u

echo "===== Basic identity ====="
date
hostname
whoami
pwd

echo "===== LANTA helper commands, if available ====="
command -v myquota >/dev/null 2>&1 && myquota || true
command -v sbalance >/dev/null 2>&1 && sbalance || true
command -v myqueue >/dev/null 2>&1 && myqueue || true

echo "===== Slurm overview ====="
command -v sinfo >/dev/null 2>&1 && sinfo -s || true
command -v squeue >/dev/null 2>&1 && squeue -u "$USER" || true

echo "===== Module system ====="
module --version 2>&1 || true
module avail 2>&1 | head -n 120 || true
module spider Mamba 2>&1 || true
module spider Python 2>&1 | head -n 80 || true
module spider GCC 2>&1 | head -n 80 || true

echo "===== Try Mamba/Conda environment ====="
module purge 2>/dev/null || true
module load Mamba/23.11.0-0 2>/dev/null || true
module list 2>&1 || true
which python || true
python -V || true
which conda || true
conda info --envs 2>/dev/null || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pytorch-2.2.2 2>/dev/null || true
fi

echo "===== Python packages in active environment ====="
python - <<'PY' || true
import importlib.util, sys
print("python:", sys.version)
for pkg in ["numpy", "pandas", "matplotlib", "torch", "mpi4py"]:
    spec = importlib.util.find_spec(pkg)
    print(f"{pkg:10s}:", "OK" if spec else "missing")
    if spec:
        try:
            mod = __import__(pkg)
            print(f"  version:", getattr(mod, "__version__", "unknown"))
        except Exception as e:
            print("  import error:", repr(e))
try:
    import torch
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("torch gpu count:", torch.cuda.device_count())
except Exception as e:
    print("torch cuda check skipped:", repr(e))
PY
BASH
chmod +x 00_env_browse.sh
```

รันสำรวจ:

```bash
bash 00_env_browse.sh |& tee 00_env_browse.log
```

สิ่งที่ควรให้ผู้เรียนดูใน log:

```bash
grep -E "numpy|pandas|matplotlib|torch|mpi4py|cuda|Mamba|partition|gpu|cpu" 00_env_browse.log | head -n 80
```

ความหมายสั้น ๆ:

- `numpy` จำเป็นสำหรับ CPU benchmark
- `torch` จำเป็นสำหรับ GPU benchmark
- `pandas` และ `matplotlib` ไม่จำเป็นต้องมีบน LANTA เพราะเราจะ plot ใน Colab ได้
- `mpi4py` จำเป็นเฉพาะ optional communication benchmark
- ถ้าไม่เห็น `pytorch-2.2.2` ให้ดูผลจาก `conda info --envs` แล้วเปลี่ยน environment ใน job script ภายหลัง

---

## 3. สร้าง benchmark script สำหรับ CPU/GPU roofline

คัดลอกบล็อกนี้เพื่อสร้าง `roofline_bench.py`:

```bash
cat > roofline_bench.py <<'PY'
#!/usr/bin/env python3
"""
Teaching-grade Roofline microbenchmarks for LANTA / Slurm clusters.

This script intentionally uses common Python packages (NumPy and optional PyTorch)
so that learners can run it quickly in a short training session.  The bytes are
algorithmic/minimum byte estimates, not hardware counter measurements.  For
publishable analysis, replace or validate the bandwidth/traffic numbers with
STREAM, LIKWID, Nsight Compute, Intel Advisor, PAPI, or vendor profilers.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


HOST = socket.gethostname()
JOB_ID = os.environ.get("SLURM_JOB_ID", "no_slurm")
RANK = env_int("SLURM_PROCID", env_int("PMI_RANK", 0))
LOCAL_RANK = env_int("SLURM_LOCALID", 0)
WORLD_SIZE = env_int("SLURM_NTASKS", env_int("PMI_SIZE", 1))
CPUS_PER_TASK = env_int("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
NODELIST = os.environ.get("SLURM_JOB_NODELIST", "")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def best_time(fn: Callable[[], object], repeat: int = 5, warmup: int = 2) -> float:
    for _ in range(max(0, warmup)):
        fn()
    times: List[float] = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def base_row(args: argparse.Namespace, backend: str, kernel: str, dtype: str, n: int) -> Dict[str, object]:
    return {
        "timestamp": now_iso(),
        "label": args.label,
        "scope": args.scope,
        "backend": backend,
        "host": HOST,
        "job_id": JOB_ID,
        "rank": RANK,
        "local_rank": LOCAL_RANK,
        "world_size": WORLD_SIZE,
        "cpus_per_task": CPUS_PER_TASK,
        "nodelist": NODELIST,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kernel": kernel,
        "dtype": dtype,
        "N": n,
    }


def add_metrics(row: Dict[str, object], seconds: float, flops: float, bytes_est: float, notes: str) -> Dict[str, object]:
    row = dict(row)
    row.update({
        "seconds": seconds,
        "flops": flops,
        "bytes_estimate": bytes_est,
        "oi_flop_per_byte": flops / bytes_est if bytes_est > 0 else float("nan"),
        "gflops": flops / seconds / 1e9 if seconds > 0 else float("nan"),
        "gbps": bytes_est / seconds / 1e9 if seconds > 0 else float("nan"),
        "notes": notes,
    })
    return row


def run_numpy_cpu(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not has_module("numpy"):
        print("NumPy not found; skip CPU NumPy benchmark.", flush=True)
        return []
    import numpy as np

    rows: List[Dict[str, object]] = []
    dtype = np.float64
    itemsize = np.dtype(dtype).itemsize

    # Vector TRIAD-like update.  NumPy may allocate a temporary, so this is a
    # teaching/algorithmic estimate, not exact DRAM traffic from counters.
    for n in args.cpu_vector_sizes:
        n = int(n)
        a = np.empty(n, dtype=dtype)
        b = np.ones(n, dtype=dtype)
        c = np.ones(n, dtype=dtype)
        scalar = dtype(3.141592653589793)

        def triad() -> object:
            a[:] = b + scalar * c
            return a

        sec = best_time(triad, repeat=args.repeat, warmup=args.warmup)
        flops = 2.0 * n
        bytes_est = 3.0 * n * itemsize
        rows.append(add_metrics(
            base_row(args, "cpu_numpy", "vector_triad", "float64", n),
            sec, flops, bytes_est,
            "algorithmic_minimum_bytes; NumPy expression may use temporaries"
        ))

    for n in args.cpu_gemm_sizes:
        n = int(n)
        a = np.random.random((n, n)).astype(dtype)
        b = np.random.random((n, n)).astype(dtype)

        def gemm() -> object:
            return a @ b

        sec = best_time(gemm, repeat=args.repeat, warmup=args.warmup)
        flops = 2.0 * n * n * n
        bytes_est = 3.0 * n * n * itemsize
        rows.append(add_metrics(
            base_row(args, "cpu_numpy", "gemm", "float64", n),
            sec, flops, bytes_est,
            "algorithmic_minimum_bytes; performance depends on linked BLAS threads"
        ))

    return rows


def run_torch_gpu(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not has_module("torch"):
        print("PyTorch not found; skip GPU benchmark.", flush=True)
        return []
    import torch

    if not torch.cuda.is_available():
        print("PyTorch found, but CUDA is not available in this job; skip GPU benchmark.", flush=True)
        return []

    device_count = torch.cuda.device_count()
    dev_id = min(LOCAL_RANK, device_count - 1)
    torch.cuda.set_device(dev_id)
    device = torch.device(f"cuda:{dev_id}")
    print(f"Using CUDA device {dev_id}: {torch.cuda.get_device_name(dev_id)}", flush=True)

    # Let PyTorch use TensorFloat-32 on A100 for a realistic training-oriented SGEMM roof.
    # This is not FP64 roofline.  For FP64, change dtype to torch.float64.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    rows: List[Dict[str, object]] = []
    dtype = torch.float32
    itemsize = 4

    for n in args.gpu_vector_sizes:
        n = int(n)
        a = torch.empty(n, device=device, dtype=dtype)
        b = torch.ones(n, device=device, dtype=dtype)
        c = torch.ones(n, device=device, dtype=dtype)
        scalar = 3.1415927

        def triad() -> object:
            a.copy_(b + scalar * c)
            torch.cuda.synchronize()
            return a

        sec = best_time(triad, repeat=args.repeat, warmup=args.warmup)
        flops = 2.0 * n
        bytes_est = 3.0 * n * itemsize
        rows.append(add_metrics(
            base_row(args, "gpu_torch", "vector_triad", "float32", n),
            sec, flops, bytes_est,
            "algorithmic_minimum_bytes; PyTorch CUDA elementwise expression"
        ))

    for n in args.gpu_gemm_sizes:
        n = int(n)
        a = torch.randn((n, n), device=device, dtype=dtype)
        b = torch.randn((n, n), device=device, dtype=dtype)

        def gemm() -> object:
            c = a @ b
            torch.cuda.synchronize()
            return c

        sec = best_time(gemm, repeat=args.repeat, warmup=args.warmup)
        flops = 2.0 * n * n * n
        bytes_est = 3.0 * n * n * itemsize
        rows.append(add_metrics(
            base_row(args, "gpu_torch", "gemm", "float32", n),
            sec, flops, bytes_est,
            "algorithmic_minimum_bytes; TF32 may be used on A100 unless disabled"
        ))

    return rows


def write_csv(rows: List[Dict[str, object]], outdir: Path, prefix: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_job{JOB_ID}_rank{RANK}_host{HOST}.csv".replace("/", "_")
    path = outdir / filename
    fieldnames = [
        "timestamp", "label", "scope", "backend", "host", "job_id", "rank", "local_rank",
        "world_size", "cpus_per_task", "nodelist", "python", "platform", "kernel", "dtype", "N",
        "seconds", "flops", "bytes_estimate", "oi_flop_per_byte", "gflops", "gbps", "notes"
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Teaching-grade Roofline benchmark")
    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--scope", default="single_node", help="label such as single_node or multinode_local")
    parser.add_argument("--label", default="lanta_roofline_training")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cpu-vector-sizes", type=parse_int_list, default=parse_int_list("10000000,30000000"))
    parser.add_argument("--cpu-gemm-sizes", type=parse_int_list, default=parse_int_list("512,1024,2048"))
    parser.add_argument("--gpu-vector-sizes", type=parse_int_list, default=parse_int_list("33554432,67108864"))
    parser.add_argument("--gpu-gemm-sizes", type=parse_int_list, default=parse_int_list("1024,2048,4096"))
    args = parser.parse_args()

    print("===== roofline_bench.py environment =====", flush=True)
    for k in ["SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        print(f"{k}={os.environ.get(k)}", flush=True)
    print(f"host={HOST} rank={RANK} local_rank={LOCAL_RANK} world_size={WORLD_SIZE}", flush=True)

    rows: List[Dict[str, object]] = []
    if args.backend in ("cpu", "auto"):
        rows.extend(run_numpy_cpu(args))
    if args.backend in ("gpu", "auto"):
        rows.extend(run_torch_gpu(args))

    if not rows:
        print("No benchmark rows were generated. Check Python packages and allocation.", flush=True)
        sys.exit(2)

    out = write_csv(rows, Path(args.outdir), prefix=f"roofline_{args.scope}_{args.backend}")
    print(f"Wrote {len(rows)} rows to {out}", flush=True)
    for r in rows:
        print(f"{r['backend']:10s} {r['kernel']:14s} N={r['N']} OI={float(r['oi_flop_per_byte']):.3g} GF/s={float(r['gflops']):.3g} GB/s={float(r['gbps']):.3g}", flush=True)


if __name__ == "__main__":
    main()
PY
chmod +x roofline_bench.py
```

script นี้จะสร้าง CSV ที่มีคอลัมน์สำคัญ:

| คอลัมน์ | ความหมาย |
|---|---|
| `kernel` | เช่น `vector_triad`, `gemm` |
| `oi_flop_per_byte` | operational intensity จาก algorithmic byte estimate |
| `gflops` | performance ที่วัดได้ |
| `gbps` | bandwidth estimate |
| `backend` | `cpu_numpy` หรือ `gpu_torch` |
| `world_size`, `rank`, `host` | ช่วยอ่านผล multi-node |

---

## 4. สร้าง optional MPI communication benchmark

ใช้เฉพาะถ้า environment มี `mpi4py`:

```bash
cat > comm_roofline_mpi.py <<'PY'
#!/usr/bin/env python3
"""
Teaching-grade MPI communication benchmark for a communication roofline.
Requires mpi4py and numpy.  It measures effective Allreduce bandwidth for
several message sizes and writes one CSV from rank 0.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from mpi4py import MPI
except Exception as e:
    print("mpi4py is not available; skip communication benchmark.")
    print("Import error:", repr(e))
    sys.exit(0)

try:
    import numpy as np
except Exception as e:
    print("numpy is not available; skip communication benchmark.")
    print("Import error:", repr(e))
    sys.exit(0)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--scope", default="multinode_comm")
    parser.add_argument("--label", default="lanta_roofline_training")
    parser.add_argument("--message-bytes", type=parse_int_list,
                        default=parse_int_list("1048576,4194304,16777216,67108864,134217728"))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm")
    hosts = comm.gather(host, root=0)

    rows = []
    for nbytes in args.message_bytes:
        count = max(1, nbytes // 8)
        actual_bytes = count * 8
        send = np.ones(count, dtype=np.float64) * (rank + 1)
        recv = np.empty_like(send)

        for _ in range(args.warmup):
            comm.Allreduce(send, recv, op=MPI.SUM)
        comm.Barrier()
        t0 = MPI.Wtime()
        for _ in range(args.repeat):
            comm.Allreduce(send, recv, op=MPI.SUM)
        comm.Barrier()
        t1 = MPI.Wtime()

        sec = (t1 - t0) / max(1, args.repeat)
        # A common ring-allreduce traffic approximation per rank.
        effective_bytes_per_rank = actual_bytes * 2.0 * (size - 1) / size if size > 1 else 0.0
        effective_gbps = effective_bytes_per_rank / sec / 1e9 if sec > 0 else 0.0

        if rank == 0:
            rows.append({
                "timestamp": now_iso(),
                "label": args.label,
                "scope": args.scope,
                "backend": "mpi_allreduce",
                "host": host,
                "all_hosts": ";".join(hosts or []),
                "job_id": job_id,
                "world_size": size,
                "message_bytes": actual_bytes,
                "seconds": sec,
                "effective_bytes_per_rank": effective_bytes_per_rank,
                "effective_GBps_per_rank": effective_gbps,
                "notes": "ring_allreduce_traffic_approximation_per_rank; not hardware counter"
            })
            print(f"size={size} message={actual_bytes} bytes time={sec:.6e}s effective_bw={effective_gbps:.3f} GB/s", flush=True)

    if rank == 0:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"comm_roofline_job{job_id}_ranks{size}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}", flush=True)


if __name__ == "__main__":
    main()
PY
chmod +x comm_roofline_mpi.py
```

ถ้า `mpi4py` ไม่มี script นี้จะไม่ทำให้คลาสพัง แต่จะ print ว่า skip communication benchmark

---

## 5. สร้าง plot script สำหรับใช้ใน Colab หรือเครื่อง local

คัดลอกเพื่อสร้าง `plot_roofline_colab.py`:

```bash
cat > plot_roofline_colab.py <<'PY'
#!/usr/bin/env python3
"""
Plot Roofline figures from LANTA CSV results.
Works in Google Colab or a local Python environment with pandas and matplotlib.
Usage in Colab:
  from google.colab import files
  files.upload()
  %run plot_roofline_colab.py --pattern "*.csv"
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_roofline_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "comm_roofline" not in Path(f).name]
    if not files:
        raise FileNotFoundError(f"No roofline benchmark CSV matched {pattern!r}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Skip {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    for col in ["N", "seconds", "flops", "bytes_estimate", "oi_flop_per_byte", "gflops", "gbps", "world_size", "rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_comm_csvs(pattern: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "comm_roofline" in Path(f).name or "comm_" in Path(f).name]
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    for col in ["world_size", "message_bytes", "seconds", "effective_bytes_per_rank", "effective_GBps_per_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def plot_node_roofline(df: pd.DataFrame, backend: str, outdir: Path) -> None:
    sub = df[(df["backend"] == backend) & (df["gflops"] > 0) & (df["oi_flop_per_byte"] > 0)].copy()
    if sub.empty:
        print(f"No data for backend={backend}")
        return

    bw = sub[sub["kernel"].str.contains("triad", case=False, na=False)]["gbps"].max()
    peak = sub[sub["kernel"].str.contains("gemm", case=False, na=False)]["gflops"].max()
    if not np.isfinite(bw) or bw <= 0 or not np.isfinite(peak) or peak <= 0:
        print(f"Cannot build roof for backend={backend}: peak={peak}, bw={bw}")
        return

    x_min = max(1e-3, sub["oi_flop_per_byte"].min() / 4)
    x_max = max(1e1, sub["oi_flop_per_byte"].max() * 4, peak / bw * 4)
    x = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    y = np.minimum(peak, bw * x)
    ridge = peak / bw

    plt.figure(figsize=(8, 6))
    plt.loglog(x, y, label=f"Roofline: peak={peak:.1f} GF/s, BW={bw:.1f} GB/s")
    plt.axvline(ridge, linestyle="--", label=f"Ridge point={ridge:.2f} FLOP/byte")

    for kernel, g in sub.groupby("kernel"):
        label = f"{kernel}"
        plt.scatter(g["oi_flop_per_byte"], g["gflops"], label=label)
        for _, row in g.iterrows():
            plt.annotate(str(int(row["N"])), (row["oi_flop_per_byte"], row["gflops"]), fontsize=8)

    plt.xlabel("Operational intensity (FLOP/byte, algorithmic estimate)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"Node Roofline - {backend}")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8)
    out = outdir / f"roofline_node_{safe_name(backend)}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


def plot_cluster_aggregate_roofline(df: pd.DataFrame, backend: str, outdir: Path) -> None:
    sub = df[(df["backend"] == backend) & (df["gflops"] > 0) & (df["oi_flop_per_byte"] > 0)].copy()
    if sub.empty or "world_size" not in sub.columns:
        return

    # Use single-node measurements to derive a per-node roof when available;
    # otherwise use the best per-rank numbers from the whole set.
    one = sub[sub["world_size"] == 1]
    base = one if not one.empty else sub
    node_bw = base[base["kernel"].str.contains("triad", case=False, na=False)]["gbps"].max()
    node_peak = base[base["kernel"].str.contains("gemm", case=False, na=False)]["gflops"].max()
    if not np.isfinite(node_bw) or node_bw <= 0 or not np.isfinite(node_peak) or node_peak <= 0:
        return

    # Aggregate rows across ranks for the same job/kernel/N.
    keys = ["job_id", "world_size", "kernel", "dtype", "N"]
    agg = sub.groupby(keys, dropna=False).agg({
        "gflops": "sum",
        "oi_flop_per_byte": "mean",
        "gbps": "sum"
    }).reset_index()
    max_nodes = int(max(1, agg["world_size"].max()))
    agg_peak = node_peak * max_nodes
    agg_bw = node_bw * max_nodes

    x_min = max(1e-3, agg["oi_flop_per_byte"].min() / 4)
    x_max = max(1e1, agg["oi_flop_per_byte"].max() * 4, agg_peak / agg_bw * 4)
    x = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    y = np.minimum(agg_peak, agg_bw * x)

    plt.figure(figsize=(8, 6))
    plt.loglog(x, y, label=f"Ideal aggregate local roofline ({max_nodes} nodes)")
    for kernel, g in agg.groupby("kernel"):
        plt.scatter(g["oi_flop_per_byte"], g["gflops"], label=kernel)
        for _, row in g.iterrows():
            plt.annotate(f"N{int(row['world_size'])}:{int(row['N'])}", (row["oi_flop_per_byte"], row["gflops"]), fontsize=8)
    plt.xlabel("Operational intensity (FLOP/byte, local memory model)")
    plt.ylabel("Aggregate performance (GFLOP/s, sum across ranks)")
    plt.title(f"Multi-node local Roofline - {backend}")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8)
    out = outdir / f"roofline_cluster_local_{safe_name(backend)}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


def plot_comm_bandwidth(comm: pd.DataFrame, outdir: Path) -> None:
    if comm is None or comm.empty:
        print("No communication CSV found; skip communication plot.")
        return
    if "effective_GBps_per_rank" not in comm.columns:
        return
    plt.figure(figsize=(8, 6))
    for ws, g in comm.groupby("world_size"):
        g = g.sort_values("message_bytes")
        plt.loglog(g["message_bytes"], g["effective_GBps_per_rank"], marker="o", label=f"{int(ws)} ranks")
    plt.xlabel("Message size per rank (bytes)")
    plt.ylabel("Effective Allreduce bandwidth per rank (GB/s)")
    plt.title("Communication bandwidth for multi-node roofline thinking")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8)
    out = outdir / "communication_bandwidth.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--outdir", default="roofline_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_roofline_csvs(args.pattern)
    print("Loaded benchmark rows:", len(df))
    print(df[["backend", "scope", "kernel", "N", "oi_flop_per_byte", "gflops", "gbps", "world_size", "host"]].head(20))

    for backend in sorted(df["backend"].dropna().unique()):
        plot_node_roofline(df, backend, outdir)
        plot_cluster_aggregate_roofline(df, backend, outdir)

    comm = read_comm_csvs(args.pattern)
    plot_comm_bandwidth(comm, outdir)


if __name__ == "__main__":
    main()
PY
chmod +x plot_roofline_colab.py
```

เราจะใช้ไฟล์นี้ภายหลังใน Google Colab หลังจากดาวน์โหลด CSV จาก LANTA แล้ว

---

## 6. สร้าง Slurm job scripts อัตโนมัติ

คัดลอกเพื่อสร้าง `make_slurm_jobs.sh`:

```bash
cat > make_slurm_jobs.sh <<'BASH'
#!/bin/bash
set -u

ACCOUNT="${LANTA_ACCOUNT:-tn999992}"

partition_exists() {
  local p="$1"
  sinfo -h -p "$p" -o "%P" >/dev/null 2>&1
}

detect_partition() {
  for p in "$@"; do
    if partition_exists "$p"; then
      echo "$p"
      return 0
    fi
  done
  sinfo -h -o "%P" 2>/dev/null | head -n 1 | sed 's/*//'
}

CPU_PARTITION="${CPU_PARTITION:-$(detect_partition cpu compute cpu-devel compute-devel normal)}"
GPU_PARTITION="${GPU_PARTITION:-$(detect_partition gpu-devel gpu)}"

if [ -z "$CPU_PARTITION" ]; then
  CPU_PARTITION="cpu"
fi
if [ -z "$GPU_PARTITION" ]; then
  GPU_PARTITION="gpu"
fi

mkdir -p jobs results logs

echo "Using account:        $ACCOUNT"
echo "Using CPU partition:  $CPU_PARTITION"
echo "Using GPU partition:  $GPU_PARTITION"

cat > jobs/job_roofline_cpu_node.slurm <<SLURM
#!/bin/bash -l
#SBATCH -p ${CPU_PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH -J roof_cpu_node
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out

module purge
module load Mamba/23.11.0-0 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pytorch-2.2.2 2>/dev/null || true
fi

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "===== CPU node roofline job ====="
hostname
lscpu | egrep 'Model name|Socket|Core|Thread|NUMA' || true
python -V
python roofline_bench.py --backend cpu --scope single_node_cpu --outdir results --repeat 5 --warmup 2
SLURM

cat > jobs/job_roofline_gpu1.slurm <<SLURM
#!/bin/bash -l
#SBATCH -p ${GPU_PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH -J roof_gpu1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out

module purge
module load Mamba/23.11.0-0 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pytorch-2.2.2 2>/dev/null || true
fi

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "===== GPU single-device roofline job ====="
hostname
nvidia-smi || true
python -V
python roofline_bench.py --backend gpu --scope single_gpu --outdir results --repeat 5 --warmup 2
SLURM

cat > jobs/job_roofline_multinode_local.slurm <<SLURM
#!/bin/bash -l
#SBATCH -p ${CPU_PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH -J roof_multi_local
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -t 00:25:00
#SBATCH -o logs/%x-%j.out

module purge
module load Mamba/23.11.0-0 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pytorch-2.2.2 2>/dev/null || true
fi

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "===== Multi-node local roofline job ====="
scontrol show hostnames "\$SLURM_JOB_NODELIST"
srun python roofline_bench.py --backend cpu --scope multinode_local_cpu --outdir results --repeat 5 --warmup 2
SLURM

cat > jobs/job_roofline_multinode_comm.slurm <<SLURM
#!/bin/bash -l
#SBATCH -p ${CPU_PARTITION}
#SBATCH -A ${ACCOUNT}
#SBATCH -J roof_multi_comm
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out

module purge
module load Mamba/23.11.0-0 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate pytorch-2.2.2 2>/dev/null || true
fi

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "===== Multi-node communication roofline helper ====="
scontrol show hostnames "\$SLURM_JOB_NODELIST"
python - <<'PY'
import importlib.util
print("mpi4py:", "OK" if importlib.util.find_spec("mpi4py") else "missing")
print("numpy:", "OK" if importlib.util.find_spec("numpy") else "missing")
PY
srun python comm_roofline_mpi.py --outdir results --repeat 20 --warmup 5
SLURM

chmod +x jobs/*.slurm

echo
printf '%s\n' "Created Slurm scripts:" jobs/*.slurm
printf '%s\n' "Submit examples:" \
  "sbatch jobs/job_roofline_cpu_node.slurm" \
  "sbatch jobs/job_roofline_gpu1.slurm" \
  "sbatch jobs/job_roofline_multinode_local.slurm" \
  "sbatch jobs/job_roofline_multinode_comm.slurm"
BASH
chmod +x make_slurm_jobs.sh
```

รันเพื่อสร้าง job scripts:

```bash
export LANTA_ACCOUNT=tn999992
bash make_slurm_jobs.sh | tee make_jobs.log
```

ดูไฟล์ที่ถูกสร้าง:

```bash
ls -lh jobs
sed -n '1,80p' jobs/job_roofline_cpu_node.slurm
```

ถ้า partition ที่ script เลือกไม่ตรงกับ LANTA ในวันอบรม ให้ดู partition จริงก่อน:

```bash
sinfo -s
```

แล้วสร้าง job ใหม่โดยกำหนด partition เอง เช่น:

```bash
CPU_PARTITION=cpu GPU_PARTITION=gpu bash make_slurm_jobs.sh
```

หรือถ้ามี `gpu-devel` และต้องการคิวสั้น:

```bash
CPU_PARTITION=cpu GPU_PARTITION=gpu-devel bash make_slurm_jobs.sh
```

---

## 7. รัน Single-node CPU Roofline

ส่งงาน:

```bash
sbatch jobs/job_roofline_cpu_node.slurm
```

ดูคิว:

```bash
myqueue
```

หรือใช้ Slurm ปกติ:

```bash
squeue -u $USER
```

เมื่อเสร็จแล้วดู log ล่าสุด:

```bash
ls -ltr logs
cat $(ls -t logs/roof_cpu_node-*.out | head -n 1)
```

ดู CSV ที่ได้:

```bash
ls -lh results
head -n 5 results/roofline_single_node_cpu_cpu_job*.csv
```

ผลที่คาดว่าจะเห็นใน log จะคล้าย ๆ แบบนี้:

```text
cpu_numpy  vector_triad   N=10000000 OI=0.0833 GF/s=... GB/s=...
cpu_numpy  gemm           N=1024     OI=85.3   GF/s=... GB/s=...
```

อ่านทันที:

- `vector_triad` มี OI ต่ำมาก จึงควรอยู่ฝั่ง memory-bound
- `gemm` มี OI สูงขึ้นมาก จึงควรขยับเข้าใกล้ compute-bound

---

## 8. รัน Single-GPU Roofline ถ้ามี GPU allocation

ส่งงาน:

```bash
sbatch jobs/job_roofline_gpu1.slurm
```

ดู log:

```bash
myqueue
ls -ltr logs
cat $(ls -t logs/roof_gpu1-*.out | head -n 1)
```

ดู CSV:

```bash
ls -lh results/*gpu*.csv
head -n 5 results/roofline_single_gpu_gpu_job*.csv
```

จุดที่ควรชี้ให้ผู้เรียนดู:

```bash
grep -E "CUDA_VISIBLE_DEVICES|nvidia-smi|CUDA|Using CUDA|gpu_torch|vector_triad|gemm" $(ls -t logs/roof_gpu1-*.out | head -n 1)
```

ถ้าไม่มี CUDA หรือไม่มี GPU allocation จะเห็นว่า script skip GPU benchmark ซึ่งเป็นพฤติกรรมที่ตั้งใจไว้

---

## 9. รัน Multi-node Local Roofline

งานนี้ใช้ 2 nodes และรัน benchmark เดิม 1 process ต่อ 1 node เพื่อให้เห็นว่าแต่ละ node ให้ผลใกล้กันไหม และเมื่อนำ GFLOP/s มารวมกันจะได้ aggregate local roofline แบบ ideal-ish

```bash
sbatch jobs/job_roofline_multinode_local.slurm
```

ดู log:

```bash
myqueue
cat $(ls -t logs/roof_multi_local-*.out | head -n 1)
```

ดูจำนวน CSV ต่อ rank:

```bash
ls -lh results/*multinode_local*.csv
for f in results/*multinode_local*.csv; do echo "===== $f"; head -n 3 "$f"; done
```

ความหมาย:

- ถ้าได้ 2 CSV แปลว่า 2 ranks จาก 2 nodes เขียนผลสำเร็จ
- ใน plot ภายหลัง เราจะรวม `gflops` ของแต่ละ rank เพื่อดู aggregate throughput
- แต่ยังไม่ใช่ communication roofline เพราะ benchmark นี้ยังไม่วัดการส่งข้อมูลข้าม network

---

## 10. Optional: รัน Multi-node Communication Benchmark ด้วย MPI Allreduce

ตรวจว่า `mpi4py` มีไหม:

```bash
python - <<'PY'
import importlib.util
print("mpi4py:", "OK" if importlib.util.find_spec("mpi4py") else "missing")
PY
```

ส่งงาน:

```bash
sbatch jobs/job_roofline_multinode_comm.slurm
```

ดู log:

```bash
myqueue
cat $(ls -t logs/roof_multi_comm-*.out | head -n 1)
```

ดู CSV:

```bash
ls -lh results/comm_roofline*.csv
cat results/comm_roofline*.csv
```

ใช้ผลนี้เพื่ออธิบายว่า multi-node application อาจมีอีกเพดานหนึ่ง:

```text
communication-bound performance <= effective_network_bandwidth x communication_intensity
communication_intensity = useful FLOPs / communicated bytes
```

ตัวอย่างเชิงคิด:

- งาน local compute เร็วขึ้นเมื่อเพิ่ม node
- แต่ถ้าแต่ละ iteration ต้อง Allreduce ใหญ่ ๆ บ่อยมาก performance อาจถูกจำกัดด้วย network
- ดังนั้น multi-node roofline ควรแยกดู local compute/memory roof กับ communication roof

---

## 11. รวมผลและเตรียมดาวน์โหลดจาก LANTA

สร้าง archive:

```bash
tar -czf roofline_results_${USER}.tar.gz results logs jobs *.py *.sh *.log 2>/dev/null || \
tar -czf roofline_results_${USER}.tar.gz results logs jobs *.py *.sh
ls -lh roofline_results_${USER}.tar.gz
```

ออกจาก LANTA:

```bash
exit
```

จากเครื่อง local ของผู้เรียน ดาวน์โหลดผ่าน transfer node:

```bash
scp YOUR_LANTA_USERNAME@transfer.lanta.nstda.or.th:/home/YOUR_LANTA_USERNAME/roofline-training/roofline_results_YOUR_LANTA_USERNAME.tar.gz .
```

แตกไฟล์ในเครื่อง local:

```bash
tar -xzf roofline_results_YOUR_LANTA_USERNAME.tar.gz
find results -maxdepth 2 -type f
```

ถ้าจะใช้ Colab ให้ upload เฉพาะไฟล์เหล่านี้ก็พอ:

```text
results/*.csv
plot_roofline_colab.py
```

---

## 12. Plot ใน Google Colab

เปิด Google Colab แล้วรัน cell นี้:

```python
from google.colab import files
uploaded = files.upload()
```

เลือก upload:

```text
plot_roofline_colab.py
results/*.csv
```

จากนั้นรัน:

```python
%run plot_roofline_colab.py --pattern "*.csv" --outdir roofline_plots
```

ดูไฟล์ภาพที่สร้าง:

```python
import os
for root, dirs, files_ in os.walk("roofline_plots"):
    for f in files_:
        print(os.path.join(root, f))
```

แสดงภาพใน Colab:

```python
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("roofline_plots/*.png")):
    print(f)
    display(Image(filename=f))
```

ดาวน์โหลดภาพทั้งหมด:

```python
!zip -r roofline_plots.zip roofline_plots
from google.colab import files
files.download("roofline_plots.zip")
```

---

## 13. การตีความผลในคลาส

### 13.1 Single-node CPU

ถามผู้เรียน:

```text
vector_triad อยู่ซ้ายหรือขวาของ ridge point?
gemm อยู่ใกล้ compute roof หรือยัง?
ถ้า gemm อยู่ต่ำกว่า roof มาก อาจเกิดจากอะไร?
```

คำตอบที่ควรนำไปสู่การอภิปราย:

- `vector_triad` มักมี OI ต่ำ จึง memory-bound
- `gemm` มี OI สูง แต่ถ้า GFLOP/s ต่ำ อาจเกี่ยวกับ BLAS, thread count, NUMA binding, matrix size, หรือ environment
- ถ้าใช้ CPU node เต็ม ควรสนใจ `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, และ binding เพิ่ม

### 13.2 Single-GPU

ถามผู้เรียน:

```text
gpu_torch vector_triad ได้ bandwidth สูงกว่า CPU ไหม?
gpu_torch gemm เข้าใกล้ compute roof มากกว่า CPU หรือไม่?
ทำไม PyTorch บน A100 อาจใช้ TF32 และทำให้ GFLOP/s สูงกว่า FP32 แบบ strict?
```

คำอธิบาย:

- A100 เหมาะกับ GEMM/AI workload มาก โดยเฉพาะเมื่อใช้ Tensor Core/TF32
- จุด `vector_triad` บน GPU แสดง memory/HBM-bound behavior
- จุด `gemm` แสดง compute-heavy behavior แต่ยังขึ้นกับ dtype, Tensor Core, batch size, และ kernel implementation

### 13.3 Multi-node

ถามผู้เรียน:

```text
เมื่อใช้ 2 nodes aggregate GFLOP/s ใกล้ 2 เท่าหรือไม่?
ถ้าไม่ใกล้ เกิดจากอะไรได้บ้าง?
ถ้า application ต้อง communicate ทุก iteration จุดจริงจะตกลงมาจาก aggregate local roofline อย่างไร?
```

คำตอบที่ควรได้:

- ถ้างาน independent และไม่มี communication อาจ scale ใกล้จำนวน node
- ถ้ามี communication เช่น Allreduce, Halo exchange, FFT transpose, distributed training gradient sync จะเริ่มติด network
- Multi-node roofline จึงควรดูอย่างน้อย 2 roof: local compute/memory roof และ communication roof

---

## 14. สูตรที่ใช้ใน plot

### 14.1 Node Roofline

```text
P_node(I) = min(Peak_node, BW_memory_node x I)
```

โดยที่

```text
I = FLOPs / bytes moved
```

### 14.2 Aggregate local roofline สำหรับ N nodes

```text
P_cluster_local(I) = min(N x Peak_node, N x BW_memory_node x I)
```

สูตรนี้ใช้เป็น **ideal local scaling guide** เท่านั้น ไม่รวม network

### 14.3 Communication roofline แบบคิดง่าย

```text
P_comm(I_comm) = BW_network_effective x I_comm
I_comm = useful FLOPs / communicated bytes
```

ถ้า application มีทั้ง compute และ communication:

```text
P_attainable <= min(P_cluster_local, P_comm)
```

---

## 15. Mini assignment สำหรับผู้เรียน

ให้ผู้เรียนตอบ 5 ข้อนี้จากกราฟของตัวเอง:

1. `vector_triad` เป็น memory-bound หรือ compute-bound?
2. `gemm` อยู่ใกล้ roofline แค่ไหน?
3. Ridge point ของ CPU/GPU อยู่ประมาณกี่ FLOP/byte?
4. เมื่อรัน multi-node aggregate performance เพิ่มขึ้นกี่เท่า?
5. ถ้ามี communication CSV, message size ใดให้ effective bandwidth สูงสุด?

---

## 16. Troubleshooting

### ปัญหา: `sbatch: error: invalid partition`

ดู partition จริง:

```bash
sinfo -s
```

สร้าง job ใหม่โดยระบุ partition เอง:

```bash
CPU_PARTITION=YOUR_CPU_PARTITION GPU_PARTITION=YOUR_GPU_PARTITION bash make_slurm_jobs.sh
```

### ปัญหา: `conda activate pytorch-2.2.2` ไม่ได้

ดู env ที่มี:

```bash
module load Mamba/23.11.0-0
conda info --envs
```

แก้ชื่อ environment ในไฟล์ job:

```bash
grep -R "conda activate" jobs
```

เช่นแก้ด้วย `sed`:

```bash
sed -i 's/conda activate pytorch-2.2.2/conda activate YOUR_ENV_NAME/g' jobs/*.slurm
```

### ปัญหา: `numpy` missing

ลอง activate environment อื่นที่มี NumPy หรือรันเฉพาะ plot ใน Colab  
ในคลาสสั้น ไม่แนะนำให้เสียเวลาสร้าง environment ใหม่ถ้า network/solver ช้า

### ปัญหา: GPU job ไม่มี CUDA

ตรวจว่าได้ GPU allocation จริงหรือไม่:

```bash
grep -E "CUDA_VISIBLE_DEVICES|nvidia-smi|CUDA|Using CUDA" logs/roof_gpu1-*.out
```

ถ้าไม่มี GPU ให้สอน CPU roofline ก่อน แล้วให้ GPU เป็น homework

### ปัญหา: `mpi4py` missing

ข้าม communication benchmark ได้ ไม่กระทบ single-node และ multi-node local roofline

```bash
ls results/*.csv
```

ถ้ามีเฉพาะ `roofline_*.csv` ก็ยัง plot node roofline ได้

---

## 17. ข้อควรย้ำกับผู้เรียน

- ห้าม benchmark หนักบน frontend node
- ผลจาก Python benchmark ใช้สอน workflow และ intuition ไม่ใช่ผลวัด hardware ceiling อย่างเป็นทางการ
- ค่า `bytes_estimate` เป็น algorithmic/minimum estimate ไม่ใช่ DRAM traffic จาก hardware counters
- Roofline ไม่ได้บอกทุกอย่าง แต่ช่วยให้ถามคำถามถูก: งานของเราติด memory, compute, หรือ communication?
- Multi-node ต้องคิด network เสมอ โดยเฉพาะงานที่มี Allreduce, Halo exchange, distributed FFT, หรือ distributed deep learning

---

## 18. แหล่งอ้างอิงสำหรับผู้สอน

- ThaiSC LANTA system overview: https://thaisc.io/thaisc-resorces/lanta
- LANTA Quick Start Guide: https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/447021064/Quick+Start+Guide
- Williams, Waterman, Patterson, “Roofline: An Insightful Visual Performance Model for Multicore Architectures”: https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
- PyTorch / CUDA performance interpretation should be validated with NVIDIA profilers for serious analysis

---

## 19. One-page command summary

```bash
# On LANTA
mkdir -p ~/roofline-training
cd ~/roofline-training
export LANTA_ACCOUNT=tn999992
bash 00_env_browse.sh |& tee 00_env_browse.log
bash make_slurm_jobs.sh | tee make_jobs.log
sbatch jobs/job_roofline_cpu_node.slurm
sbatch jobs/job_roofline_gpu1.slurm
sbatch jobs/job_roofline_multinode_local.slurm
sbatch jobs/job_roofline_multinode_comm.slurm
myqueue
ls -lh results logs
tar -czf roofline_results_${USER}.tar.gz results logs jobs *.py *.sh *.log

# On local machine
scp YOUR_LANTA_USERNAME@transfer.lanta.nstda.or.th:/home/YOUR_LANTA_USERNAME/roofline-training/roofline_results_YOUR_LANTA_USERNAME.tar.gz .
```

```python
# In Colab
from google.colab import files
uploaded = files.upload()
%run plot_roofline_colab.py --pattern "*.csv" --outdir roofline_plots
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("roofline_plots/*.png")):
    display(Image(filename=f))
```
