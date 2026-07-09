# คู่มือ Copy-Paste: สำรวจ Environment และรัน PyTorch GPU บน LANTA HPC

> ใช้สำหรับอบรมเวลาสั้น หลังจากผู้เรียน SSH เข้า LANTA แล้ว  
> Account/Project ที่ใช้ในตัวอย่างนี้: `tn999992`  
> เป้าหมาย: สำรวจระบบ → สำรวจ modules → activate conda/mamba environment → ตรวจ package → ส่งงาน GPU ด้วย SLURM → อ่านผลลัพธ์

---

## 0) แนวคิดก่อนเริ่ม

ในคลาสสั้นนี้ เราจะ **ไม่ติดตั้ง package ใหม่** ระหว่างเรียน แต่จะตรวจว่า LANTA มี module และ conda environment ที่ใช้รัน PyTorch อยู่แล้วหรือไม่ จากนั้นส่งงานไปยัง GPU node ด้วย SLURM

สิ่งที่ผู้เรียนควรเห็นเมื่อสำเร็จ:

```text
CUDA available: True
GPU count: 1        # สำหรับงาน 1 GPU
GPU 0: NVIDIA A100
Training OK
```

ถ้าเป็นงาน 4 GPUs จะเห็นประมาณนี้:

```text
GPU count: 4
Using DataParallel on 4 GPUs
Training OK
```

---

## 1) สร้าง workspace สำหรับคลาสนี้

คัดลอกทั้งบล็อกนี้ไปวางบน LANTA frontend node หลังจาก login แล้ว

```bash
export PROJECT_ID=tn999992
export ENV_NAME=pytorch-2.2.2
export TRAIN_DIR=$HOME/lanta_gpu_short_training

mkdir -p "$TRAIN_DIR"
cd "$TRAIN_DIR"

pwd
whoami
hostname
date
```

---

## 2) สำรวจ shell, path, storage และ SLURM แบบเร็ว

```bash
echo "===== BASIC ENVIRONMENT ====="
echo "USER=$USER"
echo "HOME=$HOME"
echo "PWD=$PWD"
echo "SHELL=$SHELL"
echo "PROJECT_ID=$PROJECT_ID"
echo "ENV_NAME=$ENV_NAME"

echo

echo "===== DISK LOCATION ====="
df -h . || true
quota -s || true

echo

echo "===== SLURM COMMANDS ====="
which sbatch || true
which srun || true
which squeue || true
which sinfo || true
which myqueue || true

echo

echo "===== SLURM PARTITIONS SUMMARY ====="
sinfo -s || true

echo

echo "===== GPU PARTITIONS ====="
sinfo -p gpu,gpu-devel -o "%20P %8D %12t %20C %20G" || true
```

สิ่งที่ต้องเข้าใจ: frontend node ใช้สำหรับเตรียมไฟล์และส่งงาน ไม่ใช่ที่รันงาน GPU จริง ดังนั้น `torch.cuda.is_available()` บน frontend อาจเป็น `False` ได้

---

## 3) สำรวจ LANTA module system

```bash
echo "===== RESET MODULES ====="
module purge
module list

echo

echo "===== MODULE COMMAND LOCATION ====="
type module

echo

echo "===== SEARCH MAMBA MODULE ====="
module avail Mamba 2>&1 | tee 01_module_avail_mamba.txt

echo

echo "===== SPIDER MAMBA MODULE ====="
module spider Mamba 2>&1 | tee 02_module_spider_mamba.txt

echo

echo "===== OPTIONAL: SEARCH PYTHON / CUDA / PYTORCH MODULE NAMES ====="
module avail Python 2>&1 | tee 03_module_avail_python.txt || true
module avail CUDA 2>&1 | tee 04_module_avail_cuda.txt || true
module avail PyTorch 2>&1 | tee 05_module_avail_pytorch.txt || true
```

---

## 4) โหลด Mamba และดู conda environments ที่มีอยู่

```bash
echo "===== LOAD MAMBA ====="
module purge
module load Mamba/23.11.0-0 || module load Mamba
module list

echo

echo "===== CONDA / MAMBA COMMANDS ====="
which conda || true
which mamba || true
conda --version || true
mamba --version || true

echo

echo "===== INITIALIZE CONDA FOR THIS SHELL ====="
eval "$(conda shell.bash hook)"

echo

echo "===== AVAILABLE CONDA ENVIRONMENTS ====="
conda env list | tee 06_conda_env_list.txt

echo

echo "===== ENVIRONMENTS RELATED TO TORCH / CUDA / TENSORFLOW / JAX ====="
conda env list | grep -Ei "torch|cuda|tensorflow|jax|pytorch" || true
```

---

## 5) Activate environment และตรวจ package ที่จำเป็นบน frontend

บล็อกนี้ตรวจว่า environment `pytorch-2.2.2` มีอยู่จริง และตรวจว่า import package สำคัญได้หรือไม่

```bash
echo "===== ACTIVATE TARGET ENVIRONMENT ====="
module purge
module load Mamba/23.11.0-0 || module load Mamba
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
else
    echo "ERROR: conda environment '$ENV_NAME' was not found."
    echo "Available environments are:"
    conda env list
    exit 2
fi

echo

echo "===== PYTHON LOCATION ====="
which python
python --version

echo

echo "===== KEY PACKAGES IN THIS ENV ====="
conda list | grep -Ei "^(python|pytorch|torch|torchvision|torchaudio|cuda|cudatoolkit|numpy|scipy|pandas|matplotlib|scikit|pillow)" || true

echo

echo "===== PYTHON IMPORT CHECK ====="
python - <<'PY'
import importlib.util
import sys

modules = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "matplotlib",
]

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))

for name in modules:
    spec = importlib.util.find_spec(name)
    print(f"{name:12s}:", "OK" if spec else "MISSING")

try:
    import torch
    print("Torch version:", torch.__version__)
    print("Torch CUDA version:", torch.version.cuda)
    print("CUDA available on this node:", torch.cuda.is_available())
    print("GPU count visible on this node:", torch.cuda.device_count())
except Exception as e:
    print("Torch import failed:", repr(e))
    raise
PY
```

ถ้า `torch` import ได้ แต่ `CUDA available on this node: False` บน frontend ถือว่ายังไม่ผิด เพราะ GPU จะเห็นเมื่อรันผ่าน SLURM บน GPU node

---

## 6) สร้างไฟล์ Python สำหรับรายงาน environment และทดสอบ GPU

คัดลอกบล็อกนี้ครั้งเดียว จะสร้างไฟล์ `lanta_env_report.py` และ `lanta_gpu_demo.py`

```bash
cat > lanta_env_report.py <<'PY'
import importlib.util
import os
import platform
import socket
import subprocess
import sys

print("===== PYTHON / SYSTEM REPORT =====")
print("Hostname:", socket.gethostname())
print("Platform:", platform.platform())
print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))

print("\n===== SLURM ENVIRONMENT =====")
for key in [
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_NODELIST",
    "CUDA_VISIBLE_DEVICES",
]:
    print(f"{key}={os.environ.get(key)}")

print("\n===== PACKAGE IMPORT CHECK =====")
for name in ["torch", "torchvision", "numpy", "PIL", "matplotlib"]:
    spec = importlib.util.find_spec(name)
    print(f"{name:12s}:", "OK" if spec else "MISSING")

print("\n===== NVIDIA-SMI =====")
try:
    result = subprocess.run(
        ["nvidia-smi"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)
except FileNotFoundError:
    print("nvidia-smi not found")

print("\n===== TORCH CUDA CHECK =====")
import torch

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))

print("Environment report OK")
PY

cat > lanta_gpu_demo.py <<'PY'
import os
import socket
import time

import torch
import torch.nn as nn

print("===== GPU DEMO START =====")
print("Hostname:", socket.gethostname())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. Did this job run on a GPU node?")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda:0")
model = TinyNet()

if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
else:
    print("Using single GPU")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

batch_size = 8192
x = torch.randn(batch_size, 1024, device=device)
y = torch.randn(batch_size, 10, device=device)

torch.cuda.synchronize()
t0 = time.time()

for step in range(1, 6):
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step={step} loss={loss.item():.6f}")

t1 = time.time()
print(f"Elapsed seconds: {t1 - t0:.3f}")
print("Training OK")
print("===== GPU DEMO END =====")
PY

ls -lh *.py
```

---

## 7) ส่งงาน 1 GPU แบบ copy-paste พร้อม fallback partition

บล็อกนี้จะลองส่งไป `gpu-devel` ก่อน ถ้าส่งไม่ได้จะลอง `gpu` ต่อให้อัตโนมัติ

```bash
cat > submit_1gpu.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-tn999992}"
ENV_NAME="${ENV_NAME:-pytorch-2.2.2}"

submit_to_partition() {
    local PARTITION="$1"
    local JOBSCRIPT="job_1gpu_${PARTITION}.slurm"

    cat > "$JOBSCRIPT" <<SLURM
#!/bin/bash -l
#SBATCH -p ${PARTITION}
#SBATCH -A ${PROJECT_ID}
#SBATCH -J gpu_1gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out

set -euo pipefail

echo "===== JOB START ====="
date
hostname
pwd

echo "===== MODULE SETUP ====="
module purge
module load Mamba/23.11.0-0 || module load Mamba
module list

echo "===== CONDA SETUP ====="
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}
which python
python --version

echo "===== CONDA KEY PACKAGE LIST ====="
conda list | grep -Ei "^(python|pytorch|torch|torchvision|torchaudio|cuda|cudatoolkit|numpy|scipy|pandas|matplotlib|scikit|pillow)" || true

echo "===== ENV REPORT ====="
python lanta_env_report.py

echo "===== GPU DEMO ====="
python lanta_gpu_demo.py

echo "===== JOB END ====="
date
SLURM

    echo "Submitting $JOBSCRIPT"
    sbatch --parsable "$JOBSCRIPT" 2>"submit_${PARTITION}.err"
}

for PARTITION in gpu-devel gpu; do
    if JOBID=$(submit_to_partition "$PARTITION"); then
        echo "$JOBID" | tee last_1gpu_jobid.txt
        echo "$PARTITION" | tee last_1gpu_partition.txt
        echo "Submitted 1-GPU job $JOBID to partition $PARTITION"
        exit 0
    else
        echo "Submit to partition $PARTITION failed:"
        cat "submit_${PARTITION}.err" || true
        echo
    fi
done

echo "ERROR: Could not submit to gpu-devel or gpu."
exit 1
BASH

chmod +x submit_1gpu.sh
./submit_1gpu.sh
```

---

## 8) ดูคิวและอ่านผล 1 GPU

บล็อกนี้จะเช็กคิวสั้น ๆ แล้วพยายามอ่าน output ถ้างานเสร็จแล้ว

```bash
JOBID=$(cat last_1gpu_jobid.txt)
PARTITION=$(cat last_1gpu_partition.txt)

echo "JOBID=$JOBID"
echo "PARTITION=$PARTITION"

echo

echo "===== CURRENT QUEUE STATUS ====="
myqueue || squeue -u "$USER" || true

echo

echo "===== WAIT UP TO 2 MINUTES ====="
for i in {1..12}; do
    if squeue -j "$JOBID" -h | grep -q .; then
        echo "Still in queue/running: attempt $i/12"
        squeue -j "$JOBID" || true
        sleep 10
    else
        echo "Job no longer in queue. It may have completed."
        break
    fi
done

OUT="gpu_1gpu-${JOBID}.out"
echo

echo "===== OUTPUT FILE ====="
if [ -f "$OUT" ]; then
    echo "Found $OUT"
    echo
    grep -E "CUDA available|GPU count|GPU [0-9]|Using|step=|Training OK|Environment report OK|ERROR" "$OUT" || true
    echo
    echo "Full output:"
    cat "$OUT"
else
    echo "Output file $OUT not found yet. The job may still be pending."
    echo "Run this later:"
    echo "cat $OUT"
fi
```

---

## 9) สรุปผลที่ผู้เรียนควรหาเจอใน log

```bash
JOBID=$(cat last_1gpu_jobid.txt)
OUT="gpu_1gpu-${JOBID}.out"

echo "===== SUCCESS CHECK ====="
grep -n "CUDA available: True" "$OUT" && echo "OK: CUDA is available"
grep -n "GPU count:" "$OUT" && echo "OK: GPU is visible"
grep -n "Training OK" "$OUT" && echo "OK: PyTorch GPU training ran successfully"
```

ถ้าบล็อกนี้ผ่าน แปลว่า environment และ SLURM script สำหรับ 1 GPU พร้อมใช้แล้ว

---

## 10) Optional: ส่งงาน 4 GPUs ด้วย DataParallel สำหรับ demo สั้น

ใช้บล็อกนี้เมื่อคิวไม่แน่น หรือใช้สำหรับ instructor demo เพราะจะขอ 4 GPUs บน node เดียว

```bash
cat > submit_4gpu_dp.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-tn999992}"
ENV_NAME="${ENV_NAME:-pytorch-2.2.2}"

submit_to_partition() {
    local PARTITION="$1"
    local JOBSCRIPT="job_4gpu_dp_${PARTITION}.slurm"

    cat > "$JOBSCRIPT" <<SLURM
#!/bin/bash -l
#SBATCH -p ${PARTITION}
#SBATCH -A ${PROJECT_ID}
#SBATCH -J gpu_dp4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out

set -euo pipefail

echo "===== JOB START ====="
date
hostname
pwd

echo "===== MODULE SETUP ====="
module purge
module load Mamba/23.11.0-0 || module load Mamba
module list

echo "===== CONDA SETUP ====="
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}
which python
python --version

echo "===== GPU REPORT ====="
python lanta_env_report.py

echo "===== 4-GPU DATAPARALLEL DEMO ====="
python lanta_gpu_demo.py

echo "===== JOB END ====="
date
SLURM

    echo "Submitting $JOBSCRIPT"
    sbatch --parsable "$JOBSCRIPT" 2>"submit_4gpu_${PARTITION}.err"
}

for PARTITION in gpu-devel gpu; do
    if JOBID=$(submit_to_partition "$PARTITION"); then
        echo "$JOBID" | tee last_4gpu_jobid.txt
        echo "$PARTITION" | tee last_4gpu_partition.txt
        echo "Submitted 4-GPU DataParallel job $JOBID to partition $PARTITION"
        exit 0
    else
        echo "Submit to partition $PARTITION failed:"
        cat "submit_4gpu_${PARTITION}.err" || true
        echo
    fi
done

echo "ERROR: Could not submit 4-GPU job."
exit 1
BASH

chmod +x submit_4gpu_dp.sh
./submit_4gpu_dp.sh
```

อ่านผล 4 GPUs:

```bash
JOBID=$(cat last_4gpu_jobid.txt)
OUT="gpu_dp4-${JOBID}.out"

echo "JOBID=$JOBID"
myqueue || squeue -u "$USER" || true

for i in {1..12}; do
    if squeue -j "$JOBID" -h | grep -q .; then
        echo "Still in queue/running: attempt $i/12"
        squeue -j "$JOBID" || true
        sleep 10
    else
        break
    fi
done

if [ -f "$OUT" ]; then
    grep -E "CUDA available|GPU count|GPU [0-9]|Using DataParallel|step=|Training OK|ERROR" "$OUT" || true
    cat "$OUT"
else
    echo "Output file $OUT not found yet. Run later: cat $OUT"
fi
```

---

## 11) Optional advanced: DDP sanity check แบบ 4 GPUs

สำหรับสอนให้เห็น pattern ที่ถูกต้องกว่าในการ train จริง: 1 process ต่อ 1 GPU

```bash
cat > simple_ddp_sanity.py <<'PY'
import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size,
)

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
).to(device)

model = DDP(model, device_ids=[local_rank])

x = torch.randn(8, 16, device=device)
y = torch.randn(8, 4, device=device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

opt.zero_grad(set_to_none=True)
out = model(x)
loss = (out - y).pow(2).mean()
loss.backward()
opt.step()

torch.cuda.synchronize()
print(
    f"host={socket.gethostname()} "
    f"rank={rank}/{world_size} "
    f"local_rank={local_rank} "
    f"device={device} "
    f"loss={loss.item():.6f}",
    flush=True,
)

dist.barrier()
dist.destroy_process_group()
PY

cat > submit_4gpu_ddp.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-tn999992}"
ENV_NAME="${ENV_NAME:-pytorch-2.2.2}"

submit_to_partition() {
    local PARTITION="$1"
    local JOBSCRIPT="job_4gpu_ddp_${PARTITION}.slurm"

    cat > "$JOBSCRIPT" <<SLURM
#!/bin/bash -l
#SBATCH -p ${PARTITION}
#SBATCH -A ${PROJECT_ID}
#SBATCH -J gpu_ddp4
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out

set -euo pipefail

module purge
module load Mamba/23.11.0-0 || module load Mamba
eval "\$(conda shell.bash hook)"
conda activate ${ENV_NAME}

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

export WORLD_SIZE=\${SLURM_NTASKS}
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=\$((10000 + SLURM_JOB_ID % 50000))

echo "===== DDP JOB INFO ====="
echo "HOSTNAME=\$(hostname)"
echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "SLURM_NTASKS=\$SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK=\$SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE=\$WORLD_SIZE"
echo "MASTER_ADDR=\$MASTER_ADDR"
echo "MASTER_PORT=\$MASTER_PORT"

nvidia-smi
srun python simple_ddp_sanity.py
SLURM

    echo "Submitting $JOBSCRIPT"
    sbatch --parsable "$JOBSCRIPT" 2>"submit_ddp_${PARTITION}.err"
}

for PARTITION in gpu-devel gpu; do
    if JOBID=$(submit_to_partition "$PARTITION"); then
        echo "$JOBID" | tee last_ddp_jobid.txt
        echo "$PARTITION" | tee last_ddp_partition.txt
        echo "Submitted 4-GPU DDP job $JOBID to partition $PARTITION"
        exit 0
    else
        echo "Submit to partition $PARTITION failed:"
        cat "submit_ddp_${PARTITION}.err" || true
        echo
    fi
done

echo "ERROR: Could not submit DDP job."
exit 1
BASH

chmod +x submit_4gpu_ddp.sh
./submit_4gpu_ddp.sh
```

อ่านผล DDP:

```bash
JOBID=$(cat last_ddp_jobid.txt)
OUT="gpu_ddp4-${JOBID}.out"

echo "JOBID=$JOBID"
myqueue || squeue -u "$USER" || true

for i in {1..12}; do
    if squeue -j "$JOBID" -h | grep -q .; then
        echo "Still in queue/running: attempt $i/12"
        squeue -j "$JOBID" || true
        sleep 10
    else
        break
    fi
done

if [ -f "$OUT" ]; then
    grep -E "DDP JOB INFO|rank=|local_rank=|device=cuda|loss=|ERROR|Traceback" "$OUT" || true
    cat "$OUT"
else
    echo "Output file $OUT not found yet. Run later: cat $OUT"
fi
```

ผลที่ดีควรเห็น 4 บรรทัด เช่น:

```text
rank=0/4 local_rank=0 device=cuda:0
rank=1/4 local_rank=1 device=cuda:1
rank=2/4 local_rank=2 device=cuda:2
rank=3/4 local_rank=3 device=cuda:3
```

---

## 12) คำสั่งช่วยแก้ปัญหาแบบเร็ว

ดูงานของตนเอง:

```bash
myqueue || squeue -u "$USER"
```

ดูรายละเอียด job:

```bash
scontrol show job <JOBID>
```

ยกเลิก job:

```bash
scancel <JOBID>
```

ดูไฟล์ใน workspace:

```bash
cd "$HOME/lanta_gpu_short_training"
ls -lh
```

อ่าน log ทั้งหมด:

```bash
cat gpu_1gpu-<JOBID>.out
cat gpu_dp4-<JOBID>.out
cat gpu_ddp4-<JOBID>.out
```

ค้นหา error:

```bash
grep -RniE "error|failed|traceback|cuda|no module|not found|permission|invalid account|partition" .
```

---

## 13) วิธีตีความปัญหาที่เจอบ่อย

### กรณี 1: `CUDA available: False`

ถ้าเห็นบน frontend node ยังไม่ผิด แต่ถ้าเห็นในไฟล์ `.out` ของ GPU job ให้เช็กว่า job ส่งไป partition GPU จริงหรือไม่ และมี `#SBATCH --gpus=1` หรือ `#SBATCH --gpus=4` หรือไม่

### กรณี 2: `conda environment 'pytorch-2.2.2' was not found`

ให้ดูรายการ environment ที่มี:

```bash
module purge
module load Mamba/23.11.0-0 || module load Mamba
eval "$(conda shell.bash hook)"
conda env list
```

จากนั้นตั้งชื่อ environment ใหม่ให้ตรงกับที่ระบบมี:

```bash
export ENV_NAME=<ชื่อ_env_ที่เห็นจาก_conda_env_list>
```

แล้วกลับไปรันตั้งแต่ส่วน submit job อีกครั้ง

### กรณี 3: `Invalid account` หรือ `Invalid account or account/partition combination`

ให้เช็กว่าใช้ project/account ถูกต้องหรือไม่:

```bash
echo "$PROJECT_ID"
```

ถ้าต้องเปลี่ยน account:

```bash
export PROJECT_ID=<account_project_ที่ได้รับจาก_LANTA>
```

แล้ว submit ใหม่

### กรณี 4: งาน pending นาน

ดูเหตุผลการ pending:

```bash
JOBID=$(cat last_1gpu_jobid.txt)
squeue -j "$JOBID" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.20R"
```

ถ้าคิวแน่น ในคลาสสั้นให้ทำเฉพาะ 1-GPU job ก่อน และเก็บ 4-GPU / DDP ไว้เป็น instructor demo

---

## 14) Cleanup หลังจบคลาส ถ้าต้องการ

```bash
cd "$HOME"
# ลบเฉพาะเมื่อแน่ใจว่าไม่ต้องใช้ไฟล์แล้ว
# rm -rf "$HOME/lanta_gpu_short_training"
```

---

## 15) Flow แนะนำสำหรับคลาส 20–30 นาที

1. Paste ส่วน 1–5 เพื่อสำรวจ environment และยืนยันว่า `pytorch-2.2.2` ใช้ได้
2. Paste ส่วน 6 เพื่อสร้าง Python scripts
3. Paste ส่วน 7 เพื่อส่ง 1-GPU job
4. Paste ส่วน 8–9 เพื่ออ่านผล
5. ถ้าเวลาพอและคิวไม่แน่น ใช้ส่วน 10 เป็น 4-GPU demo
6. ถ้าต้องการสอนหลักการ training จริง ใช้ส่วน 11 เป็น DDP sanity check

