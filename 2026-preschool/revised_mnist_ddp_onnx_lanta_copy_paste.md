# 14 รู้จำตัวเลขที่เขียนด้วยลายมือ  
## คู่มือแบบ Copy-Paste เท่านั้น: ฝึก CNN บน LANTA GPU ด้วย PyTorch DDP แล้วแปลงเป็น ONNX เพื่อรันบนเว็บ

> เวอร์ชันนี้ออกแบบสำหรับเวลาฝึกอบรมสั้น ผู้เรียน **ไม่ต้องพิมพ์โค้ดเอง** และ **ไม่ต้องใช้ `nano`**  
> ให้ผู้เรียนคัดลอกคำสั่งทีละบล็อก วางใน Terminal แล้วรอดูผลลัพธ์ตามที่ระบุ

---

## ภาพรวมของบทเรียน

ในบทเรียนนี้ ผู้เรียนจะทำงานครบเส้นทางตั้งแต่ HPC ไปจนถึง Web App

1. สำรวจสภาพแวดล้อมบน LANTA ว่ามี module, conda/mamba, Python, PyTorch, CUDA และ ONNX พร้อมหรือไม่
2. ดาวน์โหลดชุดข้อมูล MNIST
3. สร้างโค้ด PyTorch CNN สำหรับรู้จำตัวเลขลายมือ
4. ส่งงานผ่าน Slurm เพื่อฝึกโมเดลบน GPU ด้วย Distributed Data Parallel หรือ DDP
5. ตรวจผลลัพธ์จาก log และไฟล์โมเดล
6. แปลงโมเดล PyTorch เป็น ONNX
7. สร้าง Web App แบบง่ายเพื่อรันโมเดลในเบราว์เซอร์

---

## สิ่งที่ผู้เรียนควรรู้ก่อนเริ่ม

### MNIST คืออะไร

MNIST คือชุดข้อมูลรูปภาพตัวเลขเขียนด้วยมือ ตั้งแต่ 0 ถึง 9  
แต่ละภาพมีขนาด 28 × 28 pixel และเป็นภาพขาวดำ

ในบทเรียนนี้ เราจะให้โมเดลดูภาพ แล้วตอบว่าเป็นเลขอะไร

ตัวอย่างแนวคิด:

```text
ภาพตัวเลข 7  →  โมเดล CNN  →  คำตอบ = 7
```

---

### CNN คืออะไร

CNN หรือ Convolutional Neural Network เป็น Neural Network ที่เหมาะกับภาพ

แนวคิดแบบง่าย:

```text
ภาพดิบ
  ↓
Convolution layer: หาเส้น ขอบ มุม ลวดลาย
  ↓
Pooling: ลดขนาดภาพ แต่เก็บลักษณะสำคัญไว้
  ↓
Fully Connected layer: ตัดสินใจว่าภาพเป็นเลขอะไร
  ↓
ผลลัพธ์ 10 ค่า สำหรับเลข 0 ถึง 9
```

---

### DDP คืออะไร

DDP หรือ Distributed Data Parallel เป็นวิธีฝึกโมเดลบนหลาย GPU พร้อมกัน

แนวคิดแบบง่าย:

```text
ข้อมูลชุดใหญ่
  ↓ แบ่งข้อมูล
GPU 0 ฝึกข้อมูลส่วนที่ 1
GPU 1 ฝึกข้อมูลส่วนที่ 2
GPU 2 ฝึกข้อมูลส่วนที่ 3
GPU 3 ฝึกข้อมูลส่วนที่ 4
  ↓
รวม gradient
  ↓
อัปเดตโมเดลให้เหมือนกันทุก GPU
```

ข้อดีคือฝึกได้เร็วขึ้นเมื่อใช้หลาย GPU

---

### ONNX คืออะไร

ONNX เป็น format กลางสำหรับโมเดล Machine Learning

ในบทเรียนนี้ เราจะฝึกโมเดลด้วย PyTorch บน LANTA แล้วแปลงเป็น ONNX เพื่อให้ Web App โหลดโมเดลไปใช้ในเบราว์เซอร์ได้

```text
PyTorch model: mnist_cnn.pt
  ↓ convert
ONNX model: mnist.onnx
  ↓
Web browser ใช้ทำนายตัวเลข
```

---

## กติกาการทำตามคู่มือ

1. คัดลอกคำสั่ง **ทั้งบล็อก**
2. วางลง Terminal
3. กด Enter
4. รอดูผลลัพธ์
5. ไปบล็อกถัดไป

ไม่ต้องพิมพ์ทีละบรรทัด  
ไม่ต้องเปิด `nano`  
ไม่ต้องแก้โค้ดเองระหว่างเรียน

---

# ส่วนที่ 1: เตรียมโฟลเดอร์ทำงานบน LANTA

## ผู้เรียนทำอะไรในขั้นนี้

เราจะสร้างโฟลเดอร์สำหรับบทเรียนนี้ใน home directory ของผู้เรียน  
จากนั้นตั้งตัวแปร `LANTA_ACCOUNT` สำหรับส่งงาน Slurm

> ในคู่มือนี้ตั้ง account เป็น `tn999992` แล้ว

## คัดลอกและวางบล็อกนี้บน LANTA login node

```bash
export LANTA_ACCOUNT="tn999992"
export LESSON_DIR="$HOME/hpcignite_mnist_ddp_web"

mkdir -p "$LESSON_DIR"
cd "$LESSON_DIR"

echo "===== Lesson directory ====="
pwd

echo "===== User and host ====="
whoami
hostname
date

echo "===== Account for Slurm ====="
echo "$LANTA_ACCOUNT"
```

## ผลลัพธ์ที่ควรเห็น

ควรเห็น path ประมาณนี้

```text
/home/your_username/hpcignite_mnist_ddp_web
```

และควรเห็น account

```text
tn999992
```

---

# ส่วนที่ 2: สำรวจ environment บน LANTA

## ทำไมต้องสำรวจ environment

บน HPC เราไม่ควรเดาว่าเครื่องมี Python, CUDA หรือ PyTorch พร้อมแล้ว  
เราต้องตรวจให้แน่ใจว่า environment ที่ใช้มีสิ่งเหล่านี้

- Python
- conda หรือ mamba
- PyTorch
- torchvision
- CUDA support
- ONNX สำหรับแปลงโมเดล
- Slurm สำหรับส่งงาน

---

## 2.1 สร้างสคริปต์ตรวจ environment

คัดลอกและวางบล็อกนี้

```bash
cat > 00_check_lanta_env.sh <<'BASH'
#!/usr/bin/env bash
set -u

echo "============================================================"
echo "00_check_lanta_env.sh"
echo "This script only inspects the environment. It does not change files."
echo "============================================================"
echo

echo "===== Basic information ====="
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PWD : $(pwd)"
echo

echo "===== Slurm commands ====="
command -v sbatch || true
command -v squeue || true
command -v sinfo || true
echo

echo "===== Slurm partitions visible to this account/user ====="
sinfo -o "%20P %10a %10l %8D %10G %N" 2>/dev/null | head -n 40 || true
echo

echo "===== Current loaded modules ====="
module list 2>&1 || true
echo

echo "===== Available modules matching useful keywords ====="
module avail 2>&1 | grep -Ei "mamba|conda|miniconda|miniforge|anaconda|python|cuda|cudnn|nccl|pytorch|gcc|cray" | head -n 160 || true
echo

echo "===== Module spider quick check ====="
for name in Mamba Miniforge3 Miniconda3 Anaconda3 Python CUDA cuda cudatoolkit NCCL nccl PyTorch pytorch; do
    echo
    echo "----- module spider $name -----"
    module spider "$name" 2>&1 | head -n 35 || true
done

echo
echo "===== Python commands before activating conda ====="
command -v python || true
python --version 2>/dev/null || true
command -v conda || true
conda --version 2>/dev/null || true
command -v mamba || true
mamba --version 2>/dev/null || true

echo
echo "Environment inspection finished."
BASH

chmod +x 00_check_lanta_env.sh
bash 00_check_lanta_env.sh | tee 00_env_report.txt
```

---

## 2.2 ความหมายของผลลัพธ์

ให้ดูจากไฟล์ `00_env_report.txt`

คำที่อยากเห็น:

```text
sbatch
squeue
sinfo
Mamba
conda
python
cuda
pytorch
```

ถ้าเห็นคำเหล่านี้ แปลว่า environment มีแนวโน้มพร้อมใช้งาน

ถ้าไม่เห็นบางคำ ไม่จำเป็นต้องตกใจ เพราะ module อาจยังไม่ได้ load  
ในขั้นถัดไปเราจะใช้สคริปต์เปิด environment ให้

---

# ส่วนที่ 3: เปิด conda/mamba environment ที่มี PyTorch

## ทำไมต้องมีขั้นนี้

โค้ดในบทเรียนต้องใช้ library เหล่านี้

- `torch`
- `torchvision`
- `onnx`

บน HPC library เหล่านี้มักถูกเตรียมไว้ใน conda environment หรือ mamba environment  
เราจะใช้ environment ที่ชื่อหรือ path ต่อไปนี้ก่อน

```text
/project/cb900907-hpctgn/envs/hpc-mesa
```

ถ้า environment นี้ไม่มีในระบบของผู้เรียน สคริปต์จะบอกให้รู้ทันที

---

## 3.1 สร้างสคริปต์เปิด environment

คัดลอกและวางบล็อกนี้

```bash
cat > 01_activate_env.sh <<'BASH'
#!/usr/bin/env bash
# Use with: source ./01_activate_env.sh

set -u

echo "===== Activating Python environment ====="

# Keep default Cray/LANTA modules. Do not purge, because some default modules may be needed.
LOADED_ENV_MODULE=""

for candidate in Mamba Miniforge3 Miniconda3 Anaconda3; do
    if module load "$candidate" 2>/dev/null; then
        LOADED_ENV_MODULE="$candidate"
        break
    fi
done

if [ -n "$LOADED_ENV_MODULE" ]; then
    echo "Loaded module: $LOADED_ENV_MODULE"
else
    echo "No Mamba/Conda module was loaded by name."
    echo "Trying to continue with existing conda/mamba in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
else
    echo "ERROR: conda command is not available after module load."
    echo "Please show 00_env_report.txt to the instructor."
    return 1 2>/dev/null || exit 1
fi

ENV_CANDIDATES=(
    "/project/cb900907-hpctgn/envs/hpc-mesa"
    "$HOME/.conda/envs/hpc-mesa"
    "$HOME/mambaforge/envs/hpc-mesa"
    "$HOME/miniforge3/envs/hpc-mesa"
)

ACTIVATED_ENV=""

for env_path in "${ENV_CANDIDATES[@]}"; do
    if [ -d "$env_path" ]; then
        conda activate "$env_path"
        ACTIVATED_ENV="$env_path"
        break
    fi
done

if [ -z "$ACTIVATED_ENV" ]; then
    if conda env list | grep -qE '(^|[[:space:]])hpc-mesa([[:space:]]|$)'; then
        conda activate hpc-mesa
        ACTIVATED_ENV="hpc-mesa"
    fi
fi

if [ -z "$ACTIVATED_ENV" ]; then
    echo "ERROR: Could not find or activate hpc-mesa environment."
    echo
    echo "Conda environments found:"
    conda env list || true
    echo
    echo "Please show this output and 00_env_report.txt to the instructor."
    return 1 2>/dev/null || exit 1
fi

echo "Activated environment: $ACTIVATED_ENV"
echo "Python path: $(command -v python)"
python --version

echo
echo "===== Python package check ====="
python - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch check failed:", repr(e))

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision check failed:", repr(e))

try:
    import onnx
    print("onnx:", onnx.__version__)
except Exception as e:
    print("onnx check warning:", repr(e))
    print("ONNX conversion may need instructor support if this package is missing.")
PY

echo
echo "Environment activation finished."
BASH

chmod +x 01_activate_env.sh
source ./01_activate_env.sh
```

---

## 3.2 ผลลัพธ์ที่ควรเห็น

บน login node อาจเห็นว่า CUDA ยังใช้ไม่ได้ เพราะ login node ไม่มี GPU ให้ใช้โดยตรง

```text
torch.cuda.is_available: False
torch.cuda.device_count: 0
```

สิ่งนี้ยังไม่ผิด

เมื่อส่งงานไป compute node ที่มี GPU ผ่าน Slurm แล้ว ควรเห็น GPU ได้ใน log ของงาน

ถ้าเห็น error ว่าไม่พบ `torch` หรือ `torchvision` ให้หยุดตรงนี้และส่งไฟล์นี้ให้ผู้สอน

```bash
cat 00_env_report.txt
```

---

# ส่วนที่ 4: สร้างไฟล์โปรแกรมทั้งหมดแบบ copy-paste

## ทำไมใช้วิธีนี้

แทนที่จะให้นักเรียนเปิด editor แล้วพิมพ์โค้ดเอง  
เราจะใช้คำสั่ง `cat > file <<'EOF'` เพื่อสร้างไฟล์อัตโนมัติ

ผู้เรียนจึงคัดลอกบล็อกเดียว แล้วได้ไฟล์ครบ

---

## 4.1 สร้างไฟล์ Python และ Slurm scripts

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"

cat > dataset_download.py <<'PY'
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR = Path("./data")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Downloading MNIST dataset if needed...")
train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

print("✅ MNIST dataset is ready.")
print(f"Training samples: {len(train)}")
print(f"Testing samples : {len(test)}")
print(f"Data folder     : {DATA_DIR.resolve()}")
PY

cat > DDP.py <<'PY'
import argparse
import os
from pathlib import Path
from socket import gethostname

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class Net(nn.Module):
    """
    Small CNN for MNIST.
    Input : 1 x 28 x 28 grayscale image
    Output: 10 log-probabilities for digits 0-9
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)       # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)      # 26x26 -> 24x24
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)           # 64 x 12 x 12 = 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST DDP Example for LANTA")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    return parser.parse_args()


def setup_distributed():
    """
    Slurm starts one process per task.
    We map one process to one GPU.

    Required environment variables are prepared in submit_02_train_mnist_ddp_1node.sh:
    - MASTER_ADDR
    - MASTER_PORT
    - WORLD_SIZE
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this script inside a GPU Slurm job.")

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))

    gpu_count = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(rank % max(gpu_count, 1)))))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size, local_rank, gpu_count


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(args, model, device, train_loader, optimizer, epoch, rank):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if rank == 0 and batch_idx % args.log_interval == 0:
            processed = batch_idx * len(data)
            total = len(train_loader.dataset)
            percent = 100.0 * batch_idx / max(len(train_loader), 1)
            print(
                f"Train Epoch: {epoch} "
                f"[rank0 local batch {processed}/{total} ({percent:.0f}%)] "
                f"Loss: {loss.item():.6f}",
                flush=True,
            )

        if args.dry_run and batch_idx >= 3:
            break


def test(model, device, test_loader):
    model.eval()

    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n",
        flush=True,
    )

    return accuracy


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    rank, world_size, local_rank, gpu_count = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("===== DDP job information =====", flush=True)
        print(f"Host                 : {gethostname()}", flush=True)
        print(f"WORLD_SIZE           : {world_size}", flush=True)
        print(f"MASTER_ADDR          : {os.environ.get('MASTER_ADDR')}", flush=True)
        print(f"MASTER_PORT          : {os.environ.get('MASTER_PORT')}", flush=True)
        print(f"CUDA devices per node: {gpu_count}", flush=True)
        print(f"Epochs               : {args.epochs}", flush=True)
        print(f"Batch size per rank  : {args.batch_size}", flush=True)
        print("================================", flush=True)

    print(
        f"Hello from host={gethostname()} rank={rank}/{world_size} "
        f"local_rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = Path(args.data_dir)
    dataset_train = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=transform)
    dataset_test = datasets.MNIST(root=str(data_dir), train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Net().to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_one_epoch(args, ddp_model, device, train_loader, optimizer, epoch, rank)

        # Only rank 0 prints validation result to keep logs readable.
        if rank == 0:
            test(ddp_model, device, test_loader)

        scheduler.step()
        dist.barrier()

    if args.save_model and rank == 0:
        torch.save(ddp_model.module.state_dict(), "mnist_cnn.pt")
        print("✅ Saved model to mnist_cnn.pt", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
PY

cat > convert_to_onnx.py <<'PY'
import torch
import onnx
from DDP import Net

MODEL_PATH = "mnist_cnn.pt"
ONNX_PATH = "mnist.onnx"

print(f"Loading PyTorch model from {MODEL_PATH}...")

model = Net()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

print(f"Exporting ONNX model to {ONNX_PATH}...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=12,
    do_constant_folding=True,
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

print(f"✅ ONNX model is ready: {ONNX_PATH}")
PY

cat > make_sample_digits.py <<'PY'
from pathlib import Path
from torchvision import datasets

out_dir = Path("webapp/sample_digits")
out_dir.mkdir(parents=True, exist_ok=True)

dataset = datasets.MNIST(root="./data", train=False, download=False)

saved = []
seen_labels = set()

for img, label in dataset:
    if label not in seen_labels:
        file_path = out_dir / f"sample_digit_{label}.png"
        img.save(file_path)
        saved.append(str(file_path))
        seen_labels.add(label)

    if len(seen_labels) == 10:
        break

print("✅ Sample digit images created:")
for path in sorted(saved):
    print(path)
PY

mkdir -p logs

cat > submit_02_train_mnist_ddp_1node.sh <<'BASH'
#!/bin/bash
#SBATCH -A tn999992
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=mnist-ddp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ./01_activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

echo "===== Slurm job information ====="
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST  = $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS        = $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "WORLD_SIZE          = $WORLD_SIZE"
echo "MASTER_ADDR         = $MASTER_ADDR"
echo "MASTER_PORT         = $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "================================="

srun --kill-on-bad-exit=1 python -u DDP.py \
    --epochs 1 \
    --batch-size 128 \
    --test-batch-size 1000 \
    --log-interval 100 \
    --save-model
BASH

chmod +x submit_02_train_mnist_ddp_1node.sh

cat > submit_03_train_mnist_ddp_2nodes.sh <<'BASH'
#!/bin/bash
#SBATCH -A tn999992
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=mnist-ddp-2n
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ./01_activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export WORLD_SIZE="${SLURM_NTASKS}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((30000 + SLURM_JOB_ID % 1000))"

echo "===== Slurm multi-node job information ====="
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST  = $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS        = $SLURM_NTASKS"
echo "WORLD_SIZE          = $WORLD_SIZE"
echo "MASTER_ADDR         = $MASTER_ADDR"
echo "MASTER_PORT         = $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "============================================"

srun --kill-on-bad-exit=1 python -u DDP.py \
    --epochs 1 \
    --batch-size 128 \
    --test-batch-size 1000 \
    --log-interval 100 \
    --save-model
BASH

chmod +x submit_03_train_mnist_ddp_2nodes.sh

echo "✅ Created files:"
ls -lh dataset_download.py DDP.py convert_to_onnx.py make_sample_digits.py submit_02_train_mnist_ddp_1node.sh submit_03_train_mnist_ddp_2nodes.sh
```

---

## 4.2 ไฟล์ที่ได้จากขั้นนี้

หลังจากรันเสร็จควรมีไฟล์เหล่านี้

```text
dataset_download.py
DDP.py
convert_to_onnx.py
make_sample_digits.py
submit_02_train_mnist_ddp_1node.sh
submit_03_train_mnist_ddp_2nodes.sh
```

ความหมายของแต่ละไฟล์:

| ไฟล์ | หน้าที่ |
|---|---|
| `dataset_download.py` | ดาวน์โหลด MNIST |
| `DDP.py` | โค้ด CNN + DDP สำหรับฝึกบนหลาย GPU |
| `convert_to_onnx.py` | แปลง `mnist_cnn.pt` เป็น `mnist.onnx` |
| `make_sample_digits.py` | สร้างรูปตัวเลขตัวอย่างสำหรับทดสอบเว็บ |
| `submit_02_train_mnist_ddp_1node.sh` | ส่งงานฝึกโมเดลบน 1 node, 4 GPU |
| `submit_03_train_mnist_ddp_2nodes.sh` | ตัวเลือกเสริมสำหรับ 2 nodes, 8 GPU |

---

# ส่วนที่ 5: ดาวน์โหลดข้อมูล MNIST

## ผู้เรียนทำอะไรในขั้นนี้

เราจะดาวน์โหลดข้อมูล MNIST ลงในโฟลเดอร์ `data`

ใช้ login node ได้ เพราะขั้นนี้ยังไม่ได้ฝึกด้วย GPU

## คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"
source ./01_activate_env.sh

python dataset_download.py

echo
echo "===== Check downloaded files ====="
find data -maxdepth 3 -type f | head -n 20
```

## ผลลัพธ์ที่ควรเห็น

ควรเห็นข้อความประมาณนี้

```text
✅ MNIST dataset is ready.
Training samples: 60000
Testing samples : 10000
```

---

# ส่วนที่ 6: ส่งงานฝึกโมเดลบน GPU ด้วย Slurm

## ทำไมต้องใช้ Slurm

บน HPC ผู้เรียนไม่ควรรันงานหนักบน login node  
การใช้ GPU ต้องส่งงานเข้าระบบ queue ด้วย Slurm

ในบทเรียนนี้เราจะใช้

```text
1 node
4 GPU
4 processes
1 process ต่อ 1 GPU
```

ภาพรวมการรัน:

```text
sbatch submit_02_train_mnist_ddp_1node.sh
  ↓
Slurm จัดสรร GPU node
  ↓
srun เปิด 4 processes
  ↓
แต่ละ process ใช้ GPU คนละตัว
  ↓
DDP รวม gradient
  ↓
บันทึกโมเดลเป็น mnist_cnn.pt
```

---

## 6.1 ส่งงานแบบ 1 node, 4 GPU

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"
mkdir -p logs

JOBID=$(sbatch submit_02_train_mnist_ddp_1node.sh | awk '{print $4}')
echo "$JOBID" | tee last_jobid.txt

echo
echo "Submitted job id: $JOBID"
echo
squeue -j "$JOBID"
```

---

## 6.2 ดูคิวและดู log ระหว่างรัน

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"

JOBID=$(cat last_jobid.txt)

echo "Watching job: $JOBID"
echo "Press Ctrl+C if you want to stop watching. The Slurm job will continue."

while squeue -j "$JOBID" -h | grep -q .; do
    echo
    echo "===== squeue at $(date) ====="
    squeue -j "$JOBID" || true

    if [ -f "logs/mnist-ddp-${JOBID}.out" ]; then
        echo
        echo "===== Last 25 lines of output ====="
        tail -n 25 "logs/mnist-ddp-${JOBID}.out"
    else
        echo "Output file is not ready yet."
    fi

    sleep 10
done

echo
echo "Job finished or left the queue."
```

---

## 6.3 ดูผลลัพธ์หลังงานจบ

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"

JOBID=$(cat last_jobid.txt)

echo "===== Output file ====="
tail -n 120 "logs/mnist-ddp-${JOBID}.out" || true

echo
echo "===== Error file ====="
cat "logs/mnist-ddp-${JOBID}.err" || true

echo
echo "===== Model file ====="
ls -lh mnist_cnn.pt || true
```

---

## ผลลัพธ์ที่ควรเห็น

ควรเห็นข้อความประมาณนี้ใน output

```text
Hello from host=...
rank=0/4 local_rank=0
rank=1/4 local_rank=1
rank=2/4 local_rank=2
rank=3/4 local_rank=3
```

และควรเห็นผลทดสอบ

```text
Test set: Average loss: ..., Accuracy: .../10000 (...%)
```

ถ้าฝึกแค่ 1 epoch ความแม่นยำอาจยังไม่สูงมาก  
เป้าหมายหลักของบทเรียนนี้คือให้ผู้เรียนเห็น workflow ครบ ไม่ใช่ทำ accuracy สูงที่สุด

ควรมีไฟล์โมเดล

```text
mnist_cnn.pt
```

---

# ส่วนที่ 7: แปลง PyTorch model เป็น ONNX

## ทำไมต้องแปลงเป็น ONNX

ไฟล์ `mnist_cnn.pt` เป็นไฟล์โมเดลของ PyTorch  
แต่ Web Browser ไม่ได้รัน PyTorch โดยตรง

เราจึงแปลงเป็น ONNX

```text
mnist_cnn.pt  →  mnist.onnx
```

---

## คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"
source ./01_activate_env.sh

python convert_to_onnx.py

echo
echo "===== ONNX file ====="
ls -lh mnist.onnx
```

---

## ผลลัพธ์ที่ควรเห็น

```text
✅ ONNX model is ready: mnist.onnx
```

และควรเห็นไฟล์

```text
mnist.onnx
```

---

# ส่วนที่ 8: สร้าง Web App สำหรับทดสอบโมเดล

## แนวคิดของ Web App

Web App นี้ทำงานแบบง่าย

```text
ผู้ใช้เลือกภาพตัวเลข
  ↓
JavaScript ย่อภาพเป็น 28 × 28
  ↓
แปลงภาพเป็น tensor
  ↓
โหลด mnist.onnx ด้วย ONNX Runtime Web
  ↓
แสดงผลว่าโมเดลทำนายเลขอะไร
```

เวอร์ชันนี้ไม่ใช้ Flask  
ใช้ Python HTTP server ที่มีมากับ Python เพื่อลดการติดตั้ง package เพิ่ม

---

## 8.1 สร้างไฟล์ Web App

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"

mkdir -p webapp
cp mnist.onnx webapp/

cat > webapp/index.html <<'HTML'
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <title>MNIST ONNX Web Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 32px auto;
      padding: 0 16px;
      line-height: 1.55;
    }
    .box {
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      margin: 16px 0;
    }
    canvas {
      border: 1px solid #aaa;
      image-rendering: pixelated;
      width: 180px;
      height: 180px;
      background: white;
    }
    button {
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid #888;
      cursor: pointer;
    }
    #result {
      font-size: 1.4rem;
      font-weight: 700;
    }
    code {
      background: #f3f3f3;
      padding: 2px 5px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <h1>ทำนายตัวเลขเขียนด้วยมือด้วย MNIST + ONNX</h1>

  <div class="box">
    <p>
      เลือกรูปภาพตัวเลข เช่น ไฟล์จากโฟลเดอร์ <code>sample_digits</code>
      แล้วกดปุ่มทำนาย
    </p>

    <input type="file" id="fileInput" accept="image/*" />
    <button id="predictButton">ทำนาย</button>

    <p id="status">กำลังโหลดโมเดล...</p>
    <p id="result">ผลลัพธ์จะปรากฏที่นี่</p>
  </div>

  <div class="box">
    <h2>ภาพที่โมเดลเห็นหลังย่อเป็น 28 × 28</h2>
    <canvas id="canvas" width="28" height="28"></canvas>
  </div>

  <script>
    let session = null;

    async function loadModel() {
      try {
        session = await ort.InferenceSession.create("./mnist.onnx");
        document.getElementById("status").innerText = "โหลดโมเดลสำเร็จ พร้อมทำนาย";
      } catch (err) {
        console.error(err);
        document.getElementById("status").innerText =
          "โหลดโมเดลไม่สำเร็จ กรุณาตรวจว่า mnist.onnx อยู่ในโฟลเดอร์เดียวกับ index.html";
      }
    }

    function softmax(logits) {
      const maxLogit = Math.max(...logits);
      const exps = logits.map(x => Math.exp(x - maxLogit));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(x => x / sum);
    }

    function preprocessImageToTensor(img) {
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");

      ctx.clearRect(0, 0, 28, 28);
      ctx.drawImage(img, 0, 0, 28, 28);

      const imageData = ctx.getImageData(0, 0, 28, 28).data;

      const grayscale = [];
      let avg = 0;

      for (let i = 0; i < 28 * 28; i++) {
        const r = imageData[i * 4 + 0];
        const g = imageData[i * 4 + 1];
        const b = imageData[i * 4 + 2];

        // Convert RGB to grayscale in [0, 1]
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        grayscale.push(gray);
        avg += gray;
      }

      avg /= grayscale.length;

      // MNIST usually has bright digit strokes on dark background.
      // If the uploaded image has white background and dark digit,
      // invert it automatically.
      const shouldInvert = avg > 0.5;

      const inputTensor = new Float32Array(1 * 1 * 28 * 28);

      for (let i = 0; i < 28 * 28; i++) {
        let pixel = shouldInvert ? (1.0 - grayscale[i]) : grayscale[i];

        // Same normalization as PyTorch training:
        // transforms.Normalize((0.1307,), (0.3081,))
        pixel = (pixel - 0.1307) / 0.3081;

        inputTensor[i] = pixel;
      }

      return inputTensor;
    }

    async function predict() {
      if (!session) {
        alert("โมเดลยังโหลดไม่เสร็จ กรุณารอสักครู่");
        return;
      }

      const fileInput = document.getElementById("fileInput");
      if (!fileInput.files.length) {
        alert("กรุณาเลือกไฟล์ภาพก่อน");
        return;
      }

      const file = fileInput.files[0];
      const img = new Image();

      img.onload = async function () {
        const inputTensor = preprocessImageToTensor(img);

        const feeds = {
          input: new ort.Tensor("float32", inputTensor, [1, 1, 28, 28])
        };

        const results = await session.run(feeds);
        const output = Array.from(results.output.data);

        const predictedDigit = output.indexOf(Math.max(...output));
        const probs = softmax(output);
        const confidence = probs[predictedDigit] * 100;

        document.getElementById("result").innerText =
          `ผลลัพธ์: ${predictedDigit}  |  ความมั่นใจประมาณ ${confidence.toFixed(2)}%`;
      };

      img.src = URL.createObjectURL(file);
    }

    document.getElementById("predictButton").addEventListener("click", predict);
    loadModel();
  </script>
</body>
</html>
HTML

cat > webapp/server.py <<'PY'
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import os

PORT = int(os.environ.get("PORT", "5000"))

print(f"Serving files at http://127.0.0.1:{PORT}/")
print("Open that URL in a browser.")
print("Press Ctrl+C to stop the server.")

server = ThreadingHTTPServer(("0.0.0.0", PORT), SimpleHTTPRequestHandler)
server.serve_forever()
PY

python make_sample_digits.py

echo
echo "✅ Web App files are ready:"
find webapp -maxdepth 2 -type f | sort
```

---

## 8.2 ผลลัพธ์ที่ควรเห็น

ควรมีไฟล์เหล่านี้

```text
webapp/index.html
webapp/server.py
webapp/mnist.onnx
webapp/sample_digits/sample_digit_0.png
...
webapp/sample_digits/sample_digit_9.png
```

---

# ส่วนที่ 9: รัน Web App

มี 2 วิธี ขึ้นกับรูปแบบการอบรม

---

## วิธี A: รันบนเครื่องส่วนตัวของผู้เรียน

วิธีนี้เหมาะถ้าผู้สอนให้ผู้เรียนดาวน์โหลดโฟลเดอร์ `webapp` มาไว้บนเครื่อง Windows แล้ว

### เปิด PowerShell แล้ววางบล็อกนี้

```powershell
cd C:\hpcignite\webapp
py -m http.server 5000
```

ถ้าเครื่องไม่มีคำสั่ง `py` ให้ใช้บล็อกนี้แทน

```powershell
cd C:\hpcignite\webapp
python -m http.server 5000
```

จากนั้นเปิดเบราว์เซอร์ไปที่

```text
http://127.0.0.1:5000/
```

เลือกไฟล์ภาพตัวเลขจากโฟลเดอร์

```text
C:\hpcignite\webapp\sample_digits
```

แล้วกดปุ่ม `ทำนาย`

---

## วิธี B: รันบน LANTA ผ่าน terminal

วิธีนี้ใช้ได้เมื่อผู้เรียนมีวิธีเปิด port หรือ tunnel จากเครื่องตัวเองมายัง LANTA  
ถ้าไม่มี tunnel ผู้เรียนอาจเปิดเว็บจาก browser ในเครื่องตัวเองไม่ได้

คัดลอกและวางบล็อกนี้บน LANTA

```bash
cd "$LESSON_DIR/webapp"
python server.py
```

จากนั้นให้ผู้สอนช่วยตั้งค่า SSH tunnel หรือใช้ระบบ web access ที่ site เตรียมไว้

---

# ส่วนที่ 10: สรุปผลจาก log แบบอัตโนมัติ

## ทำไมต้องมีขั้นนี้

ผู้เรียนมักดู log ยาว ๆ แล้วไม่แน่ใจว่าต้องมองตรงไหน  
สคริปต์นี้ช่วยดึงบรรทัดสำคัญออกมา

---

## คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"

cat > summarize_training_log.py <<'PY'
from pathlib import Path
import re

log_dir = Path("logs")
logs = sorted(log_dir.glob("mnist-ddp*.out"))

if not logs:
    print("No mnist-ddp output logs found.")
    raise SystemExit(0)

for log in logs:
    print("=" * 80)
    print(log)
    print("=" * 80)

    text = log.read_text(errors="replace")

    for line in text.splitlines():
        if (
            "Hello from host=" in line
            or "WORLD_SIZE" in line
            or "MASTER_ADDR" in line
            or "CUDA devices per node" in line
            or "Test set:" in line
            or "Saved model" in line
        ):
            print(line)

    acc = re.findall(r"Accuracy:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", text)
    if acc:
        correct, total, percent = acc[-1]
        print(f"Final parsed accuracy: {correct}/{total} = {percent}%")

    print()
PY

python summarize_training_log.py
```

---

# ส่วนที่ 11: ตัวเลือกเสริมสำหรับผู้สอน — รันแบบ 2 nodes

> ส่วนนี้เป็น optional  
> ใช้เฉพาะเมื่อ queue ว่างและผู้สอนต้องการให้เห็น multi-node DDP จริง

แนวคิด:

```text
2 nodes
4 GPU ต่อ node
รวม 8 GPU
8 processes
```

คัดลอกและวางบล็อกนี้

```bash
cd "$LESSON_DIR"
mkdir -p logs

JOBID=$(sbatch submit_03_train_mnist_ddp_2nodes.sh | awk '{print $4}')
echo "$JOBID" | tee last_jobid_multinode.txt

echo
echo "Submitted multi-node job id: $JOBID"
squeue -j "$JOBID"
```

ดูผลลัพธ์

```bash
cd "$LESSON_DIR"

JOBID=$(cat last_jobid_multinode.txt)

while squeue -j "$JOBID" -h | grep -q .; do
    echo
    echo "===== squeue at $(date) ====="
    squeue -j "$JOBID" || true

    if [ -f "logs/mnist-ddp-2n-${JOBID}.out" ]; then
        echo
        echo "===== Last 25 lines of multi-node output ====="
        tail -n 25 "logs/mnist-ddp-2n-${JOBID}.out"
    else
        echo "Output file is not ready yet."
    fi

    sleep 10
done

echo
tail -n 120 "logs/mnist-ddp-2n-${JOBID}.out" || true
cat "logs/mnist-ddp-2n-${JOBID}.err" || true
```

สิ่งที่ควรเห็นคือ `WORLD_SIZE=8` และ rank ตั้งแต่ 0 ถึง 7

---

# ส่วนที่ 12: อธิบาย workflow ให้ผู้เรียนเข้าใจหลังทำเสร็จ

## 12.1 จากภาพไปเป็น tensor

ภาพ MNIST มีขนาด 28 × 28  
แต่ละ pixel ถูกแปลงเป็นตัวเลข

```text
ภาพ 28 × 28
  ↓
ตัวเลข 784 ค่า
  ↓
tensor shape = [1, 1, 28, 28]
```

ความหมายของ shape:

| ค่า | ความหมาย |
|---|---|
| 1 ตัวแรก | batch size |
| 1 ตัวที่สอง | จำนวน channel เพราะเป็นภาพขาวดำ |
| 28 | ความสูง |
| 28 | ความกว้าง |

---

## 12.2 ทำไมต้อง Normalize

ตอนฝึกโมเดล เรา normalize ข้อมูลด้วยค่า mean และ std ของ MNIST

```python
transforms.Normalize((0.1307,), (0.3081,))
```

ดังนั้นใน Web App เราต้อง normalize แบบเดียวกัน

```text
pixel_normalized = (pixel - 0.1307) / 0.3081
```

ถ้า normalize ไม่ตรงกัน โมเดลอาจทำนายผิดง่าย

---

## 12.3 ทำไมต้องใช้ DDP

ถ้าใช้ GPU เดียว

```text
GPU 0 ฝึกข้อมูลทั้งหมด
```

ถ้าใช้ 4 GPU ด้วย DDP

```text
GPU 0 ฝึกข้อมูลส่วนหนึ่ง
GPU 1 ฝึกข้อมูลส่วนหนึ่ง
GPU 2 ฝึกข้อมูลส่วนหนึ่ง
GPU 3 ฝึกข้อมูลส่วนหนึ่ง
```

จากนั้น DDP จะช่วย sync gradient เพื่อให้ทุก GPU อัปเดตโมเดลไปในทิศทางเดียวกัน

---

## 12.4 Slurm ทำหน้าที่อะไร

Slurm เป็นระบบจัดคิวงานบน HPC

ผู้เรียนส่งคำสั่ง

```bash
sbatch submit_02_train_mnist_ddp_1node.sh
```

Slurm จะทำสิ่งนี้ให้

```text
รับคำขอใช้ GPU
รอ queue
จัดสรร node
ตั้ง environment ของ job
เรียก srun เพื่อเปิดหลาย processes
เก็บ output และ error log
```

---

## 12.5 ไฟล์สำคัญหลังจบบทเรียน

| ไฟล์ | ความหมาย |
|---|---|
| `00_env_report.txt` | รายงาน environment บน LANTA |
| `mnist_cnn.pt` | โมเดล PyTorch ที่ฝึกเสร็จ |
| `mnist.onnx` | โมเดล ONNX สำหรับ Web App |
| `logs/mnist-ddp-*.out` | output จาก Slurm job |
| `logs/mnist-ddp-*.err` | error จาก Slurm job |
| `webapp/index.html` | หน้าเว็บสำหรับทดสอบ |
| `webapp/mnist.onnx` | โมเดลที่เว็บโหลดไปใช้ |

---

# ส่วนที่ 13: Troubleshooting แบบ copy-paste

## ปัญหา 1: ไม่แน่ใจว่างานอยู่ในคิวหรือรันแล้ว

```bash
cd "$LESSON_DIR"
JOBID=$(cat last_jobid.txt)
squeue -j "$JOBID"
```

ถ้าไม่มี output แปลว่างานจบแล้วหรือออกจากคิวแล้ว

---

## ปัญหา 2: งานจบแต่ไม่เห็นผลลัพธ์

```bash
cd "$LESSON_DIR"
JOBID=$(cat last_jobid.txt)

echo "===== OUT ====="
cat "logs/mnist-ddp-${JOBID}.out" || true

echo "===== ERR ====="
cat "logs/mnist-ddp-${JOBID}.err" || true
```

---

## ปัญหา 3: ไม่พบ `mnist_cnn.pt`

```bash
cd "$LESSON_DIR"
find . -name "mnist_cnn.pt" -ls
```

ถ้าไม่พบ ให้เปิด error log

```bash
cd "$LESSON_DIR"
JOBID=$(cat last_jobid.txt)
cat "logs/mnist-ddp-${JOBID}.err"
```

---

## ปัญหา 4: error เรื่อง account หรือ partition

ถ้าเห็นข้อความคล้าย ๆ นี้

```text
Invalid account
Invalid partition
```

ให้ตรวจ account และ partition

```bash
echo "Current LANTA_ACCOUNT=$LANTA_ACCOUNT"
sinfo -o "%20P %10a %10l %8D %10G %N" | head -n 40
```

ในคู่มือนี้ script ตั้งค่าไว้ว่า

```bash
#SBATCH -A tn999992
#SBATCH -p gpu
```

ถ้า site ใช้ชื่อ partition อื่น ผู้สอนต้องแก้ในไฟล์ submit script ให้ตรงกับระบบจริง

---

## ปัญหา 5: error เรื่อง dataset ไม่พบ

ให้รันใหม่

```bash
cd "$LESSON_DIR"
source ./01_activate_env.sh
python dataset_download.py
```

แล้วส่งงานใหม่

```bash
cd "$LESSON_DIR"
JOBID=$(sbatch submit_02_train_mnist_ddp_1node.sh | awk '{print $4}')
echo "$JOBID" | tee last_jobid.txt
squeue -j "$JOBID"
```

---

## ปัญหา 6: Web App โหลด `mnist.onnx` ไม่ได้

ตรวจว่าไฟล์อยู่ในโฟลเดอร์เดียวกับ `index.html`

```bash
cd "$LESSON_DIR"
ls -lh webapp/index.html webapp/mnist.onnx
```

ถ้าไม่มี `webapp/mnist.onnx` ให้คัดลอกใหม่

```bash
cd "$LESSON_DIR"
cp mnist.onnx webapp/
ls -lh webapp/mnist.onnx
```

---

# ส่วนที่ 14: แบบฝึกหัดท้ายบท

## แบบฝึกหัด 1: เปลี่ยนจำนวน epoch

ส่งงานใหม่โดยฝึก 2 epochs

```bash
cd "$LESSON_DIR"

cp submit_02_train_mnist_ddp_1node.sh submit_02_train_mnist_ddp_1node_2epochs.sh

python - <<'PY'
from pathlib import Path

p = Path("submit_02_train_mnist_ddp_1node_2epochs.sh")
text = p.read_text()
text = text.replace("--epochs 1", "--epochs 2")
text = text.replace("#SBATCH --job-name=mnist-ddp", "#SBATCH --job-name=mnist-ddp-2ep")
text = text.replace("logs/%x-%j.out", "logs/%x-%j.out")
p.write_text(text)

print("✅ Created submit_02_train_mnist_ddp_1node_2epochs.sh")
PY

chmod +x submit_02_train_mnist_ddp_1node_2epochs.sh

JOBID=$(sbatch submit_02_train_mnist_ddp_1node_2epochs.sh | awk '{print $4}')
echo "$JOBID" | tee last_jobid_2epochs.txt
squeue -j "$JOBID"
```

---

## แบบฝึกหัด 2: เปรียบเทียบ log

หลังงาน 2 epochs จบแล้ว รันคำสั่งนี้

```bash
cd "$LESSON_DIR"
python summarize_training_log.py
```

ให้ผู้เรียนตอบคำถาม:

1. Accuracy ของ 1 epoch เท่าไร
2. Accuracy ของ 2 epochs เท่าไร
3. ใช้ GPU กี่ตัว
4. `WORLD_SIZE` คืออะไร
5. ทำไม `rank` มีหลายค่า

---

# สรุปบทเรียน

เมื่อจบบทนี้ ผู้เรียนได้ทำครบ workflow ต่อไปนี้

```text
สำรวจ environment บน LANTA
  ↓
activate conda/mamba environment
  ↓
download MNIST
  ↓
create PyTorch CNN + DDP code
  ↓
submit Slurm GPU job
  ↓
read log and accuracy
  ↓
save PyTorch model
  ↓
convert model to ONNX
  ↓
create web app
  ↓
run inference in browser
```

สิ่งสำคัญที่สุดของบทเรียนนี้ไม่ใช่ accuracy สูงสุด  
แต่คือการเห็นกระบวนการครบตั้งแต่ HPC training ไปจนถึงการนำโมเดลออกไปใช้งานจริง

---

# Checklist สำหรับผู้สอน

ก่อนเริ่มอบรม ควรตรวจ 5 อย่างนี้

```bash
cd "$HOME/hpcignite_mnist_ddp_web" 2>/dev/null || true
```

1. Account `tn999992` ใช้ได้กับ partition `gpu` หรือไม่
2. มี module `Mamba` หรือ conda module ที่ใช้งานได้หรือไม่
3. environment `/project/cb900907-hpctgn/envs/hpc-mesa` ยังมีอยู่หรือไม่
4. environment มี `torch`, `torchvision`, `onnx` หรือไม่
5. compute node มี GPU และสามารถรัน NCCL DDP ได้หรือไม่

คำสั่งตรวจแบบเร็ว:

```bash
cd "$HOME/hpcignite_mnist_ddp_web"
bash 00_check_lanta_env.sh | tee 00_env_report_latest.txt
source ./01_activate_env.sh
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda available on this node:", torch.cuda.is_available())
PY
```
