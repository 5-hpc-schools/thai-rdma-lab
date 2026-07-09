# แล็บ Copy-Paste: Scientific Workloads บน LANTA HPC

> เวอร์ชันสำหรับอบรมเวลาสั้น: ผู้เรียนคัดลอกทีละบล็อกแล้วรันได้เลย  
> เป้าหมาย: ให้เห็น workflow จริงของงานวิทยาศาสตร์บน LANTA ได้แก่ GROMACS, Quantum ESPRESSO, WRF/WPS smoke test และ Bioinformatics toy pipeline

---

## 0) แนวคิดของแล็บนี้

แล็บนี้ออกแบบให้ใช้เวลาสั้น โดยไม่ให้ผู้เรียนต้องแก้ไฟล์ด้วย editor เอง ทุกอย่างใช้ `cat > file <<'EOF'` เพื่อสร้าง script และใช้ `sbatch` เพื่อส่งงานเข้า Slurm

สิ่งที่จะได้ฝึก:

1. สำรวจ environment บน LANTA
2. ใช้ `module`, `module avail`, `module load`, `module list`
3. เตรียมพื้นที่งานใน `/project` และ `/scratch` แบบอัตโนมัติ
4. รันงาน molecular dynamics ด้วย GROMACS
5. รันงาน electronic structure ด้วย Quantum ESPRESSO
6. ตรวจ environment ของ WRF/WPS บน compute node
7. รัน toy bioinformatics pipeline ด้วย BWA/SAMtools/FastQC ถ้ามี module พร้อม
8. ดู queue, log, และผลลัพธ์แบบ copy-paste

ค่าเริ่มต้นของบัญชีโครงการในแล็บนี้คือ

```bash
LANTA_ACCOUNT=tn999992
```

ถ้าผู้สอนต้องใช้ account อื่น ให้รันคำสั่งนี้ก่อนเริ่มแล็บ แล้วค่อยคัดลอกบล็อกถัดไป

```bash
export LANTA_ACCOUNT=your_project_account
```

---

## 1) Login เข้า LANTA

จากเครื่องของผู้เรียน ใช้ SSH เข้า login node

```bash
ssh your_username@lanta.nstda.or.th
```

หลังจาก login แล้ว ให้ทำทุกคำสั่งต่อไปนี้บน LANTA

---

## 2) สร้างพื้นที่แล็บแบบอัตโนมัติ

คัดลอกทั้งบล็อกนี้ไปวางใน terminal

```bash
cat > 00_setup_lanta_science_lab.sh <<'BASH'
#!/bin/bash
set -euo pipefail

export LANTA_ACCOUNT="${LANTA_ACCOUNT:-tn999992}"

PROJECT_DIR="$(ls -d /project/${LANTA_ACCOUNT}* 2>/dev/null | head -n 1 || true)"
SCRATCH_BASE="$(ls -d /scratch/${LANTA_ACCOUNT}* 2>/dev/null | head -n 1 || true)"

if [ -z "${PROJECT_DIR}" ]; then
  echo "WARNING: Cannot find /project/${LANTA_ACCOUNT}*. Falling back to HOME."
  PROJECT_DIR="${HOME}"
fi

if [ -z "${SCRATCH_BASE}" ]; then
  echo "WARNING: Cannot find /scratch/${LANTA_ACCOUNT}*. Falling back to PROJECT_DIR."
  SCRATCH_BASE="${PROJECT_DIR}"
fi

if [ "${PROJECT_DIR}" = "${HOME}" ]; then
  LABROOT="${HOME}/lanta_science_workload_lab"
else
  LABROOT="${PROJECT_DIR}/${USER}/lanta_science_workload_lab"
fi

SCRATCH_ROOT="${SCRATCH_BASE}/${USER}/lanta_science_workload_lab"

mkdir -p "${LABROOT}"/{logs,jobs,results,src,gromacs,qe,wrf,bio}
mkdir -p "${SCRATCH_ROOT}"

cat > "${LABROOT}/lanta_science_lab.env" <<EOF
export LANTA_ACCOUNT="${LANTA_ACCOUNT}"
export LABROOT="${LABROOT}"
export SCRATCH_ROOT="${SCRATCH_ROOT}"
EOF

ln -sf "${LABROOT}/lanta_science_lab.env" "${HOME}/lanta_science_lab.env"

cat <<EOF

===== LANTA Scientific Workload Lab =====
LANTA_ACCOUNT = ${LANTA_ACCOUNT}
LABROOT       = ${LABROOT}
SCRATCH_ROOT  = ${SCRATCH_ROOT}
ENV FILE      = ${HOME}/lanta_science_lab.env

Next command:
source ${HOME}/lanta_science_lab.env
cd ${LABROOT}

EOF
BASH

bash 00_setup_lanta_science_lab.sh
source ${HOME}/lanta_science_lab.env
cd ${LABROOT}
```

ตรวจสอบว่าอยู่ในพื้นที่แล็บแล้ว

```bash
pwd
ls -la
cat ${HOME}/lanta_science_lab.env
```

---

## 3) สำรวจระบบและ software environment

บล็อกนี้รันบน login node ได้ เพราะเป็นการดูข้อมูลเท่านั้น ไม่ใช่งานหนัก

```bash
cat > 01_browse_environment.sh <<'BASH'
#!/bin/bash
set +e
source ${HOME}/lanta_science_lab.env
cd "${LABROOT}"

{
  echo "===== who / where ====="
  date
  hostname
  whoami
  pwd

  echo
  echo "===== quota ====="
  myquota 2>/dev/null || true

  echo
  echo "===== Slurm overview ====="
  sinfo -s 2>/dev/null || sinfo 2>/dev/null || true

  echo
  echo "===== module command ====="
  module --version 2>&1 || true

  echo
  echo "===== currently loaded modules ====="
  module list 2>&1 || true

  echo
  echo "===== module overview: first 100 lines ====="
  module overview 2>&1 | head -n 100 || true

  echo
  echo "===== GROMACS modules ====="
  module avail GROMACS 2>&1 || true

  echo
  echo "===== Quantum ESPRESSO modules ====="
  module avail QuantumESPRESSO 2>&1 || true

  echo
  echo "===== WRF/WPS modules ====="
  module avail WRF 2>&1 || true
  module avail WPS 2>&1 || true
  module avail WRFchem 2>&1 || true

  echo
  echo "===== Bioinformatics module path ====="
  module use /lustrefs/disk/modules/easybuild/modules/bio 2>/dev/null || true
  module avail BWA 2>&1 || true
  module avail SAMtools 2>&1 || true
  module avail FastQC 2>&1 || true
  module avail BLAST 2>&1 || true

  echo
  echo "===== Conda/Mamba ====="
  module avail Mamba 2>&1 || true
  module load Mamba/23.11.0-0 2>/dev/null || true
  which conda 2>/dev/null || true
  which mamba 2>/dev/null || true
  conda env list 2>/dev/null || true

  echo
  echo "===== Apptainer ====="
  module avail Apptainer 2>&1 || true

} | tee "${LABROOT}/logs/01_browse_environment.log"

cat <<EOF

Environment report saved to:
${LABROOT}/logs/01_browse_environment.log

EOF
BASH

bash 01_browse_environment.sh
```

---

## 4) สร้าง Slurm job scripts ทั้งหมด

คัดลอกบล็อกนี้เพื่อสร้าง job scripts สำหรับทุก workload

```bash
cat > 02_make_science_jobs.sh <<'BASH'
#!/bin/bash
set -euo pipefail
source ${HOME}/lanta_science_lab.env
mkdir -p "${LABROOT}/jobs" "${LABROOT}/logs" "${LABROOT}/results"

cat > "${LABROOT}/jobs/gromacs_cpu_1node.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J gmx_cpu_1n
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:20:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module restore
module load GROMACS/2023.2-cpeGNU-23.09-CUDA-12.0 2>/dev/null || module load GROMACS/2023.5-cpeGNU-23.09-CUDA-12.0 2>/dev/null || module load GROMACS
module list
which gmx

cd "${LABROOT}/gromacs"
if [ ! -d GROMACS ] && [ -d /project/common/GROMACS ]; then
  cp -r /project/common/GROMACS .
fi

if [ -d GROMACS ]; then
  cd GROMACS
fi

TPR=\$(find . -path "*/INPUTs/*" -name "benchPEP-h*" | head -n 1 || true)
if [ -z "\${TPR}" ]; then
  TPR=\$(find . -name "*.tpr" | head -n 1 || true)
fi

if [ -z "\${TPR}" ]; then
  echo "ERROR: No GROMACS .tpr input found. Check /project/common/GROMACS."
  exit 2
fi

RUNDIR="RUN_CPU_\${SLURM_JOB_ID}"
mkdir -p "\${RUNDIR}"
cd "\${RUNDIR}"

echo "===== GROMACS CPU job ====="
echo "Host: \$(hostname)"
echo "Input TPR: ../\${TPR#./}"
echo "SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK}"

gmx mdrun -s "../\${TPR#./}" -nsteps 1000 -maxh 0.05 | tee gromacs_cpu.log

grep -E "Performance:|Finished mdrun|Running on" gromacs_cpu.log || true
SLURM

cat > "${LABROOT}/jobs/gromacs_gpu_1card.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J gmx_gpu_1g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module restore
module load GROMACS/2023.2-cpeGNU-23.09-CUDA-12.0 2>/dev/null || module load GROMACS/2023.5-cpeGNU-23.09-CUDA-12.0 2>/dev/null || module load GROMACS
module list
which gmx
nvidia-smi || true

cd "${LABROOT}/gromacs"
if [ ! -d GROMACS ] && [ -d /project/common/GROMACS ]; then
  cp -r /project/common/GROMACS .
fi

if [ -d GROMACS ]; then
  cd GROMACS
fi

TPR=\$(find . -path "*/INPUTs/*" -name "benchPEP-h*" | head -n 1 || true)
if [ -z "\${TPR}" ]; then
  TPR=\$(find . -name "*.tpr" | head -n 1 || true)
fi

if [ -z "\${TPR}" ]; then
  echo "ERROR: No GROMACS .tpr input found. Check /project/common/GROMACS."
  exit 2
fi

RUNDIR="RUN_GPU_\${SLURM_JOB_ID}"
mkdir -p "\${RUNDIR}"
cd "\${RUNDIR}"

echo "===== GROMACS GPU job ====="
echo "Host: \$(hostname)"
echo "Input TPR: ../\${TPR#./}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not_set}"

gmx mdrun -s "../\${TPR#./}" -nsteps 1000 -maxh 0.05 | tee gromacs_gpu.log

grep -E "Performance:|Finished mdrun|compatible GPU|GPU selected|Mapping of GPU" gromacs_gpu.log || true
SLURM

cat > "${LABROOT}/jobs/qe_cpu_1node.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J qe_cpu_1n
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module purge
module load QuantumESPRESSO/7.3.1-libxc-6.2.2-cpu 2>/dev/null || module load QuantumESPRESSO/7.2-libxc-6.1.0-cpu 2>/dev/null || module load QuantumESPRESSO
module list
which pw.x

cd "${LABROOT}/qe"
if [ ! -d QuantumEspresso ] && [ -d /project/common/QuantumEspresso ]; then
  cp -r /project/common/QuantumEspresso .
fi

QE_IN=\$(find . -name "ausurf.in" | head -n 1 || true)
if [ -z "\${QE_IN}" ]; then
  QE_IN=\$(find . -name "*.in" | head -n 1 || true)
fi

if [ -z "\${QE_IN}" ]; then
  echo "ERROR: No QE input found. Check /project/common/QuantumEspresso."
  exit 2
fi

QEDIR=\$(dirname "\${QE_IN}")
QEBASE=\$(basename "\${QE_IN}")
cd "\${QEDIR}"

mkdir -p "${SCRATCH_ROOT}/qe_cpu_\${SLURM_JOB_ID}"
export ESPRESSO_TMPDIR="${SCRATCH_ROOT}/qe_cpu_\${SLURM_JOB_ID}"
export ESPRESSO_PSEUDO="\${PWD}"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
ulimit -s unlimited

echo "===== QE CPU job ====="
echo "Host: \$(hostname)"
echo "Input: \${PWD}/\${QEBASE}"
echo "ESPRESSO_TMPDIR=\${ESPRESSO_TMPDIR}"
echo "OMP_NUM_THREADS=\${OMP_NUM_THREADS}"

srun --cpus-per-task="\${SLURM_CPUS_PER_TASK}" pw.x -inp "\${QEBASE}" > qe_cpu.out

grep -E "Program PWSCF|Parallel version|MPI processes|Threads/MPI process|JOB DONE|PWSCF" qe_cpu.out | tail -n 30 || true
SLURM

cat > "${LABROOT}/jobs/qe_gpu_1card.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J qe_gpu_1g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module purge
module load QuantumESPRESSO/7.3.1-libxc-6.2.2-gpu 2>/dev/null || module load QuantumESPRESSO/7.2-libxc-6.1.0-gpu 2>/dev/null || module load QuantumESPRESSO
module list
which pw.x
nvidia-smi || true

cd "${LABROOT}/qe"
if [ ! -d QuantumEspresso ] && [ -d /project/common/QuantumEspresso ]; then
  cp -r /project/common/QuantumEspresso .
fi

QE_IN=\$(find . -name "ausurf.in" | head -n 1 || true)
if [ -z "\${QE_IN}" ]; then
  QE_IN=\$(find . -name "*.in" | head -n 1 || true)
fi

if [ -z "\${QE_IN}" ]; then
  echo "ERROR: No QE input found. Check /project/common/QuantumEspresso."
  exit 2
fi

QEDIR=\$(dirname "\${QE_IN}")
QEBASE=\$(basename "\${QE_IN}")
cd "\${QEDIR}"

mkdir -p "${SCRATCH_ROOT}/qe_gpu_\${SLURM_JOB_ID}"
export ESPRESSO_TMPDIR="${SCRATCH_ROOT}/qe_gpu_\${SLURM_JOB_ID}"
export ESPRESSO_PSEUDO="\${PWD}"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
ulimit -s unlimited

echo "===== QE GPU job ====="
echo "Host: \$(hostname)"
echo "Input: \${PWD}/\${QEBASE}"
echo "ESPRESSO_TMPDIR=\${ESPRESSO_TMPDIR}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not_set}"

srun --cpus-per-task="\${SLURM_CPUS_PER_TASK}" pw.x -inp "\${QEBASE}" > qe_gpu.out

grep -E "Program PWSCF|Parallel version|MPI processes|Threads/MPI process|GPU acceleration|GPU-aware MPI|JOB DONE|PWSCF" qe_gpu.out | tail -n 40 || true
SLURM

cat > "${LABROOT}/jobs/wrf_wps_smoke.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J wrf_smoke
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module purge
module load WRF/4.4.2-DMSM-cpeCray-23.03 2>/dev/null || module load WRF 2>/dev/null || true
module load WPS/4.4-DM-cpeCray-23.03 2>/dev/null || module load WPS 2>/dev/null || true
module list

cd "${LABROOT}/wrf"
RUNDIR="SMOKE_\${SLURM_JOB_ID}"
mkdir -p "\${RUNDIR}"
cd "\${RUNDIR}"

{
  echo "===== WRF/WPS smoke test ====="
  echo "Host: \$(hostname)"
  echo "This checks WRF/WPS environment on a compute node. It does not run a full forecast."
  echo
  echo "Executables found:"
  command -v wrf.exe || true
  command -v real.exe || true
  command -v geogrid.exe || true
  command -v metgrid.exe || true
  command -v ncdump || true
  command -v link_wps || true
  command -v link_emreal || true
  echo
  echo "Try WPS helper links:"
  link_wps 2>&1 || true
  ls -la | head -n 40
  unlink_wps 2>&1 || true
  echo
  echo "Try WRF helper links:"
  link_emreal 2>&1 || true
  ls -la | head -n 40
  unlink_emreal 2>&1 || true
  echo
  echo "WRF/WPS smoke test completed."
} | tee wrf_wps_smoke.log
SLURM

cat > "${LABROOT}/jobs/bio_toy_bwa_samtools.slurm" <<SLURM
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -A ${LANTA_ACCOUNT}
#SBATCH -J bio_toy
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -o ${LABROOT}/logs/%x-%j.out

set -euo pipefail

module purge
module use /lustrefs/disk/modules/easybuild/modules/bio 2>/dev/null || true
module load BWA 2>/dev/null || module load bwa 2>/dev/null || true
module load SAMtools 2>/dev/null || module load samtools 2>/dev/null || true
module load FastQC 2>/dev/null || module load fastqc 2>/dev/null || true
module list

cd "${LABROOT}/bio"
RUNDIR="BIO_TOY_\${SLURM_JOB_ID}"
mkdir -p "\${RUNDIR}"
cd "\${RUNDIR}"

cat > make_toy_data.py <<'PYDATA'
from pathlib import Path
ref = ("ACGT" * 300)[:1200]
Path("ref.fa").write_text(">toy_ref\n" + "\n".join(ref[i:i+80] for i in range(0, len(ref), 80)) + "\n")
with open("reads.fq", "w") as f:
    for i in range(500):
        start = (i * 7) % (len(ref) - 75)
        seq = ref[start:start+75]
        qual = "I" * len(seq)
        f.write(f"@read_{i}\n{seq}\n+\n{qual}\n")
PYDATA

python3 make_toy_data.py

if ! command -v bwa >/dev/null 2>&1; then
  echo "WARNING: bwa was not found after module load."
  echo "Run this on login node to inspect available modules:"
  echo "module use /lustrefs/disk/modules/easybuild/modules/bio"
  echo "module avail BWA"
  exit 0
fi

if ! command -v samtools >/dev/null 2>&1; then
  echo "WARNING: samtools was not found after module load."
  echo "Run this on login node to inspect available modules:"
  echo "module use /lustrefs/disk/modules/easybuild/modules/bio"
  echo "module avail SAMtools"
  exit 0
fi

{
  echo "===== Bioinformatics toy pipeline ====="
  echo "Host: \$(hostname)"
  echo "Threads: \${SLURM_CPUS_PER_TASK}"
  echo "bwa: \$(command -v bwa)"
  echo "samtools: \$(command -v samtools)"
  command -v fastqc >/dev/null 2>&1 && echo "fastqc: \$(command -v fastqc)" || echo "fastqc: not available"
  echo
  echo "Index reference"
  bwa index ref.fa
  echo
  echo "Align reads"
  bwa mem -t "\${SLURM_CPUS_PER_TASK}" ref.fa reads.fq > aln.sam
  echo
  echo "Convert/sort/index"
  samtools view -@ "\${SLURM_CPUS_PER_TASK}" -bS aln.sam > aln.bam
  samtools sort -@ "\${SLURM_CPUS_PER_TASK}" aln.bam -o aln.sorted.bam
  samtools index aln.sorted.bam
  echo
  echo "Flagstat"
  samtools flagstat aln.sorted.bam | tee flagstat.txt
  echo
  if command -v fastqc >/dev/null 2>&1; then
    echo "FastQC"
    fastqc -t "\${SLURM_CPUS_PER_TASK}" reads.fq -o . || true
  fi
  echo
  echo "Files produced:"
  ls -lh
} | tee bio_toy.log
SLURM

cat > "${LABROOT}/jobs/README_jobs.txt" <<EOF
Generated jobs:
1. gromacs_cpu_1node.slurm      CPU GROMACS short run
2. gromacs_gpu_1card.slurm      GPU GROMACS short run
3. qe_cpu_1node.slurm           CPU Quantum ESPRESSO short run
4. qe_gpu_1card.slurm           GPU Quantum ESPRESSO short run
5. wrf_wps_smoke.slurm          WRF/WPS environment smoke test
6. bio_toy_bwa_samtools.slurm   Toy BWA/SAMtools pipeline
EOF

chmod +x "${LABROOT}/jobs"/*.slurm
ls -lh "${LABROOT}/jobs"
BASH

bash 02_make_science_jobs.sh
```

---

## 5) ส่งงานชุดเบาเข้า queue

บล็อกนี้ส่งงานที่เหมาะกับอบรมเวลาสั้นก่อน ได้แก่ WRF smoke test, Bioinformatics toy, GROMACS GPU 1 card, QE CPU 1 node

ถ้า partition `*-devel` ใช้ไม่ได้ script จะ fallback ไป partition ปกติ

```bash
cat > 03_submit_light_jobs.sh <<'BASH'
#!/bin/bash
set -euo pipefail
source ${HOME}/lanta_science_lab.env
cd "${LABROOT}"

submit_try() {
  local preferred="$1"
  local fallback="$2"
  local script="$3"

  echo
  echo "===== submit ${script} ====="
  if sinfo -h -p "${preferred}" >/dev/null 2>&1; then
    sbatch -p "${preferred}" "${script}" || sbatch -p "${fallback}" "${script}"
  else
    sbatch -p "${fallback}" "${script}"
  fi
}

submit_try compute-devel compute "${LABROOT}/jobs/wrf_wps_smoke.slurm"
submit_try compute-devel compute "${LABROOT}/jobs/bio_toy_bwa_samtools.slurm"
submit_try gpu-devel gpu "${LABROOT}/jobs/gromacs_gpu_1card.slurm"
submit_try compute-devel compute "${LABROOT}/jobs/qe_cpu_1node.slurm"

echo
echo "===== current queue ====="
myqueue 2>/dev/null || squeue -u "${USER}"

echo
echo "Logs will appear in: ${LABROOT}/logs"
BASH

bash 03_submit_light_jobs.sh
```

---

## 6) ดู queue และดู log

ใช้บล็อกนี้ดูสถานะงาน

```bash
source ${HOME}/lanta_science_lab.env
myqueue 2>/dev/null || squeue -u ${USER}
ls -ltr ${LABROOT}/logs
```

เมื่อเห็นว่างานจบแล้ว ให้ดู log ล่าสุด

```bash
source ${HOME}/lanta_science_lab.env
cd ${LABROOT}
for f in logs/*.out; do
  echo
  echo "==================== $f ===================="
  tail -n 60 "$f"
done
```

---

## 7) สรุปผลอัตโนมัติจาก log

หลังงานส่วนใหญ่จบแล้ว คัดลอกบล็อกนี้

```bash
cat > 04_collect_results.sh <<'BASH'
#!/bin/bash
set +e
source ${HOME}/lanta_science_lab.env
cd "${LABROOT}"

OUT="${LABROOT}/results/summary_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "===== LANTA Scientific Workload Lab Summary ====="
  echo "Date: $(date)"
  echo "User: ${USER}"
  echo "LABROOT: ${LABROOT}"
  echo "SCRATCH_ROOT: ${SCRATCH_ROOT}"
  echo

  echo "===== Slurm accounting today ====="
  sacct -u "${USER}" --format=JobID,JobName,Partition,State,Elapsed,AllocTRES -S today 2>/dev/null || true
  echo

  echo "===== GROMACS summary ====="
  grep -R "Performance:\|Finished mdrun\|compatible GPU\|GPU selected" "${LABROOT}/logs" "${LABROOT}/gromacs" 2>/dev/null || true
  echo

  echo "===== QE summary ====="
  grep -R "JOB DONE\|GPU acceleration\|GPU-aware MPI\|Parallel version\|MPI processes\|Threads/MPI" "${LABROOT}/logs" "${LABROOT}/qe" 2>/dev/null || true
  echo

  echo "===== WRF/WPS smoke summary ====="
  grep -R "WRF/WPS smoke test\|wrf.exe\|real.exe\|geogrid.exe\|metgrid.exe\|completed" "${LABROOT}/logs" "${LABROOT}/wrf" 2>/dev/null || true
  echo

  echo "===== Bioinformatics summary ====="
  grep -R "Bioinformatics toy pipeline\|mapped\|primary\|FastQC\|Files produced" "${LABROOT}/logs" "${LABROOT}/bio" 2>/dev/null || true
  echo

  echo "===== Generated files ====="
  find "${LABROOT}" -maxdepth 4 -type f | sed "s#${LABROOT}/##" | sort | head -n 200
} | tee "${OUT}"

echo
echo "Summary saved to: ${OUT}"
BASH

bash 04_collect_results.sh
```

---

## 8) Optional: ส่งงานที่เหลือ

ถ้ามีเวลาเพิ่ม ให้ลองงาน CPU/GPU เปรียบเทียบเพิ่ม

```bash
source ${HOME}/lanta_science_lab.env
cd ${LABROOT}

sbatch -p compute-devel jobs/gromacs_cpu_1node.slurm || sbatch -p compute jobs/gromacs_cpu_1node.slurm
sbatch -p gpu-devel jobs/qe_gpu_1card.slurm || sbatch -p gpu jobs/qe_gpu_1card.slurm

myqueue 2>/dev/null || squeue -u ${USER}
```

---

## 9) Optional: สร้างตารางเปรียบเทียบแบบง่าย

บล็อกนี้สร้าง CSV จาก log เท่าที่ parse ได้ เพื่อเอาไปใช้ทำกราฟต่อในเครื่องตัวเองหรือ Colab

```bash
cat > 05_make_simple_csv.py <<'PYCSV'
from pathlib import Path
import re
import csv
import os

labroot = Path(os.environ.get("LABROOT", "."))
rows = []

for path in list((labroot / "logs").glob("*.out")) + list(labroot.glob("**/*.log")) + list(labroot.glob("**/*.out")):
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        continue

    job = path.name
    if "gmx" in job.lower() or "gromacs" in txt.lower():
        m = re.search(r"Performance:\s+([0-9.]+)", txt)
        if m:
            rows.append({"workload":"GROMACS", "metric":"ns_per_day", "value":float(m.group(1)), "source":str(path)})

    if "qe" in job.lower() or "Program PWSCF" in txt:
        done = "JOB DONE" in txt
        gpu = "GPU acceleration is ACTIVE" in txt
        rows.append({"workload":"QE_GPU" if gpu else "QE_CPU", "metric":"job_done", "value":1 if done else 0, "source":str(path)})

    if "flagstat" in txt or "Bioinformatics toy pipeline" in txt:
        m = re.search(r"(\d+) \+ \d+ mapped", txt)
        if m:
            rows.append({"workload":"Bioinformatics", "metric":"mapped_reads", "value":int(m.group(1)), "source":str(path)})

out = labroot / "results" / "science_workload_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["workload", "metric", "value", "source"])
    writer.writeheader()
    writer.writerows(rows)

print(out)
PYCSV

source ${HOME}/lanta_science_lab.env
python3 05_make_simple_csv.py
cat ${LABROOT}/results/science_workload_summary.csv
```

---

## 10) วิธี download log กลับเครื่องตัวเอง

บน LANTA ให้รันคำสั่งนี้เพื่อพิมพ์คำสั่ง `scp` ที่ควรใช้บนเครื่องตัวเอง

```bash
source ${HOME}/lanta_science_lab.env
echo "Run this command on your local machine, not on LANTA:"
echo "scp -r ${USER}@transfer.lanta.nstda.or.th:${LABROOT}/logs ./lanta_science_logs"
echo "scp -r ${USER}@transfer.lanta.nstda.or.th:${LABROOT}/results ./lanta_science_results"
```

---

## 11) สิ่งที่ผู้เรียนควรสังเกต

### GROMACS

ดูใน log ว่ามีข้อความเหล่านี้หรือไม่

```bash
grep -R "Performance:\|compatible GPU\|GPU selected\|Finished mdrun" ${LABROOT}/logs ${LABROOT}/gromacs 2>/dev/null
```

คำถามชวนคิด:

1. ทำไมงาน MD มักเหมาะกับ GPU?
2. ถ้าใช้ GPU หลายใบแล้วเร็วขึ้นไม่มาก ควรเพิ่ม GPU หรือรันหลาย simulation แยกกัน?
3. `Performance: ... ns/day` แปลว่าอะไร?

### Quantum ESPRESSO

ดูใน log ว่ามีข้อความเหล่านี้หรือไม่

```bash
grep -R "JOB DONE\|GPU acceleration\|MPI processes\|Threads/MPI" ${LABROOT}/logs ${LABROOT}/qe 2>/dev/null
```

คำถามชวนคิด:

1. ทำไม QE ต้องแยก module `-cpu` และ `-gpu`?
2. ทำไมต้องตั้ง `ESPRESSO_TMPDIR` ไปที่ scratch?
3. เมื่อไรควรใช้ `--ntasks-per-node` สูง และเมื่อไรควรใช้ OpenMP เพิ่ม?

### WRF/WPS

ดูใน log ว่าพบ executable และ helper scripts หรือไม่

```bash
grep -R "wrf.exe\|real.exe\|geogrid.exe\|metgrid.exe\|link_wps\|link_emreal" ${LABROOT}/logs ${LABROOT}/wrf 2>/dev/null
```

คำถามชวนคิด:

1. WRF workflow มี preprocessing หลายขั้น ทำไมไม่ใช่แค่ `wrf.exe` อย่างเดียว?
2. ทำไม WRF ต้องสนใจ domain decomposition, halo exchange และจำนวน MPI tasks?
3. ทำไม hybrid MPI+OpenMP ช่วยลด communication ได้ในบางกรณี?

### Bioinformatics

ดูผล `flagstat`

```bash
grep -R "mapped\|properly paired\|Bioinformatics toy" ${LABROOT}/logs ${LABROOT}/bio 2>/dev/null
```

คำถามชวนคิด:

1. งาน bioinformatics แบบ mapping ใช้ CPU threads อย่างไร?
2. ทำไมบาง pipeline ใช้ module, บาง pipeline ใช้ conda/mamba, และบาง pipeline ใช้ Apptainer?
3. ถ้าใช้ CPU เพิ่มแล้วไม่เร็วขึ้นต่อ ควรทำอย่างไร?

---

## 12) Troubleshooting แบบเร็ว

### 12.1 Job รอนาน

```bash
myqueue 2>/dev/null || squeue -u ${USER}
sinfo -s
```

ลองใช้ `*-devel` สำหรับงานสั้น หรือรอ queue ปกติ

### 12.2 Account ผิด

เปิดดู job script

```bash
grep "#SBATCH -A" ${LABROOT}/jobs/*.slurm
```

ถ้าต้องเปลี่ยน account ให้เริ่มใหม่แบบนี้

```bash
export LANTA_ACCOUNT=your_project_account
bash 00_setup_lanta_science_lab.sh
source ${HOME}/lanta_science_lab.env
bash 02_make_science_jobs.sh
```

### 12.3 หา module ไม่เจอ

```bash
module avail GROMACS
module avail QuantumESPRESSO
module avail WRF
module avail WPS
module use /lustrefs/disk/modules/easybuild/modules/bio
module avail BWA
module avail SAMtools
```

### 12.4 หา input ตัวอย่างไม่เจอ

```bash
ls -la /project/common
ls -la /project/common/GROMACS 2>/dev/null || true
ls -la /project/common/QuantumEspresso 2>/dev/null || true
```

ถ้าไม่มี input ใน `/project/common` ให้ผู้สอนเตรียม input ไว้ใน project directory ก่อนอบรม

### 12.5 ไม่ควรรันงานหนักบน login node

หลักจำง่าย:

- login node: เตรียม script, จัดไฟล์, compile เบา ๆ, สำรวจ module
- compute/gpu/memory node: รันงานหนักผ่าน `sbatch`
- transfer node: โอนไฟล์ขนาดใหญ่

---

## 13) Cleanup หลังอบรม

ลบเฉพาะ temporary scratch ของแล็บนี้

```bash
source ${HOME}/lanta_science_lab.env
rm -rf ${SCRATCH_ROOT}
echo "Removed ${SCRATCH_ROOT}"
```

ถ้าต้องการลบแล็บทั้งหมดใน project/home ด้วย

```bash
source ${HOME}/lanta_science_lab.env
echo "About to remove: ${LABROOT}"
rm -rf ${LABROOT}
```

---

## 14) Instructor notes

แนะนำลำดับสำหรับคลาส 45-90 นาที:

1. 5 นาที: อธิบาย LANTA node types และ Slurm
2. 5 นาที: รัน section 2-3 เพื่อสำรวจระบบ
3. 5 นาที: สร้าง job scripts ด้วย section 4
4. 10-30 นาที: ส่งงานชุดเบาด้วย section 5
5. 10 นาที: ดู log และ collect results ด้วย section 6-7
6. 10-20 นาที: อภิปราย scientific workload แต่ละชนิด

ถ้า queue GPU ยาว ให้เน้น `bio_toy`, `wrf_smoke`, `qe_cpu` ก่อน แล้วเปิด log ตัวอย่างของผู้สอนสำหรับ GPU

---

## 15) Quick command card

```bash
source ${HOME}/lanta_science_lab.env
cd ${LABROOT}

bash 01_browse_environment.sh
bash 02_make_science_jobs.sh
bash 03_submit_light_jobs.sh
myqueue 2>/dev/null || squeue -u ${USER}
bash 04_collect_results.sh
```
