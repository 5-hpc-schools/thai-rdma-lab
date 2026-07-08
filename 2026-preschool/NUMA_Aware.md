# คู่มือ NUMA-Aware MPI บน LANTA แบบ Copy-Paste Run

**Account/Project ที่ใช้ในคู่มือนี้:** `tn999992`

คู่มือนี้พาผู้เรียนทดลองจริงบน **LANTA HPC** โดยสร้าง MPI benchmark ขนาดเล็กขึ้นมาเอง แล้วเปรียบเทียบการรัน 3 แบบ:

1. ไม่ผูกคอร์และไม่ผูกหน่วยความจำ
2. ผูก MPI rank กับ CPU core
3. ผูก MPI rank กับ CPU core และผูก memory ให้ local กับ NUMA domain

เป้าหมายของบทเรียนนี้ไม่ใช่การจำ option แต่คือการ “มองเห็น” ว่า MPI rank ของเราถูกวางบน CPU core / NUMA domain อย่างไร และการ binding ส่งผลต่อ memory-bound workload อย่างไร

---

## 0. สิ่งที่ควรรู้ก่อนเริ่ม

LANTA เป็นระบบ **HPE Cray EX** ของ ThaiSC โดย compute node ฝั่ง CPU ใช้ **AMD EPYC 7713** ซึ่ง 1 compute node มีโครงสร้างหลักคือ

- 2 CPU sockets ต่อ node
- 64 cores ต่อ socket
- รวม 128 CPU cores ต่อ node
- hardware thread per core = 1

ดังนั้นงาน pure MPI แบบเต็ม node มักเริ่มจากการใช้

```bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
```

ระบบ LANTA ใช้ **Slurm** เป็นระบบจัดคิวงาน ดังนั้นผู้ใช้ควรส่งงานด้วย `sbatch` ไม่ควรรัน job script โดยตรงด้วย `bash script.sbatch` หรือ `./script.sbatch` เพราะจะทำให้ script ไปรันบน login node แทนที่จะเข้า queue ของ Slurm

---

## 1. สร้างโฟลเดอร์ทดลอง

ให้ login เข้า LANTA แล้วคัดลอกคำสั่งนี้ไปรัน

```bash
mkdir -p ~/numa_mpi_tutorial
cd ~/numa_mpi_tutorial
pwd
```

---

## 2. สร้างโค้ด MPI benchmark

โค้ดนี้ทำ 2 อย่าง

1. แสดงว่าแต่ละ MPI rank รันอยู่บน CPU ไหน และ CPU นั้นอยู่ใน NUMA node ใด
2. รัน memory triad benchmark แบบง่าย ๆ เพื่อวัด aggregate memory bandwidth โดยประมาณ

คัดลอกทั้งหมดนี้ไปรัน

```bash
cat > numa_mpi_bench.c <<'EOF_C'
#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <unistd.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static int cpu_to_numa_node(int cpu) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);

    DIR *dir = opendir(path);
    if (!dir) return -1;

    struct dirent *entry;
    int node = -1;

    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "node", 4) == 0) {
            node = atoi(entry->d_name + 4);
            break;
        }
    }

    closedir(dir);
    return node;
}

static int affinity_summary(char *buf, size_t buflen) {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        snprintf(buf, buflen, "sched_getaffinity_error=%s", strerror(errno));
        return -1;
    }

    int count = 0;
    size_t used = 0;

    used += snprintf(buf + used, buflen - used, "allowed_cpus=");

    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            count++;

            if (count <= 24 && used < buflen) {
                used += snprintf(buf + used, buflen - used, "%s%d",
                                 count == 1 ? "" : ",", i);
            }
        }
    }

    if (count > 24 && used < buflen) {
        used += snprintf(buf + used, buflen - used, ",...");
    }

    if (used < buflen) {
        snprintf(buf + used, buflen - used, " count=%d", count);
    }

    return count;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size, local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    int mb_per_array = 32;
    int iters = 20;

    if (argc >= 2) mb_per_array = atoi(argv[1]);
    if (argc >= 3) iters = atoi(argv[2]);

    if (mb_per_array <= 0) mb_per_array = 32;
    if (iters <= 0) iters = 20;

    int cpu = sched_getcpu();
    int numa = cpu_to_numa_node(cpu);

    char affinity[512];
    int allowed_count = affinity_summary(affinity, sizeof(affinity));

    int show_ranks = 16;
    char *env_show = getenv("SHOW_RANKS");
    if (env_show) show_ranks = atoi(env_show);

    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (r == rank && (rank < show_ranks || rank == size - 1)) {
            printf("RANKMAP rank=%d local_rank=%d cpu=%d numa=%d allowed_count=%d %s\n",
                   rank, local_rank, cpu, numa, allowed_count, affinity);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    size_t n = ((size_t)mb_per_array * 1024UL * 1024UL) / sizeof(double);

    double *a = NULL;
    double *b = NULL;
    double *c = NULL;

    if (posix_memalign((void **)&a, 64, n * sizeof(double)) ||
        posix_memalign((void **)&b, 64, n * sizeof(double)) ||
        posix_memalign((void **)&c, 64, n * sizeof(double))) {
        fprintf(stderr, "rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* First-touch initialization: memory pages are touched by the rank that will use them. */
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    for (int it = 0; it < iters; it++) {
        double scalar = 1.000001 + (double)it * 0.000001;
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }

    double t1 = MPI_Wtime();

    double local_time = t1 - t0;
    double max_time = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_checksum = 0.0;
    size_t stride = n / 16 + 1;
    for (size_t i = 0; i < n; i += stride) {
        local_checksum += a[i];
    }

    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double bytes_per_rank = 3.0 * (double)n * sizeof(double) * (double)iters;
        double total_bytes = bytes_per_rank * (double)size;
        double gbps = total_bytes / max_time / 1.0e9;

        printf("RESULT ranks=%d mb_per_array=%d arrays=3 iters=%d max_time_sec=%.6f aggregate_triad_GBps=%.3f checksum=%.3f\n",
               size, mb_per_array, iters, max_time, gbps, global_checksum);
        fflush(stdout);
    }

    free(a);
    free(b);
    free(c);

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
EOF_C
```

---

## 3. Compile ด้วย Cray compiler wrapper

บน LANTA ควรใช้ compiler wrapper ของ HPE Cray ได้แก่

- `cc` สำหรับ C
- `CC` สำหรับ C++
- `ftn` สำหรับ Fortran

ให้ compile ด้วยคำสั่งนี้

```bash
module reset
module list

cc --version
cc -O3 -o numa_mpi_bench.exe numa_mpi_bench.c

ls -lh numa_mpi_bench.exe
```

ถ้า compile ผ่าน จะเห็นไฟล์

```bash
numa_mpi_bench.exe
```

---

## 4. ตรวจสอบ topology ของ compute node

สร้าง job สำหรับดู topology และทดลอง rank mapping แบบเล็ก ๆ ก่อน

```bash
cat > 00_check_topology.sbatch <<'EOF_SBATCH'
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH -J numa_check
#SBATCH -A tn999992
#SBATCH -o 00_check_topology_%j.out

module reset

echo "===== Job information ====="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

echo
echo "===== lscpu summary ====="
lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket|NUMA' || true

echo
echo "===== numactl -H, if available ====="
if command -v numactl >/dev/null 2>&1; then
    numactl -H
else
    echo "numactl command not found; continue without numactl"
fi

echo
echo "===== MPI rank map with CPU binding ====="
export SHOW_RANKS=8
srun --cpu-bind=verbose,cores --mem-bind=verbose,local ./numa_mpi_bench.exe 1 1
EOF_SBATCH
```

ส่งงาน

```bash
sbatch 00_check_topology.sbatch
```

เช็คคิว

```bash
myqueue
```

ดู output โดยแทน `JOBID` ด้วยเลข job จริง

```bash
ls -lh 00_check_topology_*.out
cat 00_check_topology_JOBID.out
```

สิ่งที่ควรดูคือบรรทัดลักษณะนี้

```text
RANKMAP rank=0 local_rank=0 cpu=... numa=... allowed_count=1 allowed_cpus=...
RANKMAP rank=1 local_rank=1 cpu=... numa=... allowed_count=1 allowed_cpus=...
```

ถ้าใช้ `--cpu-bind=cores` แล้วเห็น `allowed_count=1` แปลว่า MPI rank ถูกผูกกับ core เดี่ยวชัดเจน

---

## 5. Run benchmark เปรียบเทียบ 3 แบบ

จุดสำคัญ: อย่าใช้ `srun ./app` เป็น baseline เสมอไป เพราะ Slurm อาจ auto-bind ให้เองตาม configuration ได้ ถ้าต้องการปิด binding จริง ๆ ให้ระบุ `--cpu-bind=none` ชัดเจน

สร้าง job เปรียบเทียบ

```bash
cat > 01_compare_binding.sbatch <<'EOF_SBATCH'
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -J numa_compare
#SBATCH -A tn999992
#SBATCH -o 01_compare_binding_%j.out

module reset

export SHOW_RANKS=16

MB_PER_ARRAY=32
ITERS=30

echo "===== Job information ====="
echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "MB_PER_ARRAY=${MB_PER_ARRAY}"
echo "ITERS=${ITERS}"

echo
echo "===== CASE A: force no CPU binding and no memory binding ====="
srun --cpu-bind=none --mem-bind=none ./numa_mpi_bench.exe ${MB_PER_ARRAY} ${ITERS}

echo
echo "===== CASE B: bind MPI ranks to CPU cores only ====="
srun --cpu-bind=verbose,cores --mem-bind=none ./numa_mpi_bench.exe ${MB_PER_ARRAY} ${ITERS}

echo
echo "===== CASE C: bind MPI ranks to CPU cores and local NUMA memory ====="
srun --cpu-bind=verbose,cores --mem-bind=verbose,local ./numa_mpi_bench.exe ${MB_PER_ARRAY} ${ITERS}
EOF_SBATCH
```

ส่งงาน

```bash
sbatch 01_compare_binding.sbatch
```

ดูสถานะ

```bash
myqueue
```

เมื่อ job เสร็จแล้ว ดูผลสรุป

```bash
ls -lh 01_compare_binding_*.out
grep -E "===== CASE|^RESULT" 01_compare_binding_*.out
```

ตัวอย่างรูปแบบผลลัพธ์ที่คาดว่าจะเห็น

```text
===== CASE A: force no CPU binding and no memory binding =====
RESULT ranks=128 mb_per_array=32 arrays=3 iters=30 max_time_sec=... aggregate_triad_GBps=...

===== CASE B: bind MPI ranks to CPU cores only =====
RESULT ranks=128 mb_per_array=32 arrays=3 iters=30 max_time_sec=... aggregate_triad_GBps=...

===== CASE C: bind MPI ranks to CPU cores and local NUMA memory =====
RESULT ranks=128 mb_per_array=32 arrays=3 iters=30 max_time_sec=... aggregate_triad_GBps=...
```

วิธีอ่านผล

- `max_time_sec` ยิ่งต่ำยิ่งดี
- `aggregate_triad_GBps` ยิ่งสูงยิ่งดี
- ถ้า Case B หรือ C ดีกว่า Case A แปลว่า binding ช่วยให้ rank ทำงานนิ่งขึ้น
- ถ้า Case C ดีกว่า Case B แปลว่า memory locality มีผลกับ workload นี้

---

## 6. คำนวณ speedup แบบเร็ว ๆ

หลังจากได้ output แล้ว ให้คัดตัวเลข `max_time_sec` ของแต่ละ case มาใส่เอง เช่น

```bash
python3 - <<'EOF_PY'
T_A = 10.0   # replace with Case A max_time_sec
T_B = 8.5    # replace with Case B max_time_sec
T_C = 7.9    # replace with Case C max_time_sec

print(f"Speedup B over A = {T_A/T_B:.2f}x")
print(f"Speedup C over A = {T_A/T_C:.2f}x")
print(f"Speedup C over B = {T_B/T_C:.2f}x")
EOF_PY
```

สูตรคือ

```text
Speedup = baseline_time / optimized_time
```

---

## 7. ใช้กับ MPI application จริง

หลังจากเข้าใจ benchmark แล้ว เวลารันโปรแกรมจริง เช่น `my_mpi_application.exe` ให้เริ่มจาก template นี้

```bash
cat > run_my_mpi_app.sbatch <<'EOF_SBATCH'
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -J my_mpi_numa
#SBATCH -A tn999992
#SBATCH -o my_mpi_numa_%j.out

module reset

echo "HOSTNAME=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"

srun --cpu-bind=cores --mem-bind=local ./my_mpi_application.exe
EOF_SBATCH
```

ส่งงาน

```bash
sbatch run_my_mpi_app.sbatch
```

สำหรับการ debug ว่า rank ถูกผูกจริงไหม ให้เปลี่ยนบรรทัด `srun` เป็นแบบ verbose ชั่วคราว

```bash
srun --cpu-bind=verbose,cores --mem-bind=verbose,local ./my_mpi_application.exe
```

เมื่อมั่นใจแล้ว ให้กลับมาใช้แบบไม่ verbose เพื่อไม่ให้ log ยาวเกินจำเป็น

```bash
srun --cpu-bind=cores --mem-bind=local ./my_mpi_application.exe
```

---

## 8. หมายเหตุเรื่อง `--cpu-bind=numa`

ใน Slurm documentation ตัวเลือกสำหรับ NUMA locality domain ใช้คำว่า

```bash
--cpu-bind=ldoms
```

ไม่ใช่ `--cpu-bind=numa` ในเอกสารหลัก โดย `ldoms` หมายถึง bind task เข้ากับ NUMA locality domain

แต่สำหรับ **pure MPI 1 rank ต่อ 1 core** บน LANTA แนวทางที่เข้าใจง่ายและควรเริ่มก่อนคือ

```bash
--cpu-bind=cores --mem-bind=local
```

เพราะแต่ละ rank จะนิ่งอยู่กับ core ของตัวเอง และให้ memory allocation พยายามอยู่ใกล้ CPU ที่ใช้งาน

---

## 9. Hybrid MPI + OpenMP

ถ้าโปรแกรมเป็น hybrid เช่น 16 MPI ranks ต่อ node และแต่ละ rank ใช้ 8 OpenMP threads ให้ใช้ pattern นี้

```bash
cat > run_hybrid_app.sbatch <<'EOF_SBATCH'
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -J hybrid_numa
#SBATCH -A tn999992
#SBATCH -o hybrid_numa_%j.out

module reset

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "OMP_PLACES=${OMP_PLACES}"
echo "OMP_PROC_BIND=${OMP_PROC_BIND}"

srun --cpu-bind=cores --mem-bind=local ./my_hybrid_application.exe
EOF_SBATCH
```

ส่งงาน

```bash
sbatch run_hybrid_app.sbatch
```

สำหรับ hybrid job ควรตั้ง

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

เพื่อให้จำนวน thread และการวาง thread ตรงกับ resource ที่ขอจาก Slurm

---

## 10. Best practices สำหรับผู้ใช้ LANTA

### 10.1 ใช้ `sbatch` เสมอ

อย่ารัน job script ด้วย

```bash
bash script.sbatch
./script.sbatch
```

ให้ใช้

```bash
sbatch script.sbatch
```

### 10.2 ใช้ compiler wrapper ของ Cray

สำหรับ MPI บน LANTA ให้เริ่มจาก

```bash
cc   # C
CC   # C++
ftn  # Fortran
```

แทนการเรียก compiler ตรง ๆ

### 10.3 เริ่มจาก binding ที่อ่านง่ายก่อน

สำหรับ pure MPI ให้เริ่มจาก

```bash
srun --cpu-bind=cores --mem-bind=local ./app.exe
```

### 10.4 Debug ด้วย verbose แต่ไม่จำเป็นต้องเปิดตลอด

ใช้ตอนตรวจสอบ

```bash
srun --cpu-bind=verbose,cores --mem-bind=verbose,local ./app.exe
```

เมื่อมั่นใจแล้วให้กลับไปใช้แบบไม่ verbose

```bash
srun --cpu-bind=cores --mem-bind=local ./app.exe
```

### 10.5 อย่าด่วนสรุปจาก run เดียว

ควรรัน benchmark อย่างน้อย 3 ครั้ง เพราะระบบ shared HPC อาจมี noise จาก

- node ที่ได้รับ
- filesystem
- network
- queue condition
- background system activity

### 10.6 อย่าเชื่อ performance number ที่ไม่ได้วัดบนงานของตัวเอง

NUMA binding มักช่วยมากกับงานที่เป็น

- memory-bound
- communication-heavy
- hybrid MPI + OpenMP
- workload ที่แตะ memory จำนวนมาก

แต่ผลจริงขึ้นกับ

- source code
- dataset size
- MPI rank layout
- OpenMP setting
- memory access pattern
- library ที่ใช้

---

## 11. Template สั้นที่สุดสำหรับ production pure MPI

```bash
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -J my_job
#SBATCH -A tn999992
#SBATCH -o my_job_%j.out

module reset

srun --cpu-bind=cores --mem-bind=local ./my_mpi_application.exe
```

---

## 12. Template สั้นที่สุดสำหรับ production MPI + OpenMP

```bash
#!/bin/bash -l
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -J my_hybrid_job
#SBATCH -A tn999992
#SBATCH -o my_hybrid_job_%j.out

module reset

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

srun --cpu-bind=cores --mem-bind=local ./my_hybrid_application.exe
```

---

## 13. สรุปสำหรับผู้เรียน

NUMA-aware ไม่ใช่การทำให้ CPU เร็วขึ้น แต่คือการทำให้ “งานอยู่ใกล้ข้อมูลของตัวเอง” มากขึ้น

บนเครื่องระดับ LANTA ที่ 1 compute node มี 2 sockets และ 128 cores การปล่อยให้ rank/thread เคลื่อนที่หรือใช้ memory ข้าม locality โดยไม่จำเป็น อาจทำให้ performance ตกได้ โดยเฉพาะงานที่แตะ memory จำนวนมากหรือสื่อสารหนัก

แนวทางที่ควรจำคือ

```bash
# Pure MPI
srun --cpu-bind=cores --mem-bind=local ./app.exe
```

```bash
# Hybrid MPI + OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun --cpu-bind=cores --mem-bind=local ./app.exe
```

และทุกครั้งที่สงสัยว่า Slurm วาง rank ถูกต้องหรือไม่ ให้เปิด verbose

```bash
srun --cpu-bind=verbose,cores --mem-bind=verbose,local ./app.exe
```

---

## 14. คำสั่งรวมสำหรับเตรียม tutorial folder อย่างรวดเร็ว

ถ้าต้องการ copy ไฟล์ Markdown นี้ขึ้น LANTA แล้วแตกเป็นไฟล์ต่าง ๆ ด้วยตนเอง ให้เริ่มจาก

```bash
mkdir -p ~/numa_mpi_tutorial
cd ~/numa_mpi_tutorial
```

จากนั้นค่อย copy code blocks ในคู่มือนี้ทีละส่วนตามลำดับ

---

## References

- ThaiSC LANTA resources: https://thaisc.io/thaisc-resorces/lanta
- ThaiSC LANTA Slurm job submission guide: https://thaisc.atlassian.net/wiki/spaces/LANTA/pages/447021064
- ThaiSC LANTA programming environment and running jobs: https://thaisc.atlassian.net/wiki/x/AQDdFg
- AMD EPYC 7713 product page: https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7713.html
- Slurm `srun` documentation: https://slurm.schedmd.com/srun.html
