#!/bin/bash
# =============================================================
# run_benchmark.sh -- รัน OSU Micro-Benchmarks สำหรับ RDMA
# =============================================================
# สคริปต์นี้ค้นหาและรัน OSU micro-benchmarks เพื่อวัดประสิทธิภาพ
# ของ MPI collective operations ต่างๆ
#
# วิธีใช้: ./run_benchmark.sh [จำนวน process] [ตำแหน่ง OSU]
# =============================================================

# กำหนดค่าเริ่มต้น
NP=${1:-4}                                  # จำนวน process (default: 4)
OSU_DIR=${2:-""}                            # ตำแหน่ง OSU benchmarks

# ค้นหาตำแหน่ง OSU micro-benchmarks
find_osu() {
    # ลองค้นหาในตำแหน่งที่พบบ่อย
    local search_paths=(
        "/usr/local/libexec/osu-micro-benchmarks/mpi/collective"
        "/usr/lib/osu-micro-benchmarks/mpi/collective"
        "/opt/osu-micro-benchmarks/mpi/collective"
        "$HOME/osu-micro-benchmarks/mpi/collective"
    )

    for path in "${search_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    # ลองใช้ find ค้นหา (จำกัดขอบเขตเพื่อไม่ให้ช้า)
    local found
    found=$(find /usr /opt "$HOME" -maxdepth 5 -name "osu_allreduce" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "$(dirname "$found")"
        return 0
    fi

    return 1
}

# ตรวจสอบว่า mpirun พร้อมใช้งาน
if ! command -v mpirun &> /dev/null; then
    echo "ไม่พบ mpirun -- กรุณาติดตั้ง MPI implementation"
    exit 1
fi

# ค้นหาตำแหน่ง OSU benchmarks
if [ -z "$OSU_DIR" ]; then
    echo "กำลังค้นหา OSU Micro-Benchmarks..."
    OSU_DIR=$(find_osu)
    if [ $? -ne 0 ] || [ -z "$OSU_DIR" ]; then
        echo "ไม่พบ OSU Micro-Benchmarks"
        echo ""
        echo "วิธีติดตั้ง:"
        echo "  git clone https://github.com/ULHPC/osu-micro-benchmarks.git"
        echo "  cd osu-micro-benchmarks"
        echo "  ./configure CC=mpicc CXX=mpicxx --prefix=\$HOME/osu-micro-benchmarks"
        echo "  make -j\$(nproc) && make install"
        echo ""
        echo "แล้วรันอีกครั้ง:"
        echo "  ./run_benchmark.sh $NP \$HOME/osu-micro-benchmarks/libexec/osu-micro-benchmarks/mpi/collective"
        exit 1
    fi
fi

echo "พบ OSU benchmarks ที่: $OSU_DIR"
echo "จำนวน process: $NP"
echo ""

# รายการ benchmark ที่ต้องการรัน
# แต่ละตัวทดสอบ collective operation ที่ต่างกัน
BENCHMARKS=(
    "osu_allreduce"       # AllReduce -- รวมค่าจากทุก rank
    "osu_allgather"       # Allgather -- รวบรวมข้อมูลจากทุก rank
    "osu_bcast"           # Broadcast -- กระจายข้อมูลจาก root
    "osu_reduce"          # Reduce -- รวมค่าไปยัง root
    "osu_barrier"         # Barrier -- synchronization
)

# พารามิเตอร์สำหรับ benchmark
# -m: ขนาดข้อมูลสูงสุด (bytes)
# -i: จำนวนรอบ iteration
MAX_SIZE=4194304    # 4 MB
ITERATIONS=100

echo "============================================"
echo " OSU Micro-Benchmarks"
echo " จำนวน process: $NP"
echo " ขนาดข้อมูลสูงสุด: $MAX_SIZE bytes"
echo " จำนวน iterations: $ITERATIONS"
echo "============================================"
echo ""

# รันแต่ละ benchmark
for bench in "${BENCHMARKS[@]}"; do
    BENCH_PATH="$OSU_DIR/$bench"

    if [ ! -f "$BENCH_PATH" ]; then
        echo "--- $bench: ไม่พบไฟล์ (ข้าม) ---"
        echo ""
        continue
    fi

    echo "--- กำลังรัน $bench ---"

    # รัน benchmark ด้วย mpirun
    # --allow-run-as-root สำหรับ container environments
    mpirun --allow-run-as-root -np $NP "$BENCH_PATH" \
        -m "$MAX_SIZE" -i "$ITERATIONS" 2>&1

    echo ""
done

echo "============================================"
echo " เสร็จสิ้นการทดสอบ OSU Micro-Benchmarks"
echo "============================================"
