#!/bin/bash
# =============================================================
# benchmark.sh -- สคริปต์เปรียบเทียบ MPI_Allreduce กับ One-Sided
# =============================================================
# สคริปต์นี้รัน onesided_allreduce ด้วยจำนวน process ต่างๆ
# และรวบรวมผลลัพธ์เพื่อเปรียบเทียบประสิทธิภาพ
#
# วิธีใช้: ./benchmark.sh [จำนวน process สูงสุด]
# =============================================================

# กำหนดค่าเริ่มต้น
MAX_NP=${1:-4}          # จำนวน process สูงสุด (default: 4)
BINARY="./onesided_allreduce"

# ตรวจสอบว่าไฟล์ binary มีอยู่
if [ ! -f "$BINARY" ]; then
    echo "ไม่พบไฟล์ $BINARY -- กรุณาคอมไพล์ก่อน (make)"
    exit 1
fi

# ตรวจสอบว่า mpirun พร้อมใช้งาน
if ! command -v mpirun &> /dev/null; then
    echo "ไม่พบ mpirun -- กรุณาติดตั้ง MPI implementation"
    exit 1
fi

echo "============================================"
echo " Benchmark: MPI_Allreduce vs One-Sided"
echo "============================================"
echo ""

# รันทดสอบด้วยจำนวน process ต่างๆ
# เริ่มจาก 2 process (ต้องการอย่างน้อย 2 สำหรับ ring)
NP=2
while [ $NP -le $MAX_NP ]; do
    echo "--- จำนวน process: $NP ---"
    mpirun --allow-run-as-root -np $NP $BINARY 2>&1
    echo ""
    NP=$((NP * 2))
done

echo "============================================"
echo " เสร็จสิ้นการทดสอบ"
echo "============================================"
