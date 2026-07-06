#!/bin/bash
# setup_env.sh — ตั้งค่าสภาพแวดล้อมสำหรับห้องปฏิบัติการอาร์ดีเอ็มเอ
# วิธีใช้: source scripts/setup_env.sh

echo "=== ตั้งค่าสภาพแวดล้อมห้องปฏิบัติการอาร์ดีเอ็มเอ ==="

# ตรวจสอบ HPC-X (หากมี)
if [ -n "$HPCX_DIR" ]; then
    echo "[พบ] HPC-X: $HPCX_DIR"
    source "$HPCX_DIR/hpcx-init.sh"
    hpcx_load
elif [ -d "/opt/hpcx" ]; then
    echo "[พบ] HPC-X: /opt/hpcx"
    source /opt/hpcx/hpcx-init.sh
    hpcx_load
else
    echo "[ไม่พบ] HPC-X — ใช้ไลบรารีระบบแทน"
fi

# ตรวจสอบอุปกรณ์อาร์ดีเอ็มเอ
if command -v ibstat &> /dev/null; then
    echo ""
    echo "=== อุปกรณ์อาร์ดีเอ็มเอที่พบ ==="
    ibstat | grep -E "^CA|Port|State|Rate" | head -10
else
    echo "[เตือน] ไม่พบคำสั่ง ibstat — อาจไม่ได้ติดตั้ง OFED"
fi

# ตรวจสอบคอมไพเลอร์
echo ""
echo "=== เครื่องมือที่พบ ==="
command -v gcc     && gcc --version | head -1     || echo "[ไม่พบ] gcc"
command -v mpicc   && mpicc --version 2>&1 | head -1 || echo "[ไม่พบ] mpicc"
command -v nvcc    && nvcc --version | tail -1     || echo "[ไม่พบ] nvcc (ไม่จำเป็นยกเว้น lab04)"

echo ""
echo "=== พร้อมใช้งาน ==="
