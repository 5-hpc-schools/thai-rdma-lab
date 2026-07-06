# Lab 05: UCC AllReduce

## วัตถุประสงค์

เรียนรู้การใช้ Unified Collective Communication (UCC) library สำหรับ collective operations
UCC เป็น framework ที่รองรับหลาย transport backends (UCX, NCCL, SHARP เป็นต้น)
และสามารถทำงานกับทั้ง CPU และ GPU memory

## แนวคิดที่เรียนรู้

- การเริ่มต้น UCC ด้วย `ucc_init` และ `ucc_context_create`
- การใช้ MPI เป็น OOB (out-of-band) channel สำหรับ bootstrap
- การสร้าง team ด้วย `ucc_team_create_post` และ progress loop
- การตั้งค่า collective arguments (`UCC_COLL_TYPE_ALLREDUCE`, `UCC_DT_FLOAT64`, `UCC_OP_SUM`)
- การเรียก `ucc_collective_post` และ progress loop จนกว่าจะเสร็จ

## ข้อกำหนดเบื้องต้น

- UCC library (>= 1.0)
- UCX library (UCC transport backend)
- MPI implementation (OpenMPI หรือ MPICH)

## การติดตั้ง UCC

```bash
# จาก source
git clone https://github.com/openucx/ucc.git
cd ucc
./autogen.sh
./configure --prefix=/usr/local --with-ucx=/usr/local
make -j$(nproc) && sudo make install
```

## วิธีคอมไพล์

```bash
make
# หรือระบุตำแหน่ง UCC
make UCC_HOME=/usr/local
```

## วิธีรัน

```bash
# รันด้วย 4 processes
mpirun -np 4 ./ucc_allreduce

# รันด้วย 8 processes
mpirun -np 8 ./ucc_allreduce
```

## ผลลัพธ์ที่คาดหวัง

ทุก rank จะรายงานว่า AllReduce สำเร็จ โดยค่าผลลัพธ์เท่ากับผลรวมของ (rank+1) จากทุก rank

## แบบฝึกหัดเพิ่มเติม

1. ลองเปลี่ยน datatype เป็น `UCC_DT_INT32` หรือ `UCC_DT_FLOAT32`
2. ลองเปลี่ยน operation เป็น `UCC_OP_MAX` หรือ `UCC_OP_MIN`
3. ลองทำ multiple AllReduce operations ติดต่อกันและวัดเวลา
4. ลองใช้ in-place AllReduce (src buffer เหมือนกับ dst buffer)
