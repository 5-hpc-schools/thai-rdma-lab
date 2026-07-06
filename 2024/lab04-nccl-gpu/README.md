# Lab 04: NCCL AllReduce บน GPU

## วัตถุประสงค์

เรียนรู้การใช้ NVIDIA Collective Communications Library (NCCL) เพื่อทำ collective operations
ระหว่าง GPU หลายตัว โดย NCCL สามารถใช้ประโยชน์จาก GPU-Direct RDMA เพื่อส่งข้อมูล
ระหว่าง GPU โดยตรงโดยไม่ต้องผ่าน CPU

## แนวคิดที่เรียนรู้

- การสร้าง NCCL communicator ด้วย `ncclGetUniqueId` และ `ncclCommInitRank`
- การจัดสรร buffer บน GPU ด้วย `cudaMalloc`
- การเรียก `ncclAllReduce` พร้อม CUDA stream
- การ synchronize stream และตรวจสอบผลลัพธ์

## ข้อกำหนดเบื้องต้น

- NVIDIA GPU ที่รองรับ CUDA
- CUDA Toolkit (>= 11.0)
- NCCL (>= 2.x)
- MPI implementation (OpenMPI หรือ MPICH)

## วิธีคอมไพล์

```bash
make
```

## วิธีรัน

```bash
# รันด้วย GPU 2 ตัว
mpirun -np 2 ./allreduce_nccl

# รันด้วย GPU 4 ตัว
mpirun -np 4 ./allreduce_nccl
```

## ผลลัพธ์ที่คาดหวัง

แต่ละ rank จะรายงานว่า AllReduce สำเร็จ พร้อมแสดงค่าผลลัพธ์ที่เท่ากับ
ผลรวมของ (rank+1) จากทุก rank เช่น ถ้ามี 4 GPU ค่าผลลัพธ์ = 1+2+3+4 = 10

## แบบฝึกหัดเพิ่มเติม

1. ลองเปลี่ยน reduction operation จาก `ncclSum` เป็น `ncclProd` หรือ `ncclMax`
2. ลองเพิ่มขนาดข้อมูลและวัดเวลาที่ใช้
3. ลองใช้ `ncclGroupStart`/`ncclGroupEnd` สำหรับ multiple operations
