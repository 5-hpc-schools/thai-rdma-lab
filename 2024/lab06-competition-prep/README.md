# Lab 06: One-Sided AllReduce - โจทย์แข่งขัน

## วัตถุประสงค์

ออกแบบและ implement AllReduce แบบ one-sided โดยใช้ UCX remote memory access (RMA)
แล้วเปรียบเทียบประสิทธิภาพกับ MPI_Allreduce มาตรฐาน

นี่คือโจทย์สำหรับการแข่งขัน -- เป้าหมายคือทำให้ one-sided AllReduce เร็วกว่า MPI_Allreduce

## อัลกอริทึม Ring AllReduce

### เฟส 1: Reduce-Scatter
```
ขั้นตอนที่ 0: rank 0 ส่ง chunk 0 -> rank 1, rank 1 ส่ง chunk 1 -> rank 2, ...
ขั้นตอนที่ 1: rank 0 ส่ง chunk 3 -> rank 1, rank 1 ส่ง chunk 0 -> rank 2, ...
...
หลัง (N-1) ขั้นตอน: แต่ละ rank มี partial sum ของ chunk ที่รับผิดชอบ
```

### เฟส 2: Allgather
```
แต่ละ rank กระจาย chunk ที่มี partial sum ไปยังทุก rank อื่น
หลัง (N-1) ขั้นตอน: ทุก rank มีผลลัพธ์ AllReduce ครบถ้วน
```

## เทคนิค UCX ที่ใช้

- `ucp_put_nbx` -- เขียนข้อมูลไปยัง remote memory โดยตรง (zero-copy)
- `ucp_atomic_op_nbx` -- atomic ADD บน remote memory เพื่อแจ้งเตือน
- `ucp_mem_map` -- ลงทะเบียนหน่วยความจำสำหรับ RDMA access
- `ucp_rkey_pack/unpack` -- แลกเปลี่ยน remote memory keys

## ข้อกำหนดเบื้องต้น

- UCX library (>= 1.12)
- MPI implementation
- InfiniBand หรือ RoCE adapter (สำหรับประสิทธิภาพสูงสุด)

## วิธีคอมไพล์

```bash
make
# หรือระบุตำแหน่ง UCX
make UCX_HOME=/usr/local
```

## วิธีรัน

```bash
# รัน benchmark
mpirun -np 4 ./onesided_allreduce

# รัน benchmark script (เปรียบเทียบ MPI vs one-sided)
./benchmark.sh
```

## เกณฑ์การให้คะแนน

1. **ความถูกต้อง** (40%) -- ผลลัพธ์ต้องตรงกับ MPI_Allreduce
2. **ประสิทธิภาพ** (40%) -- เวลาที่ใช้เทียบกับ MPI_Allreduce
3. **คุณภาพโค้ด** (20%) -- ความเข้าใจอัลกอริทึมและการจัดการ resource

## แบบฝึกหัดเพิ่มเติม

1. ลอง implement แบบ recursive halving แทน ring
2. ลองใช้ pipelining เพื่อ overlap computation กับ communication
3. ลองรองรับ GPU memory ด้วย GPU-Direct RDMA
4. ลองปรับขนาด chunk ให้เหมาะกับ network bandwidth
