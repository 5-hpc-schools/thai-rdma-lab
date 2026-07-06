# Lab 01 — พื้นฐาน Verbs API

## คำอธิบาย

แบบฝึกหัดนี้สาธิตการใช้ libibverbs API เพื่อส่งข้อมูลระหว่างสองกระบวนการผ่าน RDMA
โดยครอบคลุมการทำงานพื้นฐานได้แก่ การเปิดอุปกรณ์ การจัดสรร Protection Domain
การลงทะเบียนหน่วยความจำ การสร้าง Completion Queue และ Queue Pair
รวมถึงการส่งและรับข้อมูลด้วย Send/Receive operation

## ไฟล์ในแบบฝึกหัด

| ไฟล์ | คำอธิบาย |
|------|----------|
| `send_recv.c` | ตัวอย่าง RDMA Send/Receive พื้นฐาน |
| `rdma_write.c` | ตัวอย่าง RDMA Write (เขียนไปยังหน่วยความจำระยะไกลโดยตรง) |
| `rdma_read.c` | ตัวอย่าง RDMA Read (อ่านจากหน่วยความจำระยะไกลโดยตรง) |

## ข้อกำหนดเบื้องต้น

- ติดตั้ง `libibverbs-dev` และ `librdmacm-dev`
- มีอุปกรณ์ RDMA (InfiniBand HCA หรือ RoCE NIC) หรือใช้ SoftRoCE (rxe)
- คอมไพเลอร์ C (gcc)

```bash
# ติดตั้งแพ็กเกจที่จำเป็น (Ubuntu/Debian)
sudo apt install libibverbs-dev librdmacm-dev rdma-core
```

## วิธีคอมไพล์

```bash
make            # คอมไพล์ทุกโปรแกรม
make send_recv  # คอมไพล์เฉพาะ send_recv
make clean      # ลบไฟล์ที่คอมไพล์แล้ว
```

## วิธีรัน

### Send/Receive

เปิดเทอร์มินัลสองหน้าต่าง:

```bash
# เทอร์มินัลที่ 1 — เซิร์ฟเวอร์ (ฝั่งรับ)
./send_recv

# เทอร์มินัลที่ 2 — ไคลเอนต์ (ฝั่งส่ง)
# ใส่ค่า qp_num และ lid ที่เซิร์ฟเวอร์แสดง
./send_recv <qp_num> <lid>
```

## สิ่งที่ได้เรียนรู้

1. ลำดับการเริ่มต้นทรัพยากร RDMA (Device -> PD -> MR -> CQ -> QP)
2. การเปลี่ยนสถานะ QP (RESET -> INIT -> RTR -> RTS)
3. การโพสต์งานส่งและรับ (Post Send / Post Receive)
4. การรอผลจาก Completion Queue (Poll CQ)
5. การทำความสะอาดทรัพยากรอย่างถูกต้อง
