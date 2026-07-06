# Lab 02 — UCX Tag Messaging

## คำอธิบาย

แบบฝึกหัดนี้สาธิตการใช้ UCX (Unified Communication X) สำหรับส่งข้อความ
ระหว่างสองกระบวนการด้วย Tag Matching API ซึ่งเป็น API ระดับสูง
ที่เลือกใช้ transport ที่เหมาะสมโดยอัตโนมัติ (RDMA, shared memory, TCP)

## ไฟล์ในแบบฝึกหัด

| ไฟล์ | คำอธิบาย |
|------|----------|
| `tag_send_recv.c` | ตัวอย่าง UCX tag send/receive แบบไคลเอนต์-เซิร์ฟเวอร์ |

## ข้อกำหนดเบื้องต้น

- ติดตั้ง UCX (`libucp-dev`, `libucs-dev`, `libuct-dev`)
- คอมไพเลอร์ C (gcc)

```bash
# ติดตั้งแพ็กเกจที่จำเป็น (Ubuntu/Debian)
sudo apt install libucx-dev

# หรือติดตั้งจากซอร์สโค้ด
# git clone https://github.com/openucx/ucx.git
# cd ucx && ./autogen.sh && ./configure --prefix=/usr/local && make -j && sudo make install
```

## วิธีคอมไพล์

```bash
make            # คอมไพล์ทุกโปรแกรม
make clean      # ลบไฟล์ที่คอมไพล์แล้ว
```

## วิธีรัน

เปิดเทอร์มินัลสองหน้าต่าง:

```bash
# เทอร์มินัลที่ 1 — เซิร์ฟเวอร์
./tag_send_recv

# เทอร์มินัลที่ 2 — ไคลเอนต์
./tag_send_recv 127.0.0.1        # หรือใส่ IP ของเซิร์ฟเวอร์
```

## สิ่งที่ได้เรียนรู้

1. การเริ่มต้น UCP context และ worker
2. การแลกเปลี่ยนที่อยู่ worker เพื่อสร้าง endpoint
3. การส่งและรับข้อความด้วย tag matching (non-blocking)
4. การใช้ callback และ worker progress loop
5. การปิด endpoint และทำความสะอาดทรัพยากร
