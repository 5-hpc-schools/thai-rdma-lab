# Makefile สำหรับ Lab 01 — พื้นฐาน Verbs API
# เป้าหมาย: send_recv, rdma_write, rdma_read

include ../common/Makefile.inc

TARGETS = send_recv rdma_write rdma_read

all: $(TARGETS)

# สร้างโปรแกรม send_recv จากไฟล์ send_recv.c
send_recv: send_recv.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# สร้างโปรแกรม rdma_write จากไฟล์ rdma_write.c
rdma_write: rdma_write.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# สร้างโปรแกรม rdma_read จากไฟล์ rdma_read.c
rdma_read: rdma_read.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ลบไฟล์ที่สร้างขึ้น
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
