# Makefile สำหรับ Lab 02 — UCX Tag Messaging
# เป้าหมาย: tag_send_recv

include ../common/Makefile.inc

# กำหนดค่าสำหรับ UCX
UCX_CFLAGS  = -Wall -O2 -I../common
UCX_LDFLAGS = $(UCX_LIBS)

TARGETS = tag_send_recv

all: $(TARGETS)

# สร้างโปรแกรม tag_send_recv ด้วย UCX libraries
tag_send_recv: tag_send_recv.c
	$(CC) $(UCX_CFLAGS) -o $@ $< $(UCX_LDFLAGS)

# ลบไฟล์ที่สร้างขึ้น
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
