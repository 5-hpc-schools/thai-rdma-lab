# Makefile สำหรับ Lab 03 — MPI Collectives
# เป้าหมาย: allreduce_test

include ../common/Makefile.inc

TARGETS = allreduce_test

all: $(TARGETS)

# สร้างโปรแกรม allreduce_test ด้วย mpicc
allreduce_test: allreduce_test.c
	$(MPICC) $(CFLAGS) -o $@ $<

# ลบไฟล์ที่สร้างขึ้น
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
