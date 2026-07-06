# Makefile สำหรับ Lab 05: UCC AllReduce
# ใช้ mpicc คอมไพล์พร้อม link UCC library

CC          = mpicc
CFLAGS      = -O2 -Wall -std=c11
LDFLAGS     = -lucc

# ค้นหาตำแหน่ง UCC (ถ้าติดตั้งไว้ที่ไม่ใช่ตำแหน่งมาตรฐาน)
UCC_HOME   ?= /usr
INCLUDES    = -I$(UCC_HOME)/include
LIBS        = -L$(UCC_HOME)/lib -lucc

TARGET      = ucc_allreduce
SRC         = ucc_allreduce.c

.PHONY: all clean

all: $(TARGET)

# คอมไพล์พร้อม link UCC library
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
