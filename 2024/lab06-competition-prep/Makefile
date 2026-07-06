# Makefile สำหรับ Lab 06: One-Sided AllReduce (Competition)
# ใช้ mpicc คอมไพล์พร้อม link UCX library

CC          = mpicc
CFLAGS      = -O2 -Wall -std=c11
LDFLAGS     = -lm

# ค้นหาตำแหน่ง UCX
UCX_HOME   ?= /usr
INCLUDES    = -I$(UCX_HOME)/include
LIBS        = -L$(UCX_HOME)/lib -lucp -lucs $(LDFLAGS)

TARGET      = onesided_allreduce
SRC         = onesided_allreduce.c

.PHONY: all clean bench

all: $(TARGET)

# คอมไพล์พร้อม link UCX libraries (ucp, ucs)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# รัน benchmark เปรียบเทียบ
bench: $(TARGET)
	./benchmark.sh

clean:
	rm -f $(TARGET)
