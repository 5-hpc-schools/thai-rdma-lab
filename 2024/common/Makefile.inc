# Makefile.inc — กฎการสร้างร่วมสำหรับทุกห้องปฏิบัติการ

CC       ?= gcc
MPICC    ?= mpicc
NVCC     ?= nvcc
CFLAGS   ?= -Wall -O2 -I../common
LDFLAGS  ?= -libverbs -lrdmacm
UCX_LIBS ?= -lucp -lucs -luct
UCC_LIBS ?= -lucc

# กฎสร้างไฟล์ .c เป็น executable
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# กฎสร้างไฟล์ที่ใช้ เอ็มพีไอ
%-mpi: %.c
	$(MPICC) $(CFLAGS) -o $@ $<

# กฎสร้างไฟล์ .cu (ซียูดีเอ)
%: %.cu
	$(NVCC) -O2 -o $@ $< -lnccl

clean:
	rm -f $(TARGETS) *.o
