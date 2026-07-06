# Makefile สำหรับ Lab 04: NCCL AllReduce บน GPU
# ใช้ nvcc ในการคอมไพล์ CUDA code พร้อม NCCL และ MPI

# คอมไพล์เลอร์
NVCC        = nvcc
MPICC_FLAGS = $(shell mpicc --showme:compile 2>/dev/null || pkg-config --cflags mpi 2>/dev/null)
MPILD_FLAGS = $(shell mpicc --showme:link 2>/dev/null || pkg-config --libs mpi 2>/dev/null)

# แฟลกสำหรับคอมไพล์
NVCC_FLAGS  = -O2 -std=c++11
INCLUDES    = $(MPICC_FLAGS)
LDFLAGS     = -lnccl $(MPILD_FLAGS)

# เป้าหมาย
TARGET      = allreduce_nccl
SRC         = allreduce_nccl.cu

.PHONY: all clean

all: $(TARGET)

# คอมไพล์ CUDA source พร้อม link NCCL และ MPI
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
