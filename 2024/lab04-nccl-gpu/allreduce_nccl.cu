/*
 * Lab 04: NCCL AllReduce บน GPU
 * ================================
 * ตัวอย่างการใช้ NCCL (NVIDIA Collective Communications Library)
 * เพื่อทำ AllReduce operation ระหว่าง GPU หลายตัว
 *
 * แนวคิดหลัก:
 *   - NCCL ใช้สำหรับ collective operations ระหว่าง GPU โดยตรง
 *   - รองรับ GPU-Direct RDMA เพื่อส่งข้อมูลระหว่าง GPU โดยไม่ผ่าน CPU
 *   - ncclAllReduce รวมผลข้อมูลจากทุก GPU แล้วกระจายผลลัพธ์กลับ
 *
 * การคอมไพล์: make
 * การรัน:     mpirun -np <จำนวน GPU> ./allreduce_nccl
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

/* ฟังก์ชันตรวจสอบข้อผิดพลาดของ CUDA */
#define CUDA_CHECK(cmd) do {                                    \
    cudaError_t err = cmd;                                      \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error: %s ที่ %s:%d\n",          \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* ฟังก์ชันตรวจสอบข้อผิดพลาดของ NCCL */
#define NCCL_CHECK(cmd) do {                                    \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
        fprintf(stderr, "NCCL error: %s ที่ %s:%d\n",          \
                ncclGetErrorString(res), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

/* จำนวนสมาชิกในอาร์เรย์ที่จะทำ AllReduce */
#define NUM_ELEMENTS 1024 * 1024

int main(int argc, char *argv[])
{
    int rank, nranks;
    ncclUniqueId nccl_id;
    ncclComm_t   nccl_comm;
    cudaStream_t stream;

    /* ========== ขั้นตอนที่ 1: เริ่มต้น MPI ========== */
    /* ใช้ MPI เป็น out-of-band channel สำหรับแลกเปลี่ยน NCCL unique ID */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    /* ========== ขั้นตอนที่ 2: สร้าง NCCL Unique ID ========== */
    /* rank 0 สร้าง ID แล้ว broadcast ไปยังทุก rank */
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ========== ขั้นตอนที่ 3: ตั้งค่า GPU และสร้าง NCCL communicator ========== */
    /* แต่ละ rank เลือก GPU ตาม rank number */
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* ncclCommInitRank สร้าง communicator สำหรับ rank นี้ */
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, nccl_id, rank));

    if (rank == 0) {
        printf("=== NCCL AllReduce Lab ===\n");
        printf("จำนวน GPU: %d, จำนวนสมาชิก: %d\n", nranks, NUM_ELEMENTS);
    }

    /* ========== ขั้นตอนที่ 4: จัดสรรหน่วยความจำบน GPU ========== */
    float *d_sendbuf, *d_recvbuf;
    float *h_sendbuf, *h_recvbuf;
    size_t buf_size = NUM_ELEMENTS * sizeof(float);

    /* จัดสรร buffer บน host สำหรับเตรียมข้อมูลและตรวจสอบผลลัพธ์ */
    h_sendbuf = (float *)malloc(buf_size);
    h_recvbuf = (float *)malloc(buf_size);

    /* จัดสรร buffer บน GPU */
    CUDA_CHECK(cudaMalloc(&d_sendbuf, buf_size));
    CUDA_CHECK(cudaMalloc(&d_recvbuf, buf_size));

    /* เตรียมข้อมูล: แต่ละ rank ใส่ค่า (rank + 1) ในทุกตำแหน่ง */
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_sendbuf[i] = (float)(rank + 1);
    }

    /* คัดลอกข้อมูลจาก host ไป GPU */
    CUDA_CHECK(cudaMemcpy(d_sendbuf, h_sendbuf, buf_size, cudaMemcpyHostToDevice));

    /* ========== ขั้นตอนที่ 5: เรียก ncclAllReduce ========== */
    /* ทำ SUM reduction: ผลลัพธ์ = ผลรวมของค่าจากทุก GPU */
    NCCL_CHECK(ncclAllReduce(d_sendbuf, d_recvbuf, NUM_ELEMENTS,
                             ncclFloat, ncclSum, nccl_comm, stream));

    /* รอให้ stream ทำงานเสร็จ */
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ========== ขั้นตอนที่ 6: ตรวจสอบผลลัพธ์ ========== */
    CUDA_CHECK(cudaMemcpy(h_recvbuf, d_recvbuf, buf_size, cudaMemcpyDeviceToHost));

    /* ค่าที่คาดหวัง = 1 + 2 + ... + nranks = nranks*(nranks+1)/2 */
    float expected = (float)(nranks * (nranks + 1)) / 2.0f;
    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (h_recvbuf[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("[rank %d] ค่าผิดพลาดที่ตำแหน่ง %d: ได้ %f, คาดหวัง %f\n",
                       rank, i, h_recvbuf[i], expected);
            }
        }
    }

    if (errors == 0) {
        printf("[rank %d] AllReduce สำเร็จ! ค่าผลลัพธ์ = %.1f (ถูกต้อง)\n",
               rank, h_recvbuf[0]);
    } else {
        printf("[rank %d] พบข้อผิดพลาด %d ตำแหน่ง\n", rank, errors);
    }

    /* ========== ขั้นตอนที่ 7: ทำความสะอาดทรัพยากร ========== */
    free(h_sendbuf);
    free(h_recvbuf);
    CUDA_CHECK(cudaFree(d_sendbuf));
    CUDA_CHECK(cudaFree(d_recvbuf));

    /* ทำลาย NCCL communicator */
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();

    return 0;
}
