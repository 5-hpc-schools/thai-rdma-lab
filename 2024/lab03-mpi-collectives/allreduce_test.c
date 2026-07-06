/*
 * allreduce_test.c — ตัวอย่าง MPI_Allreduce พร้อมวัดประสิทธิภาพ
 *
 * โปรแกรมนี้สาธิตการใช้ MPI_Allreduce สำหรับรวมค่าจากทุกโหนด
 * พร้อมวัดเวลาแฝงสำหรับขนาดข้อมูลต่าง ๆ
 *
 * การใช้งาน:
 *   mpirun -np 4 ./allreduce_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "timer.h"

#define NUM_ITERATIONS  100    /* จำนวนรอบสำหรับวัดค่าเฉลี่ย */
#define NUM_WARMUP      10     /* จำนวนรอบอุ่นเครื่อง (ไม่นับผล) */

/* ขนาดข้อความที่จะทดสอบ (จำนวน double) */
static const int test_counts[] = {1, 10, 100, 1000, 10000, 100000};
#define NUM_TESTS (int)(sizeof(test_counts) / sizeof(test_counts[0]))

/* ฟังก์ชันทดสอบ Allreduce สำหรับขนาดข้อมูลที่กำหนด */
static void benchmark_allreduce(int rank, int size, int count)
{
    /* จัดสรรบัฟเฟอร์ส่งและรับ */
    double *send_buf = (double *)malloc(sizeof(double) * count);
    double *recv_buf = (double *)malloc(sizeof(double) * count);

    if (!send_buf || !recv_buf) {
        fprintf(stderr, "[Rank %d] จัดสรรหน่วยความจำล้มเหลว\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* เติมค่าเริ่มต้น: แต่ละ rank ใส่ค่า rank ของตัวเอง */
    for (int i = 0; i < count; i++) {
        send_buf[i] = (double)rank;   /* rank 0 ใส่ 0.0, rank 1 ใส่ 1.0, ... */
    }

    /* รอบอุ่นเครื่อง: รันโดยไม่วัดเวลา เพื่อให้ระบบเข้าสู่สภาวะคงที่ */
    for (int i = 0; i < NUM_WARMUP; i++) {
        MPI_Allreduce(send_buf, recv_buf, count, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    /* ซิงโครไนซ์ทุก rank ก่อนเริ่มวัดเวลา */
    MPI_Barrier(MPI_COMM_WORLD);

    /* วัดเวลาสำหรับรอบทดสอบจริง */
    uint64_t start = get_time_us();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        /* MPI_Allreduce: รวมค่าจากทุก rank ด้วย SUM แล้วกระจายผลลัพธ์ */
        MPI_Allreduce(send_buf, recv_buf, count, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    uint64_t end = get_time_us();

    /* คำนวณเวลาแฝงเฉลี่ยต่อรอบ */
    double avg_us = (double)(end - start) / NUM_ITERATIONS;
    int msg_bytes = count * (int)sizeof(double);

    /* ตรวจสอบความถูกต้องของผลลัพธ์ */
    /* ผลรวมของ 0 + 1 + 2 + ... + (size-1) = size*(size-1)/2 */
    double expected = (double)size * (size - 1) / 2.0;
    int correct = 1;

    for (int i = 0; i < count; i++) {
        if (recv_buf[i] != expected) {
            correct = 0;   /* ผลลัพธ์ไม่ตรงกับที่คาดไว้ */
            break;
        }
    }

    /* Rank 0 พิมพ์ผลลัพธ์ */
    if (rank == 0) {
        printf("  %-12d %-14d %12.2f     %8.2f       %s\n",
               count, msg_bytes, avg_us,
               (avg_us > 0) ? (double)msg_bytes / avg_us : 0.0,
               correct ? "ถูกต้อง" : "ผิดพลาด!");
    }

    /* คืนหน่วยความจำ */
    free(send_buf);
    free(recv_buf);
}

int main(int argc, char *argv[])
{
    int rank, size;

    /* เริ่มต้น MPI */
    MPI_Init(&argc, &argv);

    /* ดึงหมายเลข rank และจำนวน rank ทั้งหมด */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* หมายเลขของกระบวนการนี้ */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* จำนวนกระบวนการทั้งหมด */

    if (rank == 0) {
        /* พิมพ์หัวตาราง (เฉพาะ rank 0) */
        printf("============================================"
               "====================================\n");
        printf("  MPI_Allreduce Benchmark — %d โพรเซส, "
               "%d รอบ (อุ่นเครื่อง %d รอบ)\n",
               size, NUM_ITERATIONS, NUM_WARMUP);
        printf("============================================"
               "====================================\n");
        printf("  %-12s %-14s %12s     %8s       %s\n",
               "จำนวน", "ขนาด(ไบต์)", "เวลาแฝง(us)", "MB/s", "ผลตรวจ");
        printf("  ------------ -------------- "
               "------------     --------       ------\n");
    }

    /* ทดสอบ Allreduce สำหรับขนาดข้อมูลแต่ละขนาด */
    for (int t = 0; t < NUM_TESTS; t++) {
        benchmark_allreduce(rank, size, test_counts[t]);
    }

    if (rank == 0) {
        printf("============================================"
               "====================================\n");
        printf("[เสร็จสิ้น] ทดสอบเสร็จเรียบร้อย\n");
    }

    /* จบการทำงาน MPI */
    MPI_Finalize();

    return 0;
}
