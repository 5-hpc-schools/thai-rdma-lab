/*
 * rdma_utils.h — ฟังก์ชันอำนวยความสะดวกสำหรับการเขียนโปรแกรมอาร์ดีเอ็มเอ
 * ใช้ร่วมกันในห้องปฏิบัติการทุกบท
 */
#ifndef RDMA_UTILS_H
#define RDMA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <infiniband/verbs.h>

/* ตรวจสอบค่าส่งกลับและออกจากโปรแกรมหากเกิดข้อผิดพลาด */
#define CHECK_PTR(ptr, msg) do { \
    if (!(ptr)) { \
        fprintf(stderr, "[ผิดพลาด] %s: %s (บรรทัด %d)\n", \
                msg, strerror(errno), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_RC(rc, msg) do { \
    if ((rc)) { \
        fprintf(stderr, "[ผิดพลาด] %s: รหัส %d (บรรทัด %d)\n", \
                msg, rc, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* โครงสร้างข้อมูลการเชื่อมต่อสำหรับแลกเปลี่ยนนอกช่องทาง */
struct conn_info {
    uint32_t qp_num;        /* หมายเลขคู่คิว */
    uint16_t lid;            /* ตัวระบุท้องถิ่น */
    uint8_t  gid[16];       /* ตัวระบุกลุ่ม */
    uint32_t rkey;           /* กุญแจระยะไกลของขอบเขตหน่วยความจำ */
    uint64_t remote_addr;    /* ที่อยู่หน่วยความจำระยะไกล */
};

/*
 * เปิดอุปกรณ์อาร์ดีเอ็มเอตัวแรกที่พบ
 * คืนค่า: ตัวชี้ไปยัง ibv_context หรือ NULL หากไม่พบอุปกรณ์
 */
static inline struct ibv_context *open_first_device(void)
{
    struct ibv_device **dev_list;
    struct ibv_context *ctx = NULL;
    int num_devices;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "[ผิดพลาด] ไม่พบอุปกรณ์อาร์ดีเอ็มเอ\n");
        return NULL;
    }

    ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        fprintf(stderr, "[ผิดพลาด] ไม่สามารถเปิดอุปกรณ์ %s\n",
                ibv_get_device_name(dev_list[0]));
    }

    ibv_free_device_list(dev_list);
    return ctx;
}

#endif /* RDMA_UTILS_H */
