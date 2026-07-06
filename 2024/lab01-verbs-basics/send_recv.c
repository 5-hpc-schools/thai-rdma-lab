/*
 * send_recv.c — ตัวอย่างพื้นฐาน RDMA Send/Receive ด้วย libibverbs
 *
 * โปรแกรมนี้สาธิตการส่งข้อความระหว่างสองกระบวนการผ่าน RDMA
 * โดยใช้คู่คิวแบบ Reliable Connection (RC)
 *
 * การใช้งาน:
 *   ฝั่งรับ (เซิร์ฟเวอร์):  ./send_recv
 *   ฝั่งส่ง (ไคลเอนต์):     ./send_recv <qp_num> <lid> <ของเซิร์ฟเวอร์>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include "rdma_utils.h"

/* ค่าคงที่สำหรับขนาดบัฟเฟอร์และคิว */
#define MSG_SIZE   1024           /* ขนาดข้อความสูงสุด (ไบต์) */
#define CQ_SIZE    16             /* ความจุของคิวเสร็จสมบูรณ์ */
#define PORT_NUM   1              /* หมายเลขพอร์ตของอุปกรณ์ RDMA */

/* ฟังก์ชันเปลี่ยนสถานะ QP จาก RESET -> INIT */
static int qp_to_init(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state        = IBV_QPS_INIT;       /* สถานะเป้าหมาย: INIT */
    attr.port_num        = PORT_NUM;            /* พอร์ตที่ใช้งาน */
    attr.pkey_index      = 0;                   /* ดัชนีของ partition key */
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |  /* อนุญาตเขียนในเครื่อง */
                           IBV_ACCESS_REMOTE_WRITE |  /* อนุญาตเขียนระยะไกล */
                           IBV_ACCESS_REMOTE_READ;    /* อนุญาตอ่านระยะไกล */

    /* เปลี่ยนสถานะ QP พร้อมระบุแอตทริบิวต์ที่ต้องการเปลี่ยน */
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                         IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

/* ฟังก์ชันเปลี่ยนสถานะ QP จาก INIT -> RTR (Ready to Receive) */
static int qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t remote_lid)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state              = IBV_QPS_RTR;           /* สถานะเป้าหมาย: RTR */
    attr.path_mtu              = IBV_MTU_1024;           /* ขนาด MTU ของเส้นทาง */
    attr.dest_qp_num           = remote_qpn;             /* หมายเลข QP ปลายทาง */
    attr.rq_psn                = 0;                      /* หมายเลขลำดับแพ็กเก็ตฝั่งรับ */
    attr.max_dest_rd_atomic    = 1;                      /* จำนวน RDMA Read ที่รอได้ */
    attr.min_rnr_timer         = 12;                     /* เวลารอก่อนส่ง RNR NAK */
    attr.ah_attr.dlid          = remote_lid;             /* LID ปลายทาง */
    attr.ah_attr.sl            = 0;                      /* ระดับบริการ */
    attr.ah_attr.src_path_bits = 0;                      /* บิตเส้นทางต้นทาง */
    attr.ah_attr.port_num      = PORT_NUM;               /* หมายเลขพอร์ต */
    attr.ah_attr.is_global     = 0;                      /* ไม่ใช้ GRH (เครือข่ายเดียวกัน) */

    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

/* ฟังก์ชันเปลี่ยนสถานะ QP จาก RTR -> RTS (Ready to Send) */
static int qp_to_rts(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.qp_state      = IBV_QPS_RTS;   /* สถานะเป้าหมาย: RTS */
    attr.timeout        = 14;            /* เวลาหมดอายุการส่งซ้ำ */
    attr.retry_cnt      = 7;             /* จำนวนครั้งที่ลองส่งซ้ำ */
    attr.rnr_retry      = 7;             /* จำนวนครั้งที่ลองซ้ำเมื่อได้ RNR */
    attr.sq_psn         = 0;             /* หมายเลขลำดับแพ็กเก็ตฝั่งส่ง */
    attr.max_rd_atomic  = 1;             /* จำนวน RDMA Read ที่ส่งพร้อมกันได้ */

    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                         IBV_QP_MAX_QP_RD_ATOMIC);
}

/* ฟังก์ชันรอผลจากคิวเสร็จสมบูรณ์ (Completion Queue) */
static int poll_completion(struct ibv_cq *cq)
{
    struct ibv_wc wc;   /* โครงสร้างผลลัพธ์งาน */
    int ne;

    /* วนรอจนกว่าจะมีผลลัพธ์อย่างน้อย 1 รายการ */
    do {
        ne = ibv_poll_cq(cq, 1, &wc);  /* ดึงผลลัพธ์จาก CQ */
    } while (ne == 0);

    if (ne < 0) {
        fprintf(stderr, "[ผิดพลาด] ibv_poll_cq ล้มเหลว\n");
        return -1;
    }

    /* ตรวจสอบสถานะของงานที่เสร็จ */
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "[ผิดพลาด] งานล้มเหลว สถานะ: %s (%d)\n",
                ibv_wc_status_str(wc.status), wc.status);
        return -1;
    }

    printf("[สำเร็จ] งานเสร็จสมบูรณ์ opcode=%d ไบต์=%u\n", wc.opcode, wc.byte_len);
    return 0;
}

int main(int argc, char *argv[])
{
    int is_server = (argc == 1);   /* ถ้าไม่มีอาร์กิวเมนต์ = ฝั่งเซิร์ฟเวอร์ */

    /* === ขั้นตอนที่ 1: เปิดอุปกรณ์ RDMA === */
    struct ibv_context *ctx = open_first_device();
    CHECK_PTR(ctx, "เปิดอุปกรณ์ RDMA");

    /* สอบถามแอตทริบิวต์ของพอร์ต เพื่อดึงค่า LID */
    struct ibv_port_attr port_attr;
    int rc = ibv_query_port(ctx, PORT_NUM, &port_attr);
    CHECK_RC(rc, "สอบถามพอร์ต");
    printf("[ข้อมูล] LID ของเครื่องนี้: %u\n", port_attr.lid);

    /* === ขั้นตอนที่ 2: จัดสรร Protection Domain === */
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    CHECK_PTR(pd, "จัดสรร Protection Domain");

    /* === ขั้นตอนที่ 3: ลงทะเบียนหน่วยความจำ (Memory Region) === */
    char *buf = calloc(1, MSG_SIZE);  /* จัดสรรบัฟเฟอร์สำหรับข้อความ */
    CHECK_PTR(buf, "จัดสรรบัฟเฟอร์");

    /* ลงทะเบียน MR พร้อมสิทธิ์อ่านเขียนทั้งในเครื่องและระยะไกล */
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, MSG_SIZE,
                                   IBV_ACCESS_LOCAL_WRITE |
                                   IBV_ACCESS_REMOTE_WRITE |
                                   IBV_ACCESS_REMOTE_READ);
    CHECK_PTR(mr, "ลงทะเบียน Memory Region");

    /* === ขั้นตอนที่ 4: สร้าง Completion Queue === */
    struct ibv_cq *cq = ibv_create_cq(ctx, CQ_SIZE, NULL, NULL, 0);
    CHECK_PTR(cq, "สร้าง Completion Queue");

    /* === ขั้นตอนที่ 5: สร้าง Queue Pair แบบ RC === */
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.send_cq  = cq;               /* CQ สำหรับงานส่ง */
    qp_init_attr.recv_cq  = cq;               /* CQ สำหรับงานรับ */
    qp_init_attr.qp_type  = IBV_QPT_RC;       /* ประเภท: Reliable Connection */
    qp_init_attr.cap.max_send_wr  = 4;        /* จำนวนงานส่งสูงสุดในคิว */
    qp_init_attr.cap.max_recv_wr  = 4;        /* จำนวนงานรับสูงสุดในคิว */
    qp_init_attr.cap.max_send_sge = 1;        /* จำนวน scatter/gather สูงสุดต่องานส่ง */
    qp_init_attr.cap.max_recv_sge = 1;        /* จำนวน scatter/gather สูงสุดต่องานรับ */

    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    CHECK_PTR(qp, "สร้าง Queue Pair");
    printf("[ข้อมูล] QP number: %u\n", qp->qp_num);

    /* === ขั้นตอนที่ 6: เปลี่ยนสถานะ QP เป็น INIT === */
    rc = qp_to_init(qp);
    CHECK_RC(rc, "เปลี่ยน QP เป็น INIT");

    /* === ขั้นตอนที่ 7: แลกเปลี่ยนข้อมูลการเชื่อมต่อ === */
    uint32_t remote_qpn;
    uint16_t remote_lid;

    if (is_server) {
        /* เซิร์ฟเวอร์: แสดงข้อมูลให้ผู้ใช้คัดลอกไปใส่ฝั่งไคลเอนต์ */
        printf("\n=== โหมดเซิร์ฟเวอร์ (ฝั่งรับ) ===\n");
        printf("กรุณารันฝั่งไคลเอนต์ด้วย:\n");
        printf("  ./send_recv %u %u\n\n", qp->qp_num, port_attr.lid);
        printf("ใส่ QP number ของไคลเอนต์: ");
        scanf("%u", &remote_qpn);
        printf("ใส่ LID ของไคลเอนต์: ");
        scanf("%hu", &remote_lid);
    } else {
        /* ไคลเอนต์: รับข้อมูลเซิร์ฟเวอร์จากอาร์กิวเมนต์ */
        if (argc < 3) {
            fprintf(stderr, "การใช้งาน: %s <remote_qpn> <remote_lid>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
        remote_qpn = (uint32_t)atoi(argv[1]);   /* หมายเลข QP ของเซิร์ฟเวอร์ */
        remote_lid = (uint16_t)atoi(argv[2]);    /* LID ของเซิร์ฟเวอร์ */
        printf("\n=== โหมดไคลเอนต์ (ฝั่งส่ง) ===\n");
        printf("[ข้อมูล] เชื่อมต่อไปยัง QP=%u LID=%u\n", remote_qpn, remote_lid);
    }

    /* === ขั้นตอนที่ 8: เปลี่ยนสถานะ QP เป็น RTR === */
    rc = qp_to_rtr(qp, remote_qpn, remote_lid);
    CHECK_RC(rc, "เปลี่ยน QP เป็น RTR");

    /* === ขั้นตอนที่ 9: เปลี่ยนสถานะ QP เป็น RTS === */
    rc = qp_to_rts(qp);
    CHECK_RC(rc, "เปลี่ยน QP เป็น RTS");
    printf("[ข้อมูล] QP พร้อมใช้งาน (สถานะ RTS)\n");

    /* === ขั้นตอนที่ 10: ส่งหรือรับข้อมูล === */
    struct ibv_sge sge;             /* โครงสร้าง scatter/gather element */
    sge.addr   = (uintptr_t)buf;   /* ที่อยู่ของบัฟเฟอร์ */
    sge.length = MSG_SIZE;          /* ขนาดข้อมูล */
    sge.lkey   = mr->lkey;         /* กุญแจท้องถิ่นของ MR */

    if (is_server) {
        /* ฝั่งเซิร์ฟเวอร์: โพสต์คำขอรับข้อความ */
        struct ibv_recv_wr recv_wr, *bad_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));

        recv_wr.wr_id   = 1;       /* รหัสงานสำหรับระบุตัวตน */
        recv_wr.sg_list = &sge;    /* รายการ scatter/gather */
        recv_wr.num_sge = 1;       /* จำนวน SGE */

        /* โพสต์งานรับเข้าคิว */
        rc = ibv_post_recv(qp, &recv_wr, &bad_wr);
        CHECK_RC(rc, "โพสต์งานรับ");
        printf("[รอ] กำลังรอข้อความจากไคลเอนต์...\n");

        /* รอจนกว่างานรับจะเสร็จ */
        rc = poll_completion(cq);
        CHECK_RC(rc, "รอผลงานรับ");
        printf("[ได้รับ] ข้อความ: \"%s\"\n", buf);
    } else {
        /* ฝั่งไคลเอนต์: เตรียมข้อความและส่ง */
        snprintf(buf, MSG_SIZE, "สวัสดีจากไคลเอนต์ RDMA!");

        struct ibv_send_wr send_wr, *bad_wr;
        memset(&send_wr, 0, sizeof(send_wr));

        send_wr.wr_id      = 2;              /* รหัสงาน */
        send_wr.sg_list    = &sge;           /* รายการ scatter/gather */
        send_wr.num_sge    = 1;              /* จำนวน SGE */
        send_wr.opcode     = IBV_WR_SEND;    /* ประเภทงาน: ส่งข้อมูล */
        send_wr.send_flags = IBV_SEND_SIGNALED;  /* ขอแจ้งเตือนเมื่อเสร็จ */

        /* โพสต์งานส่งเข้าคิว */
        rc = ibv_post_send(qp, &send_wr, &bad_wr);
        CHECK_RC(rc, "โพสต์งานส่ง");
        printf("[ส่ง] กำลังส่งข้อความ...\n");

        /* รอจนกว่างานส่งจะเสร็จ */
        rc = poll_completion(cq);
        CHECK_RC(rc, "รอผลงานส่ง");
        printf("[สำเร็จ] ส่งข้อความเรียบร้อย\n");
    }

    /* === ขั้นตอนที่ 11: ทำความสะอาดทรัพยากร === */
    ibv_destroy_qp(qp);       /* ทำลาย Queue Pair */
    ibv_destroy_cq(cq);       /* ทำลาย Completion Queue */
    ibv_dereg_mr(mr);         /* ยกเลิกการลงทะเบียน Memory Region */
    ibv_dealloc_pd(pd);       /* คืน Protection Domain */
    ibv_close_device(ctx);    /* ปิดอุปกรณ์ */
    free(buf);                /* คืนหน่วยความจำบัฟเฟอร์ */

    printf("[เสร็จสิ้น] ทำความสะอาดทรัพยากรเรียบร้อย\n");
    return 0;
}
