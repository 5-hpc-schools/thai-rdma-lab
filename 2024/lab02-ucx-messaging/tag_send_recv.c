/*
 * tag_send_recv.c — ตัวอย่างการส่งข้อความแบบ Tag Matching ด้วย UCX
 *
 * โปรแกรมนี้สาธิตการใช้ UCX (Unified Communication X) สำหรับ
 * ส่งและรับข้อความระหว่างเซิร์ฟเวอร์กับไคลเอนต์
 * โดยใช้ Tag Send/Receive API ที่รองรับทั้ง RDMA และ TCP
 *
 * การใช้งาน:
 *   เซิร์ฟเวอร์:  ./tag_send_recv
 *   ไคลเอนต์:     ./tag_send_recv <server_ip>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <ucp/api/ucp.h>

#define EXCHANGE_PORT  13333        /* พอร์ต TCP สำหรับแลกเปลี่ยนที่อยู่ */
#define MSG_TAG        0xCAFE       /* ค่า tag สำหรับจับคู่ข้อความ */
#define MSG_SIZE       256          /* ขนาดข้อความ (ไบต์) */

/* โครงสร้างเก็บบริบทของคำขอที่ยังไม่เสร็จ */
typedef struct {
    int complete;   /* ธงบอกว่างานเสร็จหรือยัง (0=ยังไม่เสร็จ, 1=เสร็จ) */
} request_ctx_t;

/* ฟังก์ชัน callback เมื่องานส่งเสร็จ */
static void send_cb(void *request, ucs_status_t status, void *user_data)
{
    request_ctx_t *ctx = (request_ctx_t *)user_data;
    ctx->complete = 1;   /* ทำเครื่องหมายว่างานเสร็จแล้ว */
    printf("[callback] ส่งเสร็จ สถานะ: %s\n", ucs_status_string(status));
}

/* ฟังก์ชัน callback เมื่องานรับเสร็จ */
static void recv_cb(void *request, ucs_status_t status,
                    const ucp_tag_recv_info_t *info, void *user_data)
{
    request_ctx_t *ctx = (request_ctx_t *)user_data;
    ctx->complete = 1;   /* ทำเครื่องหมายว่างานเสร็จแล้ว */
    printf("[callback] รับเสร็จ สถานะ: %s ขนาด: %lu ไบต์\n",
           ucs_status_string(status), (unsigned long)info->length);
}

/* ฟังก์ชันวนดำเนินการจนกว่างานจะเสร็จ */
static void wait_for_completion(ucp_worker_h worker, request_ctx_t *ctx)
{
    /* วนเรียก ucp_worker_progress เพื่อขับเคลื่อนงานสื่อสาร */
    while (!ctx->complete) {
        ucp_worker_progress(worker);  /* ประมวลผลเหตุการณ์ที่ค้างอยู่ */
    }
}

/* แลกเปลี่ยนที่อยู่ worker ผ่าน TCP socket */
static ucp_ep_h exchange_address_and_connect(
    ucp_worker_h worker, int is_server, const char *server_ip)
{
    ucp_address_t *local_addr;
    size_t local_addr_len;
    ucp_address_t *remote_addr = NULL;
    size_t remote_addr_len;

    /* ดึงที่อยู่ของ worker ในเครื่อง */
    ucs_status_t st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
    if (st != UCS_OK) {
        fprintf(stderr, "[ผิดพลาด] ดึงที่อยู่ worker ล้มเหลว\n");
        exit(EXIT_FAILURE);
    }

    int sock, conn;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(EXCHANGE_PORT);

    if (is_server) {
        /* เซิร์ฟเวอร์: เปิด socket รอรับการเชื่อมต่อ */
        sock = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(sock, (struct sockaddr *)&addr, sizeof(addr));
        listen(sock, 1);
        printf("[เซิร์ฟเวอร์] รอการเชื่อมต่อบนพอร์ต %d...\n", EXCHANGE_PORT);
        conn = accept(sock, NULL, NULL);   /* รอรับการเชื่อมต่อ */
        close(sock);
    } else {
        /* ไคลเอนต์: เชื่อมต่อไปยังเซิร์ฟเวอร์ */
        conn = socket(AF_INET, SOCK_STREAM, 0);
        addr.sin_addr.s_addr = inet_addr(server_ip);
        printf("[ไคลเอนต์] กำลังเชื่อมต่อไปยัง %s:%d...\n", server_ip, EXCHANGE_PORT);
        if (connect(conn, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("connect");
            exit(EXIT_FAILURE);
        }
    }

    /* ส่งขนาดและที่อยู่ของเครื่องนี้ไปยังอีกฝั่ง */
    write(conn, &local_addr_len, sizeof(local_addr_len));
    write(conn, local_addr, local_addr_len);

    /* รับขนาดและที่อยู่จากอีกฝั่ง */
    read(conn, &remote_addr_len, sizeof(remote_addr_len));
    remote_addr = malloc(remote_addr_len);
    read(conn, remote_addr, remote_addr_len);
    close(conn);

    printf("[ข้อมูล] แลกเปลี่ยนที่อยู่สำเร็จ (ขนาด: %lu <-> %lu ไบต์)\n",
           (unsigned long)local_addr_len, (unsigned long)remote_addr_len);

    /* สร้าง endpoint เชื่อมต่อไปยัง worker ระยะไกล */
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = remote_addr;   /* ที่อยู่ของ worker ปลายทาง */

    ucp_ep_h ep;
    st = ucp_ep_create(worker, &ep_params, &ep);
    if (st != UCS_OK) {
        fprintf(stderr, "[ผิดพลาด] สร้าง endpoint ล้มเหลว\n");
        exit(EXIT_FAILURE);
    }

    /* คืนหน่วยความจำที่อยู่ */
    ucp_worker_release_address(worker, local_addr);
    free(remote_addr);

    return ep;
}

int main(int argc, char *argv[])
{
    int is_server = (argc == 1);   /* ไม่มีอาร์กิวเมนต์ = เซิร์ฟเวอร์ */
    const char *server_ip = is_server ? NULL : argv[1];

    /* === ขั้นตอนที่ 1: เริ่มต้น UCP context === */
    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG;   /* เปิดใช้ tag matching */

    ucp_config_t *config;
    ucp_config_read(NULL, NULL, &config);   /* อ่านค่าตั้งต้นจากสภาพแวดล้อม */

    ucp_context_h ucp_ctx;
    ucs_status_t st = ucp_init(&ucp_params, config, &ucp_ctx);
    ucp_config_release(config);   /* คืนค่าตั้งต้นหลังใช้งาน */
    if (st != UCS_OK) {
        fprintf(stderr, "[ผิดพลาด] ucp_init ล้มเหลว: %s\n", ucs_status_string(st));
        return EXIT_FAILURE;
    }

    /* === ขั้นตอนที่ 2: สร้าง worker === */
    ucp_worker_params_t wk_params;
    memset(&wk_params, 0, sizeof(wk_params));
    wk_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wk_params.thread_mode = UCS_THREAD_MODE_SINGLE;  /* ใช้เธรดเดียว */

    ucp_worker_h worker;
    st = ucp_worker_create(ucp_ctx, &wk_params, &worker);
    if (st != UCS_OK) {
        fprintf(stderr, "[ผิดพลาด] สร้าง worker ล้มเหลว\n");
        return EXIT_FAILURE;
    }

    /* === ขั้นตอนที่ 3: แลกเปลี่ยนที่อยู่และสร้าง endpoint === */
    ucp_ep_h ep = exchange_address_and_connect(worker, is_server, server_ip);

    /* === ขั้นตอนที่ 4: ส่ง/รับข้อความด้วย tag matching === */
    char buf[MSG_SIZE];
    request_ctx_t req_ctx = { .complete = 0 };   /* บริบทสำหรับติดตามสถานะ */

    if (is_server) {
        /* เซิร์ฟเวอร์: รับข้อความ */
        ucp_request_param_t recv_param;
        memset(&recv_param, 0, sizeof(recv_param));
        recv_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_USER_DATA;
        recv_param.cb.recv      = recv_cb;        /* callback เมื่อรับเสร็จ */
        recv_param.user_data    = &req_ctx;       /* ส่งบริบทเข้า callback */

        /* โพสต์คำขอรับข้อความที่ tag ตรงกัน */
        ucs_status_ptr_t req = ucp_tag_recv_nbx(worker, buf, MSG_SIZE,
                                                 MSG_TAG, 0xFFFF, &recv_param);

        /* ตรวจสอบว่าเสร็จทันทีหรือต้องรอ */
        if (UCS_PTR_IS_ERR(req)) {
            fprintf(stderr, "[ผิดพลาด] tag_recv_nbx ล้มเหลว\n");
        } else if (req != NULL) {
            wait_for_completion(worker, &req_ctx);   /* รอจนเสร็จ */
            ucp_request_free(req);                   /* คืนคำขอ */
        }

        printf("[ได้รับ] ข้อความ: \"%s\"\n", buf);
    } else {
        /* ไคลเอนต์: ส่งข้อความ */
        snprintf(buf, MSG_SIZE, "สวัสดีจากไคลเอนต์ UCX!");

        ucp_request_param_t send_param;
        memset(&send_param, 0, sizeof(send_param));
        send_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_USER_DATA;
        send_param.cb.send      = send_cb;        /* callback เมื่อส่งเสร็จ */
        send_param.user_data    = &req_ctx;       /* บริบทสำหรับ callback */

        /* ส่งข้อความพร้อมกำหนด tag */
        ucs_status_ptr_t req = ucp_tag_send_nbx(ep, buf, strlen(buf) + 1,
                                                 MSG_TAG, &send_param);

        if (UCS_PTR_IS_ERR(req)) {
            fprintf(stderr, "[ผิดพลาด] tag_send_nbx ล้มเหลว\n");
        } else if (req != NULL) {
            wait_for_completion(worker, &req_ctx);   /* รอจนเสร็จ */
            ucp_request_free(req);                   /* คืนคำขอ */
        }

        printf("[สำเร็จ] ส่งข้อความเรียบร้อย\n");
    }

    /* === ขั้นตอนที่ 5: ปิด endpoint อย่างสุภาพ === */
    ucp_request_param_t close_param;
    memset(&close_param, 0, sizeof(close_param));
    close_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    close_param.flags        = UCP_EP_CLOSE_FLAG_FORCE;   /* บังคับปิด */
    ucs_status_ptr_t close_req = ucp_ep_close_nbx(ep, &close_param);

    /* รอจน endpoint ปิดเสร็จ */
    if (close_req != NULL && !UCS_PTR_IS_ERR(close_req)) {
        while (ucp_request_check_status(close_req) == UCS_INPROGRESS) {
            ucp_worker_progress(worker);
        }
        ucp_request_free(close_req);
    }

    /* === ขั้นตอนที่ 6: ทำความสะอาดทรัพยากร === */
    ucp_worker_destroy(worker);     /* ทำลาย worker */
    ucp_cleanup(ucp_ctx);           /* ทำลาย UCP context */
    printf("[เสร็จสิ้น] ทำความสะอาดทรัพยากรเรียบร้อย\n");

    return 0;
}
