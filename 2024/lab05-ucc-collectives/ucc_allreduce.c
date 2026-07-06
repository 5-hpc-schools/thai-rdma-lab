/*
 * Lab 05: UCC AllReduce
 * =======================
 * ตัวอย่างการใช้ Unified Collective Communication (UCC) library
 * เพื่อทำ AllReduce operation โดยใช้ MPI เป็น out-of-band (OOB) channel
 *
 * แนวคิดหลัก:
 *   - UCC เป็น collective communication framework ที่รองรับหลาย transport
 *   - รองรับทั้ง CPU และ GPU memory
 *   - ใช้ MPI สำหรับ OOB (out-of-band) communication ในการตั้งค่า
 *   - collective operations ทำงานแบบ non-blocking ต้องใช้ progress loop
 *
 * การคอมไพล์: make
 * การรัน:     mpirun -np <จำนวน process> ./ucc_allreduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <ucc/api/ucc.h>

/* จำนวนสมาชิกในอาร์เรย์ */
#define NUM_ELEMENTS (1024 * 1024)

/* ตัวแปร global สำหรับ MPI context (ใช้ใน OOB callbacks) */
static MPI_Comm g_mpi_comm;

/* ========== OOB Callbacks สำหรับ UCC ========== */

/*
 * ฟังก์ชัน allgather สำหรับ OOB
 * UCC ต้องการ allgather เพื่อแลกเปลี่ยนข้อมูลระหว่าง rank ในช่วงเริ่มต้น
 * เราใช้ MPI_Allgather เป็น backend
 */
static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **request)
{
    MPI_Comm comm = *(MPI_Comm *)coll_info;
    /* ใช้ MPI_Allgather เพื่อรวบรวมข้อมูลจากทุก rank */
    MPI_Allgather(sbuf, (int)msglen, MPI_BYTE,
                  rbuf, (int)msglen, MPI_BYTE, comm);
    /* คืนค่า request เป็น non-NULL เพื่อบอกว่าเสร็จแล้ว */
    *request = (void *)1;
    return UCC_OK;
}

/*
 * ฟังก์ชันตรวจสอบว่า OOB request เสร็จหรือยัง
 * เนื่องจาก allgather ของเราเป็น blocking จึงเสร็จทันที
 */
static ucc_status_t oob_allgather_test(void *request)
{
    (void)request;
    return UCC_OK;
}

/*
 * ฟังก์ชันคืนทรัพยากรของ OOB request
 */
static ucc_status_t oob_allgather_free(void *request)
{
    (void)request;
    return UCC_OK;
}

/* ========== ฟังก์ชันหลัก ========== */
int main(int argc, char *argv[])
{
    int rank, size;
    ucc_lib_h         lib;
    ucc_context_h     context;
    ucc_team_h        team;
    ucc_coll_req_h    coll_req;
    ucc_status_t       status;

    /* ========== ขั้นตอนที่ 1: เริ่มต้น MPI ========== */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_mpi_comm = MPI_COMM_WORLD;

    if (rank == 0) {
        printf("=== UCC AllReduce Lab ===\n");
        printf("จำนวน process: %d, จำนวนสมาชิก: %d\n", size, NUM_ELEMENTS);
    }

    /* ========== ขั้นตอนที่ 2: สร้าง UCC Library handle ========== */
    /* ucc_init เริ่มต้น UCC library พร้อมกำหนดพารามิเตอร์ */
    ucc_lib_config_h lib_config;
    ucc_lib_params_t lib_params;
    memset(&lib_params, 0, sizeof(lib_params));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;

    status = ucc_lib_config_read(NULL, NULL, &lib_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_lib_config_read ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    status = ucc_init(&lib_params, lib_config, &lib);
    ucc_lib_config_release(lib_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_init ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ========== ขั้นตอนที่ 3: สร้าง UCC Context ========== */
    /* Context ต้องการ OOB allgather เพื่อแลกเปลี่ยนข้อมูล transport */
    ucc_context_config_h ctx_config;
    ucc_context_params_t ctx_params;
    memset(&ctx_params, 0, sizeof(ctx_params));
    ctx_params.mask             = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather    = oob_allgather;
    ctx_params.oob.req_test     = oob_allgather_test;
    ctx_params.oob.req_free     = oob_allgather_free;
    ctx_params.oob.coll_info    = &g_mpi_comm;
    ctx_params.oob.n_oob_eps    = (uint32_t)size;
    ctx_params.oob.oob_ep       = (uint32_t)rank;

    status = ucc_context_config_read(lib, NULL, &ctx_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_context_config_read ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    status = ucc_context_create(lib, &ctx_params, ctx_config, &context);
    ucc_context_config_release(ctx_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_context_create ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ========== ขั้นตอนที่ 4: สร้าง UCC Team ========== */
    /* Team คือกลุ่มของ process ที่จะทำ collective operations ร่วมกัน */
    ucc_team_params_t team_params;
    memset(&team_params, 0, sizeof(team_params));
    team_params.mask          = UCC_TEAM_PARAM_FIELD_OOB |
                                UCC_TEAM_PARAM_FIELD_EP |
                                UCC_TEAM_PARAM_FIELD_EP_RANGE;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = &g_mpi_comm;
    team_params.oob.n_oob_eps = (uint32_t)size;
    team_params.oob.oob_ep    = (uint32_t)rank;
    team_params.ep            = (uint64_t)rank;
    team_params.ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;

    /* post แล้ว loop รอจนกว่า team จะสร้างเสร็จ */
    status = ucc_team_create_post(&context, 1, &team_params, &team);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_team_create_post ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* รอจนกว่า team จะพร้อมใช้งาน */
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        ucc_context_progress(context);
    }
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_team_create_test ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ========== ขั้นตอนที่ 5: เตรียมข้อมูล ========== */
    size_t buf_size = NUM_ELEMENTS * sizeof(double);
    double *sendbuf = (double *)malloc(buf_size);
    double *recvbuf = (double *)malloc(buf_size);

    /* แต่ละ rank ใส่ค่า (rank + 1) ในทุกตำแหน่ง */
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        sendbuf[i] = (double)(rank + 1);
    }
    memset(recvbuf, 0, buf_size);

    /* ========== ขั้นตอนที่ 6: ตั้งค่าและเรียก AllReduce ========== */
    ucc_coll_args_t coll_args;
    memset(&coll_args, 0, sizeof(coll_args));
    coll_args.mask              = 0;
    coll_args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    coll_args.src.info.buffer   = sendbuf;
    coll_args.src.info.count    = NUM_ELEMENTS;
    coll_args.src.info.datatype = UCC_DT_FLOAT64;
    coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    coll_args.dst.info.buffer   = recvbuf;
    coll_args.dst.info.count    = NUM_ELEMENTS;
    coll_args.dst.info.datatype = UCC_DT_FLOAT64;
    coll_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
    coll_args.op                = UCC_OP_SUM;

    /* เริ่มต้น collective operation */
    status = ucc_collective_init(&coll_args, &coll_req, team);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_collective_init ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ส่ง collective request เข้าคิว */
    status = ucc_collective_post(coll_req);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_collective_post ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ========== ขั้นตอนที่ 7: Progress loop ========== */
    /* UCC ทำงานแบบ non-blocking ต้องเรียก progress จนกว่าจะเสร็จ */
    while (UCC_INPROGRESS == (status = ucc_collective_test(coll_req))) {
        ucc_context_progress(context);
    }
    if (status != UCC_OK) {
        fprintf(stderr, "AllReduce ล้มเหลว: %s\n", ucc_status_string(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* คืนทรัพยากรของ request */
    ucc_collective_finalize(coll_req);

    /* ========== ขั้นตอนที่ 8: ตรวจสอบผลลัพธ์ ========== */
    /* ค่าที่คาดหวัง = 1 + 2 + ... + size = size*(size+1)/2 */
    double expected = (double)(size * (size + 1)) / 2.0;
    int errors = 0;
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (recvbuf[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("[rank %d] ค่าผิดพลาดที่ตำแหน่ง %d: ได้ %f, คาดหวัง %f\n",
                       rank, i, recvbuf[i], expected);
            }
        }
    }

    if (errors == 0) {
        printf("[rank %d] UCC AllReduce สำเร็จ! ค่าผลลัพธ์ = %.1f (ถูกต้อง)\n",
               rank, recvbuf[0]);
    } else {
        printf("[rank %d] พบข้อผิดพลาด %d ตำแหน่ง\n", rank, errors);
    }

    /* ========== ขั้นตอนที่ 9: ทำความสะอาดทรัพยากร ========== */
    free(sendbuf);
    free(recvbuf);

    /* ทำลาย team, context, และ library ตามลำดับ (ย้อนกลับจากการสร้าง) */
    while (UCC_INPROGRESS == (status = ucc_team_destroy(team))) {
        ucc_context_progress(context);
    }
    ucc_context_destroy(context);
    ucc_finalize(lib);

    MPI_Finalize();

    return 0;
}
