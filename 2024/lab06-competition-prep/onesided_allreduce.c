/*
 * Lab 06: One-Sided AllReduce ด้วย UCX Put + Atomic
 * =====================================================
 * โปรแกรมแข่งขัน: สร้าง AllReduce แบบ one-sided โดยใช้ UCX
 *
 * อัลกอริทึม Ring AllReduce:
 *   ขั้นตอนที่ 1 - Reduce-Scatter (ring):
 *     แต่ละ rank ส่งส่วนของข้อมูลไปยัง rank ถัดไปในวงแหวน
 *     ใช้ ucp_put_nbx เพื่อเขียนข้อมูลตรงไปยัง remote memory
 *     ใช้ ucp_atomic_op_nbx (ADD) เพื่อแจ้งเตือนว่าเขียนเสร็จแล้ว
 *     หลัง size-1 รอบ แต่ละ rank จะมีผลรวมของส่วนที่รับผิดชอบ
 *
 *   ขั้นตอนที่ 2 - Allgather (ring):
 *     แต่ละ rank ส่งส่วนที่มีผลรวมแล้วไปยัง rank ถัดไป
 *     หลัง size-1 รอบ ทุก rank จะมีผลลัพธ์ AllReduce ครบถ้วน
 *
 *   สุดท้าย: เปรียบเทียบกับ MPI_Allreduce เพื่อตรวจสอบความถูกต้อง
 *            และวัด benchmark เปรียบเทียบประสิทธิภาพ
 *
 * การคอมไพล์: make
 * การรัน:     mpirun -np <จำนวน process> ./onesided_allreduce
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <ucp/api/ucp.h>

/* จำนวนสมาชิกในอาร์เรย์ (ต้องหารด้วยจำนวน rank ลงตัว) */
#define NUM_ELEMENTS (1024 * 1024)
/* จำนวนรอบ benchmark */
#define NUM_ITERS    100

/* ========== โครงสร้างข้อมูลสำหรับ UCX ========== */
typedef struct {
    ucp_context_h   context;
    ucp_worker_h    worker;
    ucp_ep_h       *eps;          /* endpoint ไปยังแต่ละ rank */
    ucp_mem_h       data_memh;    /* memory handle สำหรับ data buffer */
    ucp_mem_h       flag_memh;    /* memory handle สำหรับ notification flags */
    void           *data_rkey_buf;
    size_t          data_rkey_len;
    void           *flag_rkey_buf;
    size_t          flag_rkey_len;
    ucp_rkey_h     *data_rkeys;   /* remote keys สำหรับเข้าถึง data ของ remote */
    ucp_rkey_h     *flag_rkeys;   /* remote keys สำหรับเข้าถึง flags ของ remote */
} ucx_state_t;

/* ========== ฟังก์ชัน callback สำหรับ request ========== */
static void send_cb(void *request, ucs_status_t status, void *user_data)
{
    (void)request; (void)status; (void)user_data;
}

/* ========== ฟังก์ชันรอจนกว่า request จะเสร็จ ========== */
static void wait_request(ucp_worker_h worker, void *request)
{
    if (request == NULL) return;
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "UCX request ล้มเหลว: %s\n",
                ucs_status_string(UCS_PTR_STATUS(request)));
        return;
    }
    /* วนรอ progress จนกว่า request จะเสร็จ */
    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        ucp_worker_progress(worker);
    }
    ucp_request_free(request);
}

/* ========== เริ่มต้น UCX context และ worker ========== */
static int init_ucx(ucx_state_t *state)
{
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucs_status_t status;

    /* กำหนดคุณสมบัติที่ต้องการ: RMA (put) และ Atomic operations */
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT;
    params.features   = UCP_FEATURE_RMA | UCP_FEATURE_AMO64;
    params.request_size = 8;
    params.request_init = NULL;

    ucp_config_h config;
    ucp_config_read(NULL, NULL, &config);
    status = ucp_init(&params, config, &state->context);
    ucp_config_release(config);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_init ล้มเหลว\n");
        return -1;
    }

    /* สร้าง worker สำหรับ progress engine */
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(state->context, &worker_params, &state->worker);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_worker_create ล้มเหลว\n");
        return -1;
    }
    return 0;
}

/* ========== แลกเปลี่ยน address และสร้าง endpoints ========== */
static int setup_endpoints(ucx_state_t *state, int rank, int size)
{
    ucp_address_t *local_addr;
    size_t local_addr_len;
    ucs_status_t status;

    /* ดึง worker address */
    status = ucp_worker_get_address(state->worker, &local_addr, &local_addr_len);
    if (status != UCS_OK) return -1;

    /* รวบรวม address ทุก rank ผ่าน MPI */
    size_t *all_lens = calloc(size, sizeof(size_t));
    MPI_Allgather(&local_addr_len, sizeof(size_t), MPI_BYTE,
                  all_lens, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    int *displs = calloc(size, sizeof(int));
    int *recvcounts = calloc(size, sizeof(int));
    size_t total = 0;
    for (int i = 0; i < size; i++) {
        displs[i]     = (int)total;
        recvcounts[i] = (int)all_lens[i];
        total += all_lens[i];
    }

    char *all_addrs = malloc(total);
    MPI_Allgatherv(local_addr, (int)local_addr_len, MPI_BYTE,
                   all_addrs, recvcounts, displs, MPI_BYTE, MPI_COMM_WORLD);

    /* สร้าง endpoint ไปยังแต่ละ rank */
    state->eps = calloc(size, sizeof(ucp_ep_h));
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        ucp_ep_params_t ep_params;
        memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = (ucp_address_t *)(all_addrs + displs[i]);
        ucp_ep_create(state->worker, &ep_params, &state->eps[i]);
    }

    ucp_worker_release_address(state->worker, local_addr);
    free(all_lens); free(displs); free(recvcounts); free(all_addrs);
    return 0;
}

/* ========== ลงทะเบียนหน่วยความจำและแลกเปลี่ยน remote keys ========== */
static int register_memory(ucx_state_t *state, double *data, uint64_t *flags,
                           size_t data_size, size_t flag_size, int rank, int size)
{
    ucp_mem_map_params_t mem_params;
    ucs_status_t status;

    /* ลงทะเบียน data buffer กับ UCX */
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mem_params.address    = data;
    mem_params.length     = data_size;
    status = ucp_mem_map(state->context, &mem_params, &state->data_memh);
    if (status != UCS_OK) return -1;

    /* ลงทะเบียน flag buffer */
    mem_params.address = flags;
    mem_params.length  = flag_size;
    status = ucp_mem_map(state->context, &mem_params, &state->flag_memh);
    if (status != UCS_OK) return -1;

    /* สร้าง packed rkey สำหรับ data buffer และ flag buffer */
    ucp_rkey_buffer_release(state->data_rkey_buf);
    ucp_mem_rkey_pack(state->context, state->data_memh,
                      &state->data_rkey_buf, &state->data_rkey_len);
    ucp_mem_rkey_pack(state->context, state->flag_memh,
                      &state->flag_rkey_buf, &state->flag_rkey_len);

    /* แลกเปลี่ยน rkey ผ่าน MPI */
    void *all_data_rkeys = malloc(state->data_rkey_len * size);
    void *all_flag_rkeys = malloc(state->flag_rkey_len * size);
    MPI_Allgather(state->data_rkey_buf, (int)state->data_rkey_len, MPI_BYTE,
                  all_data_rkeys, (int)state->data_rkey_len, MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(state->flag_rkey_buf, (int)state->flag_rkey_len, MPI_BYTE,
                  all_flag_rkeys, (int)state->flag_rkey_len, MPI_BYTE, MPI_COMM_WORLD);

    /* Unpack rkeys ของแต่ละ remote rank */
    state->data_rkeys = calloc(size, sizeof(ucp_rkey_h));
    state->flag_rkeys = calloc(size, sizeof(ucp_rkey_h));
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;
        ucp_ep_rkey_unpack(state->eps[i],
                           (char *)all_data_rkeys + i * state->data_rkey_len,
                           &state->data_rkeys[i]);
        ucp_ep_rkey_unpack(state->eps[i],
                           (char *)all_flag_rkeys + i * state->flag_rkey_len,
                           &state->flag_rkeys[i]);
    }

    /* แลกเปลี่ยน base address ของ data buffer และ flag buffer */
    free(all_data_rkeys);
    free(all_flag_rkeys);
    return 0;
}

/* ========== One-Sided Ring AllReduce ========== */
static void onesided_allreduce(ucx_state_t *state, double *data, double *scratch,
                               uint64_t *flags, uint64_t *remote_data_addrs,
                               uint64_t *remote_flag_addrs,
                               int count, int rank, int size)
{
    int chunk = count / size;

    /*
     * เฟส 1: Reduce-Scatter แบบ Ring
     * --------------------------------
     * ในแต่ละรอบ (step) ของ ring:
     *   - rank i ส่ง chunk ที่ (i - step) ไปยัง rank (i + 1)
     *   - rank (i + 1) รับข้อมูลแล้วบวกเข้ากับ local data
     *   - ใช้ atomic ADD บน flag เพื่อแจ้งเตือนการรับ
     * หลัง (size-1) รอบ แต่ละ rank จะมีผลรวมของ chunk ที่รับผิดชอบ
     */
    int next = (rank + 1) % size;

    for (int step = 0; step < size - 1; step++) {
        int send_chunk = ((rank - step) % size + size) % size;
        size_t offset  = send_chunk * chunk * sizeof(double);
        size_t nbytes  = chunk * sizeof(double);

        /* ใช้ ucp_put_nbx เขียน chunk ไปยัง scratch area ของ rank ถัดไป */
        ucp_request_param_t param;
        memset(&param, 0, sizeof(param));
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        param.cb.send      = send_cb;

        void *req = ucp_put_nbx(state->eps[next],
                                &data[send_chunk * chunk], nbytes,
                                remote_data_addrs[next] + (size_t)(count * sizeof(double)) + offset,
                                state->data_rkeys[next], &param);
        wait_request(state->worker, req);

        /* ใช้ atomic ADD เพื่อเพิ่ม flag ของ remote rank เป็นการแจ้งเตือน */
        uint64_t one = 1;
        ucp_request_param_t atom_param;
        memset(&atom_param, 0, sizeof(atom_param));
        atom_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        atom_param.cb.send      = send_cb;

        req = ucp_atomic_op_nbx(state->eps[next], UCP_ATOMIC_OP_ADD,
                                &one, sizeof(one),
                                remote_flag_addrs[next] + step * sizeof(uint64_t),
                                state->flag_rkeys[next], &atom_param);
        wait_request(state->worker, req);

        /* รอจนกว่าจะได้รับข้อมูลจาก rank ก่อนหน้า */
        while (__sync_fetch_and_add(&flags[step], 0) == 0) {
            ucp_worker_progress(state->worker);
        }

        /* บวก scratch data เข้ากับ local data */
        int recv_chunk = ((rank - step - 1) % size + size) % size;
        double *scratch_ptr = scratch + recv_chunk * chunk;
        double *local_ptr   = data    + recv_chunk * chunk;
        for (int i = 0; i < chunk; i++) {
            local_ptr[i] += scratch_ptr[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    memset(flags, 0, size * sizeof(uint64_t));

    /*
     * เฟส 2: Allgather แบบ Ring
     * ---------------------------
     * ในแต่ละรอบ:
     *   - rank i ส่ง chunk ที่มีผลรวมแล้วไปยัง rank ถัดไป
     *   - หลัง (size-1) รอบ ทุก rank จะมีข้อมูลครบ
     */
    for (int step = 0; step < size - 1; step++) {
        int send_chunk = ((rank - step + 1) % size + size) % size;
        size_t offset  = send_chunk * chunk * sizeof(double);
        size_t nbytes  = chunk * sizeof(double);

        ucp_request_param_t param;
        memset(&param, 0, sizeof(param));
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        param.cb.send      = send_cb;

        void *req = ucp_put_nbx(state->eps[next],
                                &data[send_chunk * chunk], nbytes,
                                remote_data_addrs[next] + offset,
                                state->data_rkeys[next], &param);
        wait_request(state->worker, req);

        /* แจ้งเตือนด้วย atomic ADD */
        uint64_t one = 1;
        ucp_request_param_t atom_param;
        memset(&atom_param, 0, sizeof(atom_param));
        atom_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        atom_param.cb.send      = send_cb;

        req = ucp_atomic_op_nbx(state->eps[next], UCP_ATOMIC_OP_ADD,
                                &one, sizeof(one),
                                remote_flag_addrs[next] + step * sizeof(uint64_t),
                                state->flag_rkeys[next], &atom_param);
        wait_request(state->worker, req);

        /* รอรับ chunk จาก rank ก่อนหน้า */
        while (__sync_fetch_and_add(&flags[step], 0) == 0) {
            ucp_worker_progress(state->worker);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/* ========== ฟังก์ชันหลัก ========== */
int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (NUM_ELEMENTS % size != 0) {
        if (rank == 0)
            fprintf(stderr, "จำนวนสมาชิก (%d) ต้องหารด้วยจำนวน rank (%d) ลงตัว\n",
                    NUM_ELEMENTS, size);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("=== One-Sided Ring AllReduce (Competition) ===\n");
        printf("จำนวน process: %d, จำนวนสมาชิก: %d\n", size, NUM_ELEMENTS);
    }

    /* จัดสรรหน่วยความจำ: data + scratch area */
    size_t data_size = NUM_ELEMENTS * sizeof(double);
    double *data     = (double *)aligned_alloc(64, data_size);
    double *scratch  = (double *)aligned_alloc(64, data_size);
    double *mpi_result = (double *)malloc(data_size);
    uint64_t *flags  = (uint64_t *)calloc(size, sizeof(uint64_t));

    /* เตรียมข้อมูลเริ่มต้น */
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        data[i] = (double)(rank + 1);
    }
    memset(scratch, 0, data_size);

    /* ========== Benchmark: MPI_Allreduce (baseline) ========== */
    double *data_copy = (double *)malloc(data_size);
    memcpy(data_copy, data, data_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        memcpy(data_copy, data, data_size);  /* รีเซ็ตข้อมูลทุกรอบ */
        MPI_Allreduce(data_copy, mpi_result, NUM_ELEMENTS,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();
    double mpi_time = (t1 - t0) / NUM_ITERS;

    /* ตรวจสอบผลลัพธ์ MPI_Allreduce */
    double expected = (double)(size * (size + 1)) / 2.0;
    if (rank == 0) {
        printf("\n--- ผลลัพธ์ MPI_Allreduce ---\n");
        printf("ค่าผลลัพธ์[0] = %.1f (คาดหวัง %.1f)\n", mpi_result[0], expected);
        printf("เวลาเฉลี่ย: %.3f ms\n", mpi_time * 1000.0);
    }

    /* ========== สรุปผลการทดสอบ ========== */
    /* หมายเหตุ: ส่วน one-sided allreduce ต้องการ UCX ที่ตั้งค่าครบถ้วน
     * นักศึกษาจะต้อง implement ส่วน init_ucx, setup_endpoints,
     * register_memory ให้สมบูรณ์ แล้วเรียก onesided_allreduce
     * เพื่อเปรียบเทียบกับ MPI_Allreduce */

    if (rank == 0) {
        printf("\n--- สรุป ---\n");
        printf("MPI_Allreduce เวลาเฉลี่ย: %.3f ms\n", mpi_time * 1000.0);
        printf("(เติม one-sided implementation เพื่อเปรียบเทียบ)\n");
        printf("\nเป้าหมายการแข่งขัน: ทำให้ one-sided เร็วกว่า MPI_Allreduce\n");
    }

    /* ทำความสะอาดทรัพยากร */
    free(data);
    free(scratch);
    free(mpi_result);
    free(data_copy);
    free(flags);

    MPI_Finalize();
    return 0;
}
