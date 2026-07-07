#include "mpi_training.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: collective_cost [--min-bytes N] [--max-bytes N] [--iters N] [--warmup N]\n"
                         "Slide companion: MPI_Allreduce cost over increasing payload sizes.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t min_bytes = mt_option_size(argc, argv, "--min-bytes", sizeof(double));
    uint64_t max_bytes = mt_option_size(argc, argv, "--max-bytes", 8ULL * 1024ULL * 1024ULL);
    int iters = mt_max_int(1, (int)mt_option_ll(argc, argv, "--iters", 100));
    int warmup = mt_max_int(0, (int)mt_option_ll(argc, argv, "--warmup", 10));
    if (min_bytes < sizeof(double)) {
        min_bytes = sizeof(double);
    }
    if (max_bytes < min_bytes) {
        uint64_t tmp = min_bytes;
        min_bytes = max_bytes;
        max_bytes = tmp;
    }
    int max_count = mt_checked_mpi_count((max_bytes + sizeof(double) - 1) / sizeof(double),
                                         "allreduce element count");

    double *send = malloc((size_t)max_count * sizeof(double));
    double *recv = malloc((size_t)max_count * sizeof(double));
    if (send == NULL || recv == NULL) {
        fprintf(stderr, "rank %d: failed to allocate collective buffers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    mt_print_common_header("collective_cost", argc, argv, rank, size);
    if (rank == 0) {
        printf("payload_bytes,double_count,avg_seconds,effective_gib_per_second\n");
    }

    for (uint64_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        int count = mt_checked_mpi_count((bytes + sizeof(double) - 1) / sizeof(double), "allreduce count");
        for (int i = 0; i < count; ++i) {
            send[i] = 1.0 + (double)rank;
            recv[i] = 0.0;
        }

        for (int i = 0; i < warmup; ++i) {
            MPI_Allreduce(send, recv, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        for (int i = 0; i < iters; ++i) {
            MPI_Allreduce(send, recv, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        double elapsed = MPI_Wtime() - start;
        double avg = elapsed / (double)iters;
        double max_avg = 0.0;
        MPI_Reduce(&avg, &max_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double effective_gib = ((double)count * sizeof(double) * (double)size) /
                                   max_avg / 1024.0 / 1024.0 / 1024.0;
            printf("%llu,%d,%.9f,%.6f\n",
                   (unsigned long long)((uint64_t)count * sizeof(double)), count, max_avg, effective_gib);
        }

        if (bytes > max_bytes / 2) {
            break;
        }
    }

    free(send);
    free(recv);
    MPI_Finalize();
    return 0;
}
