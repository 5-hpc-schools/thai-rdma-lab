#include "mpi_training.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: ping_pong [--min-bytes N] [--max-bytes N] [--iters N] [--warmup N]\n"
                         "RDMA prep: explicit two-rank send/receive latency and bandwidth.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "ping_pong requires at least 2 MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    uint64_t min_bytes = mt_option_size(argc, argv, "--min-bytes", 1);
    uint64_t max_bytes = mt_option_size(argc, argv, "--max-bytes", 4ULL * 1024ULL * 1024ULL);
    int iters = mt_max_int(1, (int)mt_option_ll(argc, argv, "--iters", 1000));
    int warmup = mt_max_int(0, (int)mt_option_ll(argc, argv, "--warmup", 100));
    if (min_bytes == 0) {
        min_bytes = 1;
    }
    if (max_bytes < min_bytes) {
        uint64_t tmp = min_bytes;
        min_bytes = max_bytes;
        max_bytes = tmp;
    }
    int max_count = mt_checked_mpi_count(max_bytes, "message size");
    char *buffer = calloc((size_t)max_count, 1);
    if (buffer == NULL) {
        fprintf(stderr, "rank %d: failed to allocate message buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    mt_print_common_header("ping_pong", argc, argv, rank, size);
    if (rank == 0) {
        printf("bytes,one_way_latency_us,round_trip_us,bidirectional_gib_per_second\n");
    }

    for (uint64_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        int count = (int)bytes;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0 || rank == 1) {
            for (int i = 0; i < warmup; ++i) {
                if (rank == 0) {
                    MPI_Send(buffer, count, MPI_BYTE, 1, 10, MPI_COMM_WORLD);
                    MPI_Recv(buffer, count, MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buffer, count, MPI_BYTE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer, count, MPI_BYTE, 0, 11, MPI_COMM_WORLD);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        if (rank == 0 || rank == 1) {
            for (int i = 0; i < iters; ++i) {
                if (rank == 0) {
                    MPI_Send(buffer, count, MPI_BYTE, 1, 20, MPI_COMM_WORLD);
                    MPI_Recv(buffer, count, MPI_BYTE, 1, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buffer, count, MPI_BYTE, 0, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer, count, MPI_BYTE, 0, 21, MPI_COMM_WORLD);
                }
            }
        }
        double elapsed = MPI_Wtime() - start;

        if (rank == 0) {
            double round_trip = elapsed / (double)iters;
            double one_way_latency_us = round_trip * 0.5 * 1.0e6;
            double gib = (2.0 * (double)bytes * (double)iters) / elapsed / 1024.0 / 1024.0 / 1024.0;
            printf("%llu,%.6f,%.6f,%.6f\n",
                   (unsigned long long)bytes, one_way_latency_us, round_trip * 1.0e6, gib);
        }

        if (bytes > max_bytes / 2) {
            break;
        }
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
