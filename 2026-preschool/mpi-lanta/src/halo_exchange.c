#include "mpi_training.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: halo_exchange [--points-per-rank N] [--steps N] [--warmup N]\n"
                         "RDMA prep: nearest-neighbor ghost-cell exchange with nonblocking MPI.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long points = mt_max_ll(1, mt_option_ll(argc, argv, "--points-per-rank", 1000000));
    int steps = mt_max_int(1, (int)mt_option_ll(argc, argv, "--steps", 100));
    int warmup = mt_max_int(0, (int)mt_option_ll(argc, argv, "--warmup", 5));

    double *current = malloc((size_t)(points + 2) * sizeof(double));
    double *next = malloc((size_t)(points + 2) * sizeof(double));
    if (current == NULL || next == NULL) {
        fprintf(stderr, "rank %d: failed to allocate halo arrays\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    for (long long i = 0; i < points + 2; ++i) {
        current[i] = (double)(rank + 1);
        next[i] = 0.0;
    }

    int left = size > 1 ? (rank + size - 1) % size : MPI_PROC_NULL;
    int right = size > 1 ? (rank + 1) % size : MPI_PROC_NULL;

    mt_print_common_header("halo_exchange", argc, argv, rank, size);

    for (int phase = 0; phase < warmup + steps; ++phase) {
        if (phase == warmup) {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Request req[4];
        MPI_Irecv(&current[0], 1, MPI_DOUBLE, left, 101, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&current[points + 1], 1, MPI_DOUBLE, right, 100, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(&current[1], 1, MPI_DOUBLE, left, 100, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&current[points], 1, MPI_DOUBLE, right, 101, MPI_COMM_WORLD, &req[3]);

        static double start = 0.0;
        if (phase == warmup) {
            start = MPI_Wtime();
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        for (long long i = 1; i <= points; ++i) {
            next[i] = 0.25 * current[i - 1] + 0.5 * current[i] + 0.25 * current[i + 1];
        }
        double *tmp = current;
        current = next;
        next = tmp;

        if (phase == warmup + steps - 1) {
            double elapsed = MPI_Wtime() - start;
            double max_elapsed = 0.0;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            double local_checksum = current[1] + current[points / 2 + 1] + current[points];
            double global_checksum = 0.0;
            MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                double total_points = (double)points * (double)size * (double)steps;
                printf("points_per_rank %lld\n", points);
                printf("steps %d\n", steps);
                printf("elapsed_seconds %.6f\n", max_elapsed);
                printf("million_points_per_second %.6f\n", total_points / max_elapsed / 1.0e6);
                printf("checksum %.6f\n", global_checksum);
            }
        }
    }

    free(current);
    free(next);
    MPI_Finalize();
    return 0;
}
