#include "mpi_training.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: memory_stream [--mib-per-rank MiB] [--iters N]\n"
                         "Performance probe: STREAM-like triad per MPI rank.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long mib = mt_max_ll(1, mt_option_ll(argc, argv, "--mib-per-rank", 128));
    int iters = mt_max_int(1, (int)mt_option_ll(argc, argv, "--iters", 10));
    size_t n = (size_t)((uint64_t)mib * 1024ULL * 1024ULL / (3 * sizeof(double)));
    if (n < 1) {
        n = 1;
    }

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "rank %d: failed to allocate memory-stream arrays\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0 + rank;
        b[i] = 2.0 + rank;
        c[i] = 3.0 + rank;
    }

    mt_print_common_header("memory_stream", argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iters; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            a[i] = b[i] + 3.141592653589793 * c[i];
        }
        double *tmp = a;
        a = b;
        b = tmp;
    }
    double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_checksum = a[0] + a[n / 2] + a[n - 1];
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double bytes_moved = (double)iters * (double)size * (double)(3 * n * sizeof(double));
        printf("mib_per_rank_requested %lld\n", mib);
        printf("elements_per_rank %zu\n", n);
        printf("iterations %d\n", iters);
        printf("elapsed_seconds %.6f\n", max_elapsed);
        printf("aggregate_gib_per_second %.6f\n", bytes_moved / max_elapsed / 1024.0 / 1024.0 / 1024.0);
        printf("checksum %.6f\n", global_checksum);
    }

    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}
