#include "mpi_training.h"

#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: pi_mpi [--steps N]\n"
                         "Exercise: parallel midpoint integration using cyclic work distribution and MPI_Reduce.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long steps = mt_max_ll(1, mt_option_ll(argc, argv, "--steps", 100000000));
    mt_print_common_header("pi_mpi", argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    long double h = 1.0L / (long double)steps;
    long double local_sum = 0.0L;
    for (long long i = rank; i < steps; i += size) {
        long double x = ((long double)i + 0.5L) * h;
        local_sum += 4.0L / (1.0L + x * x);
    }

    long double global_sum = 0.0L;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long double pi = h * global_sum;
        printf("steps %lld\n", steps);
        printf("pi %.18f\n", (double)pi);
        printf("absolute_error %.18Le\n", fabsl(pi - acosl(-1.0L)));
        printf("elapsed_seconds %.6f\n", max_elapsed);
        printf("million_steps_per_second %.6f\n", (double)steps / max_elapsed / 1.0e6);
    }

    MPI_Finalize();
    return 0;
}
