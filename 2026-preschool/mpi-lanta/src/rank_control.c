#include "mpi_training.h"

#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: rank_control\n"
                         "Exercise: rank 0 prints number of ranks; every rank prints its own message.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mt_print_common_header("rank_control", argc, argv, rank, size);

    if (rank == 0) {
        printf("Rank 0 sees %d MPI processes\n", size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            printf("Hello from rank %d\n", rank);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
