#include "mpi_training.h"

#include <stdio.h>

static void block_bounds(long long n, int rank, int size, long long *low, long long *high) {
    long long c = n / size;
    long long r = n % size;
    *low = (long long)rank * c + (rank < r ? rank : r);
    if (rank < r) {
        ++c;
    }
    *high = *low + c;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: work_distribution [--items N] [--mode cyclic|block]\n"
                         "Slide companion: cyclic and block-balanced work distribution.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = mt_max_ll(1, mt_option_ll(argc, argv, "--items", 32));
    const char *mode = mt_option_value(argc, argv, "--mode");
    if (mode == NULL) {
        mode = "cyclic";
    }

    mt_print_common_header("work_distribution", argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            if (strcmp(mode, "block") == 0) {
                long long low = 0;
                long long high = 0;
                block_bounds(n, rank, size, &low, &high);
                printf("rank %d block indices [%lld, %lld) count %lld\n",
                       rank, low, high, high - low);
            } else {
                long long count = 0;
                printf("rank %d cyclic indices", rank);
                for (long long i = rank; i < n; i += size) {
                    printf(" %lld", i);
                    ++count;
                }
                printf(" count %lld\n", count);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
