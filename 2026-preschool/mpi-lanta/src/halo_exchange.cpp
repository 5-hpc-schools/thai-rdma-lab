#include "mpi_training.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: halo_exchange [--points-per-rank N] [--steps N] [--warmup N]\n"
        "Run a 1D nearest-neighbor halo exchange and stencil update.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long requested_points = mpi_training::option_ll(argc, argv, "--points-per-rank", 1000000);
    const std::size_t points_per_rank = static_cast<std::size_t>(std::max<long long>(1, requested_points));
    const int steps = std::max(1, mpi_training::option_int(argc, argv, "--steps", 100));
    const int warmup = std::max(0, mpi_training::option_int(argc, argv, "--warmup", 5));

    std::vector<double> current(points_per_rank + 2, static_cast<double>(rank + 1));
    std::vector<double> next(points_per_rank + 2, 0.0);

    const int left = size > 1 ? (rank + size - 1) % size : MPI_PROC_NULL;
    const int right = size > 1 ? (rank + 1) % size : MPI_PROC_NULL;

    auto one_step = [&]() {
        MPI_Request requests[4];
        MPI_Irecv(&current[0], 1, MPI_DOUBLE, left, 101, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&current[points_per_rank + 1], 1, MPI_DOUBLE, right, 100, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&current[1], 1, MPI_DOUBLE, left, 100, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&current[points_per_rank], 1, MPI_DOUBLE, right, 101, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        for (std::size_t i = 1; i <= points_per_rank; ++i) {
            next[i] = 0.25 * current[i - 1] + 0.5 * current[i] + 0.25 * current[i + 1];
        }
        std::swap(current, next);
    };

    mpi_training::print_common_header("halo_exchange", argc, argv, rank, size);

    for (int i = 0; i < warmup; ++i) {
        one_step();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int i = 0; i < steps; ++i) {
        one_step();
    }
    const double elapsed = MPI_Wtime() - start;

    const double local_checksum = current[1] + current[points_per_rank / 2 + 1] + current[points_per_rank];
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double total_points = static_cast<double>(points_per_rank) * static_cast<double>(size) *
                                    static_cast<double>(steps);
        std::cout << "points_per_rank " << points_per_rank << '\n';
        std::cout << "steps " << steps << '\n';
        std::cout << "elapsed_seconds " << max_elapsed << '\n';
        std::cout << "million_points_per_second " << total_points / max_elapsed / 1.0e6 << '\n';
        std::cout << "checksum " << global_checksum << '\n';
    }

    MPI_Finalize();
    return 0;
}
