#include "mpi_training.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: memory_stream [--mib-per-rank MiB] [--iters N]\n"
        "Run a STREAM-like triad per MPI rank. Useful for rank density and CPU binding comparisons.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long requested_mib = mpi_training::option_ll(argc, argv, "--mib-per-rank", 128);
    const std::uint64_t mib_per_rank = static_cast<std::uint64_t>(std::max<long long>(1, requested_mib));
    const int iters = std::max(1, mpi_training::option_int(argc, argv, "--iters", 10));
    const std::uint64_t bytes_per_rank = mib_per_rank * 1024ULL * 1024ULL;
    const std::size_t n =
        std::max<std::size_t>(1, static_cast<std::size_t>(bytes_per_rank / (3 * sizeof(double))));

    std::vector<double> a(n, 1.0 + rank);
    std::vector<double> b(n, 2.0 + rank);
    std::vector<double> c(n, 3.0 + rank);
    const double scalar = 3.141592653589793;

    mpi_training::print_common_header("memory_stream", argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iters; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = b[i] + scalar * c[i];
        }
        std::swap(a, b);
    }
    const double elapsed = MPI_Wtime() - start;

    const double local_checksum = a.front() + a[n / 2] + a.back();
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double bytes_moved = static_cast<double>(iters) * static_cast<double>(size) *
                                   static_cast<double>(3 * n * sizeof(double));
        const double bandwidth_gib = bytes_moved / max_elapsed / 1024.0 / 1024.0 / 1024.0;
        std::cout << "mib_per_rank_requested " << mib_per_rank << '\n';
        std::cout << "elements_per_rank " << n << '\n';
        std::cout << "iterations " << iters << '\n';
        std::cout << "elapsed_seconds " << max_elapsed << '\n';
        std::cout << "aggregate_gib_per_second " << bandwidth_gib << '\n';
        std::cout << "checksum " << global_checksum << '\n';
    }

    MPI_Finalize();
    return 0;
}
