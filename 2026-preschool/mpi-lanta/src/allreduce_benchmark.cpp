#include "mpi_training.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: allreduce_benchmark [--min-bytes N] [--max-bytes N] [--iters N] [--warmup N]\n"
        "Measure MPI_Allreduce time over increasing payload sizes.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::uint64_t min_bytes = mpi_training::option_size(argc, argv, "--min-bytes", 8);
    std::uint64_t max_bytes = mpi_training::option_size(argc, argv, "--max-bytes", 8ULL * 1024ULL * 1024ULL);
    const int iters = std::max(1, mpi_training::option_int(argc, argv, "--iters", 100));
    const int warmup = std::max(0, mpi_training::option_int(argc, argv, "--warmup", 10));

    min_bytes = std::max<std::uint64_t>(min_bytes, sizeof(double));
    if (max_bytes < min_bytes) {
        std::swap(min_bytes, max_bytes);
    }
    mpi_training::checked_mpi_count((max_bytes + sizeof(double) - 1) / sizeof(double), "allreduce element count");

    mpi_training::print_common_header("allreduce_benchmark", argc, argv, rank, size);
    if (rank == 0) {
        std::cout << "payload_bytes,double_count,avg_seconds,effective_gib_per_second\n";
    }

    for (std::uint64_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        const std::size_t count = static_cast<std::size_t>((bytes + sizeof(double) - 1) / sizeof(double));
        const int mpi_count = static_cast<int>(count);
        std::vector<double> send(count, 1.0 + rank);
        std::vector<double> recv(count, 0.0);

        for (int i = 0; i < warmup; ++i) {
            MPI_Allreduce(send.data(), recv.data(), mpi_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        for (int i = 0; i < iters; ++i) {
            MPI_Allreduce(send.data(), recv.data(), mpi_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        const double elapsed = MPI_Wtime() - start;
        const double avg = elapsed / static_cast<double>(iters);
        double max_avg = 0.0;
        MPI_Reduce(&avg, &max_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double effective_gib =
                static_cast<double>(count * sizeof(double) * static_cast<std::size_t>(size)) / max_avg / 1024.0 /
                1024.0 / 1024.0;
            std::cout << count * sizeof(double) << ',' << count << ',' << max_avg << ',' << effective_gib << '\n';
        }

        if (bytes > max_bytes / 2) {
            break;
        }
    }

    MPI_Finalize();
    return 0;
}
