#include "mpi_training.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: ping_pong [--min-bytes N] [--max-bytes N] [--iters N] [--warmup N]\n"
        "Measure rank 0 <-> rank 1 round-trip latency and bandwidth.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "ping_pong requires at least 2 MPI ranks\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::uint64_t min_bytes = mpi_training::option_size(argc, argv, "--min-bytes", 1);
    std::uint64_t max_bytes = mpi_training::option_size(argc, argv, "--max-bytes", 4ULL * 1024ULL * 1024ULL);
    const int iters = std::max(1, mpi_training::option_int(argc, argv, "--iters", 1000));
    const int warmup = std::max(0, mpi_training::option_int(argc, argv, "--warmup", 100));

    if (min_bytes == 0) {
        min_bytes = 1;
    }
    if (max_bytes < min_bytes) {
        std::swap(min_bytes, max_bytes);
    }
    mpi_training::checked_mpi_count(max_bytes, "message size");

    std::vector<char> buffer(static_cast<std::size_t>(max_bytes), 0);
    mpi_training::print_common_header("ping_pong", argc, argv, rank, size);

    if (rank == 0) {
        std::cout << "bytes,one_way_latency_us,round_trip_us,bidirectional_gib_per_second\n";
    }

    for (std::uint64_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        const int count = static_cast<int>(bytes);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0 || rank == 1) {
            for (int i = 0; i < warmup; ++i) {
                if (rank == 0) {
                    MPI_Send(buffer.data(), count, MPI_BYTE, 1, 10, MPI_COMM_WORLD);
                    MPI_Recv(buffer.data(), count, MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buffer.data(), count, MPI_BYTE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer.data(), count, MPI_BYTE, 0, 11, MPI_COMM_WORLD);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        if (rank == 0 || rank == 1) {
            for (int i = 0; i < iters; ++i) {
                if (rank == 0) {
                    MPI_Send(buffer.data(), count, MPI_BYTE, 1, 20, MPI_COMM_WORLD);
                    MPI_Recv(buffer.data(), count, MPI_BYTE, 1, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buffer.data(), count, MPI_BYTE, 0, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buffer.data(), count, MPI_BYTE, 0, 21, MPI_COMM_WORLD);
                }
            }
        }
        const double elapsed = MPI_Wtime() - start;

        if (rank == 0) {
            const double round_trip = elapsed / static_cast<double>(iters);
            const double one_way_latency_us = round_trip * 0.5 * 1.0e6;
            const double bidirectional_gib =
                (2.0 * static_cast<double>(bytes) * static_cast<double>(iters)) / elapsed / 1024.0 / 1024.0 / 1024.0;
            std::cout << bytes << ',' << one_way_latency_us << ',' << round_trip * 1.0e6 << ',' << bidirectional_gib
                      << '\n';
        }

        if (bytes > max_bytes / 2) {
            break;
        }
    }

    MPI_Finalize();
    return 0;
}
