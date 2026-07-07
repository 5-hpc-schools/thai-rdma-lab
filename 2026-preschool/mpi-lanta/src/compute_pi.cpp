#include "mpi_training.hpp"

#include <cmath>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: compute_pi [--steps N]\n"
        "Compute pi by midpoint integration. Useful for compiler and optimization comparisons.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long requested_steps = mpi_training::option_ll(argc, argv, "--steps", 200000000LL);
    const std::uint64_t steps = static_cast<std::uint64_t>(std::max<long long>(1, requested_steps));

    mpi_training::print_common_header("compute_pi", argc, argv, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const long double h = 1.0L / static_cast<long double>(steps);
    long double local_sum = 0.0L;
    for (std::uint64_t i = static_cast<std::uint64_t>(rank); i < steps; i += static_cast<std::uint64_t>(size)) {
        const long double x = (static_cast<long double>(i) + 0.5L) * h;
        local_sum += 4.0L / (1.0L + x * x);
    }

    long double global_sum = 0.0L;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long double pi = h * global_sum;
        const long double error = std::abs(pi - std::acos(-1.0L));
        const double million_steps_per_second = static_cast<double>(steps) / max_elapsed / 1.0e6;
        std::cout << std::setprecision(18);
        std::cout << "steps " << steps << '\n';
        std::cout << "pi " << static_cast<double>(pi) << '\n';
        std::cout << "absolute_error " << static_cast<double>(error) << '\n';
        std::cout << std::setprecision(6);
        std::cout << "elapsed_seconds " << max_elapsed << '\n';
        std::cout << "million_steps_per_second " << million_steps_per_second << '\n';
    }

    MPI_Finalize();
    return 0;
}
