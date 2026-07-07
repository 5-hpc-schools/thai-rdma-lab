#include "mpi_training.hpp"

#include <array>
#include <cstring>
#include <map>

namespace {

std::string compiler_string() {
    std::ostringstream out;
#if defined(__INTEL_LLVM_COMPILER)
    out << "Intel oneAPI " << __INTEL_LLVM_COMPILER;
#elif defined(__INTEL_COMPILER)
    out << "Intel classic " << __INTEL_COMPILER;
#elif defined(_CRAYC)
#if defined(_RELEASE_MAJOR) && defined(_RELEASE_MINOR)
    out << "Cray CCE " << _RELEASE_MAJOR << "." << _RELEASE_MINOR;
#else
    out << "Cray CCE " << _CRAYC;
#endif
#elif defined(__clang__)
    out << "Clang " << __clang_version__;
#elif defined(__GNUC__)
    out << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
    out << "unknown compiler";
#endif
#if defined(__VERSION__)
    out << " (" << __VERSION__ << ")";
#endif
    return out.str();
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi_training::print_usage_if_requested(
        argc,
        argv,
        "Usage: hello_environment\n"
        "Print MPI rank placement, compiler identity, and selected Slurm/module variables.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mpi_training::print_common_header("hello_environment", argc, argv, rank, size);

    std::array<char, MPI_MAX_PROCESSOR_NAME> local_name{};
    const std::string host = mpi_training::processor_name();
    std::strncpy(local_name.data(), host.c_str(), local_name.size() - 1);

    std::vector<char> all_names(static_cast<std::size_t>(size) * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(local_name.data(),
               MPI_MAX_PROCESSOR_NAME,
               MPI_CHAR,
               all_names.data(),
               MPI_MAX_PROCESSOR_NAME,
               MPI_CHAR,
               0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "compiler " << compiler_string() << '\n';
        std::cout << "cxx_standard " << __cplusplus << '\n';
        std::cout << '\n';
        std::cout << "rank,node\n";
        std::map<std::string, int> ranks_per_node;
        for (int r = 0; r < size; ++r) {
            const char *name = all_names.data() + static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME;
            ranks_per_node[name] += 1;
            std::cout << r << "," << name << '\n';
        }
        std::cout << '\n';
        std::cout << "node,ranks_on_node\n";
        for (const auto &entry : ranks_per_node) {
            std::cout << entry.first << "," << entry.second << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}
