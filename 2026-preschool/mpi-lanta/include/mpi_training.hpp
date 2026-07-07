#ifndef MPI_TRAINING_HPP
#define MPI_TRAINING_HPP

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpi_training {

inline const char *option_value(int argc, char **argv, const std::string &name) {
    const std::string prefix = name + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == name && i + 1 < argc) {
            return argv[i + 1];
        }
        if (arg.rfind(prefix, 0) == 0) {
            return argv[i] + prefix.size();
        }
    }
    return nullptr;
}

inline bool has_flag(int argc, char **argv, const std::string &name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return true;
        }
    }
    return false;
}

inline long long option_ll(int argc, char **argv, const std::string &name, long long default_value) {
    const char *raw = option_value(argc, argv, name);
    if (raw == nullptr) {
        return default_value;
    }
    return std::stoll(raw);
}

inline int option_int(int argc, char **argv, const std::string &name, int default_value) {
    return static_cast<int>(option_ll(argc, argv, name, default_value));
}

inline double option_double(int argc, char **argv, const std::string &name, double default_value) {
    const char *raw = option_value(argc, argv, name);
    if (raw == nullptr) {
        return default_value;
    }
    return std::stod(raw);
}

inline std::uint64_t parse_size(const std::string &raw) {
    if (raw.empty()) {
        throw std::runtime_error("empty size value");
    }

    std::string value = raw;
    std::uint64_t multiplier = 1;
    char suffix = value.back();
    if (suffix == 'B' || suffix == 'b') {
        value.pop_back();
        if (!value.empty()) {
            suffix = value.back();
        }
    }

    if (!value.empty()) {
        suffix = value.back();
        switch (suffix) {
        case 'K':
        case 'k':
            multiplier = 1024ULL;
            value.pop_back();
            break;
        case 'M':
        case 'm':
            multiplier = 1024ULL * 1024ULL;
            value.pop_back();
            break;
        case 'G':
        case 'g':
            multiplier = 1024ULL * 1024ULL * 1024ULL;
            value.pop_back();
            break;
        default:
            break;
        }
    }

    const long double parsed = std::stold(value);
    if (parsed < 0.0L) {
        throw std::runtime_error("size value must be non-negative: " + raw);
    }
    return static_cast<std::uint64_t>(parsed * multiplier);
}

inline std::uint64_t option_size(int argc, char **argv, const std::string &name, std::uint64_t default_value) {
    const char *raw = option_value(argc, argv, name);
    if (raw == nullptr) {
        return default_value;
    }
    return parse_size(raw);
}

inline std::string env_or(const char *name, const std::string &fallback = "unset") {
    const char *value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : std::string(value);
}

inline std::string processor_name() {
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    int len = 0;
    MPI_Get_processor_name(name, &len);
    return std::string(name, static_cast<std::size_t>(len));
}

inline void print_args(int argc, char **argv) {
    for (int i = 0; i < argc; ++i) {
        std::cout << (i == 0 ? "" : " ") << argv[i];
    }
    std::cout << '\n';
}

inline void print_common_header(const std::string &program, int argc, char **argv, int rank, int size) {
    if (rank != 0) {
        return;
    }
    std::cout << "# " << program << '\n';
    std::cout << "mpi_ranks " << size << '\n';
    std::cout << "command ";
    print_args(argc, argv);
    std::cout << "slurm_job_id " << env_or("SLURM_JOB_ID") << '\n';
    std::cout << "slurm_nodes " << env_or("SLURM_JOB_NUM_NODES") << '\n';
    std::cout << "slurm_ntasks " << env_or("SLURM_NTASKS") << '\n';
    std::cout << "slurm_tasks_per_node " << env_or("SLURM_TASKS_PER_NODE") << '\n';
    std::cout << "slurm_cpus_per_task " << env_or("SLURM_CPUS_PER_TASK") << '\n';
    std::cout << "slurm_cpu_bind " << env_or("SLURM_CPU_BIND") << '\n';
    std::cout << "slurm_cpu_bind_list " << env_or("SLURM_CPU_BIND_LIST") << '\n';
    std::cout << "omp_num_threads " << env_or("OMP_NUM_THREADS") << '\n';
    std::cout << "pe_env " << env_or("PE_ENV") << '\n';
    std::cout << "cray_cpu_target " << env_or("CRAY_CPU_TARGET") << '\n';
    std::cout << "loaded_modules " << env_or("LOADEDMODULES") << '\n';
}

inline std::uint64_t checked_mpi_count(std::uint64_t count, const std::string &what) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(what + " exceeds MPI int count limit in this teaching example");
    }
    return count;
}

inline void print_usage_if_requested(int argc, char **argv, const std::string &usage) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << usage << '\n';
        }
        MPI_Finalize();
        std::exit(0);
    }
}

} // namespace mpi_training

#endif
