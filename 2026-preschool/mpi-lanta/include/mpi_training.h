#ifndef MPI_TRAINING_H
#define MPI_TRAINING_H

#include <mpi.h>

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline const char *mt_option_value(int argc, char **argv, const char *name) {
    size_t name_len = strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
        if (strncmp(argv[i], name, name_len) == 0 && argv[i][name_len] == '=') {
            return argv[i] + name_len + 1;
        }
    }
    return NULL;
}

static inline int mt_has_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static inline long long mt_parse_ll(const char *raw, const char *name) {
    char *end = NULL;
    errno = 0;
    long long value = strtoll(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, raw);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return value;
}

static inline long long mt_option_ll(int argc, char **argv, const char *name, long long default_value) {
    const char *raw = mt_option_value(argc, argv, name);
    return raw == NULL ? default_value : mt_parse_ll(raw, name);
}

static inline double mt_parse_double(const char *raw, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(raw, &end);
    if (errno != 0 || end == raw || *end != '\0') {
        fprintf(stderr, "Invalid floating-point value for %s: %s\n", name, raw);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return value;
}

static inline double mt_option_double(int argc, char **argv, const char *name, double default_value) {
    const char *raw = mt_option_value(argc, argv, name);
    return raw == NULL ? default_value : mt_parse_double(raw, name);
}

static inline uint64_t mt_parse_size(const char *raw, const char *name) {
    char *end = NULL;
    errno = 0;
    double value = strtod(raw, &end);
    if (errno != 0 || end == raw || value < 0.0) {
        fprintf(stderr, "Invalid size for %s: %s\n", name, raw);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    uint64_t multiplier = 1;
    if (*end != '\0') {
        if ((end[0] == 'K' || end[0] == 'k') && (end[1] == '\0' || end[1] == 'B' || end[1] == 'b')) {
            multiplier = 1024ULL;
        } else if ((end[0] == 'M' || end[0] == 'm') && (end[1] == '\0' || end[1] == 'B' || end[1] == 'b')) {
            multiplier = 1024ULL * 1024ULL;
        } else if ((end[0] == 'G' || end[0] == 'g') && (end[1] == '\0' || end[1] == 'B' || end[1] == 'b')) {
            multiplier = 1024ULL * 1024ULL * 1024ULL;
        } else {
            fprintf(stderr, "Invalid size suffix for %s: %s\n", name, raw);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    return (uint64_t)(value * (double)multiplier);
}

static inline uint64_t mt_option_size(int argc, char **argv, const char *name, uint64_t default_value) {
    const char *raw = mt_option_value(argc, argv, name);
    return raw == NULL ? default_value : mt_parse_size(raw, name);
}

static inline const char *mt_env_or(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return (value == NULL || value[0] == '\0') ? fallback : value;
}

static inline void mt_print_command(int argc, char **argv) {
    for (int i = 0; i < argc; ++i) {
        printf("%s%s", i == 0 ? "" : " ", argv[i]);
    }
    printf("\n");
}

static inline void mt_print_common_header(const char *program, int argc, char **argv, int rank, int size) {
    if (rank != 0) {
        return;
    }
    printf("# %s\n", program);
    printf("mpi_ranks %d\n", size);
    printf("command ");
    mt_print_command(argc, argv);
    printf("slurm_job_id %s\n", mt_env_or("SLURM_JOB_ID", "unset"));
    printf("slurm_nodes %s\n", mt_env_or("SLURM_JOB_NUM_NODES", "unset"));
    printf("slurm_ntasks %s\n", mt_env_or("SLURM_NTASKS", "unset"));
    printf("slurm_tasks_per_node %s\n", mt_env_or("SLURM_TASKS_PER_NODE", "unset"));
    printf("slurm_cpus_per_task %s\n", mt_env_or("SLURM_CPUS_PER_TASK", "unset"));
    printf("slurm_cpu_bind %s\n", mt_env_or("SLURM_CPU_BIND", "unset"));
    printf("omp_num_threads %s\n", mt_env_or("OMP_NUM_THREADS", "unset"));
    printf("pe_env %s\n", mt_env_or("PE_ENV", "unset"));
    printf("cray_cpu_target %s\n", mt_env_or("CRAY_CPU_TARGET", "unset"));
    printf("loaded_modules %s\n", mt_env_or("LOADEDMODULES", "unset"));
}

static inline void mt_usage_or_continue(int argc, char **argv, const char *usage) {
    if (mt_has_flag(argc, argv, "--help") || mt_has_flag(argc, argv, "-h")) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            printf("%s\n", usage);
        }
        MPI_Finalize();
        exit(0);
    }
}

static inline int mt_checked_mpi_count(uint64_t count, const char *what) {
    if (count > (uint64_t)INT_MAX) {
        fprintf(stderr, "%s exceeds MPI int count limit in this teaching example\n", what);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return (int)count;
}

static inline long long mt_max_ll(long long a, long long b) {
    return a > b ? a : b;
}

static inline int mt_max_int(int a, int b) {
    return a > b ? a : b;
}

#endif
