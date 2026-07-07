#include "mpi_training.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_compiler(void) {
#if defined(_CRAYC)
    printf("compiler Cray CCE %d\n", _CRAYC);
#elif defined(__INTEL_LLVM_COMPILER)
    printf("compiler Intel oneAPI %d\n", __INTEL_LLVM_COMPILER);
#elif defined(__INTEL_COMPILER)
    printf("compiler Intel classic %d\n", __INTEL_COMPILER);
#elif defined(__clang__)
    printf("compiler Clang %s\n", __clang_version__);
#elif defined(__GNUC__)
    printf("compiler GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("compiler unknown\n");
#endif
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: hello_mpi\n"
                         "Slide companion: MPI_Init, MPI_Comm_rank, MPI_Comm_size, and rank placement.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mt_print_common_header("hello_mpi", argc, argv, rank, size);

    char local_name[MPI_MAX_PROCESSOR_NAME] = {0};
    int name_len = 0;
    MPI_Get_processor_name(local_name, &name_len);

    char *all_names = NULL;
    if (rank == 0) {
        all_names = calloc((size_t)size, MPI_MAX_PROCESSOR_NAME);
        if (all_names == NULL) {
            fprintf(stderr, "rank 0: failed to allocate hostname buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }

    MPI_Gather(local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_compiler();
        printf("c_standard %ld\n", (long)__STDC_VERSION__);
        printf("\nrank,node\n");
        for (int r = 0; r < size; ++r) {
            printf("%d,%s\n", r, all_names + (size_t)r * MPI_MAX_PROCESSOR_NAME);
        }
        free(all_names);
    }

    MPI_Finalize();
    return 0;
}
