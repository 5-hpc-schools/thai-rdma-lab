#include "mpi_training.h"

#include <float.h>
#include <stdio.h>

static double poly(double coeff[4], double x) {
    return coeff[3] * x * x * x + coeff[2] * x * x + coeff[1] * x + coeff[0];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mt_usage_or_continue(argc, argv,
                         "Usage: poly_mpi [--input input/poly.dat]\n"
                         "Exercise: rank 0 reads input, MPI_Bcast distributes it, MPI_Reduce finds global maximum.");

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *input = mt_option_value(argc, argv, "--input");
    if (input == NULL) {
        input = "input/poly.dat";
    }

    double coeff[4] = {0.0, 0.0, 0.0, 0.0};
    double domain[2] = {0.0, 0.0};
    long long nsteps = 0;

    if (rank == 0) {
        FILE *fp = fopen(input, "r");
        if (fp == NULL) {
            perror(input);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fscanf(fp, "%lf %lf %lf %lf", &coeff[0], &coeff[1], &coeff[2], &coeff[3]) != 4 ||
            fscanf(fp, "%lf %lf %lld", &domain[0], &domain[1], &nsteps) != 3) {
            fprintf(stderr, "Invalid polynomial input file: %s\n", input);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(fp);
        if (nsteps < 2) {
            nsteps = 2;
        }
    }

    MPI_Bcast(coeff, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(domain, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsteps, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    mt_print_common_header("poly_mpi", argc, argv, rank, size);

    double step = (domain[1] - domain[0]) / (double)(nsteps - 1);
    double local_max = -DBL_MAX;
    for (long long i = rank; i < nsteps; i += size) {
        double x = domain[0] + (double)i * step;
        double y = poly(coeff, x);
        if (y > local_max) {
            local_max = y;
        }
    }

    double global_max = -DBL_MAX;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("input %s\n", input);
        printf("nsteps %lld\n", nsteps);
        printf("maximum %.18e\n", global_max);
    }

    MPI_Finalize();
    return 0;
}
