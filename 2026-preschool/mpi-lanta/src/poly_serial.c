#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double poly(double coeff[4], double x) {
    return coeff[3] * x * x * x + coeff[2] * x * x + coeff[1] * x + coeff[0];
}

int main(int argc, char **argv) {
    const char *input = "input/poly.dat";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strncmp(argv[i], "--input=", 8) == 0) {
            input = argv[i] + 8;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: poly_serial [--input input/poly.dat]\n");
            return 0;
        }
    }

    FILE *fp = fopen(input, "r");
    if (fp == NULL) {
        perror(input);
        return 1;
    }

    double coeff[4] = {0.0, 0.0, 0.0, 0.0};
    double xmin = 0.0;
    double xmax = 0.0;
    long long nsteps = 0;
    if (fscanf(fp, "%lf %lf %lf %lf", &coeff[0], &coeff[1], &coeff[2], &coeff[3]) != 4 ||
        fscanf(fp, "%lf %lf %lld", &xmin, &xmax, &nsteps) != 3) {
        fprintf(stderr, "Invalid polynomial input file: %s\n", input);
        fclose(fp);
        return 1;
    }
    fclose(fp);
    if (nsteps < 2) {
        nsteps = 2;
    }

    double step = (xmax - xmin) / (double)(nsteps - 1);
    double ymax = -DBL_MAX;
    double xmax_at_ymax = xmin;
    for (long long i = 0; i < nsteps; ++i) {
        double x = xmin + (double)i * step;
        double y = poly(coeff, x);
        if (y > ymax) {
            ymax = y;
            xmax_at_ymax = x;
        }
    }

    printf("# poly_serial\n");
    printf("input %s\n", input);
    printf("nsteps %lld\n", nsteps);
    printf("maximum %.18e\n", ymax);
    printf("x_at_maximum %.18e\n", xmax_at_ymax);
    return 0;
}
