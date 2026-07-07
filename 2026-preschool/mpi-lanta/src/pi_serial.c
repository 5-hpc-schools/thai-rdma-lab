#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    long long steps = 100000000;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--steps") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            steps = atoll(argv[++i]);
        } else if (strncmp(argv[i], "--steps=", 8) == 0) {
            steps = atoll(argv[i] + 8);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: pi_serial [--steps N]\n");
            return 0;
        }
    }
    if (steps < 1) {
        steps = 1;
    }

    double start = now_seconds();
    long double h = 1.0L / (long double)steps;
    long double sum = 0.0L;
    for (long long i = 0; i < steps; ++i) {
        long double x = ((long double)i + 0.5L) * h;
        sum += 4.0L / (1.0L + x * x);
    }
    long double pi = h * sum;
    double elapsed = now_seconds() - start;

    printf("# pi_serial\n");
    printf("steps %lld\n", steps);
    printf("pi %.18f\n", (double)pi);
    printf("absolute_error %.18Le\n", fabsl(pi - acosl(-1.0L)));
    printf("elapsed_seconds %.6f\n", elapsed);
    printf("million_steps_per_second %.6f\n", (double)steps / elapsed / 1.0e6);
    return 0;
}
