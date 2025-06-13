#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>

#include "tnns/data.h"
#include "tnns/tnns_cpu.h"
#include "tnns/common.h"
#include "utils.h"

#define OPTIMIZE_1x1

static void time_candidates(testcase_tnns_gemm_t* testcase, FILE* result_file, char* version, bool test) {
    float* ref_y = NULL;

    if(test) {
        ref_y = direct_mmm(testcase->x, testcase->w, testcase->m, testcase->n, testcase->p);
    }

    struct timeval start, end;
    double seconds = 0;
    int num_runs = 30;

    // warmup-loop
    for(int i = 0; i < 10; ++i) {
        tnns_gemm(testcase);
    }

    for(int i = 0; i < num_runs; ++i) {
        gettimeofday(&start, NULL);
        tnns_gemm(testcase);
        gettimeofday(&end, NULL);
        seconds += (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);

        if(test && i == 0 && !check_matrices(testcase->output, ref_y, testcase->m, testcase->n)) {
            fprintf(stderr, "Function did not return the correct result!\n");
            if(result_file) {
                fprintf(result_file, "error\n");
            }
            abort();
        }
    }
    seconds /= num_runs;

    if(result_file) {
        //m,n,p,Cycles,Runtime,Type
        char shape[300];
        snprintf(shape,300,"%d_%d_%dm=n=p/32",
            testcase->m,
            testcase->n,
            testcase->p);
        fprintf(
            result_file, 
            "%d,%d,%d,%lf,%s\n",
            testcase->m,
            testcase->n,
            testcase->p,
            seconds,
            version
        );
    }
}

static int benchmark_random(bool do_test, char* version) {
    srand(0);
    time_t rawtime = time(NULL);
    struct tm* t = localtime(&rawtime);

    char result_file[300];
    snprintf(result_file, 300, "out/measurements/gemm/measurement_%d_%d_%d_%d_%d_%d.csv",t->tm_year, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    FILE* measurement = fopen(result_file, "w");
    if(!measurement) {
        fprintf(stderr, "failed to open measurement output file:\n");
        perror("");
        return EXIT_FAILURE;
    }
    fprintf(measurement, "m,n,p,Runtime,Type\n");

    testcase_tnns_gemm_t* test = create_testcase_gemm();
    float* input = generate_array(4096*4096*8,true);
    float* weights = generate_array(4096*4096*8,true);
 
    test->x = input;
    test->w = weights;

    for(int k = 32; k <= 4096; k *= 2) {
        int m = k;
        int n = k;
        int p = k*8;
        test->m = m;
        test->n = n;
        test->p = p;
        test->p_packed = (p/64 + !!(p%64));
        test->x_packing0 = bitpacking0(input, m, p);
        test->w_packing0 = bitpacking0(weights, n, p);
        test->x_packing1 = bitpacking1(input, m, p);
        test->w_packing1 = bitpacking1(weights, n, p);
        test->x_packing2 = bitpacking2(input, m, p);
        test->w_packing2 = bitpacking2(weights, n, p);

        time_candidates(test, measurement, version, do_test);
        free(test->output);
        test->output = NULL;

        free(test->x_packing0);
        free(test->w_packing0);
        free(test->x_packing1);
        free(test->w_packing1);
        free(test->x_packing2);
        free(test->w_packing2);
        test->x_packing0 = NULL;
        test->w_packing0 = NULL;
        test->x_packing1 = NULL;
        test->w_packing1 = NULL;
        test->x_packing2 = NULL;
        test->w_packing2 = NULL;
    }

    fclose(measurement);
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printf("Usage: %s version", argv[0]);
        return EXIT_FAILURE;
    } else {
        return benchmark_random(false, argv[1]);
    }
}
