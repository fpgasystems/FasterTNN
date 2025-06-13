#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "utils.h"
#include "tnns/data.h"
#include "tnns/tnns_cpu.h"
#include "tnns/common.h"
#include "timing/shapes_e2e.h"
#include "timing/shapes_layers.h"
#include "timing/shapes_test.h"

#define OPTIMIZE_1x1

#ifdef AVX2
#define ternarize_opt ternarize_opt_avx2
#define binarize_opt binarize_opt_avx2
#else 
#ifdef ARM_NEON
#define ternarize_opt ternarize_opt_arm_neon
#define binarize_opt binarize_opt_arm_neon
#endif
#endif

inline void extract_test_params(testcase_t *test, int shape[PARAMCOUNT]) {
    test->channelcount = shape[0];
    test->inputH = shape[1];
    test->inputW = shape[2];
    test->kernelcount = shape[3];
    test->kernelW = shape[4];
    test->kernelH = shape[5];
    test->paddingW = shape[6];
    test->paddingH = shape[6];
    test->strideH = shape[7];
    test->strideW = shape[7];
}

static long long MAX_INPUT_SIZE = 128ll*4096*14*14;
static long long MAX_NUM_KERNELS = 16384;
static long long MAX_KERNEL_SIZE = 16384ll*16384ll;

static int BATCH_SIZE = 8;
static int WARMUP_RUNS = 5;
static int MEASUREMENT_RUNS = 10;

enum shape_type { LAYER, E2E, TEST };
const char* shape_type_to_string(enum shape_type t) {
    switch (t) {
        case LAYER: return "LAYER";
        case E2E: return "E2E";
        case TEST: return "TEST";
        default:  return "UNKNOWN";
    }
}

static int benchmark_layers(char* version, enum tnns_type tnns_type, enum shape_type shape_type) {
    // set up shapes
    int shapecount;
    int* layercounts;
    int (*(*shapes))[PARAMCOUNT];
    switch (shape_type) {
        case LAYER:
            shapecount = shapecount_layers;
            layercounts = layercounts_layers;
            shapes = shapes_layers;
            break;
        case E2E:
            shapecount = shapecount_e2e;
            layercounts = layercounts_e2e;
            shapes = shapes_e2e;
            break;
        case TEST:
            shapecount = shapecount_test;
            layercounts = layercounts_test;
            shapes = shapes_test;
            break;
    }

    // prepare timestamped measurement file
    time_t rawtime = time(NULL);
    struct tm* t = localtime(&rawtime);
    char result_file[300];
    snprintf(result_file, 300, "out/measurements/measurement_%d_%d_%d_%d_%d_%d.csv",t->tm_year, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    FILE* measurement = fopen(result_file, "w");
    if(!measurement) {
        fprintf(stderr, "failed to open measurement output file:\n");
        perror("");
        return EXIT_FAILURE;
    }
    fprintf(measurement, "TNNs-type, Version, Shape-type, Shape, Runtime\n");

    // prepare input, kernel and threshold arrays
    srand(0);
    float* input = generate_array(MAX_INPUT_SIZE, (tnns_type == TNN || tnns_type == TBN));
    float* kernel = generate_array(MAX_KERNEL_SIZE, (tnns_type == TNN || tnns_type == BTN));
    float input_thresholds[BATCH_SIZE];
    float kernel_thresholds[MAX_NUM_KERNELS];
    for (int i = 0; i < BATCH_SIZE; i++) { 
        input_thresholds[i] = ((tnns_type == BTN || tnns_type == BNN) ? 0.0 : 0.5); 
    }
    for (int i = 0; i < MAX_NUM_KERNELS; i++) { 
        kernel_thresholds[i] = ((tnns_type == TBN || tnns_type == BNN) ? 0.0 : 0.5); 
    }

    // iterate over each shape, which in turn may consist of multiple layers
    for (int shape = 0; shape < shapecount; shape++) {
        int layercount = layercounts[shape];

        printf("SHAPE %d\n", shape);

        // prepare testcase
        testcase_t* test = create_testcase();
        test->type = tnns_type;
        test->input = input;
        test->input_thresholds = input_thresholds;
        test->kernel = kernel;
        test->preluAlpha = 0.5;
        test->batchcount = BATCH_SIZE;

        // prepare an array of quantized kernels
        int64_t* quantized_kernels_packing0[layercount];
        int64_t* quantized_kernels_packing2[layercount];
        for (int i = 0; i < layercount; ++i) {
            extract_test_params(test, shapes[shape][i]);
            if (tnns_type == TBN || tnns_type == BNN) {
                quantized_kernels_packing0[i] = binarize_naive(kernel, kernel_thresholds, test->kernelcount, test->channelcount, test->kernelH, test->kernelW, 0, 0);
                quantized_kernels_packing2[i] = binarize_opt(kernel, kernel_thresholds, test->kernelcount, test->channelcount, test->kernelH, test->kernelW);
            } else {
                quantized_kernels_packing0[i] = ternarize_naive(kernel, kernel_thresholds, test->kernelcount, test->channelcount, test->kernelH, test->kernelW, 0, 0);
                quantized_kernels_packing2[i] = ternarize_opt(kernel, kernel_thresholds, test->kernelcount, test->channelcount, test->kernelH, test->kernelW);
            }

            test->quantized_kernel_packing0 = quantized_kernels_packing0[i];
            test->quantized_kernel_packing2 = quantized_kernels_packing2[i];
        }

        // test against reference implementation
        if (shape_type == TEST) {
            for (int i = 0; i < layercount; ++i) {
                extract_test_params(test, shapes[shape][i]);
                test->quantized_kernel_packing0 = quantized_kernels_packing0[i];
                test->quantized_kernel_packing2 = quantized_kernels_packing2[i];

                tnns_conv(test);

                int pad_val = (tnns_type == BTN || tnns_type == BNN) ? 1 : 0;
                float* px = direct_pad(test->input, test->paddingH, test->paddingW, test->batchcount, test->channelcount, test->inputH, test->inputW, pad_val);
                float* ref_y = direct_conv2d(px, test->kernel, test->strideH, test->strideW, test->batchcount, test->channelcount, test->inputH + 2*test->paddingH, test->inputW + 2*test->paddingW, test->kernelcount, test->kernelH, test->kernelW, test->preluAlpha);
                
                if(!check_nhwc(test->output, ref_y, test->batchcount, test->kernelcount, get_output_height(test), get_output_width(test))) {
                    fprintf(stderr, "Function did not return the correct result!\n");
                    fprintf((void *) result_file, "error\n");
                    free(test->output);
                    abort();
                }
                free(test->output);
                test->output = NULL;
                free(px);
                free(ref_y);
            }
        }

        // warm-up CPU
        for (int r = 0; r < WARMUP_RUNS; r++) {
            for (int i = 0; i < layercount; ++i) {
                extract_test_params(test, shapes[shape][i]);
                test->quantized_kernel_packing0 = quantized_kernels_packing0[i];
                test->quantized_kernel_packing2 = quantized_kernels_packing2[i];

                tnns_conv(test);

                free(test->output);
                test->output = NULL;
            }
        }

        // measure average runtime over multiple runs
        struct timeval start, end;
        double seconds = 0;
        for (int r = 0; r < MEASUREMENT_RUNS; r++) {

            gettimeofday(&start, NULL);
            test->input = input;
            for (int i = 0; i < layercount; ++i) {

                extract_test_params(test, shapes[shape][i]);
                test->quantized_kernel_packing0 = quantized_kernels_packing0[i];
                test->quantized_kernel_packing2 = quantized_kernels_packing2[i];

                tnns_conv(test);

                // output of current layer becomes input of next layer
                if (test->input != input) {
                    free(test->input);
                }
                test->input = test->output;
            }
            gettimeofday(&end, NULL);
            seconds += (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6); 

            free(test->output);
            test->output = NULL;
        }
        seconds /= MEASUREMENT_RUNS;
        fprintf(measurement, "%s, %s, %s, %d, %lf\n", tnns_type_to_string(tnns_type), version, shape_type_to_string(shape_type), shape, seconds);

        // free quantized kernels
        for (int i = 0; i < layercount; i++) {
            free(quantized_kernels_packing0[i]);
            free(quantized_kernels_packing2[i]);
        }
    }

    fclose(measurement);

    free(input);
    free(kernel);

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if(argc != 4) {
        printf("Usage: %s version tnns_type shape_type", argv[0]);
        return EXIT_FAILURE;
    } else {
        char* version = argv[1];

        enum tnns_type tnns_type;
        if (strcmp(argv[2], "tnn") == 0) {
            tnns_type = TNN;
        } else if (strcmp(argv[2], "tbn") == 0) {
            tnns_type = TBN;
        } else if (strcmp(argv[2], "btn") == 0) {
            tnns_type = BTN;
        } else if (strcmp(argv[2], "bnn") == 0) {
            tnns_type = BNN;
        } else {
            printf("Invalid tnns_type %s", argv[2]);
            return EXIT_FAILURE;
        }

        enum shape_type shape_type; 
        if (strcmp(argv[3], "layer") == 0) {
            shape_type = LAYER;
        } else if (strcmp(argv[3], "e2e") == 0) {
            shape_type = E2E;
        } else if (strcmp(argv[3], "test") == 0) {
            shape_type = TEST;
        } else {
            printf("Invalid shape_type %s", argv[3]);
            return EXIT_FAILURE;
        }

        return benchmark_layers(version, tnns_type, shape_type);
    }
}
