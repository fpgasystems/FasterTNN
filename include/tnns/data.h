#pragma once
#include <tnns/common.h>

typedef struct testcase {
    enum tnns_type type;
    int batchcount;
    int channelcount;
    int inputW;
    int inputH;
    int kernelcount;
    int kernelW;
    int kernelH;
    int paddingW;
    int paddingH;
    int strideW;
    int strideH;
    float preluAlpha;
    float* input_thresholds;
    float* input;
    float* kernel;
    int64_t* quantized_kernel_packing0;
    int64_t* quantized_kernel_packing1;
    int64_t* quantized_kernel_packing2;
    float* output;
} testcase_t;

typedef struct testcase_tnns_gemm {
    int m;
    int p;
    int p_packed;
    int n;
    float* x;
    float* w;
    int64_t* x_packing0;
    int64_t* x_packing1;
    int64_t* x_packing2;
    int64_t* w_packing0;
    int64_t* w_packing1;
    int64_t* w_packing2;
    float* output;
} testcase_tnns_gemm_t;

void free_testcase(testcase_t* testcase);
testcase_t* create_testcase(void);
int get_input_len(testcase_t* testcase);
int get_kernel_len(testcase_t* testcase);
int get_output_len(testcase_t* testcase);
int get_output_width(testcase_t* testcase);
int get_output_height(testcase_t* testcase);

void free_testcase_gemm(testcase_tnns_gemm_t* testcase);
testcase_tnns_gemm_t* create_testcase_gemm(void);