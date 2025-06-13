#include "tnns/data.h"

void free_testcase(testcase_t* testcase) {
    if(testcase->input_thresholds) free(testcase->input_thresholds);
    if(testcase->input) free(testcase->input);
    if(testcase->kernel) free(testcase->kernel);
    if(testcase->quantized_kernel_packing0) free(testcase->quantized_kernel_packing0);
    if(testcase->quantized_kernel_packing1) free(testcase->quantized_kernel_packing1);
    if(testcase->quantized_kernel_packing2) free(testcase->quantized_kernel_packing2);
    if(testcase->output) free(testcase->output);
    free(testcase);
}

testcase_t* create_testcase(void) {
    testcase_t* result = (testcase_t*)calloc(1,sizeof(testcase_t));
    return result;
}

int get_input_len(testcase_t* testcase) {
    return testcase->batchcount * testcase->channelcount * testcase->inputH * testcase->inputW;
}

int get_kernel_len(testcase_t* testcase) {
    return testcase->kernelcount * testcase->kernelH * testcase->kernelW * testcase->channelcount;
}

int get_output_len(testcase_t* testcase) {
    int H_pack = testcase->inputH + 2 * testcase->paddingH;
    int W_pack = testcase->inputW + 2 * testcase->paddingW;
    int OH = (H_pack - testcase->kernelH) / testcase->strideH + 1;
    int OW = (W_pack - testcase->kernelW) / testcase->strideW + 1;
    return OH * OW * testcase->kernelcount * testcase->batchcount;
}

int get_output_width(testcase_t* testcase) {
    int W_pack = testcase->inputW + 2 * testcase->paddingW;
    int OW = (W_pack - testcase->kernelW) / testcase->strideW + 1;
    return OW;
}

int get_output_height(testcase_t* testcase) {
    int H_pack = testcase->inputH + 2 * testcase->paddingH;
    int OH = (H_pack - testcase->kernelH) / testcase->strideH + 1;
    return OH;
}

void free_testcase_gemm(testcase_tnns_gemm_t* testcase) {
    if(testcase->x) free(testcase->x);
    if(testcase->w) free(testcase->w);
    if(testcase->x_packing0) free(testcase->x_packing0);
    if(testcase->x_packing1) free(testcase->x_packing1);
    if(testcase->x_packing2) free(testcase->x_packing2);
    if(testcase->w_packing0) free(testcase->w_packing0);
    if(testcase->w_packing1) free(testcase->w_packing1);
    if(testcase->w_packing2) free(testcase->w_packing2);
    if(testcase->output) free(testcase->output);
    free(testcase);
}

testcase_tnns_gemm_t* create_testcase_gemm(void) {
    testcase_tnns_gemm_t* result = (testcase_tnns_gemm_t*)calloc(1,sizeof(testcase_tnns_gemm_t));
    return result;
}