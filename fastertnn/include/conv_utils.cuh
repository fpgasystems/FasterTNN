#ifndef CONV_UTILS_CUH
#define CONV_UTILS_CUH
#include <random>
#include <assert.h>
#include "utils.cuh"

enum QuanType {
    Binary,
    Ternary
};

enum ConvType {
    FP32,
    TAB_TNN,
    TAB_TBN,
    TAB_BTN,
    TAB_BNN
};

template<typename T>
void generate_random_array(
    T* dst, 
    int size, 
    ConvType type
) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution_tab(-1, 1);
    std::uniform_int_distribution<int> distribution_fp(-10, 10);

    switch (type)
    {
    /* generate ternary: -1, 0, 1 */
    case TAB_TNN:
        for (int i = 0; i < size; i++) {
            dst[i] = distribution_tab(generator);
        }
        break;
    /* generate binary: -1, 1 */
    case TAB_BNN:
        for (int i = 0; i < size; i++) {
            int tmp = distribution_tab(generator);
            dst[i] = tmp == 0 ? 1 : tmp;
        }
        break;
    default:
        for (int i = 0; i < size; i++) {
            dst[i] = ((T)distribution_fp(generator)/2);
        }
        break;
    }
}

typedef struct testcase {
    enum ConvType type;
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

    //
    int packed_channel;
    int paddedH;
    int paddedW;
    int outputH;
    int outputW;

    // host side
    float* input_thresholds;
    float* input;
    float* kernel;
    int32_t* quantized_kernel_packing0;
    int32_t* quantized_kernel_packing1;
    int32_t* quantized_kernel_packing2;
    int32_t* output;

    // device side
    float* dev_input_thresholds;
    float* dev_kernel_thresholds;
    float* dev_input;
    float* dev_kernel;
    int32_t* dev_quantized_kernel;
    int32_t* dev_quantized_output;
    int32_t* dev_padded_output;
    int32_t* dev_img2row_output;
    int32_t* dev_output;

    cudaEvent_t start_event;
    cudaEvent_t end_event;
} testcase_t;

testcase_t* create_testcase(void);
void free_testcase(testcase_t* data);

void setup_conv_data(
    testcase_t* data,
    ConvType ctype,
    int batch_size,
    int shape[8]
);

void advance_conv_data(
    testcase_t* data,
    ConvType ctype,
    int32_t* prev_output,
    int batch_size,
    int shape[8],
    int quant_size,
    int pack_cnt
);

#endif /* CONV_UTILS_CUH */