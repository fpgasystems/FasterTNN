#include "conv_utils.cuh"
#include <iostream>

testcase_t* create_testcase(void) {
    testcase_t* result = (testcase_t*)calloc(1,sizeof(testcase_t));
    return result;
}

void free_testcase(testcase_t* data) {
    cudaFree(data->dev_input);
    cudaFree(data->dev_img2row_output);
    cudaFree(data->dev_input_thresholds);
    cudaFree(data->dev_kernel);
    cudaFree(data->dev_kernel_thresholds);
    cudaFree(data->dev_output);
    cudaFree(data->dev_padded_output);
    cudaFree(data->dev_quantized_kernel);
    cudaFree(data->dev_quantized_output);
    cudaEventDestroy(data->start_event);
    cudaEventDestroy(data->end_event);
    free(data);
}

void setup_conv_data(
    testcase_t* data,
    ConvType ctype,
    int batch_size,
    int shape[8]
) { 
    int quant_size = 32;
    int pack_cnt_a = (ctype == TAB_TNN || ctype == TAB_TBN) ? 2 : 1;
    int pack_cnt_w = (ctype == TAB_BNN || ctype == TAB_TBN) ? 1 : 2;
    // setup metadata field
    data->type = ctype;
    data->batchcount = batch_size;
    data->channelcount = shape[0];
    data->inputH = shape[1];
    data->inputW = shape[2];
    data->kernelcount = shape[3];
    data->kernelW = shape[4];
    data->kernelH = shape[5];
    data->paddingW = shape[6];
    data->paddingH = shape[6];
    data->strideH = shape[7];
    data->strideW = shape[7];

    // for (int i = 0; i < 8; i++)
    //     std::cout << shape[i] << "-";
    // std::cout << "\n";

    int C_pack = ceil(1. * (data->channelcount/quant_size));    // channel number after quantization
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH + 1) / data->strideH;   // output height
    int OW = (W_pad - data->kernelW + 1) / data->strideW;   // output width

    data->packed_channel = C_pack;
    data->paddedH = H_pad;
    data->paddedW = W_pad;
    data->outputH = OH;
    data->outputW = OW;

    // host side kernel allocation
    int kernel_size = data->kernelcount * data->kernelW * data->kernelH * data->channelcount;
    float* kernel = (float*)aligned_alloc(32, kernel_size*sizeof(float));
    float* kernel_threshold = (float*)aligned_alloc(32, data->kernelcount*sizeof(float));
    generate_random_array<float>(kernel, kernel_size, FP32);
    generate_random_array<float>(kernel_threshold, data->kernelcount, FP32);

    // device side kernel allocation
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_kernel, kernel_size * sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_kernel_thresholds, data->kernelcount * sizeof(float)));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_kernel_thresholds, kernel_threshold, data->kernelcount * sizeof(float), cudaMemcpyHostToDevice));

    // host side input allocation
    int input_size = data->batchcount*data->inputH*data->inputW*data->channelcount;
    int threshold_size = data->batchcount;
    float* input = (float*)aligned_alloc(32, input_size*sizeof(float));
    float* threshold = (float*)aligned_alloc(32, threshold_size*sizeof(float));
    generate_random_array<float>(input, input_size, FP32);
    generate_random_array<float>(threshold, threshold_size, FP32);

    // device side input allocation
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_input, input_size * sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_input_thresholds, threshold_size * sizeof(float)));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_input_thresholds, threshold, threshold_size * sizeof(float), cudaMemcpyHostToDevice));
    
    int quantized_input_size = data->batchcount * data->inputH * data->inputW * C_pack * pack_cnt_a;
    int padded_input_size = data->batchcount * H_pad * W_pad * C_pack * pack_cnt_a;
    int img2row_input_size = data->batchcount * H_pad * W_pad * data->kernelW * data->kernelH * C_pack * pack_cnt_a;
    int quantized_kernel_size = data->kernelcount * data->kernelW * data->kernelH * C_pack * pack_cnt_w;
    int output_size = data->batchcount * data->kernelcount * OH * OW;

    // device side allocations
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_quantized_kernel, quantized_kernel_size*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_quantized_output, quantized_input_size*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_padded_output, padded_input_size*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_img2row_output, img2row_input_size*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_output, output_size*sizeof(int32_t)));

    // create events
    cudaEventCreate(&data->start_event);
    cudaEventCreate(&data->end_event);
}

void advance_conv_data(
    testcase_t* data,
    ConvType ctype,
    int32_t* prev_output,
    int batch_size,
    int shape[8],
    int quant_size,
    int pack_cnt
) {
    // setup metadata field
    data->type = ctype;
    data->channelcount = shape[0];
    data->inputH = shape[1];
    data->inputW = shape[2];
    data->kernelcount = shape[3];
    data->kernelW = shape[4];
    data->kernelH = shape[5];
    data->paddingW = shape[6];
    data->paddingH = shape[6];
    data->strideH = shape[7];
    data->strideW = shape[7];

    for (int i = 0; i < 8; i++)
        std::cout << shape[i] << "-";
    std::cout << "\n";

    int C_pack = ceil(1. * (data->channelcount/quant_size));    // channel number after quantization
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH + 1) / data->strideH;   // output height
    int OW = (W_pad - data->kernelW + 1) / data->strideW;   // output width

    data->packed_channel = C_pack;
    data->paddedH = H_pad;
    data->paddedW = W_pad;
    data->outputH = OH;
    data->outputW = OW;

    // setup input
    data->dev_input = reinterpret_cast<float*>(prev_output);

    // host side kernel allocation
    int kernel_size = data->kernelcount * data->kernelW * data->kernelH * data->channelcount;
    float* kernel = (float*)aligned_alloc(32, kernel_size*sizeof(float));
    float* kernel_threshold = (float*)aligned_alloc(32, data->kernelcount*sizeof(float));
    generate_random_array<float>(kernel, kernel_size, FP32);
    generate_random_array<float>(kernel_threshold, data->kernelcount, FP32);

    // device side kernel allocation
    CUDA_CALL_CHECK(cudaFree(data->dev_kernel));
    CUDA_CALL_CHECK(cudaFree(data->dev_kernel_thresholds));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_kernel, kernel_size * sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)&data->dev_kernel_thresholds, data->kernelcount * sizeof(float)));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(data->dev_kernel_thresholds, kernel_threshold, data->kernelcount * sizeof(float), cudaMemcpyHostToDevice));
}