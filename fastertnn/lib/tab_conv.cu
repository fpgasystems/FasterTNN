#include "tab_quan.cuh"
#include "sm80_tnn_gemm.cuh"
#include "sm80_tbn_gemm.cuh"
#include "sm80_bnn_gemm.cuh"
#include "sm80_btn_gemm.cuh"
#include "tab_conv.h"


void tab_conv(testcase_t* data, bool baseline) {
    switch (data->type)
    {
    case ConvType::TAB_TNN:
        tnn_conv(data, baseline);
        break;
    case ConvType::TAB_BNN:
        bnn_conv(data, baseline);
        break;
    case ConvType::TAB_TBN:
        tbn_conv(data, baseline);
        break;
    case ConvType::TAB_BTN:
        btn_conv(data, baseline);
        break;
    default:
        break;
    }
}

void pre_conv(
    testcase_t* data
) {
    // quantize weight
    if (data->type == TAB_BNN || data->type == TAB_TBN) {
        tab_quan::binary_quantization(
            data->dev_quantized_kernel,
            data->dev_kernel,
            data->dev_kernel_thresholds,
            data->kernelcount,
            data->kernelH,
            data->kernelW,
            data->channelcount
        );
    } else {
        tab_quan::ternary_quantization(
            data->dev_quantized_kernel,
            data->dev_kernel,
            data->dev_kernel_thresholds,
            data->kernelcount,
            data->kernelH,
            data->kernelW,
            data->channelcount
        );
    }
}

void tnn_conv(
    testcase_t* data,
    bool baseline
) {
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH) / data->strideH + 1; // Output height
    int OW = (W_pad - data->kernelW) / data->strideW + 1; // Output width
    int C_pack = ceil(1.*data->channelcount / tab_quan::BIT_PACK_SIZE); // Channels after bit-packing
    int K = C_pack * data->kernelH * data->kernelW;

    cudaEventRecord(data->start_event);
    tab_quan::ternary_quantization(
        data->dev_quantized_output,
        data->dev_input,
        data->dev_input_thresholds,
        data->batchcount,
        data->inputH,
        data->inputW,
        data->channelcount
    );

    tab_quan::pad<int32_t>(
        data->dev_padded_output,
        data->dev_quantized_output,
        data->batchcount,
        data->inputH,
        data->inputW, 
        C_pack*tab_quan::BIT_PACK_COUNT,
        data->paddingH,
        data->paddingW
    );

    tab_quan::img2row<int32_t>(
        data->dev_img2row_output,
        data->dev_padded_output,
        data->batchcount,
        H_pad,
        W_pad,
        C_pack*tab_quan::BIT_PACK_COUNT,
        data->kernelH,
        data->kernelW,
        data->strideH,
        data->strideW
    );

    // std::cout << data->batchcount * OH * OW << " " << data->kernelcount << " " << K << "\n";
    if (baseline) {
        sm80_tnn::tnn_gemm_baseline(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    } else {
        sm80_tnn::sm80_tnn_gemm_m8n8k128(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    }
    cudaEventRecord(data->end_event);
    cudaEventSynchronize(data->end_event);
    float conv_ms = 0;
    cudaEventElapsedTime(&conv_ms, data->start_event, data->end_event);
    std::cout << "End..." << conv_ms << "ms\n";
}

void bnn_conv(
    testcase_t* data,
    bool baseline
) {
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH) / data->strideH + 1; // Output height
    int OW = (W_pad - data->kernelW) / data->strideW + 1; // Output width
    int C_pack = ceil(1.*data->channelcount / tab_quan::BIT_PACK_SIZE); // Channels after bit-packing
    int K = C_pack * data->kernelH * data->kernelH;

    cudaEventRecord(data->start_event);
    tab_quan::binary_quantization(
        data->dev_quantized_output,
        data->dev_input,
        data->dev_input_thresholds,
        data->batchcount,
        data->inputH,
        data->inputW,
        data->channelcount
    );

    tab_quan::pad<int32_t>(
        data->dev_padded_output,
        data->dev_quantized_output,
        data->batchcount,
        data->inputH,
        data->inputW, 
        C_pack,
        data->paddingH,
        data->paddingW
    );

    tab_quan::img2row<int32_t>(
        data->dev_img2row_output,
        data->dev_padded_output,
        data->batchcount,
        H_pad,
        W_pad,
        C_pack,
        data->kernelH,
        data->kernelW,
        data->strideH,
        data->strideW
    );

    if (baseline) {
        sm80_bnn::bnn_gemm_baseline(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K,
            data->channelcount * data->kernelH * data->kernelW
        );
    } else {
        sm80_bnn::sm80_bnn_gemm_m8n8k128(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K,
            data->channelcount * data->kernelH * data->kernelW
        );
    }
    cudaEventRecord(data->end_event);
    cudaEventSynchronize(data->end_event);
    float conv_ms = 0;
    cudaEventElapsedTime(&conv_ms, data->start_event, data->end_event);
    // std::cout << "End..." << conv_ms << "ms\n";
}

void tbn_conv(
    testcase_t* data,
    bool baseline
) {
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH) / data->strideH + 1; // Output height
    int OW = (W_pad - data->kernelW) / data->strideW + 1; // Output width
    int C_pack = ceil(1.*data->channelcount / tab_quan::BIT_PACK_SIZE); // Channels after bit-packing
    int K = C_pack * data->kernelH * data->kernelH;

    cudaEventRecord(data->start_event);
    tab_quan::ternary_quantization(
        data->dev_quantized_output,
        data->dev_input,
        data->dev_input_thresholds,
        data->batchcount,
        data->inputH,
        data->inputW,
        data->channelcount
    );

    tab_quan::pad<int32_t>(
        data->dev_padded_output,
        data->dev_quantized_output,
        data->batchcount,
        data->inputH,
        data->inputW, 
        C_pack*tab_quan::BIT_PACK_COUNT,
        data->paddingH,
        data->paddingW
    );

    tab_quan::img2row<int32_t>(
        data->dev_img2row_output,
        data->dev_padded_output,
        data->batchcount,
        H_pad,
        W_pad,
        C_pack*tab_quan::BIT_PACK_COUNT,
        data->kernelH,
        data->kernelW,
        data->strideH,
        data->strideW
    );

    if (baseline) {
        sm80_tbn::tbn_gemm_baseline(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    } else {
        sm80_tbn::sm80_tbn_gemm_m8n8k128(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    }
    cudaEventRecord(data->end_event);
    cudaEventSynchronize(data->end_event);
    float conv_ms = 0;
    cudaEventElapsedTime(&conv_ms, data->start_event, data->end_event);
    // std::cout << "End..." << conv_ms << "ms\n";
}

void btn_conv(
    testcase_t* data,
    bool baseline
) {
    int H_pad = data->inputH + 2 * data->paddingH; // Height after bit-packing
    int W_pad = data->inputW + 2 * data->paddingW; // Width after bit-packing
    int OH = (H_pad - data->kernelH) / data->strideH + 1; // Output height
    int OW = (W_pad - data->kernelW) / data->strideW + 1; // Output width
    int C_pack = ceil(1.*data->channelcount / tab_quan::BIT_PACK_SIZE); // Channels after bit-packing
    int K = C_pack * data->kernelH * data->kernelH;

    cudaEventRecord(data->start_event);
    tab_quan::binary_quantization(
        data->dev_quantized_output,
        data->dev_input,
        data->dev_input_thresholds,
        data->batchcount,
        data->inputH,
        data->inputW,
        data->channelcount
    );

    tab_quan::pad<int32_t>(
        data->dev_padded_output,
        data->dev_quantized_output,
        data->batchcount,
        data->inputH,
        data->inputW, 
        C_pack,
        data->paddingH,
        data->paddingW
    );

    tab_quan::img2row<int32_t>(
        data->dev_img2row_output,
        data->dev_padded_output,
        data->batchcount,
        H_pad,
        W_pad,
        C_pack,
        data->kernelH,
        data->kernelW,
        data->strideH,
        data->strideW
    );

    if (baseline) {
        sm80_btn::btn_gemm_baseline(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    } else {
        sm80_btn::sm80_btn_gemm_multi_stage(
            data->dev_img2row_output,
            data->dev_quantized_kernel,
            data->dev_output,
            data->batchcount * OH * OW,
            data->kernelcount,
            K
        );
    }
    cudaEventRecord(data->end_event);
    cudaEventSynchronize(data->end_event);
    float conv_ms = 0;
    cudaEventElapsedTime(&conv_ms, data->start_event, data->end_event);
    // std::cout << "End..." << conv_ms << "ms\n";
}