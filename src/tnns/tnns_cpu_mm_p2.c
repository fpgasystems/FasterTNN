#include "tnns/data.h"
#include "tnns/binarize2row.h"
#include "tnns/ternarize2row.h"
#include "tnns/mm.h"

static float* tnn_conv_p2(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t* qx = t2r_p2_opt(x, q_thresholds, N, C, H, W, H_pad, W_pad, KH, KW, H_stride, W_stride);

    // Bitwise GEMM and PReLU activation
    int C_pack = (C + cntbits - 1) / cntbits; // Channels after bit-packing
    int K = C_pack * KH * KW;
    float* y = mm_tnn(qx, q_weights, N, OH, OW, KN, K, prelu_alpha);

    free(qx);

    return y;
}

static float* tbn_conv_p2(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t *qx = t2r_p2_opt(x, q_thresholds, N, C, H, W, H_pad, W_pad, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM and PReLU activation
    int C_pack = (C + cntbits - 1) / cntbits; // Channels after bit-packing
    int K = C_pack * KH * KW;
    float* y = mm_tbn(qx, q_weights, N, OH, OW, KN, K, prelu_alpha);

    free(qx);

    return y;
}

static float* btn_conv_p2(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t *qx = b2r(x, q_thresholds, N, C, H, W, H_pad, W_pad, KH, KW, H_stride, W_stride); 
       
    // Bitwise GEMM and PReLU activation
    int C_pack = (C + cntbits - 1) / cntbits; // Channels after bit-packing
    int K = C_pack * KH * KW;
    float* y = mm_btn(qx, q_weights, N, OH, OW, KN, K, prelu_alpha);

    free(qx);

    return y;
}

static float* bnn_conv_p2(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t *qx = b2r(x, q_thresholds, N, C, H, W, H_pad, W_pad, KH, KW, H_stride, W_stride); 
       
    // Bitwise GEMM and PReLU activation
    int C_pack = (C + cntbits - 1) / cntbits; // Channels afterbit-packing
    int K = C_pack * KH * KW;
    float* y = mm_bnn(qx, q_weights, N, OH, OW, KN, K, C * KH * KW, prelu_alpha);

    free(qx);

    return y;
}


void tnns_conv(testcase_t* data) {
    float* (*tnn_conv_function)(float*, float*, int64_t*, int, int, int, int, int, int, int, int, int, int, int, float) = tnn_conv_p2;

    switch (data->type) {
        case TNN:
            tnn_conv_function = tnn_conv_p2;
            break;
        case TBN:
            tnn_conv_function = tbn_conv_p2;
            break;
        case BTN:
            tnn_conv_function = btn_conv_p2;
            break;
        case BNN:
            tnn_conv_function = bnn_conv_p2;
            break;
    }

    float* result = tnn_conv_function(
        data->input,
        data->input_thresholds,
        data->quantized_kernel_packing2,
        data->paddingH,
        data->paddingW,
        data->strideH,
        data->strideW,
        data->batchcount,
        data->channelcount,
        data->inputH,
        data->inputW,
        data->kernelcount,
        data->kernelH,
        data->kernelW,
        data->preluAlpha
        );

    data->output = result;
}

void tnns_gemm(testcase_tnns_gemm_t* data) {
    float prelu_alpha = 1;
    float* result = mm_tnn(data->x_packing2, data->w_packing2, data->m, 1, 1, data->n, data->p_packed, prelu_alpha);

    if(data->output) {
        int output_len = data->m * data->n;
        memcpy(data->output, result, output_len * sizeof(float));
        free(result);
    } else {
        data->output = result;
    }
}
