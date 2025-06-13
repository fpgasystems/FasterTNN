#include "tnns/data.h"
#include "tnns/ternarize2row.h"
#include "tnns/mm.h"

static float* tnn_conv_p1(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int C_pack = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // Channels after bit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t *qx = t2r_p1(x, q_thresholds, N, C, H, W, H_pad, W_pad, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM and PReLU activation
    float* y = mm_tnn(qx, q_weights, N, OH, OW, KN, C_pack * KH * KW, prelu_alpha);

    free(qx);

    return y;
}

void tnns_conv(testcase_t* data) {
    float* result = tnn_conv_p1(
        data->input,
        data->input_thresholds,
        data->quantized_kernel_packing1,
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
    if(data->output) {
        int output_len = get_output_len(data);
        memcpy(data->output, result, output_len * sizeof(float));
        free(result);
    } else {
        data->output = result;
    }
}

void tnns_gemm(testcase_tnns_gemm_t* data) {
    float prelu_alpha = 1;
    float* result = mm_tnn(data->x_packing1, data->w_packing1, data->m, 1, 1, data->n, data->p_packed, prelu_alpha);

    if(data->output) {
        int output_len = data->m * data->n;
        memcpy(data->output, result, output_len * sizeof(float));
        free(result);
    } else {
        data->output = result;
    }
}