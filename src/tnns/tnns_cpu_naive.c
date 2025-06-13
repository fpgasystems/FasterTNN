#include <string.h>
#include "tnns/data.h"
#include "utils.h"

static float* prelu_naive(int* x, int N, int C, int H, int W, float alpha) {
    float* y = calloc(N * C * H * W, sizeof(float));

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float current = (float) x[((n * C + c) * H + h) * W + w];
                    if (current > 0)
                        y[((n * C + c) * H + h) * W + w] = current;
                    else
                        y[((n * C + c) * H + h) * W + w] = current * alpha;
                }
            }
        }
    }

    return y;
}

static int64_t* img2row_naive(int64_t* qx_img, int N, int C, int H, int W, int KH, int KW, int H_stride, int W_stride) {
    const int OH = (H - KH) / H_stride + 1;
    const int OW = (W - KW) / W_stride + 1;
    const int H_fused = OH * OW;
    const int W_fused = KH * KW * C; 
    int alloc_size = (N * H_fused * W_fused) * sizeof(int64_t);
    alloc_size = alloc_size % 32 == 0 ? alloc_size : alloc_size + 32 - (alloc_size % 32);
    int64_t* qx_row = aligned_alloc(32, alloc_size);
    memset(qx_row, 0, alloc_size);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int c = 0; c < C; c++) {
                            // qx_row[N, OH, OW, KH, KW, C] = qx_img[N, H+kh, W+kw, C]
                            qx_row[(n * H_fused + oh * OW + ow) * W_fused + kh * KW * C + kw * C + c] = qx_img[((n * H + oh * H_stride + kh) * W + ow * W_stride + kw) * C + c];
                        }
                    }
                }
            }
        }
    }

    return qx_row;
}

static int* gemm_p0_naive(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int KB = K * BITS;

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                for (int c = 0; c < KN; c++) {
                    int cntp1 = 0;
                    int cntp2 = 0;
                    for (int ik = 0; ik < KB; ik += BITS) {
                        int64_t p1 = a[((n * OH + h) * OW + w) * KB + ik + 0] ^ b[c * KB + ik + 0]; 
                        int64_t p2 = a[((n * OH + h) * OW + w) * KB + ik + 1] & b[c * KB + ik + 1];
                        cntp1 = cntp1 + popcnt64(p2);
                        cntp2 = cntp2 + popcnt64(p1 & p2);
                    }
                    // output format: (N, KN, OH, OW) to match (N, C, H, W)
                    y[((n * KN + c) * OH + h) * OW + w] = cntp1 - cntp2 - cntp2;
                }
            }
        }
    }
    return y;
}

static int* gemm_tbn_p0_naive(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int KB = K * BITS;

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                for (int c = 0; c < KN; c++) {
                    int cntp1 = 0;
                    int cntp2 = 0;
                    for (int ik = 0; ik < KB; ik += BITS) {
                        int64_t p1 = a[((n * OH + h) * OW + w) * KB + ik + 0] ^ b[(c * KB + ik) / BITS]; 
                        int64_t p2 = a[((n * OH + h) * OW + w) * KB + ik + 1];
                        cntp1 = cntp1 + popcnt64(p2);
                        cntp2 = cntp2 + popcnt64(p1 & p2);
                    }
                    // output format: (N, KN, OH, OW) to match (N, C, H, W)
                    y[((n * KN + c) * OH + h) * OW + w] = cntp1 - cntp2 - cntp2;
                }
            }
        }
    }
    return y;
}

static int* gemm_btn_p0_naive(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int KB = K * BITS;

    // TAB-BTN precomputes an array of cntp1
    int* cntp1 = calloc(KN, sizeof(int));
    for (int c = 0; c < KN; c++) {
        int c1 = 0;
        for (int ik = 0; ik < KB; ik += BITS) {
            int64_t p2 = b[c * KB + ik + 1];
            c1 = c1 + popcnt64(p2);
        }
        cntp1[c] = c1;
    }

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < OH; h++) {
            for (int w = 0; w < OW; w++) {
                for (int c = 0; c < KN; c++) {
                    int cntp2 = 0;
                    for (int ik = 0; ik < KB; ik += BITS) {
                        int64_t p1 = a[(((n * OH + h) * OW + w) * KB + ik) / BITS] ^ b[c * KB + ik + 0]; 
                        int64_t p2 = b[c * KB + ik + 1];
                        cntp2 = cntp2 + popcnt64(p1 & p2);
                    }
                    // output format: (N, KN, OH, OW) to match (N, C, H, W)
                    y[((n * KN + c) * OH + h) * OW + w] = cntp1[c] - cntp2 - cntp2;
                }
            }
        }
    }
    return y;
}

static int* gemm_bnn_p0_naive(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, int K_unpack) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int kn = 0; kn < KN; kn++) {
                    int cntp2 = 0;
                    for (int ik = 0; ik < K; ik++) {
                        int64_t p1 = a[((n * OH + oh) * OW + ow) * K + ik] ^ b[kn * K + ik]; 
                        cntp2 = cntp2 + popcnt64(p1);
                    }
                    // output format: (N, KN, OH, OW) to match (N, C, H, W)
                    y[((n * KN + kn) * OH + oh) * OW + ow] = K_unpack - cntp2 - cntp2;
                }
            }
        }
    }

    return y;
}

static float* tnn_conv_naive(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int C_pack = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // Channels afterbit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t* qxi = ternarize_naive(x, q_thresholds, N, C, H, W, H_pad, W_pad);
    int64_t* qx = img2row_naive(qxi, N, C_pack * BITS, H_pack, W_pack, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM and PReLU activation
    int* yi = gemm_p0_naive(qx, q_weights, N, OH, OW, KN, C_pack * KH * KW);

    // Activation function: PReLU
    float* y = prelu_naive(yi, N, KN, OH, OW, prelu_alpha);

    free(qxi);
    free(qx);
    free(yi);

    return y;
}

static float* tbn_conv_naive(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int C_pack = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // Channels afterbit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t* qxi = ternarize_naive(x, q_thresholds, N, C, H, W, H_pad, W_pad);
    int64_t* qx = img2row_naive(qxi, N, C_pack * BITS, H_pack, W_pack, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM and PReLU activation
    int* yi = gemm_tbn_p0_naive(qx, q_weights, N, OH, OW, KN, C_pack * KH * KW);

    // Activation function: PReLU
    float* y = prelu_naive(yi, N, KN, OH, OW, prelu_alpha);

    free(qxi);
    free(qx);
    free(yi);

    return y;
}

static float* btn_conv_naive(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int C_pack = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // Channels afterbit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t* qxi = binarize_naive(x, q_thresholds, N, C, H, W, H_pad, W_pad);
    int64_t* qx = img2row_naive(qxi, N, C_pack, H_pack, W_pack, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM and PReLU activation
    int* yi = gemm_btn_p0_naive(qx, q_weights, N, OH, OW, KN, C_pack * KH * KW);

    // Activation function: PReLU
    float* y = prelu_naive(yi, N, KN, OH, OW, prelu_alpha);

    free(qxi);
    free(qx);
    free(yi);

    return y;
}

static float* bnn_conv_naive(float* x, float* q_thresholds, int64_t* q_weights, int H_pad, int W_pad, int H_stride, int W_stride, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    int H_pack = H + 2 * H_pad; // Height after bit-packing
    int W_pack = W + 2 * W_pad; // Width  after bit-packing
    int C_pack = (C % cntbits) ? ((C / cntbits) + 1) : (C / cntbits); // Channels afterbit-packing
    int OH = (H_pack - KH) / H_stride + 1; // Output height
    int OW = (W_pack - KW) / W_stride + 1; // Output width

    // Quantize and Img2Row
    int64_t* qxi = binarize_naive(x, q_thresholds, N, C, H, W, H_pad, W_pad);
    int64_t* qx = img2row_naive(qxi, N, C_pack, H_pack, W_pack, KH, KW, H_stride, W_stride);
       
    // Bitwise GEMM
    int* yi = gemm_bnn_p0_naive(qx, q_weights, N, OH, OW, KN, C_pack * KH * KW, C * KH * KW);

    // Activation function: PReLU
    float* y = prelu_naive(yi, N, KN, OH, OW, prelu_alpha);

    free(qxi);
    free(qx);
    free(yi);

    return y;
}

void tnns_conv(testcase_t* data) {
    float* (*tnn_conv_function)(float*, float*, int64_t*, int, int, int, int, int, int, int, int, int, int, int, float) = tnn_conv_naive;

    switch (data->type) {
        case TNN:
            tnn_conv_function = tnn_conv_naive;
            break;
        case TBN:
            tnn_conv_function = tbn_conv_naive;
            break;
        case BTN:
            tnn_conv_function = btn_conv_naive;
            break;
        case BNN:
            tnn_conv_function = bnn_conv_naive;
            break;
    }

    float* result = tnn_conv_function(
            data->input,
            data->input_thresholds,
            data->quantized_kernel_packing0,
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
    float prelu_alpha = 0.5;
    int* gemm_result = gemm_p0_naive(data->x_packing0, data->w_packing0, data->m, 1, 1, data->n, data->p_packed);
    float* result = prelu_naive(gemm_result, data->m, data->n, 1, 1, prelu_alpha);

    if(data->output) {
        int output_len = data->m * data->n;
        memcpy(data->output, result, output_len * sizeof(float));
        free(result);
    } else {
        data->output = result;
    }
}
