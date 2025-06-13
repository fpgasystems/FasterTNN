#include "utils.h"
#include <assert.h>

/* 
 * Generate a pseudorandom array of given size. 
 * The elements will be in {-1, 0, 1} if ternary is `true` and in {-1, 1} otherwise. 
 */
float* generate_array(int64_t size, bool ternary) {
    if (size * sizeof(float) % 32 != 0) {
        fprintf(stderr, "Tried to generate array of size not divisible by 32\n");
        return NULL;
    }
    printf("Generating array of size %llu bytes\n", (unsigned long long int)(size * sizeof(float)));
    float *arr = (float *)aligned_alloc(32, size * sizeof(float));
    if (!arr) {
        printf("Error generating array: malloc returned null!\n");
        return 0;
    }

    if (ternary) {
        for (int64_t i=0; i<size; i++) {
            arr[i] = (rand() % 3) - 1;
        }
    }
    else {
        for (int64_t i=0; i<size; i++) {
            arr[i] = 2*(rand() % 2) - 1;
        }
    }

    return arr;
}

/*
 * Compare two tensors in the NHWC format.
 */
bool check_nhwc(float* X, float* X2, int N, int C, int H, int W) {
    assert(X2 && "Ref was null");
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    // Use N_H_W_C format
                    float xx = X[((n * H + h) * W + w) * C + c] - X2[((n * H + h) * W + w) * C + c];
                    if ((xx > 0.01) || (xx < -0.01)) {
                        printf("Check failed in: n: %i, h: %i, w: %i, c: %i\n", n, h, w, c);
                        printf("X1: %f, X2: %f\n -> diff: %f\n", X[((n * H + h) * W + w) * C + c], X2[((n * H + h) * W + w) * C + c], xx);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/*
 * Compare two tensors in the NCHW format.
 */
bool check_nchw(float* X, float* X2, int N, int C, int H, int W) {
    assert(X2 && "Ref was null");
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    // Use N_C_H_W format
                    float xx = X[((n * C + c) * H + h) * W + w] - X2[((n * C + c) * H + h) * W + w];
                    if ((xx > 0.01) || (xx < -0.01)) {
                        printf("Check failed in: n: %i, c: %i, h: %i, w: %i\n", n, c, h, w);
                        printf("X1: %f, X2: %f\n -> diff: %f\n", X[((n * C + c) * H + h) * W + w], X2[((n * C + c) * H + h) * W + w], xx);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/*
 * Compare two integer matrices.
 */
bool check_matrices(float* X1, float* X2, int m, int n) {
    assert(X2 && "Ref was null");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float diff = X1[i*n + j] - X2[i*n + j];
            if (diff > 0.001 || diff < -0.001) {
                printf("Check failed in: m: %i, n: %i; i: %i, j: %i\n", m, n, i, j);
                printf("X1: %f, X2: %f\n -> diff: %f\n", X1[i*n + j], X2[i*n + j], diff);
                return false;
            }
        }
    }
    return true;
}

/*
 * Pad an image with a given padding value.
 */
float *direct_pad(float* x, int padding1, int padding2, int N, int C, int H, int W, int pad_val) {
    const int packH = H + 2 * padding1;
    const int packW = W + 2 * padding2;
    float *qx = (float *)calloc(N * C * packH * packW, sizeof(float));
    if (pad_val != 0) {
        for (int i = 0; i < N * C * packH * packW; i++) {
            qx[i] = pad_val;
        }
    }

    for (int in = 0; in < N; in++) {
        for (int ic = 0; ic < C; ic++) {
            for (int ih = 0; ih < H; ih++) {
                for (int iw = 0; iw < W; iw++) {
                    qx[((in * C + ic) * packH + (ih + padding1)) * packW + iw + padding2] = x[((in * C + ic) * H + ih) * W + iw];
                }
            }
        }
    }

    return qx;
}

/*
 * Perform a 2D convolution in N_C_H_W format.
 */
float *direct_conv2d(float* x, float* w, int stride1, int stride2, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha) {
    const int FH = (H - KH) / stride1 + 1;
    const int FW = (W - KW) / stride2 + 1;

    float *y = (float *)malloc(N * KN * FH * FW * sizeof(float));

    for (int on = 0; on < N; on++) {
        for (int kn = 0; kn < KN; kn++) {
            for (int oh = 0; oh < FH; oh++) {
                for (int ow = 0; ow < FW; ow++) {
                    float sum = 0;
                    for (int kc = 0; kc < C; kc++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                // Use N_C_H_W format
                                sum += x[((on * C + kc) * H + (oh * stride1 + kh)) * W + ow * stride2 + kw] * w[((kn * C + kc) * KH + kh) * KW + kw];
                            }
                        }
                    }
                    // y[((on * FH + oh) * FW + ow) * KN + kn] = sum >= 0 ? sum : sum * prelu_alpha;
                    // Use N_C_H_W format
                    y[((on * KN + kn) * FH + oh) * FW + ow] = sum >= 0 ? sum : sum * prelu_alpha;
                }
            }
        }
    }

    return y;
}

/*
 * Compute the matrix product of two ternary integer (-1, 0, 1) matrices X and W. 
 * We assume that W is stored in column-major format (or equivalently, W^T is stored in row-major format).
 */
float *direct_mmm(float* x, float* w, int m, int n, int p) {

    float *y = (float *) calloc(m * n, sizeof(float));

    if (!y) { return NULL; }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                y[i*n + j] += x[i*p + k] * w[j*p + k];
            }
        }
    }

    return y;
}

/*
 * Convert a matrix to bitpacking format 0 (64-bit sign vector followed by 64-bit abs vector)
 */
int64_t *bitpacking0(float* x, int I, int J) {
    assert(J % 64 == 0 && "J needs to be divisible by 64");

    int64_t *y = (int64_t *) aligned_alloc(8, I * J / 32 * sizeof(int64_t));
    if (!y) {
        printf("Error generating array: malloc returned null!\n");
        return 0;
    }

    for (int i = 0; i < I; i++) {
        for (int j0 = 0; j0 < J; j0 += 64) {
            int start_sign = i*J/32 + 2*(j0 / 64);
            int start_abs = start_sign + 1;
            y[start_sign] = 0L;
            y[start_abs] = 0L;
            for (int j1 = 0; j1 < 64; j1++) {
                float val = x[i*J + j0 + j1];
                int64_t mask = ((int64_t) 1) << j1;
                if (val == -1) {
                    y[start_sign] |= mask;
                } 
                if (val != 0) {
                    y[start_abs] |= mask;
                }
            }
        }
    }

    return y;
}

/*
 * Convert a matrix to bitpacking format 1 (256-bit sign vector followed by 256-bit abs vector)
 * assuming the input satisfies suitnnle divisibility constraints.
 */
int64_t *bitpacking1(float* x, int I, int J) {
    assert(J % 256 == 0 && "J needs to be divisible by 256");

    int64_t *y = (int64_t *) aligned_alloc(32, I * J / 32 * sizeof(int64_t));
    if (!y) {
        printf("Error generating array: malloc returned null!\n");
        return 0;
    }

    for (int i = 0; i < I; i++) {
        for (int j0 = 0; j0 < J; j0 += 256) {
            int start_sign = i*J/32 + 8*(j0 / 256);
            int start_abs = start_sign + 4;
            y[start_sign  ] = 0L;
            y[start_sign+1] = 0L;
            y[start_sign+2] = 0L;
            y[start_sign+3] = 0L;
            y[start_abs   ] = 0L;
            y[start_abs +1] = 0L;
            y[start_abs +2] = 0L;
            y[start_abs +3] = 0L;
            for (int j1 = 0; j1 < 256; j1++) {
                float val = x[i*J + j0 + j1];
                int64_t mask = ((int64_t) 1) << (j1 % 64);
                if (val == -1) {
                    y[start_sign + j1/64] |= mask;
                } 
                if (val != 0) {
                    y[start_abs + j1/64] |= mask;
                }
            }
        }
    }

    return y;
}

/*
 * Convert a matrix to bitpacking format 2 (256-bit sign vector followed by 256-bit zero vector)
 * assuming the input satisfies suitnnle divisibility constraints.
 */
int64_t *bitpacking2(float* x, int I, int J) {
    assert(J % 256 == 0 && "J needs to be divisible by 256");

    int64_t *y = (int64_t *) aligned_alloc(32, I * J / 32 * sizeof(int64_t));
    if (!y) {
        printf("Error generating array: malloc returned null!\n");
        return 0;
    }

    for (int i = 0; i < I; i++) {
        for (int j0 = 0; j0 < J; j0 += 256) {
            int start_sign = i*J/32 + 8*(j0 / 256);
            int start_zero = start_sign + 4;
            y[start_sign  ] = 0L;
            y[start_sign+1] = 0L;
            y[start_sign+2] = 0L;
            y[start_sign+3] = 0L;
            y[start_zero  ] = 0L;
            y[start_zero+1] = 0L;
            y[start_zero+2] = 0L;
            y[start_zero+3] = 0L;
            for (int j1 = 0; j1 < 256; j1++) {
                float val = x[i*J + j0 + j1];
                int64_t mask = ((int64_t) 1) << (j1 % 64);
                if (val == -1) {
                    y[start_sign + j1/64] |= mask;
                } else if (val == 0) {
                    y[start_zero + j1/64] |= mask;
                }
            }
        }
    }

    return y;
}


/*
  Quantize the input x to be {+1, 0, -1} using N,H,W,C,B format based on the thresholds.
*/
int64_t* ternarize_naive(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int H_pack = H + 2 * H_pad;
    const int W_pack = W + 2 * W_pad;

    // quantized qx, in N_H_W_C_B format
    size_t alloc_size = (N * H_pack * W_pack * C_pack * BITS) * sizeof(int64_t);
    alloc_size = alloc_size % 32 == 0 ? alloc_size : alloc_size + 32 - (alloc_size % 32);
    int64_t* qx = (int64_t*)aligned_alloc(32, alloc_size);
    memset(qx, 0, alloc_size);

    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                // pack the first part: 0 ~ priChannel*cntbits
                for (int ic = 0; ic < priChannel; ic++) {
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < cntbits; bit++) {
                        float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
                        if (currentx > q_thresholds[in]) {
                            p2 = p2 | onebit[bit]; // Pack 1: 01
                        } else if (currentx < -q_thresholds[in]) {
                            p1 = p1 | onebit[bit]; // Pack -1: 11
                            p2 = p2 | onebit[bit];
                        }
                    }
                    // store the ternarized and packed data in N_H_W_C_B format
                    //qx.index({ in, ih + H_pad, iw + W_pad, priChannel * 2 + 0 }) = p1;
                    //qx.index({ in, ih + H_pad, iw + W_pad, priChannel * 2 + 1 }) = p2;
                    qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 0] = p1;
                    qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 1] = p2;
                }

                // pack the tail: priChannel*cntbits ~ C
                if ((C % cntbits) > 0) {
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < (C % cntbits); bit++) {
                        float currentx = x[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw];
                        if (currentx > q_thresholds[in]) {
                            p2 = p2 | onebit[bit]; // Pack 1: 01
                        }
                        else if (currentx < (-q_thresholds[in])) {
                            p1 = p1 | onebit[bit]; // Pack -1: 11
                            p2 = p2 | onebit[bit];
                        }
                    }

                    qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + priChannel) * BITS + 0] = p1;
                    qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + priChannel) * BITS + 1] = p2;
                }
            }
        }
    }
    return qx;
}

/*
 * Packing 1: Instead of packing the sign and abs bitvectors of each packed channel alternatingly,
 * pack four sign bitvectors and four abs bitvectors alternatingly.
 * This works under our divisibility constraint that 256 divides C. 
 * Used in gemm-variants/mm_v{5,6,7}_p1_{...}.c
 */
int64_t* ternarize_packing1(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int H_pack = H + 2 * H_pad;
    const int W_pack = W + 2 * W_pad;

    // quantized qx, in N_H_W_C_B format
    int64_t* qx = (int64_t*)aligned_alloc(32, (N * H_pack * W_pack * C_pack * BITS) * sizeof(int64_t));
    memset(qx, 0, (N * H_pack * W_pack * C_pack * BITS) * sizeof(int64_t));

    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int ic = 0; ic < priChannel; ic += 4) {
                    for (int ic0 = 0; ic0 < 4; ic0++) {
                        int64_t p1 = 0;
                        int64_t p2 = 0;
                        for (int bit = 0; bit < cntbits; bit++) {
                            float currentx = x[((in * C + ((ic + ic0) * cntbits + bit)) * H + ih) * W + iw];
                            if (currentx > q_thresholds[in]) {
                                p2 = p2 | onebit[bit]; // Pack 1: 01
                            }
                            else if (currentx < (-q_thresholds[in])) {
                                p1 = p1 | onebit[bit]; // Pack -1: 11
                                p2 = p2 | onebit[bit];
                            }
                        }

                        qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 0 + ic0] = p1;
                        qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 4 + ic0] = p2;
                    }
                }
                // divisibility constraints: we assume there is no tail
            }
        }
    }
    return qx;
}


/*
 * Ternarize a matrix to bitpacking format 2 (256-bit sign vector followed by 256-bit zero vector) 
 * assuming that there is no padding and that 256 divides C.
 */
int64_t* ternarize_packing2(float* x, float* q_thresholds, int N, int C, int H, int W) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    const int H_pad = 0;
    const int W_pad = 0;

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int H_pack = H + 2 * H_pad;
    const int W_pack = W + 2 * W_pad;

    // quantized qx, in N_H_W_C_B format
    int64_t* qx = (int64_t*)aligned_alloc(32, (N * H_pack * W_pack * C_pack * BITS) * sizeof(int64_t));
    memset(qx, 0, (N * H_pack * W_pack * C_pack * BITS) * sizeof(int64_t));

    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int ic = 0; ic < priChannel; ic += 4) {
                    for (int ic0 = 0; ic0 < 4; ic0++) {
                        int64_t p1 = 0;
                        int64_t p2 = 0;
                        for (int bit = 0; bit < cntbits; bit++) {
                            float currentx = x[((in * C + ((ic + ic0) * cntbits + bit)) * H + ih) * W + iw];
                            if (currentx > q_thresholds[in]) {
                               // Pack 1: 00
                            } else { if (currentx < (-q_thresholds[in])) {
                                p1 = p1 | onebit[bit]; // Pack -1: 10
                            } else {
                                p2 = p2 | onebit[bit]; // Pack 0: 01
                            } }
                        }
                        qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 0 + ic0] = p1;
                        qx[(((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic) * BITS + 4 + ic0] = p2;
                    }
                }
            }
        }
    }
    return qx;
}

/*
 * Optimized ternarization for AVX2. 
 * We take the product H*W*C_pack and pad it to be a multiple of 4 such that it matches the SIMD width 4x64 = 256 bit.
 */
int64_t* ternarize_opt_avx2(float* x, float* q_thresholds, int N, int C, int H, int W) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int K_padded = 4 * ((H * W * C_pack + 3) / 4);
    const int K_fused = BITS * K_padded;

    // quantized qx, in N_H_W_C_B format
    int64_t* qx = (int64_t*)aligned_alloc(32, N * K_fused * sizeof(int64_t));

    for (int i = 0; i < N * K_fused; i += 8) {
        for (int j1 = 0; j1 < 4; j1++) {
            qx[i + j1] = 0;
        }
        for (int j2 = 4; j2 < 8; j2++) {
            qx[i + j2] = -1;
        }
    }
    
    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int ic = 0; ic < C_pack; ic++) {
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < cntbits; bit++) {
                        float currentx = 0.0;
                        if (ic < priChannel || (C%cntbits > 0 && bit < C%cntbits)) {
                            currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
                        }
                        if (currentx > q_thresholds[in]) {
                            // Pack 1: 00
                        } else { if (currentx < (-q_thresholds[in])) {
                            p1 = p1 | onebit[bit]; // Pack -1: 10
                        } else {
                            p2 = p2 | onebit[bit]; // Pack 0: 01
                        } }
                    }
                    int idx = (ih*W+iw)*C_pack + ic;
                    int qx_idx = (idx / 4) * 4 * BITS + (idx % 4);
                    qx[(in * K_fused) + qx_idx + 0] = p1; 
                    qx[(in * K_fused) + qx_idx + 4] = p2;
                }
            }
        }
    }

    return qx;
}

/*
 * Optimized ternarization for ARM Neon. 
 * We take the product H*W*C_pack and pad it to be a multiple of 2 such that it matches the SIMD width 2x64 = 128 bit.
 */
int64_t* ternarize_opt_arm_neon(float* x, float* q_thresholds, int N, int C, int H, int W) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int K_padded = 2 * ((H * W * C_pack + 1) / 2);
    const int K_fused = BITS * K_padded;

    // quantized qx, in N_H_W_C_B format
    int64_t* qx = (int64_t*)aligned_alloc(32, N * K_fused * sizeof(int64_t));
    memset(qx, 0, N * K_fused * sizeof(int64_t));
    
    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int ic = 0; ic < C_pack; ic++) {
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < cntbits; bit++) {
                        float currentx = 0.0;
                        if (ic < priChannel || (C%cntbits > 0 && bit < C%cntbits)) {
                            currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
                        }
                        if (currentx > q_thresholds[in]) {
                            p2 = p2 | onebit[bit]; // Pack 1: 01
                        } else { if (currentx < (-q_thresholds[in])) {
                            p1 = p1 | onebit[bit]; // Pack -1: 11
                            p2 = p2 | onebit[bit]; 
                        }}
                    }
                    int idx = (ih*W+iw)*C_pack + ic;
                    int qx_idx = (idx / 2) * 2 * BITS + (idx % 2);
                    qx[(in * K_fused) + qx_idx + 0] = p1; 
                    qx[(in * K_fused) + qx_idx + 2] = p2;
                }
            }
        }
    }

    return qx;
}

/*
 * Binarize the input according to the thresholds.
 */
int64_t* binarize_naive(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int H_pack = H + 2 * H_pad;
    const int W_pack = W + 2 * W_pad;

    // quantized qx, in N_H_W_C format
    size_t alloc_size = (N * H_pack * W_pack * C_pack) * sizeof(int64_t);
    alloc_size = alloc_size % 32 == 0 ? alloc_size : alloc_size + 32 - (alloc_size % 32);
    int64_t* qx = (int64_t*)aligned_alloc(32, alloc_size);

    for (int i = 0; i < (N * H_pack * W_pack * C_pack); i++) {
        qx[i] = 0L;
    }

    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                // Pack the first part: 0 ~ priChannel*cntbits
                for (int ic = 0; ic < priChannel; ic++) {
                    int64_t p1 = 0L;
                    for (int bit = 0; bit < cntbits; bit++) {
                        float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
                        if (currentx < q_thresholds[in]) {
                            p1 = p1 | onebit[bit]; // Pack -1: 1
                        } 
                    }
                    qx[((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + ic] = p1;
                }

                // Pack the tail: priChannel*cntbits ~ C
                if ((C % cntbits) > 0) {
                    int64_t p1 = 0L;
                    for (int bit = 0; bit < (C % cntbits); bit++) {
                        float currentx = x[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw];
                        if (currentx < q_thresholds[in]) {
                            p1 = p1 | onebit[bit]; // Pack -1: 1
                        } 
                    }
                    qx[((in * H_pack + ih + H_pad) * W_pack + iw + W_pad) * C_pack + priChannel] = p1;
                }
            }
        }
    }

    return qx;
}

/*
 * Optimized binarization. 
 * We take the product H*W*C_pack and pad it to be a multiple of the SIMD width given by the parameter packing. 
 */
int64_t* binarize_opt_helper(float* x, float* q_thresholds, int N, int C, int H, int W, int packing) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int K_fused = packing * ((H * W * C_pack + packing-1) / packing);

    // quantized qx, in N_H_W_C_B format
    int64_t* qx = (int64_t*)aligned_alloc(32, N * K_fused * sizeof(int64_t));
    memset(qx, 0, N * K_fused * sizeof(int64_t));
    
    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int ic = 0; ic < C_pack; ic++) {
                    int64_t p1 = 0;
                    for (int bit = 0; bit < cntbits; bit++) {
                        if (ic < priChannel || (C%cntbits > 0 && bit < C%cntbits)) {
                            float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];                            
                            if (currentx < q_thresholds[in]) {
                                p1 = p1 | onebit[bit]; // Pack -1: 1
                            } 
                        }
                    }
                    int idx = (ih*W+iw)*C_pack + ic;
                    qx[(in * K_fused) + idx] = p1;
                }
            }
        }
    }

    return qx;
}

/* Optimized binarization for AVX2 with SIMD width 4x64 = 256.  */
int64_t* binarize_opt_avx2(float* x, float* q_thresholds, int N, int C, int H, int W) {
    return binarize_opt_helper(x, q_thresholds, N, C, H, W, 4);
}

/* Optimized binarization for ARM Neon with SIMD width 2x64 = 128.  */
int64_t* binarize_opt_arm_neon(float* x, float* q_thresholds, int N, int C, int H, int W) {
    return binarize_opt_helper(x, q_thresholds, N, C, H, W, 2);
}
