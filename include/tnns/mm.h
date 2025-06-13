#pragma once
#include "tnns/common.h"
#include <assert.h>

float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha);
float* mm_tbn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha);
float* mm_btn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha);
float* mm_bnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, int K_unpack, float prelu_alpha);


#define mm_blocked_transposed(i1s, i2s, iks, kernel) \
    int M1 = N * OH * OW; \
    int OHW = OH * OW; \
    int M2 = KN; \
    int* y = calloc(N * KN * OH * OW, sizeof(int)); \
    const int KB = K * BITS; \
    const int i1_stride = i1s; \
    const int i2_stride = i2s; \
    const int ik_stride = iks; \
    const int i1_limit = M1 - M1 % i1_stride; \
    const int i2_limit = M2 - M2 % i2_stride; \
    const int ik_limit = KB - KB % ik_stride; \
    \
    int i1 = 0; \
    for (; i1 < i1_limit; i1 += i1_stride) { \
        int i2 = 0; \
        for (; i2 < i2_limit; i2 += i2_stride) { \
            int ik = 0; \
            for (; ik < ik_limit; ik += ik_stride) { \
				kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), i1_stride, i2_stride, ik_stride, N, M1, M2, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), i1_stride, i2_stride, KB % ik_stride, N, M1, M2, KB, prelu_alpha, true); \
            } \
        } \
    \
        if(M2 % i2_stride != 0) { \
            int ik = 0; \
            for (; ik < ik_limit; ik += ik_stride) { \
				kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), i1_stride, M2 % i2_stride, ik_stride, N, M1, M2, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), i1_stride, M2 % i2_stride, KB % ik_stride, N, M1, M2, KB, prelu_alpha, true); \
            } \
        } \
    } \
    \
    if (M1 % i1_stride != 0) { \
        int i2 = 0; \
        for (; i2 < i2_limit; i2 += i2_stride) { \
            int ik = 0; \
            for (; ik < ik_limit; ik += ik_stride) { \
				kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), M1 % i1_stride, i2_stride, ik_stride, N, M1, M2, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), M1 % i1_stride, i2_stride, KB % ik_stride, N, M1, M2, KB, prelu_alpha, true); \
            } \
        } \
    \
        if(M2 % i2_stride != 0) { \
            int ik = 0; \
            for (; ik < ik_limit; ik += ik_stride) { \
				kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), M1 % i1_stride, M2 % i2_stride, ik_stride, N, M1, M2, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + i1*KB + ik, b + i2*KB + ik, y + ((i1/OHW) * M2 + i2) * OHW + (i1%OHW), M1 % i1_stride, M2 % i2_stride, KB % ik_stride, N, M1, M2, KB, prelu_alpha, true); \
            } \
        } \
    } \
    \
    return (float*)y; 



#define mm_blocked(ohs, ows, iks, kernel) \
    int* y = calloc(M * N, sizeof(int)); \
    const int KB = K * BITS; \
    const int oh_stride = ohs; \
    const int ow_stride = ows; \
    const int ik_stride = iks; \
    const int h_limit = M - M % oh_stride; \
    const int w_limit = N - N % ow_stride; \
    const int k_limit = KB - KB % ik_stride; \
    \
    int oh = 0; \
    for (; oh < h_limit; oh += oh_stride) { \
        int ow = 0; \
        for (; ow < w_limit; ow += ow_stride) { \
            int ik = 0; \
            for (; ik < k_limit; ik += ik_stride) { \
				kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, oh_stride, ow_stride, ik_stride, N, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, oh_stride, ow_stride, KB % ik_stride, N, KB, prelu_alpha, true); \
            } \
        } \
    \
        if(N % ow_stride != 0) { \
            int ik = 0; \
            for (; ik < k_limit; ik += ik_stride) { \
				kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, oh_stride, N % ow_stride, ik_stride, N, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, oh_stride, N % ow_stride, KB % ik_stride, N, KB, prelu_alpha, true); \
            } \
        } \
    } \
    \
    if (M % oh_stride != 0) { \
        int ow = 0; \
        for (; ow < w_limit; ow += ow_stride) { \
            int ik = 0; \
            for (; ik < k_limit; ik += ik_stride) { \
				kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, M % oh_stride, ow_stride, ik_stride, N, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, M % oh_stride, ow_stride, KB % ik_stride, N, KB, prelu_alpha, true); \
            } \
        } \
    \
        if(N % ow_stride != 0) { \
            int ik = 0; \
            for (; ik < k_limit; ik += ik_stride) { \
				kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, M % oh_stride, N % ow_stride, ik_stride, N, KB, prelu_alpha, ik == KB - ik_stride); \
			} \
            if (KB % ik_stride != 0) { \
			    kernel(a + oh*KB + ik, b + ow*KB + ik, y + oh*N + ow, M % oh_stride, N % ow_stride, KB % ik_stride, N, KB, prelu_alpha, true); \
            } \
        } \
    } \
    \
    return (float*)y; 
