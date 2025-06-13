#include "tnns/mm.h"

#ifdef AVX2
#include <nmmintrin.h>
#include <immintrin.h>

/**
 * Extracted from libpopcnt.h
*/
static inline __m256i popcnt256(__m256i v)
{
  __m256i lookup1 = _mm256_setr_epi8(
      4, 5, 5, 6, 5, 6, 6, 7,
      5, 6, 6, 7, 6, 7, 7, 8,
      4, 5, 5, 6, 5, 6, 6, 7,
      5, 6, 6, 7, 6, 7, 7, 8
  );

  __m256i lookup2 = _mm256_setr_epi8(
      4, 3, 3, 2, 3, 2, 2, 1,
      3, 2, 2, 1, 2, 1, 1, 0,
      4, 3, 3, 2, 3, 2, 2, 1,
      3, 2, 2, 1, 2, 1, 1, 0
  );

  __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i lo = _mm256_and_si256(v, low_mask);
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
  __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
  __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);

  return _mm256_sad_epu8(popcnt1, popcnt2);
}

/**
 * Extracted from libpopcnt.h
*/
static inline void CSA256(__m256i* h, __m256i* l, __m256i a, __m256i b, __m256i c)
{
  __m256i u = _mm256_xor_si256(a, b);
  *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
  *l = _mm256_xor_si256(u, c);
}

static inline void CSA256_custom3(__m256i* h, __m256i* l, __m256i a, __m256i neg, __m256i zero)
{ 
  // h = and(a, nonpos) or and(a xor nonpos, neg) 
  //   = and(a, nonpos) or and(not a, neg) 
  //   = and(a, nonpos) or neg
  //   = and(a, zero) or neg
  // l = xor(a, nonpos, neg) = xor(a, zero)
  *h = _mm256_or_si256(_mm256_and_si256(a, zero), neg);
  *l = _mm256_xor_si256(a, zero);
}

static inline void gemm_nano_packing2_opt3_4_1(int64_t* a, int64_t* b, int* y, int K, int N, int KB, float alpha, bool last) {

    __m256i cnt_0_0 = _mm256_setzero_si256();
    __m256i ones_0_0 = _mm256_setzero_si256();
    __m256i twos_0_0 = _mm256_setzero_si256();
    __m256i fours_0_0 = _mm256_setzero_si256();
    __m256i eights_0_0 = _mm256_setzero_si256();
    __m256i twosA_0_0;
    __m256i cnt_1_0 = _mm256_setzero_si256();
    __m256i ones_1_0 = _mm256_setzero_si256();
    __m256i twos_1_0 = _mm256_setzero_si256();
    __m256i fours_1_0 = _mm256_setzero_si256();
    __m256i eights_1_0 = _mm256_setzero_si256();
    __m256i twosA_1_0;
    __m256i cnt_2_0 = _mm256_setzero_si256();
    __m256i ones_2_0 = _mm256_setzero_si256();
    __m256i twos_2_0 = _mm256_setzero_si256();
    __m256i fours_2_0 = _mm256_setzero_si256();
    __m256i eights_2_0 = _mm256_setzero_si256();
    __m256i twosA_2_0;
    __m256i cnt_3_0 = _mm256_setzero_si256();
    __m256i ones_3_0 = _mm256_setzero_si256();
    __m256i twos_3_0 = _mm256_setzero_si256();
    __m256i fours_3_0 = _mm256_setzero_si256();
    __m256i eights_3_0 = _mm256_setzero_si256();
    __m256i twosA_3_0;
    int k = 0;
    int limit4 = K - K % (BITS * 4);
    assert(limit4 == K && "K must be divisible by 4 * BITS at least");
    for (; k < limit4; k += (BITS*4)) {
        __m256i a0_sign_0_0 = _mm256_loadu_si256((__m256i*)(a + 0*KB + k + 8*0));
        __m256i a0_zero_0_0 = _mm256_loadu_si256((__m256i*)(a + 0*KB + k + 8*0 + 4));
        __m256i b0_sign_0_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0));
        __m256i b0_zero_0_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0 + 4));
        __m256i p0_zero_0_0 = _mm256_or_si256(a0_zero_0_0, b0_zero_0_0);
        __m256i p0_sign_0_0 = _mm256_xor_si256(a0_sign_0_0, b0_sign_0_0);
        __m256i p0_neg_0_0  = _mm256_andnot_si256(p0_zero_0_0, p0_sign_0_0);
        CSA256_custom3(&twosA_0_0, &ones_0_0, ones_0_0, p0_neg_0_0, p0_zero_0_0);

        twos_0_0 = _mm256_add_epi64(twos_0_0, popcnt256(twosA_0_0));
        __m256i a0_sign_1_0 = _mm256_loadu_si256((__m256i*)(a + 1*KB + k + 8*0));
        __m256i a0_zero_1_0 = _mm256_loadu_si256((__m256i*)(a + 1*KB + k + 8*0 + 4));
        __m256i b0_sign_1_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0));
        __m256i b0_zero_1_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0 + 4));
        __m256i p0_zero_1_0 = _mm256_or_si256(a0_zero_1_0, b0_zero_1_0);
        __m256i p0_sign_1_0 = _mm256_xor_si256(a0_sign_1_0, b0_sign_1_0);
        __m256i p0_neg_1_0  = _mm256_andnot_si256(p0_zero_1_0, p0_sign_1_0);
        CSA256_custom3(&twosA_1_0, &ones_1_0, ones_1_0, p0_neg_1_0, p0_zero_1_0);

        twos_1_0 = _mm256_add_epi64(twos_1_0, popcnt256(twosA_1_0));
        __m256i a0_sign_2_0 = _mm256_loadu_si256((__m256i*)(a + 2*KB + k + 8*0));
        __m256i a0_zero_2_0 = _mm256_loadu_si256((__m256i*)(a + 2*KB + k + 8*0 + 4));
        __m256i b0_sign_2_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0));
        __m256i b0_zero_2_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0 + 4));
        __m256i p0_zero_2_0 = _mm256_or_si256(a0_zero_2_0, b0_zero_2_0);
        __m256i p0_sign_2_0 = _mm256_xor_si256(a0_sign_2_0, b0_sign_2_0);
        __m256i p0_neg_2_0  = _mm256_andnot_si256(p0_zero_2_0, p0_sign_2_0);
        CSA256_custom3(&twosA_2_0, &ones_2_0, ones_2_0, p0_neg_2_0, p0_zero_2_0);

        twos_2_0 = _mm256_add_epi64(twos_2_0, popcnt256(twosA_2_0));
        __m256i a0_sign_3_0 = _mm256_loadu_si256((__m256i*)(a + 3*KB + k + 8*0));
        __m256i a0_zero_3_0 = _mm256_loadu_si256((__m256i*)(a + 3*KB + k + 8*0 + 4));
        __m256i b0_sign_3_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0));
        __m256i b0_zero_3_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0 + 4));
        __m256i p0_zero_3_0 = _mm256_or_si256(a0_zero_3_0, b0_zero_3_0);
        __m256i p0_sign_3_0 = _mm256_xor_si256(a0_sign_3_0, b0_sign_3_0);
        __m256i p0_neg_3_0  = _mm256_andnot_si256(p0_zero_3_0, p0_sign_3_0);
        CSA256_custom3(&twosA_3_0, &ones_3_0, ones_3_0, p0_neg_3_0, p0_zero_3_0);

        twos_3_0 = _mm256_add_epi64(twos_3_0, popcnt256(twosA_3_0));
    }

    __m256i s1_0_0 = _mm256_add_epi64(_mm256_slli_epi64(eights_0_0, 3), _mm256_slli_epi64(fours_0_0, 2));
    __m256i s2_0_0 = _mm256_add_epi64(_mm256_slli_epi64(twos_0_0, 1), popcnt256(ones_0_0));
    cnt_0_0 = _mm256_add_epi64(_mm256_slli_epi64(cnt_0_0, 4), _mm256_add_epi64(s1_0_0, s2_0_0));

    uint64_t* cnt64_0_0;
    cnt64_0_0 = (uint64_t*) &cnt_0_0;

    *(y + 0*N + 0) += K*32 - (cnt64_0_0[0] + cnt64_0_0[1] + cnt64_0_0[2] + cnt64_0_0[3]);
    __m256i s1_1_0 = _mm256_add_epi64(_mm256_slli_epi64(eights_1_0, 3), _mm256_slli_epi64(fours_1_0, 2));
    __m256i s2_1_0 = _mm256_add_epi64(_mm256_slli_epi64(twos_1_0, 1), popcnt256(ones_1_0));
    cnt_1_0 = _mm256_add_epi64(_mm256_slli_epi64(cnt_1_0, 4), _mm256_add_epi64(s1_1_0, s2_1_0));

    uint64_t* cnt64_1_0;
    cnt64_1_0 = (uint64_t*) &cnt_1_0;

    *(y + 1*N + 0) += K*32 - (cnt64_1_0[0] + cnt64_1_0[1] + cnt64_1_0[2] + cnt64_1_0[3]);
    __m256i s1_2_0 = _mm256_add_epi64(_mm256_slli_epi64(eights_2_0, 3), _mm256_slli_epi64(fours_2_0, 2));
    __m256i s2_2_0 = _mm256_add_epi64(_mm256_slli_epi64(twos_2_0, 1), popcnt256(ones_2_0));
    cnt_2_0 = _mm256_add_epi64(_mm256_slli_epi64(cnt_2_0, 4), _mm256_add_epi64(s1_2_0, s2_2_0));

    uint64_t* cnt64_2_0;
    cnt64_2_0 = (uint64_t*) &cnt_2_0;

    *(y + 2*N + 0) += K*32 - (cnt64_2_0[0] + cnt64_2_0[1] + cnt64_2_0[2] + cnt64_2_0[3]);
    __m256i s1_3_0 = _mm256_add_epi64(_mm256_slli_epi64(eights_3_0, 3), _mm256_slli_epi64(fours_3_0, 2));
    __m256i s2_3_0 = _mm256_add_epi64(_mm256_slli_epi64(twos_3_0, 1), popcnt256(ones_3_0));
    cnt_3_0 = _mm256_add_epi64(_mm256_slli_epi64(cnt_3_0, 4), _mm256_add_epi64(s1_3_0, s2_3_0));

    uint64_t* cnt64_3_0;
    cnt64_3_0 = (uint64_t*) &cnt_3_0;

    *(y + 3*N + 0) += K*32 - (cnt64_3_0[0] + cnt64_3_0[1] + cnt64_3_0[2] + cnt64_3_0[3]);
    if(last) {
        float activation_0_0 = (float) *(y + 0*N + 0);
        if(activation_0_0 <= 0) {
            activation_0_0 *= alpha;
        }
        float* fview_0_0 = (float*)(y + 0*N + 0);
        *fview_0_0 = activation_0_0;
        float activation_1_0 = (float) *(y + 1*N + 0);
        if(activation_1_0 <= 0) {
            activation_1_0 *= alpha;
        }
        float* fview_1_0 = (float*)(y + 1*N + 0);
        *fview_1_0 = activation_1_0;
        float activation_2_0 = (float) *(y + 2*N + 0);
        if(activation_2_0 <= 0) {
            activation_2_0 *= alpha;
        }
        float* fview_2_0 = (float*)(y + 2*N + 0);
        *fview_2_0 = activation_2_0;
        float activation_3_0 = (float) *(y + 3*N + 0);
        if(activation_3_0 <= 0) {
            activation_3_0 *= alpha;
        }
        float* fview_3_0 = (float*)(y + 3*N + 0);
        *fview_3_0 = activation_3_0;
    }
}

static inline void gemm_nano_packing2_opt3_1_1(int64_t* a, int64_t* b, int* y, int K, int N, int KB, float alpha, bool last) {

    __m256i cnt_0_0 = _mm256_setzero_si256();
    __m256i ones_0_0 = _mm256_setzero_si256();
    __m256i twos_0_0 = _mm256_setzero_si256();
    __m256i fours_0_0 = _mm256_setzero_si256();
    __m256i eights_0_0 = _mm256_setzero_si256();
    __m256i twosA_0_0;
    int k = 0;
    int limit4 = K - K % (BITS * 4);
    assert(limit4 == K && "K must be divisible by 4 * BITS at least");
    for (; k < limit4; k += (BITS*4)) {
        __m256i a0_sign_0_0 = _mm256_loadu_si256((__m256i*)(a + 0*KB + k + 8*0));
        __m256i a0_zero_0_0 = _mm256_loadu_si256((__m256i*)(a + 0*KB + k + 8*0 + 4));
        __m256i b0_sign_0_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0));
        __m256i b0_zero_0_0 = _mm256_loadu_si256((__m256i*)(b + 0*KB + k + 8*0 + 4));
        __m256i p0_zero_0_0 = _mm256_or_si256(a0_zero_0_0, b0_zero_0_0);
        __m256i p0_sign_0_0 = _mm256_xor_si256(a0_sign_0_0, b0_sign_0_0);
        __m256i p0_neg_0_0  = _mm256_andnot_si256(p0_zero_0_0, p0_sign_0_0);
        CSA256_custom3(&twosA_0_0, &ones_0_0, ones_0_0, p0_neg_0_0, p0_zero_0_0);

        twos_0_0 = _mm256_add_epi64(twos_0_0, popcnt256(twosA_0_0));
    }

    __m256i s1_0_0 = _mm256_add_epi64(_mm256_slli_epi64(eights_0_0, 3), _mm256_slli_epi64(fours_0_0, 2));
    __m256i s2_0_0 = _mm256_add_epi64(_mm256_slli_epi64(twos_0_0, 1), popcnt256(ones_0_0));
    cnt_0_0 = _mm256_add_epi64(_mm256_slli_epi64(cnt_0_0, 4), _mm256_add_epi64(s1_0_0, s2_0_0));

    uint64_t* cnt64_0_0;
    cnt64_0_0 = (uint64_t*) &cnt_0_0;

    *(y + 0*N + 0) += K*32 - (cnt64_0_0[0] + cnt64_0_0[1] + cnt64_0_0[2] + cnt64_0_0[3]);
    if(last) {
        float activation_0_0 = (float) *(y + 0*N + 0);
        if(activation_0_0 <= 0) {
            activation_0_0 *= alpha;
        }
        float* fview_0_0 = (float*)(y + 0*N + 0);
        *fview_0_0 = activation_0_0;
    }
}

static inline void micro_mm_p2_lib_1pc(int64_t* a, int64_t* b, int* y, int I, int J, int K, int N, int KB, float alpha, bool last) {
    #define i_stride 4
    #define j_stride 1
    int i_limit = I - I % i_stride;
    int j_limit = J - J % j_stride;
    int i = 0;
    for (; i < i_limit; i += i_stride) {
        int j = 0;
        for (; j < j_limit; j += j_stride) {
            gemm_nano_packing2_opt3_4_1(a + i*KB, b+j*KB, y + i*N + j, K, N, KB, alpha, last);
        }
        for(; j < J; ++j) {
            gemm_nano_packing2_opt3_4_1(a + i*KB, b+j*KB, y + i*N + j, K, N, KB, alpha, last);
        }
    }
    for (; i < I; ++i) {
        int j = 0;
        for(; j < j_limit; j += j_stride) {
            gemm_nano_packing2_opt3_1_1(a + i*KB, b+j*KB, y + i*N + j, K, N, KB, alpha, last);
        }
        for(; j < J; ++j) {
            gemm_nano_packing2_opt3_1_1(a + i*KB, b+j*KB, y + i*N + j, K, N, KB, alpha, last);
        }
    }
}

static float* mm_p2_lib_csas(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    mm_blocked(M, N, KB, micro_mm_p2_lib_1pc)
}

/**
 * Optimization 8:
 * We switch to a new packing to get rid of one bitwise instruction in the innermost loop
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p2_lib_csas(a, b, N*OH*OW, KN, K,prelu_alpha);
}

#endif