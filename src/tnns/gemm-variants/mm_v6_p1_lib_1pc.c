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

static inline void micro_mm_p1_lib_1pc(int64_t* a, int64_t* b, int* y, int I, int J, int K, int N, int KB, float alpha, bool last) {
    __m256i negator = _mm256_set1_epi8(-1);
    
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            __m256i cnt = _mm256_setzero_si256();
            __m256i ones = _mm256_setzero_si256();
            __m256i twos = _mm256_setzero_si256();
            __m256i fours = _mm256_setzero_si256();
            __m256i eights = _mm256_setzero_si256();
            __m256i twosA;
            __m256i twosB, foursA;
            __m256i foursB, eightsA;
            int k = 0;
            int limit4 = K - K % (BITS * 4);
            assert(limit4 == K && "K must be divisible by 4 * BITS at least");
            int limit3 = K - K % (BITS * 8);
            int limit2 = K - K % (BITS * 16);
            for (; k < limit2; k+=(BITS*16)) {
                __m256i a0_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0));
                __m256i a0_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0 + 4));
                __m256i b0_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0));
                __m256i b0_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0 + 4));

                __m256i p0_abs = _mm256_and_si256(a0_abs, b0_abs);
                __m256i p0_sign = _mm256_xor_si256(a0_sign, b0_sign);
                __m256i p0_zero = _mm256_xor_si256(p0_abs, negator);
                __m256i p0_nonpos = _mm256_or_si256(p0_zero, p0_sign);
                __m256i p0_neg = _mm256_and_si256(p0_abs, p0_sign);

                __m256i a1_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*1));
                __m256i a1_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*1 + 4));
                __m256i b1_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*1));
                __m256i b1_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*1 + 4));

                __m256i p1_abs = _mm256_and_si256(a1_abs, b1_abs);
                __m256i p1_sign = _mm256_xor_si256(a1_sign, b1_sign);
                __m256i p1_zero = _mm256_xor_si256(p1_abs, negator);
                __m256i p1_nonpos = _mm256_or_si256(p1_zero, p1_sign);
                __m256i p1_neg = _mm256_and_si256(p1_abs, p1_sign);

                __m256i a2_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*2));
                __m256i a2_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*2 + 4));
                __m256i b2_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*2));
                __m256i b2_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*2 + 4));

                __m256i p2_abs = _mm256_and_si256(a2_abs, b2_abs);
                __m256i p2_sign = _mm256_xor_si256(a2_sign, b2_sign);
                __m256i p2_zero = _mm256_xor_si256(p2_abs, negator);
                __m256i p2_nonpos = _mm256_or_si256(p2_zero, p2_sign);
                __m256i p2_neg = _mm256_and_si256(p2_abs, p2_sign);

                __m256i a3_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*3));
                __m256i a3_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*3 + 4));
                __m256i b3_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*3));
                __m256i b3_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*3 + 4));

                __m256i p3_abs = _mm256_and_si256(a3_abs, b3_abs);
                __m256i p3_sign = _mm256_xor_si256(a3_sign, b3_sign);
                __m256i p3_zero = _mm256_xor_si256(p3_abs, negator);
                __m256i p3_nonpos = _mm256_or_si256(p3_zero, p3_sign);
                __m256i p3_neg = _mm256_and_si256(p3_abs, p3_sign);

                CSA256(&twosA, &ones, ones, p0_neg, p0_nonpos);
                CSA256(&twosB, &ones, ones, p1_neg, p1_nonpos);
                CSA256(&foursA, &twos, twos, twosA, twosB);

                CSA256(&twosA, &ones, ones, p2_neg, p2_nonpos);
                CSA256(&twosB, &ones, ones, p3_neg, p3_nonpos);
                CSA256(&foursB, &twos, twos, twosA, twosB);

                CSA256(&eightsA, &fours, fours, foursA, foursB);

                eights = _mm256_add_epi64(eights, popcnt256(eightsA));
            }

            fours = popcnt256(fours);
            for (; k < limit3; k += (BITS*8)) {
                __m256i a0_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0));
                __m256i a0_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0 + 4));
                __m256i b0_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0));
                __m256i b0_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0 + 4));

                __m256i p0_abs = _mm256_and_si256(a0_abs, b0_abs);
                __m256i p0_sign = _mm256_xor_si256(a0_sign, b0_sign);
                __m256i p0_zero  = _mm256_xor_si256(p0_abs, negator);
                __m256i p0_nonpos  = _mm256_or_si256(p0_zero, p0_sign);
                __m256i p0_neg  = _mm256_and_si256(p0_abs, p0_sign);

                __m256i a1_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*1));
                __m256i a1_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*1 + 4));
                __m256i b1_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*1));
                __m256i b1_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*1 + 4));

                __m256i p1_abs = _mm256_and_si256(a1_abs, b1_abs);
                __m256i p1_sign = _mm256_xor_si256(a1_sign, b1_sign);
                __m256i p1_zero  = _mm256_xor_si256(p1_abs, negator);
                __m256i p1_nonpos  = _mm256_or_si256(p1_zero, p1_sign);
                __m256i p1_neg  = _mm256_and_si256(p1_abs, p1_sign);

                CSA256(&twosA, &ones, ones, p0_neg, p0_nonpos);
                CSA256(&twosB, &ones, ones, p1_neg, p1_nonpos);
                CSA256(&foursA, &twos, twos, twosA, twosB);

                fours = _mm256_add_epi64(fours, popcnt256(foursA));
            }

            twos = popcnt256(twos);
            for (; k < limit4; k += (BITS*4)) {
                __m256i a0_sign = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0));
                __m256i a0_abs = _mm256_load_si256((__m256i*)(a + i*KB + k + 8*0 + 4));
                __m256i b0_sign = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0));
                __m256i b0_abs = _mm256_load_si256((__m256i*)(b + j*KB + k + 8*0 + 4));

                __m256i p0_abs = _mm256_and_si256(a0_abs, b0_abs);
                __m256i p0_sign = _mm256_xor_si256(a0_sign, b0_sign);
                __m256i p0_zero  = _mm256_xor_si256(p0_abs, negator);
                __m256i p0_nonpos  = _mm256_or_si256(p0_zero, p0_sign);
                __m256i p0_neg  = _mm256_and_si256(p0_abs, p0_sign);

                CSA256(&twosA, &ones, ones, p0_neg, p0_nonpos);

                twos = _mm256_add_epi64(twos, popcnt256(twosA));
            }

            __m256i s1 = _mm256_add_epi64(_mm256_slli_epi64(eights, 3), _mm256_slli_epi64(fours, 2));
            __m256i s2 = _mm256_add_epi64(_mm256_slli_epi64(twos, 1), popcnt256(ones));
            cnt = _mm256_add_epi64(_mm256_slli_epi64(cnt, 4), _mm256_add_epi64(s1, s2));

            uint64_t* cnt64;
            cnt64 = (uint64_t*) &cnt;

            *(y + i*N + j) += K*32 - (cnt64[0] + cnt64[1] + cnt64[2] + cnt64[3]);
            if(last) {
                float activation = (float) *(y + i*N + j);
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)(y + i*N + j);
                *fview = activation;
            }
        }
    }
}


static float* mm_p1_lib_1pc(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    mm_blocked(1, N, KB, micro_mm_p1_lib_1pc)
}

/**
 * Optimization 6:
 * We transform the computation to compute one large popcount instead of two smaller ones
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p1_lib_1pc(a,b,N*OH*OW,KN,K,prelu_alpha);
}
#endif
