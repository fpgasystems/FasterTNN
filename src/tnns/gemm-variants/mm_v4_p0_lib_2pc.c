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

static inline void micro_mm_p0_lib_2pc(int64_t* a, int64_t* b, int* y, int I, int J, int K, int N, int KB, float alpha, bool last) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {

            __m256i cntp1 = _mm256_setzero_si256();
            __m256i cntp2 = _mm256_setzero_si256();

            int k = 0;
            for (; k + (BITS * 4) <= K; k += (BITS*4)) {
                __m256i a1 = _mm256_loadu_si256((__m256i*)((__m256i*)(a + (i*KB + k + 0))));
                __m256i a2 = _mm256_loadu_si256((__m256i*)((__m256i*)(a + (i*KB + k + 4))));
                __m256i b1 = _mm256_loadu_si256((__m256i*)((__m256i*)(b + (j*KB + k + 0))));
                __m256i b2 = _mm256_loadu_si256((__m256i*)((__m256i*)(b + (j*KB + k + 4))));
                __m256i a_lower = _mm256_unpacklo_epi64(a1, a2);
                __m256i a_upper = _mm256_unpackhi_epi64(a1, a2);
                __m256i b_lower = _mm256_unpacklo_epi64(b1, b2);
                __m256i b_upper = _mm256_unpackhi_epi64(b1, b2);

                __m256i p1 = _mm256_xor_si256(a_lower, b_lower);
                __m256i p2 = _mm256_and_si256(a_upper, b_upper);
                __m256i p1a2 = _mm256_and_si256(p1,p2);

                cntp1 = _mm256_add_epi64(cntp1, popcnt256(p2));
                cntp2 = _mm256_add_epi64(cntp2, popcnt256(p1a2));
            }

            int cntp1_0 = 0;
            int cntp2_0 = 0;
            for (; k < K; k += BITS) {
                int64_t p1 = a[i*KB + k + 0] ^ b[j*KB + k + 0];
                int64_t p2 = a[i*KB + k + 1] & b[j*KB + k + 1];
                cntp1_0 += popcnt64(p2);
                cntp2_0 += popcnt64(p1 & p2);
            }

            int64_t cntp1i[4] __attribute__((aligned(32)));
            int64_t cntp2i[4] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i*)cntp1i, cntp1);
            _mm256_store_si256((__m256i*)cntp2i, cntp2);
            cntp1i[0] -= 2 * cntp2i[0];
            cntp1i[1] -= 2 * cntp2i[1];
            cntp1i[2] -= 2 * cntp2i[2];
            cntp1i[3] -= 2 * cntp2i[3];

            cntp1_0 -= 2 * cntp2_0;
            y[i*N+j] += cntp1_0 + cntp1i[0] + cntp1i[1] + cntp1i[2] + cntp1i[3];
            if(last) {
                float activation = (float) y[i*N+j];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[i*N+j];
                *fview = activation;
            }
        }
    }
}


static float* mm_p0_lib_2pc(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    mm_blocked(M, N, KB, micro_mm_p0_lib_2pc)
}

/**
 * Optimization 4:
 * We vectorize the popcount using the library libpopcnt
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p0_lib_2pc(a,b,N*OH*OW,KN,K,prelu_alpha);
}

#endif