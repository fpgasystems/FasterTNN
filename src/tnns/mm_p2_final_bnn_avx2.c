#include "tnns/mm.h"

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

float* mm_bnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, int K_unpack, float prelu_alpha) {
    K = 4 * ((K + 3) / 4);
    
    float* y = calloc(N * KN * OH * OW, sizeof(float));

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int kn = 0; kn < KN; kn++) {
                    __m256i neg_popcounts = _mm256_setzero_si256();
                    int ik = 0;
                    for (; ik < K; ik += 4) {
                        __m256i a_sign = _mm256_load_si256((__m256i*)(a + ((n * OH + oh) * OW + ow) * K + ik));
                        __m256i b_sign = _mm256_load_si256((__m256i*)(b + kn*K + ik));

                        __m256i p1 = _mm256_xor_si256(a_sign, b_sign);
                        __m256i neg_pc = popcnt256(p1);

                        neg_popcounts = _mm256_add_epi64(neg_popcounts, neg_pc);
                        
                    }
                    int negative = ((int64_t *) &neg_popcounts)[0] + ((int64_t *) &neg_popcounts)[1] + ((int64_t *) &neg_popcounts)[2] + ((int64_t *) &neg_popcounts)[3];

                    // K_unpack is C*KH*KW without padding, i.e. the total number of binary values
                    int result = K_unpack - negative - negative;

                    y[((n * KN + kn) * OH + oh) * OW + ow] = result >= 0 ? (float) result : prelu_alpha * result;
                }
            }
        }
    }
    
    return y;
}
