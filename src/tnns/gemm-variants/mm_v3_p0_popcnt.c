#include "tnns/mm.h"

#ifdef AVX2
#include <nmmintrin.h>
#include <immintrin.h>

static inline void micro_mm_p0_popcnt(int64_t* a, int64_t* b, int* y, int I, int J, int K, int N, int KB, float alpha, bool last) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            int cntp1_0 = 0;
            int cntp1_1 = 0;
            int cntp1_2 = 0;
            int cntp1_3 = 0;
            int cntp2_0 = 0;
            int cntp2_1 = 0;
            int cntp2_2 = 0;
            int cntp2_3 = 0;
            int k = 0;
            int k_lim = K - K % (BITS*4);
            for (; k < k_lim; k += (BITS*4)) {
                __m256i a1 = _mm256_loadu_si256((__m256i*)(a + (i*KB + k + 0)));
                __m256i a2 = _mm256_loadu_si256((__m256i*)(a + (i*KB + k + 4)));
                __m256i b1 = _mm256_loadu_si256((__m256i*)(b + (j*KB + k + 0)));
                __m256i b2 = _mm256_loadu_si256((__m256i*)(b + (j*KB + k + 4)));
                __m256i a_lower = _mm256_unpacklo_epi64(a1, a2);
                __m256i a_upper = _mm256_unpackhi_epi64(a1, a2);
                __m256i b_lower = _mm256_unpacklo_epi64(b1, b2);
                __m256i b_upper = _mm256_unpackhi_epi64(b1, b2);

                __m256i p1 = _mm256_xor_si256(a_lower, b_lower);
                __m256i p2 = _mm256_and_si256(a_upper, b_upper);
                __m256i p1a2 = _mm256_and_si256(p1,p2);


                int64_t p2i[4] __attribute__((aligned(32)));
                int64_t p1a2i[4] __attribute__((aligned(32)));
                _mm256_store_si256((__m256i*)p2i, p2);
                _mm256_store_si256((__m256i*)p1a2i, p1a2);
                cntp1_0 += popcnt64(p2i[0]);
                cntp1_1 += popcnt64(p2i[1]);
                cntp1_2 += popcnt64(p2i[2]);
                cntp1_3 += popcnt64(p2i[3]);
                cntp2_0 += popcnt64(p1a2i[0]);
                cntp2_1 += popcnt64(p1a2i[1]);
                cntp2_2 += popcnt64(p1a2i[2]);
                cntp2_3 += popcnt64(p1a2i[3]);
            }
            
            int cntp1_4 = 0;
            int cntp2_4 = 0;
            for (; k < K; k += BITS) {
                int64_t p1 = a[i*KB + k + 0] ^ b[j*KB + k + 0];
                int64_t p2 = a[i*KB + k + 1] & b[j*KB + k + 1];
                cntp1_0 += popcnt64(p2);
                cntp2_0 += popcnt64(p1 & p2);
            }

            cntp1_0 -= 2 * cntp2_0;
            cntp1_1 -= 2 * cntp2_1;
            cntp1_2 -= 2 * cntp2_2;
            cntp1_3 -= 2 * cntp2_3;
            y[i*N+j] += + cntp1_0 + cntp1_1 + cntp1_2 + cntp1_3 + cntp1_4 - 2 * cntp2_4;
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



static float* mm_p0_popcnt(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    mm_blocked(8, 4, KB, micro_mm_p0_popcnt)
}

/**
 * Optimization 3:
 * We vectorize the computation but still use scalar popcount instructions
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p0_popcnt(a,b,N*OH*OW,KN,K,prelu_alpha);
}

#endif
