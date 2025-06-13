#include "tnns/mm.h"

static inline void micro_mm_p0_naive(int64_t* a, int64_t* b, int* y, int I, int J, int K, int N, int KB, float alpha, bool last) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            int cntp1 = 0;
            int cntp2 = 0;
            for (int k = 0; k < K; k += BITS) {
                int64_t p1 = a[i*KB + k + 0] ^ b[j*KB + k + 0]; 
                int64_t p2 = a[i*KB + k + 1] & b[j*KB + k + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[i*N+j] = y[i*N+j] + cntp1 - cntp2 - cntp2;
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


static float* mm_p0_blocked_naive(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    mm_blocked(8, 4, KB, micro_mm_p0_naive)
}

/**
 * Optimization 2:
 * We block the computation for cache
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p0_blocked_naive(a,b,N*OH*OW,KN,K,prelu_alpha);
}
