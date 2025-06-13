#include "tnns/mm.h"

static float* mm_p0_naive(int64_t* a, int64_t* b, int M, int N, int K, float prelu_alpha) {
    float* y = calloc(M * N, sizeof(float));
    const int KB = K * BITS;

    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
            int cntp2 = 0;
            for (int ik = 0; ik < KB; ik += BITS) {
                int64_t p1 = a[oh * KB + ik + 0] ^ b[ow * KB + ik + 0]; 
                int64_t p2 = a[oh * KB + ik + 1] & b[ow * KB + ik + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            int tmp = cntp1 - cntp2 - cntp2;
            y[oh * N + ow] = tmp < 0 ? tmp * prelu_alpha : tmp;
        }
    }

    return y;
}

/**
 * Optimization 1:
 * We fuse the activation function (PReLU) into the GEMM
*/
float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    return mm_p0_naive(a,b,N*OH*OW,KN,K,prelu_alpha);
}
