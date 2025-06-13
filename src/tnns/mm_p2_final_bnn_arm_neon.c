#include "tnns/mm.h"

#include <arm_neon.h>


float* bnn_gemm_packing1(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, int K_unpack, float prelu_alpha) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int M1 = N * OH * OW;
    const int M2 = KN;
    const int OHW = OH * OW;

    for (int i = 0; i < M1; i++) {
        for (int j = 0; j < M2; j++) {
            int64_t cntp2 = 0;
            int8x16_t cntp2_tmp = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 1 < K; k += 2) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + i*K + k));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + j*K + k));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);

                uint8x16_t pcnt_p1 = vcntq_u8(vreinterpretq_u8_s64(p1));
                
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                    cntp2_tmp = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

            // K_unpack is C*KH*KW without padding, i.e. the total number of binary values                
            float activation = (float) (K_unpack - 2 * cntp2);
            
            if(activation <= 0) {
                activation *= prelu_alpha;
            }
            float* fview = (float*)&y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
            *fview = activation;
        }
    }
    return (float*)y;
}




float* mm_bnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, int K_unpack, float prelu_alpha) {
    K = 2 * ((K + 1) / 2);
    return bnn_gemm_packing1(a, b, N, OH, OW, KN, K, K_unpack, prelu_alpha);
}
