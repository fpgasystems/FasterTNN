#include "tnns/mm.h"

#include <arm_neon.h>

static inline void tbn_gemm_micro_arm_packedx2(int64_t* a, int64_t* b, int* y, int I1, int I2, int K, int N, int M1, int M2, int KB, float alpha, bool last) {
    int OHW = M1 / N;
    
    for (int i = 0; i < I1; i++) {
        int64_t cntp1 = 0;
        int8x16_t cntp1_tmp = vdupq_n_s64(0);
        
        int k = 0;
        int cnt = 0;
        for (; k + 3 < K; k += 4) {
            int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
            int64x2_t p2 = a_lower;
            uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
            
            cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);

            ++cnt;
            if(cnt == 16) {
                uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);

                cntp1_tmp = vdupq_n_s64(0);
                cnt = 0;
            }
        }

        uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
        uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
        uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
        cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
        cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
        
        for (int j = 0; j < I2; j++) {

            int64_t cntp2 = 0;
            int8x16_t cntp2_tmp = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 3 < K; k += 4) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)/BITS));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = a_lower;
                int64x2_t p1a2 = vandq_s64(p1, p2);

                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

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


            y[((i/OHW) * M2 + j) * OHW + (i%OHW)] += cntp1 - 2 * cntp2;
            if(last) {
                float activation = (float) y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                *fview = activation;
            }
        }
    }
}

static inline void tbn_gemm_micro_arm_packedx2_unrolled(int64_t* a, int64_t* b, int* y, int I1, int I2, int K, int N, int M1, int M2, int KB, float alpha, bool last) {
    int OHW = M1 / N;
    
    for (int i = 0; i < I1; i++) {
        int64_t cntp1 = 0;
        int8x16_t cntp1_tmp = vdupq_n_s64(0);
        
        int k = 0;
        int cnt = 0;
        for (; k + 3 < K; k += 4) {
            int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
            int64x2_t p2 = a_lower;
            uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
            
            cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);

            ++cnt;
            if(cnt == 16) {
                uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);

                cntp1_tmp = vdupq_n_s64(0);
                cnt = 0;
            }
        }

        uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
        uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
        uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
        cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
        cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
        
        for (int j = 0; j < I2; j++) {

            int64_t cntp2 = 0;
            int64_t cntp2_00 = 0;
            int64_t cntp2_01 = 0;
            int64_t cntp2_10 = 0;
            int64_t cntp2_11 = 0;
            int8x16_t cntp2_tmp = vdupq_n_s64(0);
            int8x16_t cntp2_tmp_1 = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 7 < K; k += 8) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t a_upper_1 = vld1q_s64((int64_t*)(a + (i*KB + k + 4)));
                int64x2_t a_lower_1 = vld1q_s64((int64_t*)(a + (i*KB + k + 6)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)/BITS));
                int64x2_t b_upper_1 = vld1q_s64((int64_t*)(b + (j*KB + k + 4)/BITS));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = a_lower;
                int64x2_t p1a2 = vandq_s64(p1, p2);
                
                int64x2_t p1_1 = veorq_s64(a_upper_1, b_upper_1);
                int64x2_t p2_1 = a_lower_1;
                int64x2_t p1a2_1 = vandq_s64(p1_1, p2_1);

                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                uint8x16_t pcnt_p1a2_1 = vcntq_u8(vreinterpretq_u8_s64(p1a2_1));
                
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);
                cntp2_tmp_1 = vaddq_u8(cntp2_tmp_1, pcnt_p1a2_1);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);
                    
                    uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
                    uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
                    uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
                    cntp2_10 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
                    cntp2_11 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);

                    cntp2_tmp = vdupq_n_s64(0);
                    cntp2_tmp_1 = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp2_1_0 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp2_2_0 = vpaddlq_u16(tmp2_1_0);
            uint64x2_t tmp2_3_0 = vpaddlq_u32(tmp2_2_0);
            cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3_0, 0);
            cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3_0, 1);
            cntp2_tmp = vdupq_n_s64(0);
            
            uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
            uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
            uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
            cntp2_10 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
            cntp2_11 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);
            
            cntp2 += cntp2_10 + cntp2_11;
            
            for (; k + 3 < K; k += 4) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)/BITS));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = a_lower;
                int64x2_t p1a2 = vandq_s64(p1, p2);

                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                    cntp2_tmp = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
            cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

            cntp2 += cntp2_00 + cntp2_01;

            y[((i/OHW) * M2 + j) * OHW + (i%OHW)] += cntp1 - 2 * cntp2;
            if(last) {
                float activation = (float) y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                *fview = activation;
            }
        }
    }
}


float* tbn_gemm_packing1(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int M1 = N * OH * OW;
    const int M2 = KN;
    const int KB = K * BITS;

    tbn_gemm_micro_arm_packedx2_unrolled(a, b, y, M1, M2, KB, N, M1, M2, KB, prelu_alpha, true);

    return (float*)y;
}




float* mm_tbn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    K = 2 * ((K + 1) / 2);
    return tbn_gemm_packing1(a, b, N, OH, OW, KN, K, prelu_alpha);
}
