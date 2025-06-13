#include "tnns/mm.h"

#include <arm_neon.h>

static inline min(int x, int y) {
    return x <= y ? x : y;
}

static inline void tnns_gemm_micro_arm(int64_t* a, int64_t* b, int* y, int I1, int I2, int K, int N, int M1, int M2, int KB, float alpha, bool last) {
    int OHW = M1 / N;

    for (int i = 0; i < I1; i++) {
        for (int j = 0; j < I2; j++) {

            int64_t cntp1 = 0;
            int64_t cntp2 = 0;
            
            int8x16_t cntp1_tmp = vdupq_n_s64(0);
            int8x16_t cntp2_tmp = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 3 < K; k += 4) {
                int64x2_t a1 = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a2 = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b1 = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                int64x2_t b2 = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));

                int64x2_t a_upper = vcombine_s64(vget_low_s64(a1), vget_low_s64(a2));
                int64x2_t a_lower = vcombine_s64(vget_high_s64(a1), vget_high_s64(a2));
                int64x2_t b_upper = vcombine_s64(vget_low_s64(b1), vget_low_s64(b2));
                int64x2_t b_lower = vcombine_s64(vget_high_s64(b1), vget_high_s64(b2));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = vandq_s64(a_lower, b_lower);
                int64x2_t p1a2 = vandq_s64(p1, p2);

                uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                
                cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                    cntp1_tmp = vdupq_n_s64(0);
                    cntp2_tmp = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

            for (; k < K; k += BITS) {
                int64_t p1 = a[i*KB + k + 0] ^ b[j*KB + k + 0];
                int64_t p2 = a[i*KB + k + 1] & b[j*KB + k + 1];
                cntp1 += popcnt64(p2);
                cntp2 += popcnt64(p1 & p2);
            }

            y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)] += cntp1 - 2 * cntp2;
            if(last) {
                float activation = (float) y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)];
                *fview = activation;
            }
        }
    }
}

static inline void tnns_gemm_micro_arm_packedx2(int64_t* a, int64_t* b, int* y, int N, int M1, int M2, int KB, float alpha, bool last) {
    int OHW = M1 / N;
         
    for (int i0_0 = 0; i0_0 < N; i0_0++) {        
        for (int j = 0; j < M2; j++) {
            for (int i0_1 = 0; i0_1 < OHW; i0_1++) {            
                int i = i0_0 * OHW + i0_1;

                int64_t cntp1 = 0;
                int64_t cntp1_00 = 0;
                int64_t cntp1_01 = 0;
                int64_t cntp2 = 0;
                int64_t cntp2_00 = 0;
                int64_t cntp2_01 = 0;
                
                int8x16_t cntp1_tmp = vdupq_n_s64(0);
                int8x16_t cntp2_tmp = vdupq_n_s64(0);

                int k = 0;
                int cnt = 0;
                for (; k + 3 < KB; k += 4) {
                    int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                    int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                    int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                    int64x2_t b_lower = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));

                    int64x2_t p1 = veorq_s64(a_upper, b_upper);
                    int64x2_t p2 = vandq_s64(a_lower, b_lower);
                    int64x2_t p1a2 = vandq_s64(p1, p2);

                    uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
                    uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                    
                    cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);
                    cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

                    ++cnt;
                    if(cnt == 16) {
                        uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                        uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                        uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                        uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                        uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                        uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                        cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                        cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                        cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                        cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                        cntp1_tmp = vdupq_n_s64(0);
                        cntp2_tmp = vdupq_n_s64(0);
                        cnt = 0;
                    }
                }

                uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);
                
                cntp1 = cntp1_00 + cntp1_01;
                cntp2 = cntp2_00 + cntp2_01;      

                float activation = (float) (cntp1 - 2 * cntp2);
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[(i0_0 * M2 + j) * OHW + i0_1];
                *fview = activation;
            }
        }
    }

}

static inline void tnns_gemm_micro_arm_packedx2_unrollj2(int64_t* a, int64_t* b, int* y, int I1, int I2, int K, int N, int M1, int M2, int KB, float alpha, bool last) {
    int OHW = M1/ N;

    for (int i = 0; i < I1; i++) {
        int j = 0;
        
        for (; j <= I2-2; j += 2) {
            int64_t cntp1 = 0;
            int64_t cntp2 = 0;
            int64_t cntp1_1 = 0;
            int64_t cntp2_1 = 0;
            
            int8x16_t cntp1_tmp = vdupq_n_s64(0);
            int8x16_t cntp2_tmp = vdupq_n_s64(0);
            int8x16_t cntp1_tmp_1 = vdupq_n_s64(0);
            int8x16_t cntp2_tmp_1 = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 3 < K; k += 4) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                int64x2_t b_lower = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));
                int64x2_t b_upper_1 = vld1q_s64((int64_t*)(b + ((j+1)*KB + k + 0)));
                int64x2_t b_lower_1 = vld1q_s64((int64_t*)(b + ((j+1)*KB + k + 2)));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = vandq_s64(a_lower, b_lower);
                int64x2_t p1a2 = vandq_s64(p1, p2);
                
                int64x2_t p1_1 = veorq_s64(a_upper, b_upper_1);
                int64x2_t p2_1 = vandq_s64(a_lower, b_lower_1);
                int64x2_t p1a2_1 = vandq_s64(p1_1, p2_1);

                uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));

                uint8x16_t pcnt_p2_1 = vcntq_u8(vreinterpretq_u8_s64(p2_1));
                uint8x16_t pcnt_p1a2_1 = vcntq_u8(vreinterpretq_u8_s64(p1a2_1));
                
                cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);
                
                cntp1_tmp_1 = vaddq_u8(cntp1_tmp_1, pcnt_p2_1);
                cntp2_tmp_1 = vaddq_u8(cntp2_tmp_1, pcnt_p1a2_1);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);
                    
                    uint16x8_t tmp1_1_1 = vpaddlq_u8(cntp1_tmp_1);
                    uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
                    uint32x4_t tmp1_2_1 = vpaddlq_u16(tmp1_1_1);
                    uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
                    uint64x2_t tmp1_3_1 = vpaddlq_u32(tmp1_2_1);
                    uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
                    
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3_1, 0);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3_1, 1);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);

                    cntp1_tmp = vdupq_n_s64(0);
                    cntp2_tmp = vdupq_n_s64(0);
                    cntp1_tmp_1 = vdupq_n_s64(0);
                    cntp2_tmp_1 = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
            cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);
            
            uint16x8_t tmp1_1_1 = vpaddlq_u8(cntp1_tmp_1);
            uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
            uint32x4_t tmp1_2_1 = vpaddlq_u16(tmp1_1_1);
            uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
            uint64x2_t tmp1_3_1 = vpaddlq_u32(tmp1_2_1);
            uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
            
            cntp1_1 += (int64_t)vgetq_lane_u64(tmp1_3_1, 0);
            cntp1_1 += (int64_t)vgetq_lane_u64(tmp1_3_1, 1);
            cntp2_1 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
            cntp2_1 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);


            y[((i/OHW) * M2 + j) * OHW + (i%OHW)] += cntp1 - 2 * cntp2;
            y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)] += cntp1_1 - 2 * cntp2_1;
            if(last) {
                float activation = (float) y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[((i/OHW) * M2 + j) * OHW + (i%OHW)];
                *fview = activation;
                float activation_1 = (float) y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)];
                if(activation_1 <= 0) {
                    activation_1 *= alpha;
                }
                float* fview_1 = (float*)&y[((i/OHW) * M2 + j+1) * OHW + (i%OHW)];
                *fview_1 = activation_1;
            }
        }
        
        
        for (; j < I2; j++) {

            int64_t cntp1 = 0;
            int64_t cntp2 = 0;
            
            int8x16_t cntp1_tmp = vdupq_n_s64(0);
            int8x16_t cntp2_tmp = vdupq_n_s64(0);

            int k = 0;
            int cnt = 0;
            for (; k + 3 < K; k += 4) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                int64x2_t b_lower = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = vandq_s64(a_lower, b_lower);
                int64x2_t p1a2 = vandq_s64(p1, p2);

                uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                
                cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                    cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                    cntp1_tmp = vdupq_n_s64(0);
                    cntp2_tmp = vdupq_n_s64(0);
                    cnt = 0;
                }
            }

            uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
            cntp1 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
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

static inline void tnns_gemm_micro_arm_packedx4(int64_t* a, int64_t* b, int* y, int I1, int I2, int K, int N, int M1, int M2, int KB, float alpha, bool last) {
    const int OHW = M1 / N;

    for (int i = 0; i < I1; i++) {
        for (int j = 0; j < I2; j++) {

            int64_t cntp1 = 0;
            int64_t cntp1_00 = 0;
            int64_t cntp1_01 = 0;
            int64_t cntp1_10 = 0;
            int64_t cntp1_11 = 0;
            int64_t cntp2 = 0;
            int64_t cntp2_00 = 0;
            int64_t cntp2_01 = 0;
            int64_t cntp2_10 = 0;
            int64_t cntp2_11 = 0;
            
            int8x16_t cntp1_tmp_0 = vdupq_n_s64(0);
            int8x16_t cntp1_tmp_1 = vdupq_n_s64(0);
            int8x16_t cntp2_tmp_0 = vdupq_n_s64(0);
            int8x16_t cntp2_tmp_1 = vdupq_n_s64(0);
            
            int k = 0;
            int cnt = 0;
            for (; k + 7 < K; k += 8) {
                int64x2_t a_upper_0 = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower_0 = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t a_upper_1 = vld1q_s64((int64_t*)(a + (i*KB + k + 4)));
                int64x2_t a_lower_1 = vld1q_s64((int64_t*)(a + (i*KB + k + 6)));
                int64x2_t b_upper_0 = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                int64x2_t b_lower_0 = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));
                int64x2_t b_upper_1 = vld1q_s64((int64_t*)(b + (j*KB + k + 4)));
                int64x2_t b_lower_1 = vld1q_s64((int64_t*)(b + (j*KB + k + 6)));

                int64x2_t p1_0 = veorq_s64(a_upper_0, b_upper_0);
                int64x2_t p1_1 = veorq_s64(a_upper_1, b_upper_1);
                int64x2_t p2_0 = vandq_s64(a_lower_0, b_lower_0);
                int64x2_t p2_1 = vandq_s64(a_lower_1, b_lower_1);
                int64x2_t p1a2_0 = vandq_s64(p1_0, p2_0);
                int64x2_t p1a2_1 = vandq_s64(p1_1, p2_1);

                uint8x16_t pcnt_p2_0 = vcntq_u8(vreinterpretq_u8_s64(p2_0));
                uint8x16_t pcnt_p2_1 = vcntq_u8(vreinterpretq_u8_s64(p2_1));
                uint8x16_t pcnt_p1a2_0 = vcntq_u8(vreinterpretq_u8_s64(p1a2_0));
                uint8x16_t pcnt_p1a2_1 = vcntq_u8(vreinterpretq_u8_s64(p1a2_1));
                
                cntp1_tmp_0 = vaddq_u8(cntp1_tmp_0, pcnt_p2_0);
                cntp1_tmp_1 = vaddq_u8(cntp1_tmp_1, pcnt_p2_1);
                cntp2_tmp_0 = vaddq_u8(cntp2_tmp_0, pcnt_p1a2_0);
                cntp2_tmp_1 = vaddq_u8(cntp2_tmp_1, pcnt_p1a2_1);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp1_1_0 = vpaddlq_u8(cntp1_tmp_0);
                    uint16x8_t tmp1_1_1 = vpaddlq_u8(cntp1_tmp_1);
                    uint16x8_t tmp2_1_0 = vpaddlq_u8(cntp2_tmp_0);
                    uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
                    uint32x4_t tmp1_2_0 = vpaddlq_u16(tmp1_1_0);
                    uint32x4_t tmp1_2_1 = vpaddlq_u16(tmp1_1_1);
                    uint32x4_t tmp2_2_0 = vpaddlq_u16(tmp2_1_0);
                    uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
                    uint64x2_t tmp1_3_0 = vpaddlq_u32(tmp1_2_0);
                    uint64x2_t tmp1_3_1 = vpaddlq_u32(tmp1_2_1);
                    uint64x2_t tmp2_3_0 = vpaddlq_u32(tmp2_2_0);
                    uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
                    cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3_0, 0);
                    cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3_0, 1);
                    cntp1_10 += (int64_t)vgetq_lane_u64(tmp1_3_1, 0);
                    cntp1_11 += (int64_t)vgetq_lane_u64(tmp1_3_1, 1);
                    cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3_0, 0);
                    cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3_0, 1);
                    cntp2_10 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
                    cntp2_11 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);

                    cntp1_tmp_0 = vdupq_n_s64(0);
                    cntp1_tmp_1 = vdupq_n_s64(0);
                    cntp2_tmp_0 = vdupq_n_s64(0);
                    cntp2_tmp_1 = vdupq_n_s64(0);
                    cnt = 0;
                }
            } 
            
            uint16x8_t tmp1_1_0 = vpaddlq_u8(cntp1_tmp_0);
            uint16x8_t tmp1_1_1 = vpaddlq_u8(cntp1_tmp_1);
            uint16x8_t tmp2_1_0 = vpaddlq_u8(cntp2_tmp_0);
            uint16x8_t tmp2_1_1 = vpaddlq_u8(cntp2_tmp_1);
            uint32x4_t tmp1_2_0 = vpaddlq_u16(tmp1_1_0);
            uint32x4_t tmp1_2_1 = vpaddlq_u16(tmp1_1_1);
            uint32x4_t tmp2_2_0 = vpaddlq_u16(tmp2_1_0);
            uint32x4_t tmp2_2_1 = vpaddlq_u16(tmp2_1_1);
            uint64x2_t tmp1_3_0 = vpaddlq_u32(tmp1_2_0);
            uint64x2_t tmp1_3_1 = vpaddlq_u32(tmp1_2_1);
            uint64x2_t tmp2_3_0 = vpaddlq_u32(tmp2_2_0);
            uint64x2_t tmp2_3_1 = vpaddlq_u32(tmp2_2_1);
            cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3_0, 0);
            cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3_0, 1);
            cntp1_10 += (int64_t)vgetq_lane_u64(tmp1_3_1, 0);
            cntp1_11 += (int64_t)vgetq_lane_u64(tmp1_3_1, 1);
            cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3_0, 0);
            cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3_0, 1);
            cntp2_10 += (int64_t)vgetq_lane_u64(tmp2_3_1, 0);
            cntp2_11 += (int64_t)vgetq_lane_u64(tmp2_3_1, 1);
            cnt = 0;
                
            int8x16_t cntp1_tmp = vdupq_n_s64(0);
            int8x16_t cntp2_tmp = vdupq_n_s64(0);
                
            for (; k + 3 < K; k += 4) {
                int64x2_t a_upper = vld1q_s64((int64_t*)(a + (i*KB + k + 0)));
                int64x2_t a_lower = vld1q_s64((int64_t*)(a + (i*KB + k + 2)));
                int64x2_t b_upper = vld1q_s64((int64_t*)(b + (j*KB + k + 0)));
                int64x2_t b_lower = vld1q_s64((int64_t*)(b + (j*KB + k + 2)));

                int64x2_t p1 = veorq_s64(a_upper, b_upper);
                int64x2_t p2 = vandq_s64(a_lower, b_lower);
                int64x2_t p1a2 = vandq_s64(p1, p2);

                uint8x16_t pcnt_p2 = vcntq_u8(vreinterpretq_u8_s64(p2));
                uint8x16_t pcnt_p1a2 = vcntq_u8(vreinterpretq_u8_s64(p1a2));
                
                cntp1_tmp = vaddq_u8(cntp1_tmp, pcnt_p2);
                cntp2_tmp = vaddq_u8(cntp2_tmp, pcnt_p1a2);

                ++cnt;
                if(cnt == 16) {
                    uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
                    uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
                    uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
                    uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
                    uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
                    uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
                    cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
                    cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
                    cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
                    cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);

                    cntp1_tmp = vdupq_n_s64(0);
                    cntp2_tmp = vdupq_n_s64(0);
                    cnt = 0;
                }
            }
            uint16x8_t tmp1_1 = vpaddlq_u8(cntp1_tmp);
            uint16x8_t tmp2_1 = vpaddlq_u8(cntp2_tmp);
            uint32x4_t tmp1_2 = vpaddlq_u16(tmp1_1);
            uint32x4_t tmp2_2 = vpaddlq_u16(tmp2_1);
            uint64x2_t tmp1_3 = vpaddlq_u32(tmp1_2);
            uint64x2_t tmp2_3 = vpaddlq_u32(tmp2_2);
            cntp1_00 += (int64_t)vgetq_lane_u64(tmp1_3, 0);
            cntp1_01 += (int64_t)vgetq_lane_u64(tmp1_3, 1);
            cntp2_00 += (int64_t)vgetq_lane_u64(tmp2_3, 0);
            cntp2_01 += (int64_t)vgetq_lane_u64(tmp2_3, 1);
            
            cntp1 += cntp1_00 + cntp1_01 + cntp1_10 + cntp1_11;
            cntp2 += cntp2_00 + cntp2_01 + cntp2_10 + cntp2_11;

            y[((i/OHW)*M2 + j)*OHW + (i%OHW)] += cntp1 - 2 * cntp2;
            if(last) {
                float activation = (float) y[((i/OHW)*M2 + j)*OHW + (i%OHW)];
                if(activation <= 0) {
                    activation *= alpha;
                }
                float* fview = (float*)&y[((i/OHW)*M2 + j)*OHW + (i%OHW)];
                *fview = activation;
            }
        }
    }
}

float* tnns_gemm_packing1(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    int* y = calloc(N * KN * OH * OW, sizeof(int));
    const int M1 = N * OH * OW;
    const int M2 = KN;
    const int KB = K * BITS;
    
    tnns_gemm_micro_arm_packedx2(a, b, y, N, M1, M2, KB, prelu_alpha, true);

    return (float*)y;
}




float* mm_tnn(int64_t* a, int64_t* b, int N, int OH, int OW, int KN, int K, float prelu_alpha) {
    K = 2 * ((K + 1) / 2);
    return tnns_gemm_packing1(a, b, N, OH, OW, KN, K, prelu_alpha);
}
