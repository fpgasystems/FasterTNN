#include <assert.h>
#include <string.h>
#include "tnns/common.h"
#include "tnns/binarize2row.h"

#include <arm_neon.h>

static inline int min(int a, int b) {
    return a < b ? a : b;
}
static inline int max(int a, int b) {
    return a > b ? a : b;
}

static int32_t p1_lo[8] = {0,0,0,0,0,0,0,0};
static int32_t p1_hi[8] = {0,0,0,0,0,0,0,0};


static inline void b2r_ARMx2(float* x, int64_t* qx_row, float* q_thresholds, int64_t* onebit, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW) {
    // initialize vectors
    int64x2_t p1 = vdupq_n_s64(0);
    float32x2_t threshold = vdup_n_f32(q_thresholds[in]);
    
    // convert float32x2_t to float64x2_t
    float64x2_t threshold64 = vcvt_f64_f32(threshold);

    for (int64_t bit = 0; bit < cntbits; bit++) {
        // load 2 float32x2_t from memory
        float32x2_t currentx = vld1_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);
        // convert float32x2_t to float64x2_t
        float64x2_t currentx64 = vcvt_f64_f32(currentx);

        // compare current values with threshold
        int64x2_t mask1 = vcltq_f64(currentx64, threshold64);
        
        // create a vector of one bit
        int64x2_t bitvec = vdupq_n_s64(onebit[bit]);

        // set bits based on mask1 and bitvec
        p1 = vorrq_s64(p1, vandq_s64(mask1, bitvec));
    }

    // store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    while((bh_base >= 0) && (bw_base < kH * kW)) {
        int bh = min(iw_pad / W_stride, OW-1);
        int bw = max(iw_pad - (OW-1)*W_stride, iw_pad % W_stride);

        int index_low = (in * H_fused + bh_base + bh) * W_fused + ((bw_base + bw) * C_pack + ic);
        int64_t iteration_diff = -W_fused + W_stride * C_pack;

        for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
            vst1q_s64(&qx_row[index_low], p1);
            index_low += iteration_diff;
        }

        bh_base -= OW;
        bw_base += kW * H_stride;
    }
}

static inline void b2r_ARMx4(float* x, int64_t* qx_row, float* q_thresholds, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW) {

    // initialize vectors
    int32x4_t v_p1 = vdupq_n_s32(0);
    int32x4_t v_onebit = vdupq_n_s32(1);

    float32x4_t v_threshold = vdupq_n_f32(q_thresholds[in]);


    // pack first 32-bits
    int limit = min(C - (ic*cntbits), 32);

    for (int bit = 0; bit < limit; bit++) {
        // load 4 floats into 128-bit vector register
        float32x4_t v_currentx = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);

        // compare current values with threshold
        uint32x4_t v_p1_mask = vcltq_f32(v_currentx, v_threshold);

        // set bit to 1 if the mask is one -> AND the mask and the onebit registers
        uint32x4_t v_p1_nextbit = vandq_s32(v_p1_mask, v_onebit);

        // insert the bits using OR
        v_p1 = vorrq_s32(v_p1, v_p1_nextbit);

        // shift the onebit left by 1 to pack the next bit
        v_onebit = vshlq_n_s32(v_onebit, 1);
    }

    // store first 32-bits temporarily
    vst1q_s32(p1_lo, v_p1);
    
    // pack second 32-bits if we have more than 32 lower bits
    if (C - (ic*cntbits) >= 32) {
            
        // reset some vectors
        v_onebit = vdupq_n_s32(1);
        v_p1 = vdupq_n_s32(0);

        // pack bits
        limit = min(C - (ic*cntbits), 64);
        for (int bit = 32; bit < limit; bit++) {
            // load 4 floats into 128-bit vector register
            float32x4_t v_currentx = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);

            // compare current values with negative threshold
            uint32x4_t v_p1_mask = vcltq_f32(v_currentx, v_threshold);

            // set bit to 1 if the mask is one -> AND the mask and the onebit registers
            uint32x4_t v_p1_nextbit = vandq_s32(v_p1_mask, v_onebit);

            // insert the bits using OR
            v_p1 = vorrq_s32(v_p1, v_p1_nextbit);

            // shift the onebit left by 1 to pack the next bit
            v_onebit = vshlq_n_s32(v_onebit, 1);

            // store second 32-bits temporarily
            vst1q_s32(p1_hi, v_p1);
        }
    } 
    // otherwise set the high 32-bits to zero
    else {
        for (int i=0; i<4; i++) {
            p1_hi[i] = 0;
        }
    }


    // Store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    int bh, bw, idx;
    int it_diff = -W_fused + W_stride * C_pack;
    while((bh_base >= 0) && (bw_base < kH * kW)) {
        for (int i = 0; i < 4; i++) {
            bh = min((iw_pad + i) / W_stride, OW-1);
            bw = max(iw_pad + i - (OW-1)*W_stride, (iw_pad + i) % W_stride);
            idx = (in * H_fused + bh_base + bh) * W_fused + ((bw_base + bw) * C_pack + ic);
            
            for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
                int32_t *data_ptr = (int32_t *)&qx_row[idx];
                data_ptr[0] = p1_lo[i];
                data_ptr[1] = p1_hi[i];
                idx += it_diff;
            }
        }

        bh_base -= OW;
        bw_base += kW * H_stride;
    }

}

static inline void b2r_ARMx8(float* x, int64_t* qx_row, float* q_thresholds, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW) {

    // initialize vectors
    int32x4_t v_p1_0 = vdupq_n_s32(0);
    int32x4_t v_p1_1 = vdupq_n_s32(0);
    int32x4_t v_onebit = vdupq_n_s32(1);

    float32x4_t v_threshold = vdupq_n_f32(q_thresholds[in]);

    // pack first 32-bits
    int limit = min(C - (ic*cntbits), 32);
    
    for (int bit = 0; bit < limit; bit++) {
        // load 8 floats into two 128-bit vector registers
        float32x4_t v_currentx_0 = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);
        float32x4_t v_currentx_1 = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw + 4]);

        // compare current values with negative threshold
        uint32x4_t v_p1_mask_0 = vcltq_f32(v_currentx_0, v_threshold);
        uint32x4_t v_p1_mask_1 = vcltq_f32(v_currentx_1, v_threshold);

        // set bit to 1 if the mask is one -> AND the mask and the onebit registers
        uint32x4_t v_p1_nextbit_0 = vandq_s32(v_p1_mask_0, v_onebit);
        uint32x4_t v_p1_nextbit_1 = vandq_s32(v_p1_mask_1, v_onebit);

        // insert the bits using OR
        v_p1_0 = vorrq_s32(v_p1_0, v_p1_nextbit_0);
        v_p1_1 = vorrq_s32(v_p1_1, v_p1_nextbit_1);

        // shift the onebit left by 1 to pack the next bit
        v_onebit = vshlq_n_s32(v_onebit, 1);
    }

    // store first 32-bits temporarily
    vst1q_s32(p1_lo, v_p1_0);
    vst1q_s32(p1_lo+4, v_p1_1);
    
    // pack second 32-bits if we have more than 32 lower bits
    if (C - (ic*cntbits) >= 32) {
            
        // reset some vectors
        v_onebit = vdupq_n_s32(1);
        v_p1_0 = vdupq_n_s32(0);
        v_p1_1 = vdupq_n_s32(0);

        // pack bits
        limit = min(C - (ic*cntbits), 64);
        for (int bit = 32; bit < limit; bit++) {
            // load 8 floats into two 128-bit vector registers
            float32x4_t v_currentx_0 = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);
            float32x4_t v_currentx_1 = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw + 4]);

            // compare current values with negative threshold
            uint32x4_t v_p1_mask_0 = vcltq_f32(v_currentx_0, v_threshold);
            uint32x4_t v_p1_mask_1 = vcltq_f32(v_currentx_1, v_threshold);

            // set bit to 1 if the mask is one -> AND the mask and the onebit registers
            uint32x4_t v_p1_nextbit_0 = vandq_s32(v_p1_mask_0, v_onebit);
            uint32x4_t v_p1_nextbit_1 = vandq_s32(v_p1_mask_1, v_onebit);

            // insert the bits using OR
            v_p1_0 = vorrq_s32(v_p1_0, v_p1_nextbit_0);
            v_p1_1 = vorrq_s32(v_p1_1, v_p1_nextbit_1);

            // shift the onebit left by 1 to pack the next bit
            v_onebit = vshlq_n_s32(v_onebit, 1);
        }

        // store second 32-bits temporarily
        vst1q_s32(p1_hi, v_p1_0);
        vst1q_s32(p1_hi+4, v_p1_1);
    } 
    // otherwise set the high 32-bits to zero
    else {
        for (int i=0; i<8; i++) {
            p1_hi[i] = 0;
        }
    }

    // Store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    int bh, bw, idx;
    int it_diff = -W_fused + W_stride * C_pack;
    while((bh_base >= 0) && (bw_base < kH * kW)) {
        for (int i = 0; i < 8; i++) {
            bh = min((iw_pad + i) / W_stride, OW-1);
            bw = max(iw_pad + i - (OW-1)*W_stride, (iw_pad + i) % W_stride);
            idx = (in * H_fused + bh_base + bh) * W_fused + ((bw_base + bw) * C_pack + ic);
            
            for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
                int32_t *data_ptr = (int32_t *)&qx_row[idx];
                data_ptr[0] = p1_lo[i];
                data_ptr[1] = p1_hi[i];
                idx += it_diff;
            }
        }

        bh_base -= OW;
        bw_base += kW * H_stride;
    }

}


static inline void b2r_ARMx4_packed(float* x, int64_t* qx_row, float* q_thresholds, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW) {

    // initialize vectors
    int32x4_t v_p1 = vdupq_n_s32(0);
    int32x4_t v_onebit = vdupq_n_s32(1);

    float32x4_t v_threshold = vdupq_n_f32(q_thresholds[in]);

    // pack first 32-bits
    int limit = min(C - (ic*cntbits), 32);
    // printf("min -> %d\n", limit);
    for (int bit = 0; bit < limit; bit++) {
        // load 4 floats into 128-bit vector register
        float32x4_t v_currentx = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);

        // compare current values with negative threshold
        uint32x4_t v_p1_mask = vcltq_f32(v_currentx, v_threshold);

        // set bit to 1 if the mask is one -> AND the mask and the onebit registers
        uint32x4_t v_p1_nextbit = vandq_s32(v_p1_mask, v_onebit);

        // insert the bits using OR
        v_p1 = vorrq_s32(v_p1, v_p1_nextbit);

        // shift the onebit left by 1 to pack the next bit
        v_onebit = vshlq_n_s32(v_onebit, 1);
    }

    // store first 32-bits temporarily
    vst1q_s32(p1_lo, v_p1);
    
    // pack second 32-bits if we have more than 32 lower bits
    if (C - (ic*cntbits) >= 32) {
            
        // reset some vectors
        v_onebit = vdupq_n_s32(1);
        v_p1 = vdupq_n_s32(0);

        // pack bits
        limit = min(C - (ic*cntbits), 64);
        for (int bit = 32; bit < limit; bit++) {
            // load 4 floats into 128-bit vector register
            float32x4_t v_currentx = vld1q_f32(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);

            // compare current values with negative threshold
            uint32x4_t v_p1_mask = vcltq_f32(v_currentx, v_threshold);

            // set bit to 1 if the mask is one -> AND the mask and the onebit registers
            uint32x4_t v_p1_nextbit = vandq_s32(v_p1_mask, v_onebit);

            // insert the bits using OR
            v_p1 = vorrq_s32(v_p1, v_p1_nextbit);

            // shift the onebit left by 1 to pack the next bit
            v_onebit = vshlq_n_s32(v_onebit, 1);

            // store second 32-bits temporarily
            vst1q_s32(p1_hi, v_p1);
        }
    } 
    // otherwise set the high 32-bits to zero
    else {
        for (int i=0; i<4; i++) {
            p1_hi[i] = 0;
        }
    }


    // Store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    int bh, bw, idx;
    int idx_base, idx_offset;
    while((bh_base >= 0) && (bw_base < kH * kW)) {
        for (int i = 0; i < 4; i++) {
            bh = min((iw_pad + i) / W_stride, OW-1);
            bw = max(iw_pad + i - (OW-1)*W_stride, (iw_pad + i) % W_stride);
            idx = (in * H_fused + bh_base + bh) * W_fused + ((bw_base + bw) * C_pack + ic);

            for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
                idx_base = (idx - idx_offset);

                int32_t *data_ptr = (int32_t *)&qx_row[idx_base];
                data_ptr[0]             = p1_lo[i];
                data_ptr[1]             = p1_hi[i];

                idx -= W_fused;
                idx += W_stride * C_pack;
            }
        }

        bh_base -= OW;
        bw_base += kW * H_stride;
    }

}


static inline void b2r_ARMx4_1x1(float* x, int64_t* qx_row, float* q_thresholds, int C, int in, int ic, int C_pack) {

    // initialize vectors
    int32x4_t v_p1 = vdupq_n_s32(0);
    int32x4_t v_onebit = {1, 2, 4, 8};

    float32x4_t v_threshold = vdupq_n_f32(q_thresholds[in]);


    // pack 32-bits
    int limit = min(C - (ic*32), 32);
    for (int bit = 0; bit < limit; bit+=4) {
        // load 4 floats into 128-bit vector register
        float32x4_t v_currentx = vld1q_f32(&x[in * C + (ic * 32 + bit)]);

        // compare current values with negative threshold
        uint32x4_t v_p1_mask = vcltq_f32(v_currentx, v_threshold);

        // set bit to 1 if the mask is one -> AND the mask and the onebit registers
        uint32x4_t v_p1_nextbit = vandq_s32(v_p1_mask, v_onebit);

        // insert the bits using OR
        v_p1 = vorrq_s32(v_p1, v_p1_nextbit);

        // shift the onebit left by 4  to pack the next channels
        v_onebit = vshlq_n_s32(v_onebit, 4);
    }


    // combine the parts
    int32_t p1_0 = vgetq_lane_s32(v_p1, 0);
    int32_t p1_1 = vgetq_lane_s32(v_p1, 1);
    int32_t p1_2 = vgetq_lane_s32(v_p1, 2);
    int32_t p1_3 = vgetq_lane_s32(v_p1, 3);

    p1_0 = p1_0 | p1_1;
    p1_2 = p1_2 | p1_3;
    p1_0 = p1_0 | p1_2;

    int idx = (in * C_pack + (ic / 2));
    int high = ic % 2;    
    int32_t *data = (int32_t *)&qx_row[idx];
    data[high] = p1_0;
}


static inline void b2r_basic(float* x, int64_t* qx_row, float* q_thresholds, int64_t* onebit, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW) {
    const int priChannel = C / cntbits;
    
    int64_t p1 = 0;
    for (int bit = 0; bit < cntbits; bit++) {
        if (ic < priChannel || (C%cntbits > 0 && bit < C%cntbits)) {
            float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
            if (currentx < q_thresholds[in]) {
                p1 = p1 | onebit[bit]; // Pack -1: 1
            }
        }
    }

    // Store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    for (; (bh_base >= 0) && (bw_base < kH * kW); bh_base -= OW, bw_base += kW * H_stride) {
        int bh = min(iw_pad / W_stride, OW-1);
        int bw = max(iw_pad - (OW-1)*W_stride, iw_pad % W_stride);

        for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
            qx_row[(in * H_fused + bh_base+bh) * W_fused + (bw_base+bw) * C_pack + ic] = p1;
        }
    }
}

int64_t *binarize2row(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride) {
    int64_t onebit[cntbits];
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = 1ll << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    const int C_pack = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int H_pack = H + 2 * H_pad;
    const int W_pack = W + 2 * W_pad;
    const int OH = (H_pack - kH) / H_stride + 1;
    const int OW = (W_pack - kW) / W_stride + 1;
    const int H_fused = OH * OW;
    int padded = kH * kW * C_pack;
    padded = (padded % 2 == 0) ? padded : padded + 2 - (padded%2);
    const int W_fused = padded; 

    // allocate memory
    int64_t* qx_row = aligned_alloc(32, N * H_fused * W_fused * sizeof(int64_t));
    assert(qx_row && "qx_row was null, maybe size was not divisible by 32?");
    memset(qx_row, 0, N * H_fused * W_fused * sizeof(int64_t));

    // special case 1x1 input (fully connected) -> use optimized kernel
    if (H_pack == 1 && W_pack == 1 && C%128 == 0) {
        for (int in = 0; in < N; in++) {
            int ic = 0;
            for (; ic < priChannel; ic++) {
                b2r_ARMx4_1x1(x, qx_row, q_thresholds, C, in, ic*2, C_pack);
                b2r_ARMx4_1x1(x, qx_row, q_thresholds, C, in, ic*2+1, C_pack);
            }
        }
    // general case
    } else {
        for (int in = 0; in < N; in++) {
            for (int ih = 0; ih < H; ih++) {

                // apply padding
                int ih_pad = ih + H_pad;

                // pack the first part (divisible by cntbits -> [0 - priChannel*cntbits])
                for (int ic = 0; ic < priChannel; ic++) {
                    int iw = 0;
                    for (; iw+3 < W; iw+=4) {
                        int iw_pad = iw + W_pad;
                        b2r_ARMx4(x, qx_row, q_thresholds, C, H, W, in, ic, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW);
                    }
                    for (; iw < W; iw++) {
                        int iw_pad = iw + W_pad;
                        b2r_basic(x, qx_row, q_thresholds, onebit, C, H, W, in, ic, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW);
                    }
                }

                // pack the tail part (-> [priChannel*cntbits - C])
                if ((C % cntbits) > 0) {
                    int iw = 0;
                    for (; iw+3 < W; iw+=4) {
                        int iw_pad = iw + W_pad;
                        b2r_ARMx4(x, qx_row, q_thresholds, C, H, W, in, priChannel, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW);
                    }
                    for (; iw < W; iw++) {
                        int iw_pad = iw + W_pad;
                        b2r_basic(x, qx_row, q_thresholds, onebit, C, H, W, in, priChannel, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW);
                    }
                }
            }
        }
    }

    return qx_row;
}

int64_t *b2r(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride) {
    return binarize2row(x, q_thresholds, N, C, H, W, H_pad, W_pad, kH, kW, H_stride, W_stride);
}
