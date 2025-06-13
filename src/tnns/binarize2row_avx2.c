#include "tnns/binarize2row.h"
#include <assert.h>

#include <nmmintrin.h>
#include <immintrin.h>


static inline int min(int a, int b) {
    return a < b ? a : b;
}
static inline int max(int a, int b) {
    return a > b ? a : b;
}

static inline void b2r_basic(float* x, int64_t* qx_row, float* q_thresholds, int64_t* onebit, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW, int bits) {
    int64_t p1 = 0;
    for (int bit = 0; bit < bits; bit++) {
        float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
        if (currentx < q_thresholds[in]) {
            p1 = p1 | onebit[bit]; 
        }
    }

    // store packed bits such that the input is flattened
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;

    for (; (bh_base >= 0) && (bw_base < kH * kW); bh_base -= OW, bw_base += kW * H_stride) {
        int bh = min(iw_pad / W_stride, OW-1);
        int bw = max(iw_pad - (OW-1)*W_stride, iw_pad % W_stride);

        for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
            qx_row[(in * H_fused + bh_base+bh) * W_fused + ((bw_base+bw) * C_pack + ic)] = p1;
        }
    }
}

static inline void b2r_AVXx4(float* x, int64_t* qx_row, float* q_thresholds, int C, int H, int W, int in, int ic, int ih, int iw, int ih_pad, int iw_pad, int H_stride, int W_stride, int kH, int kW, int C_pack, int H_fused, int W_fused, int OH, int OW, int bits) {
    // initialize vectors
    __m256i p1 = _mm256_setzero_si256();
    __m256d threshold64 = _mm256_set1_pd(q_thresholds[in]);
    __m256i bitvec = _mm256_set1_epi64x(1);

    for (int64_t bit = 0; bit < bits; bit++) {
        // load 2 float32x4_t from memory and convert to float64x4_t
        __m128 currentx = _mm_load_ps(&x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]);
        __m256d currentx64 = _mm256_cvtps_pd(currentx);

        __m256d mask1 = _mm256_cmp_pd(currentx64, threshold64, _MM_CMPINT_LT);

        // set bits of p1 and p2 based on mask1 and mask2 and bitvec
        p1 = _mm256_or_si256(p1, _mm256_and_si256((__m256i)mask1, bitvec));

        // shift bit to left
        bitvec = _mm256_slli_epi64(bitvec, 1);
    }

    // store as int64 arrays
    int64_t p1i[4] __attribute__((__aligned__(32)));
    _mm256_store_si256((__m256i*)p1i, p1);

    // write packed bits to correct position in img2row matrix
    int bh_base = min(ih_pad / H_stride, OH-1) * OW;
    int bw_base = max(ih_pad - (OH-1)*H_stride, ih_pad % H_stride) * kW;
    while((bh_base >= 0) && (bw_base < kH * kW)) {
        for(int i = 0; i < 4; ++i) {
            int bh = min((iw_pad + i) / W_stride, OW-1);
            int bw = max((iw_pad + i) - (OW-1)*W_stride, (iw_pad + i) % W_stride);

            int index_qrow = (in * H_fused + bh_base + bh) * W_fused + ((bw_base + bw) * C_pack + ic);
            int64_t iteration_diff = -W_fused + W_stride * C_pack;

            for (; (bh >= 0) && (bw < kW); bh -= 1, bw += W_stride) {
                qx_row[index_qrow] = p1i[i];
                index_qrow += iteration_diff;
            }
        }
        bh_base -= OW;
        bw_base += kW * H_stride;
    }
}

static inline void b2r_AVXx8_packing2_1x1(float* x, int64_t* qx_row, float* q_thresholds, int C, int in, int ic, int C_pack) {
    // initialize vectors
    __m256i v_p1 = _mm256_setzero_si256();
    __m256i v_p2 = _mm256_setzero_si256();
    __m256i v_onebit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256 v_thresh = _mm256_set1_ps(q_thresholds[in]);

    // pack 32-bits
    int limit = min(C - (ic*32), 32);
    for (int bit = 0; bit < limit; bit += 8) {
        // load 8 floats into 256-bit vector register
        float* base_ptr = &x[((in * C + (ic * 32 + bit)))];

        __m256 v_currentx;
        if((size_t)base_ptr % 32 != 0) {
            v_currentx = _mm256_loadu_ps(base_ptr);
        } else {
            v_currentx = _mm256_load_ps(base_ptr);
        }

        // compare mask based on negative and positive thresholds
        __m256 v_p1_mask = _mm256_cmp_ps(v_currentx, v_thresh, _MM_CMPINT_LT);
        __m256i v_p1_nextbit = _mm256_and_si256((__m256i)v_p1_mask, v_onebit);

        // insert the bits using OR
        v_p1 = _mm256_or_si256(v_p1, v_p1_nextbit);

        // shift the onebit left by 4 to pack the next bit
        v_onebit = _mm256_slli_epi32(v_onebit, 8);
    }

    // combine the parts 
    // or/add are equivalent here as we bit i is set in at most one vector
    __m256i res1 = _mm256_hadd_epi32(v_p1, v_p2);   
    __m256i res2 = _mm256_hadd_epi32(res1, res1);
    int64_t p12_0 = _mm256_extract_epi64(res2, 0);
    int64_t p12_4 = _mm256_extract_epi64(res2, 2);
    int64_t p12 = p12_0 | p12_4;
    int32_t p1_0 = (int32_t)(p12 & 0xFFFFFFFF);

    // store to qx_row
    int idx = (in * C_pack) + ((ic / 2) / 4) * 4 + ((ic / 2) % 4);
    int high = ic % 2;
    
    int32_t *data = (int32_t *) &qx_row[idx];
    data[high] = p1_0;
}

int64_t *b2r(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride) {
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

    const int W_fused = 4 * ((kH * kW * C_pack + 3) / 4); 

    // allocate memory
    int64_t* qx_row = aligned_alloc(32, N * H_fused * W_fused * sizeof(int64_t));
    assert(qx_row && "qx_row was null, maybe size was not divisible by 32?");
    memset(qx_row, 0, N * H_fused * W_fused * sizeof(int64_t));

    if (H_pack == 1 && W_pack == 1 && C % 256 == 0) {
        for (int in = 0; in < N; in++) {
            int ic = 0;
            for (; ic < priChannel; ic++) {
                b2r_AVXx8_packing2_1x1(x, qx_row, q_thresholds, C, in, 2*ic, C_pack);
                b2r_AVXx8_packing2_1x1(x, qx_row, q_thresholds, C, in, 2*ic+1, C_pack);

            }
        }
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
                        b2r_AVXx4(x, qx_row, q_thresholds, C, H, W, in, ic, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW, cntbits);
                    }
                    for (; iw < W; iw++) {
                        int iw_pad = iw + W_pad;
                        b2r_basic(x, qx_row, q_thresholds, onebit, C, H, W, in, ic, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW, cntbits);
                    }
                }

                // pack the tail part (-> [priChannel*cntbits - C])
                if ((C % cntbits) > 0) {
                    int iw = 0;
                    for (; iw+3 < W; iw+=4) {
                        int iw_pad = iw + W_pad;
                        b2r_AVXx4(x, qx_row, q_thresholds, C, H, W, in, priChannel, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW, (C % cntbits));
                    }
                    for (; iw < W; iw++) {
                        int iw_pad = iw + W_pad;
                        b2r_basic(x, qx_row, q_thresholds, onebit, C, H, W, in, priChannel, ih, iw, ih_pad, iw_pad, H_stride, W_stride, kH, kW, C_pack, H_fused, W_fused, OH, OW, (C % cntbits));
                    }
                }
            }
        }
    }

    return qx_row;
}