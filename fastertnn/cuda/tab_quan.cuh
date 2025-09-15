#ifndef TAB_QUAN_CUH
#define TAB_QUAN_CUH

#include "utils.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <iostream>

namespace tab_quan
{

#define FULL_MASK 0xFFFFFFFF
constexpr int BIT_PACK_SIZE = 32;   // channel size
constexpr int BIT_PACK_COUNT = 2;   // idot name
constexpr int ELE_PER_THREAD = 4;   // element to process each thread
constexpr int CHANNEL_PER_WARP = ELE_PER_THREAD;
constexpr int THREAD_PER_WARP = 32; 
constexpr int ELE_PER_WARP = THREAD_PER_WARP*ELE_PER_THREAD;
constexpr int VEC_PER_WARP = ELE_PER_WARP/ELE_PER_THREAD;
constexpr int BIT_LEN_PER_WARP = ELE_PER_WARP*2;
constexpr int BYTE_LEN_PER_WARP = BIT_LEN_PER_WARP/8;
constexpr int BYTE_LEN_PER_THREAD = BYTE_LEN_PER_WARP/32;

constexpr int BLOCK_SIZE = 128;
constexpr int WARP_PER_BLOCK = BLOCK_SIZE/THREAD_PER_WARP;
constexpr int ELE_PER_BLOCK = BLOCK_SIZE*ELE_PER_THREAD;

/**
 * Refered from TAB
*/
__global__ void ternary_quantization_baseline_kernel(
    int32_t* t, 
    const float* a, 
    const float* ths, 
    const int N, const  int H, const int W, const int C
) {
    int packedC = ceil((float)C/BIT_PACK_SIZE);
    int currentC = threadIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N*H*W)   return;

    uint32_t bit0 = 0;
    uint32_t bit1 = 0;
    uint32_t mask_msb = (1UL << 31);
    float threshold = ths[row/(H*W)];

#pragma unroll
    for (int i = 0; i < BIT_PACK_SIZE-1; i++) {
        int offset = threadIdx.x*BIT_PACK_SIZE + i;
        if (row < N*H*W && offset < C) {
            float val = a[row*C + offset];
            if (val > threshold) {
                bit1 |= mask_msb;
            } else if (val < -threshold) {
                bit0 |= mask_msb;
                bit1 |= mask_msb;
            }
        } 
        bit0 >>= 1;
        bit1 >>= 1;
    }

    if ((row < N*H*W) && (threadIdx.x*BIT_PACK_SIZE + BIT_PACK_SIZE - 1 < C)) {
        float val = a[row*C + threadIdx.x*BIT_PACK_SIZE + BIT_PACK_SIZE - 1];
        if (val > threshold) {
            bit1 |= mask_msb;
        } else if (val < -threshold) {
            bit0 |= mask_msb;
            bit1 |= mask_msb;
        } 
    }

    /* write back */
    t[row*packedC*BIT_PACK_COUNT + currentC*BIT_PACK_COUNT + 0] = bit0;
    t[row*packedC*BIT_PACK_COUNT + currentC*BIT_PACK_COUNT + 1] = bit1;
}

void ternary_quantization_baseline(
    int32_t* t, 
    float* a,
    float* threshold,
    const int N, const  int H, const int W, const int C
) {
    int blkx = ceil((float)C/BIT_PACK_SIZE);;
    int blky = 32;
    int gridx = 1;
    int gridy = ceil((float)N*H*W/blky);
    dim3 dimBlock(blkx, blky);
    dim3 dimGrid(gridx, gridy);

    ternary_quantization_baseline_kernel<<<dimGrid, dimBlock>>>(
        t, a, threshold, 
        N, H, W, C
    );
    CUDA_ERR_CHECK();
}


__global__ void ternary_quantization_kernel(
    int32_t* t, 
    const float* __restrict__ a, 
    const float* __restrict__ ths, 
    const int N, const  int H, const int W, const int C
) {
    using InputVec = CustomVec<float, ELE_PER_THREAD>;
    using QuanVec = CustomVec<float, BIT_PACK_SIZE>;
    using OutputVec = CustomVec<int2, ELE_PER_THREAD>;

    const InputVec* a_vec = reinterpret_cast<const InputVec*>(a + blockIdx.x * ELE_PER_BLOCK + threadIdx.x * ELE_PER_THREAD);
    OutputVec* t_vec = reinterpret_cast<OutputVec*>(t);
    OutputVec output_fragment;
    int2 local_bits_set;
    __shared__ InputVec smem[BLOCK_SIZE];

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREAD_PER_WARP;
    int local_warp_id = threadIdx.x / THREAD_PER_WARP;
    int lane_id = threadIdx.x % THREAD_PER_WARP;
    bool lane_predicate = (lane_id == 0);

    // vec load into smem
    smem[threadIdx.x] = *a_vec;
    __syncwarp();

    // reinterpret
    QuanVec* quan_smem = reinterpret_cast<QuanVec*>(smem + local_warp_id*THREAD_PER_WARP);
#pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        int thrId = (warp_id*ELE_PER_WARP + i*BIT_PACK_SIZE)/(H*W*C);
        const float threshold = ths[thrId];

        float ele = quan_smem[i].ReadAsScalar(lane_id);
        bool pos_one = ele > threshold;
        bool neg_one = ele < -threshold;

        // quantization
        local_bits_set.x = __ballot_sync(FULL_MASK, neg_one);
        local_bits_set.y = __ballot_sync(FULL_MASK, pos_one || neg_one);
        output_fragment.StoreByScalar(local_bits_set, i);
    }

    // only one thread writes back
    if (lane_predicate) {
        t_vec[warp_id] = output_fragment;
    }
}

void ternary_quantization(
    int32_t* t, 
    float* a,
    float* threshold,
    const int N, const  int H, const int W, const int C
) {
    assert(C%BIT_PACK_SIZE == 0 && "C has to be multiple of 32");
    assert((N*H*W*C)%ELE_PER_BLOCK == 0 && "NHWC has to be multiple of 512");
    int total_element = N*H*W*C;
    int blk_num = total_element/ELE_PER_BLOCK;

    ternary_quantization_kernel<<<blk_num, BLOCK_SIZE>>>(
        t, a, threshold, 
        N, H, W, C
    );
    CUDA_ERR_CHECK();
}

/**
 * Refered from TAB
*/
__global__ void binary_quantization_baseline_kernel(
    int32_t* t, 
    const float* a, 
    const float* ths, 
    const int N, const  int H, const int W, const int C
) {
    int packedC = ceil((float)C/BIT_PACK_SIZE);
    int currentC = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N*H*W)   return;

    uint32_t bit1 = 0;
    uint32_t mask_msb = (1UL << 31);
    float threshold = ths[row/(H*W)];

#pragma unroll
    for (int i = 0; i < BIT_PACK_SIZE-1; i++) {
        int offset = threadIdx.x*BIT_PACK_SIZE + i;
        if (offset < C) {
            float val = a[row*C + offset];
            if (val < -threshold) {
                bit1 |= mask_msb;
            }
        } 
        bit1 >>= 1;
    }

    if (threadIdx.x*BIT_PACK_SIZE + BIT_PACK_SIZE - 1 < C) {
        float val = a[row*C + threadIdx.x*BIT_PACK_SIZE + BIT_PACK_SIZE - 1];
        if (val < -threshold) {
            bit1 |= mask_msb;
        } 
    }

    t[row*packedC + currentC] = bit1;
}

void binary_quantization_baseline(
    int32_t* t, 
    float* a,
    float* threshold,
    const int N, const  int H, const int W, const int C
) {
    int blkx = ceil((float)C/BIT_PACK_SIZE);;
    int blky = 32;
    int gridx = 1;
    int gridy = ceil((float)N*H*W/blky);
    dim3 dimBlock(blkx, blky);
    dim3 dimGrid(gridx, gridy);

    binary_quantization_baseline_kernel<<<dimGrid, dimBlock>>>(
        t, a, threshold, 
        N, H, W, C
    );
    CUDA_ERR_CHECK();
}

__global__ void binary_quantization_kernel(
    int32_t* t, 
    const float* __restrict__ a, 
    const float* __restrict__ ths, 
    const int N, const  int H, const int W, const int C) {
    using InputVec = CustomVec<float, ELE_PER_THREAD>;
    using QuanVec = CustomVec<float, BIT_PACK_SIZE>;
    using OutputVec = CustomVec<int32_t, ELE_PER_THREAD>;

    const InputVec* a_vec = reinterpret_cast<const InputVec*>(a + blockIdx.x * ELE_PER_BLOCK + threadIdx.x * ELE_PER_THREAD);
    OutputVec* t_vec = reinterpret_cast<OutputVec*>(t);
    OutputVec output_fragment;
    int32_t local_bits_set;
    __shared__ InputVec smem[BLOCK_SIZE];

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREAD_PER_WARP;
    int local_warp_id = threadIdx.x / THREAD_PER_WARP;
    int lane_id = threadIdx.x % THREAD_PER_WARP;
    bool lane_predicate = (lane_id == 0);

    // vec load into smem
    smem[threadIdx.x] = *a_vec;
    __syncwarp();

    // reinterpret
    QuanVec* quan_smem = reinterpret_cast<QuanVec*>(smem + local_warp_id*THREAD_PER_WARP);
#pragma unroll
    for (int i = 0; i < ELE_PER_THREAD; i++) {
        int thrId = (warp_id*ELE_PER_WARP + i*BIT_PACK_SIZE)/(H*W*C);
        const float threshold = ths[thrId];

        float ele = quan_smem[i].ReadAsScalar(lane_id);
        bool neg_one = ele < -threshold;

        // quantization
        local_bits_set = __ballot_sync(FULL_MASK, neg_one);
        output_fragment.StoreByScalar(local_bits_set, i);
    }

    // only one thread writes back
    if (lane_predicate) {
        t_vec[warp_id] = output_fragment;
    }
}

void binary_quantization(
    int32_t* t, 
    float* a,
    float* threshold,
    const int N, const  int H, const int W, const int C
) {
    assert((C%BIT_PACK_SIZE == 0) && "C has to be multiple of 32");
    assert(((N*H*W*C)%ELE_PER_BLOCK == 0) && "NHWC has to be multiple of 512");
    int total_element = N*H*W*C;
    int blk_num = total_element/ELE_PER_BLOCK;

    binary_quantization_kernel<<<blk_num, BLOCK_SIZE>>>(
        t, a, threshold, 
        N, H, W, C
    );
    CUDA_ERR_CHECK();
}

// refered from TAB
template<typename T>
__global__ void pad_kernel(
    T* y, 
    const T* a, 
    const int N, const int H, const int W, const int C, 
    const int P1, const int P2
) {
    const int n = blockIdx.x;
    const int h = threadIdx.x;

    const int PH = H + P1 * 2;
    const int PW = W + P2 * 2;

    T* ybase = y + ((n * PH + h + P1) * PW + P2) * C;
    const T* abase = a + (n * H + h) * W * C;

    for (int wc = 0; wc < W * C; wc++) {
        ybase[wc] = abase[wc];
    }
}

template<typename T>
void pad(
    T* output, 
    const T* input, 
    const int N, const int H, const int W, const int C, 
    const int P1, const int P2
) {
    pad_kernel<T><<<N, H>>>(
        output, input,
        N, H, W, C,
        P1, P2
    );
    CUDA_ERR_CHECK();
}

// refered from TAB
template<typename T>
__global__ void img2row_kernel(
    T* y, 
    const T* a, 
    const int N, const int H, const int W, const int C, 
    const int KH, const int KW, const int S1, const int S2
) {
    const int n = blockIdx.x;
    const int oh = blockIdx.y;
    const int ow = threadIdx.x;

    const int OH = (H - KH + 1) / S1;
    const int OW = (W - KW + 1) / S2;
    const int H1 = OH * OW;
    const int W1 = C * KH * KW;
    const int KWC = KW * C;

    int h = oh * S1;
    int w = ow * S2;
    T* ybase = y + (n * H1 + oh * OW + ow) * W1;
    const T* abase = a + ((n * H + h) * W + w) * C;
    for (int kh = 0; kh < KH; kh++) {
        for (int kwc = 0; kwc < KWC; kwc++) {
            ybase[(kh * KWC + kwc)] = abase[kh * W * C + kwc];
        }
    }
}

template<typename T>
void img2row(
    T* output, 
    const T* input, 
    const int N, const int H, const int W, const int C, 
    const int KH, const int KW, const int S1, const int S2
) {
    int OH = (H - KH + 1) / S1;     // output height
    int OW = (W - KW + 1) / S2;     // output width
    dim3 dimGrid(N, OH);
    img2row_kernel<T><<<dimGrid, OW>>>(
        output, input, N, H, W, C,
        KH, KW, S1, S2
    );
    CUDA_ERR_CHECK();
}
    
} // namespace tab_quan



#endif