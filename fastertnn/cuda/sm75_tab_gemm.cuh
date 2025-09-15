#ifndef TAB_GEMM_CUH
#define TAB_GEMM_CUH

#include <cuda.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "utils.cuh"
#include "ptx.cuh"

namespace sm75_tab {

// popcnt in cuda
#define popcnt64(a) __popcll(a) 
#define popcnt32(a) __popc(a)
#define MULTIPLY(a, b) ((a) * (b))

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

constexpr int TILE_K = 16;  // num vec on K-dim of a tile
constexpr int TILE_K_PADDED = TILE_K + 1;

constexpr int NUM_VEC_PER_BUFFER = TILE_K*TILE_K_PADDED;

__global__ void tnn_gemm_baseline_kernel(
    const int32_t* a, 
    const int32_t* b, 
    int32_t* c, 
    int M, 
    int N, 
    int K
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= N || ty >= M)
        return;

    const int KB = K * BIT_PACK_COUNT;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;
    int s4 = 0;

    // int p1 = 0;
    // int p2 = 0;
    // int cntp1 = 0;
    // int cntp2 = 0;
// #pragma unroll
    for (int i = 0; i < KB; i = i + BIT_PACK_COUNT) {
        // curr
        s1 += popcnt32(a[ty * KB + i + 0] & b[tx * KB + i + 0]);
        s2 += popcnt32(a[ty * KB + i + 0] & b[tx * KB + i + 1]);
        s3 += popcnt32(a[ty * KB + i + 1] & b[tx * KB + i + 0]);
        s4 += popcnt32(a[ty * KB + i + 1] & b[tx * KB + i + 1]);

        // prev
        // p1 = a[ty * KB + i + 0] ^ b[tx * KB + i + 0];
        // p2 = a[ty * KB + i + 1] & b[tx * KB + i + 1];
        // cntp1 = cntp1 + popcnt32(p2);
        // cntp2 = cntp2 + popcnt32(p1 & p2);
    }
    // int prev_res = cntp1 - cntp2*2;
    c[ty*N + tx] = 4*s1 - 2*s2 - 2*s3 + s4;
}

void tnn_gemm_baseline(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    tnn_gemm_baseline_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}


__global__ void sm75_tnn_gemm_two_stage_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K       // num of bits pack
) {
    constexpr int stage = 2;
    constexpr int num_scalar_per_vec = 4;
    using InputVec = CustomVec<int32_t, num_scalar_per_vec>; // 128bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _K = K/InputVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_K/(TILE_K)); // num of tiles on K dim
    constexpr int kBlockNum = TILE_K;   // num of blocks each tile
    
    const int tIdx = threadIdx.x;
    const int tIdy = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* cast gmem pointer */
    const InputVec* a_vec = reinterpret_cast<const InputVec*>(a);
    const InputVec* b_vec = reinterpret_cast<const InputVec*>(b);
    a_vec = partition_gmem_thread<InputVec>(a_vec, row, _K, tIdx);
    b_vec = partition_gmem_thread<InputVec>(b_vec, col, _K, tIdy);

    /* double smem buffer */
    __shared__ InputVec smem_a_vec[stage*TILE_K_PADDED*TILE_K];
    __shared__ InputVec smem_b_vec[stage*TILE_K_PADDED*TILE_K];

    /* double buffer ptr */
    int buf = 0;

    // prefetch the first tile into first buffer
    // NOTE: transpose is performed on fly
    smem_a_vec[tIdx*TILE_K_PADDED + tIdy] = a_vec[0];
    smem_b_vec[tIdy*TILE_K_PADDED + tIdx] = b_vec[0];

    /* mainloop */
    int cntp1 = 0;
    int cntp2 = 0;
    int tileId = 0;
    InputVec fragment_a;
    InputVec fragment_b;
    do {
        // sync gmem to smem pipe write of previous buffer
        __syncthreads();

        tileId++;
        // launch async copy from gmem to smem
        buf ^= 1;
        if (tileId < kTileNum) {
            smem_a_vec[buf*NUM_VEC_PER_BUFFER + tIdx*TILE_K_PADDED + tIdy] = partition_gmem_tile<InputVec, TILE_K>(a_vec, tileId)[0];
            smem_b_vec[buf*NUM_VEC_PER_BUFFER + tIdy*TILE_K_PADDED + tIdx] = partition_gmem_tile<InputVec, TILE_K>(b_vec, tileId)[0];
        }

        // start tile calculation of prefetched tile
        buf ^= 1;
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            fragment_a.LoadByVec(smem_a_vec, buf*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdy);
            fragment_b.LoadByVec(smem_b_vec, buf*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdx);
            tnn_loop_mma<InputScalar, InputVec>(fragment_a, fragment_b, cntp1, cntp2);
        }

        // sync gmem to smem pipe write
        buf ^= 1;
    } while (tileId < kTileNum);

    /* epilogue */
    c[row*N + col] = cntp1 - cntp2 - cntp2;
}


void sm75_tnn_gemm(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    assert((M%16 == 0) && "M-dim has to be multiple of 64");
    assert((N%16 == 0) && "N-dim has to be multiple of 64");
    assert((K%32 == 0) && "K-dim has to be multiple of 32");
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    sm75_tnn_gemm_two_stage_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}


__global__ void bnn_gemm_baseline_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, int N, int K, int NUM) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= N || ty >= M)
        return;

    int cnt = 0;
#pragma unroll
    for (int i = 0; i < K; i++) {
        cnt = cnt + __popc(a[ty * K + i] ^ b[tx * K + i]);
    }
    c[ty * N + tx] = NUM - cnt - cnt;
}

void bnn_gemm_baseline(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K, int NUM
) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    bnn_gemm_baseline_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K, NUM
    );
    CUDA_ERR_CHECK();
}


__global__ void sm75_bnn_gemm_two_stage_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K,       // num of bits pack,
    int NUM
) {
    constexpr int stage = 2;
    constexpr int num_scalar_per_vec = 4;
    using InputVec = CustomVec<int32_t, num_scalar_per_vec, int32_t>; // 128bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _K = K/InputVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_K/(TILE_K)); // num of tiles on K dim
    constexpr int kBlockNum = TILE_K;   // num of blocks each tile
    
    const int tIdx = threadIdx.x;
    const int tIdy = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* cast gmem pointer */
    const InputVec* a_vec = reinterpret_cast<const InputVec*>(a);
    const InputVec* b_vec = reinterpret_cast<const InputVec*>(b);
    a_vec = partition_gmem_thread<InputVec>(a_vec, row, _K, tIdx);
    b_vec = partition_gmem_thread<InputVec>(b_vec, col, _K, tIdy);

    /* double smem buffer */
    __shared__ InputVec smem_a_vec[stage*TILE_K_PADDED*TILE_K];
    __shared__ InputVec smem_b_vec[stage*TILE_K_PADDED*TILE_K];

    /* double buffer ptr */
    int buf = 0;

    // prefetch the first tile into first buffer
    // NOTE: transpose is performed on fly
    smem_a_vec[tIdx*TILE_K_PADDED + tIdy] = a_vec[0];
    smem_b_vec[tIdy*TILE_K_PADDED + tIdx] = b_vec[0];

    /* mainloop */
    int cntp = 0;
    int tileId = 0;
    InputVec fragment_a;
    InputVec fragment_b;
    do {
        // sync gmem to smem pipe write of previous buffer
        cp_async_wait_group<0>();
        __syncthreads();

        tileId++;
        // launch async copy from gmem to smem
        buf ^= 1;
        if (tileId < kTileNum) {
            smem_a_vec[buf*NUM_VEC_PER_BUFFER + tIdx*TILE_K_PADDED + tIdy] = partition_gmem_tile<InputVec, TILE_K>(a_vec, tileId)[0];
            smem_b_vec[buf*NUM_VEC_PER_BUFFER + tIdy*TILE_K_PADDED + tIdx] = partition_gmem_tile<InputVec, TILE_K>(b_vec, tileId)[0];
        }

        // start tile calculation of prefetched tile
        buf ^= 1;
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            fragment_a.LoadByVec(smem_a_vec, buf*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdy);
            fragment_b.LoadByVec(smem_b_vec, buf*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdx);
            bnn_loop_mma<InputScalar, InputVec>(fragment_a, fragment_b, cntp);
        }

        // sync gmem to smem pipe write
        buf ^= 1;
    } while (tileId < kTileNum);

    /* epilogue */
    c[row*N + col] = NUM - 2*cntp;
}

void sm75_bnn_gemm(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K, int NUM
) {
    assert((M%16 == 0) && "M-dim has to be multiple of 64");
    assert((N%16 == 0) && "N-dim has to be multiple of 64");
    assert((K%64 == 0) && "K-dim has to be multiple of 64");
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    sm75_bnn_gemm_two_stage_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K, NUM
    );
    CUDA_ERR_CHECK();
}



} // namespace sm75_tab
#endif /* TAB_GEMM_CUH */