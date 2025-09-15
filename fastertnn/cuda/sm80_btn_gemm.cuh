#ifndef SM80_BTN_GEMM_CUH
#define SM80_BTN_GEMM_CUH

#include <cuda.h>
// #include <assert.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "utils.cuh"
#include "ptx.cuh"

namespace sm80_btn {

constexpr int BIT_PACK_SIZE = 32;   // channel size
constexpr int BIT_PACK_COUNT = 2;   // idot name
constexpr int THREAD_PER_WARP = 32; 
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 16;
constexpr int WARP_X = 4;
constexpr int WARP_Y = 8;
constexpr int NUM_WARP_PER_BLOCK = BLOCK_X*BLOCK_Y/THREAD_PER_WARP;
constexpr int NUM_WARP_PER_BLOCK_X = BLOCK_X/WARP_X;
constexpr int NUM_WARP_PER_BLOCK_Y = BLOCK_Y/WARP_Y;
constexpr int TILE_WIDTH_X = BLOCK_X;   // num of vectors on X-dim
constexpr int TILE_WIDTH_Y = BLOCK_Y;
constexpr int TILE_WIDTH_Y_PER_WARP = TILE_WIDTH_Y/NUM_WARP_PER_BLOCK;

constexpr int NUM_VEC_PER_TILE = TILE_WIDTH_X*TILE_WIDTH_Y;
constexpr int NUM_VEC_PER_WARP = TILE_WIDTH_X*TILE_WIDTH_Y_PER_WARP;

//
constexpr int MMA_TILE_X = 4;   // num of bits-pack per mma on X
constexpr int MMA_TILE_Y = 8;

//
constexpr int TILE_K = 16;  // num vec on K-dim of a tile
constexpr int TILE_K_PADDED = TILE_K + 1;
constexpr int NUM_VEC_PER_BUFFER = TILE_K*TILE_K_PADDED;

/**
 * Refered from TAB
*/
__global__ void btn_gemm_baseline_kernel(
    const int32_t* a, 
    const int32_t* b, 
    int32_t* c, 
    int M, int N, int K
) {

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= N || ty >= M)
        return;

    int32_t cntp1 = 0;
    int32_t cntp2 = 0;
    for (int k = 0; k < K; k++) {
        int32_t p1 = (a[ty * K + k] ^ b[(tx * K + k) * BIT_PACK_COUNT + 0]);
        int32_t p2 = b[(tx * K + k) * BIT_PACK_COUNT + 1];
        cntp1 += __popc(p2);
        cntp2 += __popc(p1 & p2);
    }

    c[ty * N + tx] = cntp1 - 2*cntp2;
}

template<int STAGE>
__global__ void sm80_btn_gemm_multi_stage_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K       // num of bits pack
) {
    static_assert(STAGE >= 2);
    constexpr int num_scalar_per_vec = 4;
    using InputAVec = CustomVec<int32_t, num_scalar_per_vec/2, int32_t>; // 64bit vector
    using InputWVec = CustomVec<int32_t, num_scalar_per_vec>; // 128bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _KA = K/InputAVec::BitsPackPerVec;  // num of vectors on K dim
    const int _KW = K/InputWVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_KA/(TILE_K)); // num of tiles on K dim
    int kTileCnt = kTileNum;
    constexpr int kBlockNum = TILE_K;   // num of blocks each tile
    
    const int tIdx = threadIdx.x;
    const int tIdy = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* cast and slice gmem */
    const InputAVec* tAgmem_r = partition_gmem_thread<InputAVec>(reinterpret_cast<const InputAVec*>(a), row, _KA, tIdx);
    const InputWVec* tBgmem_r = partition_gmem_thread<InputWVec>(reinterpret_cast<const InputWVec*>(b), col, _KW, tIdy);

    /* double smem buffer */
    __shared__ InputAVec smem_a_vec[STAGE*TILE_K_PADDED*TILE_K];
    __shared__ InputWVec smem_b_vec[STAGE*TILE_K_PADDED*TILE_K];
    InputAVec* tAsmem_w = partition_smem_thread<InputAVec, TILE_K_PADDED, false>(smem_a_vec, tIdx, tIdy);
    InputWVec* tBsmem_w = partition_smem_thread<InputWVec, TILE_K_PADDED, true>(smem_b_vec, tIdx, tIdy);

    /* stage ptr */
    int stageWriteIterator = 0;
    int stageReadIterator = 0;
    int tileId = 0;

    // prefetch the tiles except for the last into buffers
    // NOTE: transpose is performed on fly
#pragma unroll
    for (; stageWriteIterator < STAGE-1; ) {
        cp_async_ca_shared_global<InputAVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputAVec, TILE_K>(tAgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
        );
        cp_async_ca_shared_global<InputWVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputWVec, TILE_K>(tBgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tBsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
        );
        cp_async_commit_group();

        // advance tile
        --kTileCnt;
        if (kTileCnt > 0)  tileId++;

        // advance stage writer
        stageWriteIterator++;
    }

    // wait for the first buffer
    cp_async_wait_group<STAGE-2>();
    __syncthreads();

    /* mainloop */
    int cntp1 = 0;
    int cntp2 = 0;
    InputAVec tAreg;
    InputWVec tBreg;
    while (kTileCnt > -(STAGE-1)) {
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            // launch from gmem to smem for next stage tile
            if (k == 0) {
                if (kTileCnt > 0) {
                    cp_async_ca_shared_global<InputAVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputAVec, TILE_K>(tAgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
                    );
                    cp_async_ca_shared_global<InputWVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputWVec, TILE_K>(tBgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tBsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
                    );
                    cp_async_commit_group();
                }
                
                // advance tile
                --kTileCnt;
                if (kTileCnt > 0)  tileId++;
                
                // advance stage writer
                stageWriteIterator = stageReadIterator;
            }

            // block calculation
            tAreg.LoadByVec(smem_a_vec, stageReadIterator*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdy);
            tBreg.LoadByVec(smem_b_vec, stageReadIterator*NUM_VEC_PER_BUFFER + k*TILE_K_PADDED + tIdx);
            btn_loop_mma<InputScalar, InputAVec, InputWVec>(tAreg, tBreg, cntp1, cntp2);

            // wait from gmem to smem for next tile
            if (k == kBlockNum - 1) {
                cp_async_wait_group<STAGE-2>();
                __syncthreads();
                ++stageReadIterator;
                stageReadIterator = stageReadIterator%STAGE;
            }
        }
        // NOTE: cannot be removed when async byte size is small
        cp_async_wait_group<0>();
        __syncthreads();
    }

    /* epilogue */
    c[row*N + col] = cntp1 - 2*cntp2;
}

// TBN host functions

void btn_gemm_baseline(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    btn_gemm_baseline_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}

void sm80_btn_gemm_multi_stage(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    assert((M%16 == 0) && "M-dim has to be multiple of 16");
    assert((N%16 == 0) && "N-dim has to be multiple of 16");
    assert((K%64 == 0) && "K-dim has to be multiple of 64");
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    sm80_btn_gemm_multi_stage_kernel<2><<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}


} // namespace sm80_btn


#endif /* SM80_BTN_GEMM_CUH  */