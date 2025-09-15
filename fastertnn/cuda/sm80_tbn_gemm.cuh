#ifndef SM80_TBN_GEMM_CUH
#define SM80_TBN_GEMM_CUH

#include <cuda.h>
// #include <assert.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "utils.cuh"
#include "ptx.cuh"

namespace sm80_tbn {

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
__global__ void tbn_gemm_baseline_kernel(
    const int32_t* a, 
    const int32_t* b, 
    int32_t* c, 
    int M, int N, int K
) {

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= N || ty >= M)
        return;

    const int KB = K * BIT_PACK_COUNT;
    int32_t s1 = 0;
    int32_t s2 = 0;
    int32_t s3 = 0;
    for (int k = 0; k < K; k++) {
        // int32_t s1 = __popc(b[tx * K + k] & b[tx * K + k]);
        s2 += __popc(a[ty * KB + k * BIT_PACK_COUNT + 0] ^ b[tx * K + k]);
        s3 += __popc(a[ty * KB + k * BIT_PACK_COUNT + 1] ^ b[tx * K + k]);
    }

    c[ty * N + tx] = s1 - 2*s2 + s3;
}

template<int STAGE>
__global__ void sm80_tbn_gemm_multi_stage_m8n8k128_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K       // num of bits pack
) {
    static_assert(STAGE >= 2);
    constexpr int num_scalar_per_vec = 4;
    using OutputVec = CustomVec<int32_t, 2>;
    using InputAVec = CustomVec<int32_t, num_scalar_per_vec>; // 128bit vector
    using InputWVec = CustomVec<int32_t, num_scalar_per_vec/2, int32_t>; // 64bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _KA = K/InputAVec::BitsPackPerVec;  // num of vectors on K dim
    const int _KW = K/InputWVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_KA/(TILE_WIDTH_X)); // num of tiles on K dim
    int kTileCnt = kTileNum;
    constexpr int kBlockNum = TILE_WIDTH_X*InputAVec::BitsPackPerVec/MMA_TILE_X;   // num of blocks each tile
    
    const int tId = threadIdx.x;
    const int laneId = tId%THREAD_PER_WARP;
    const int warpId = tId/THREAD_PER_WARP;

    const int tIdx = tId%BLOCK_X;
    const int tIdy = tId/BLOCK_X;
    const int laneIdx = laneId%WARP_X;
    const int laneIdy = laneId/WARP_X;
    const int warpIdx = warpId%NUM_WARP_PER_BLOCK_X;
    const int warpIdy = warpId/NUM_WARP_PER_BLOCK_X;
    const int blkOffsetA = blockIdx.y * TILE_WIDTH_Y * _KA;
    const int blkOffsetB = blockIdx.x * TILE_WIDTH_Y * _KW;

    /* cast and slice gmem */
    OutputVec* tCgmem_w = reinterpret_cast<OutputVec*>(
        c + blockIdx.y*TILE_WIDTH_Y*N + blockIdx.x*TILE_WIDTH_Y +
        warpIdy*TILE_WIDTH_Y/2*N + warpIdx*TILE_WIDTH_Y/2 +
        laneIdy*N + laneIdx*2
    );
    const InputAVec* tAgmem_r = partition_gmem_thread<InputAVec>(reinterpret_cast<const InputAVec*>(a) + blkOffsetA, tIdy, _KA, tIdx);
    const InputWVec* tBgmem_r = partition_gmem_thread<InputWVec>(reinterpret_cast<const InputWVec*>(b) + blkOffsetB, tIdy, _KW, tIdx);

    /* double smem buffer */
    __shared__ InputAVec smem_a_vec[STAGE*TILE_WIDTH_X*TILE_WIDTH_Y];
    __shared__ InputWVec smem_b_vec[STAGE*TILE_WIDTH_X*TILE_WIDTH_Y];
    InputAVec* tAsmem_w = partition_smem_thread<InputAVec, TILE_WIDTH_X, true>(smem_a_vec, tIdx, tIdy);
    InputWVec* tBsmem_w = partition_smem_thread<InputWVec, TILE_WIDTH_X, true>(smem_b_vec, tIdx, tIdy);

    /* stage ptr */
    int stageWriteIterator = 0;
    int stageReadIterator = 0;
    int tileId = 0;

    // prefetch the tiles except for the last into buffers
    // NOTE: transpose is performed on fly
#pragma unroll
    for (; stageWriteIterator < STAGE-1; ) {
        cp_async_ca_shared_global<InputAVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputAVec, TILE_WIDTH_X>(tAgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
        );
        cp_async_ca_shared_global<InputWVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputWVec, TILE_WIDTH_X>(tBgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tBsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
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
    int cntp1[2] = {0, 0};
    int cntp2[2] = {0, 0};
    int cntp3[2] = {0, 0};
    InputAVec::BitsPackType tAreg;
    InputWVec::BitsPackType tBreg;
    InputAVec::BitsPackType* tAsmem_r = reinterpret_cast<InputAVec::BitsPackType*>(smem_a_vec + warpIdy * NUM_VEC_PER_TILE/2 + laneIdy*TILE_WIDTH_X);
    InputWVec::BitsPackType* tBsmem_r = reinterpret_cast<InputWVec::BitsPackType*>(smem_b_vec + warpIdx * NUM_VEC_PER_TILE/2 + laneIdy*TILE_WIDTH_X);
    while (kTileCnt > -(STAGE-1)) {
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            // launch from gmem to smem for next stage tile
            if (k == 0) {
                if (kTileCnt > 0) {
                    cp_async_ca_shared_global<InputAVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputAVec, TILE_WIDTH_X>(tAgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
                    );
                    cp_async_ca_shared_global<InputWVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputWVec, TILE_WIDTH_X>(tBgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tBsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
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
            tAreg = tAsmem_r[stageReadIterator*NUM_VEC_PER_TILE*InputAVec::BitsPackPerVec + k*MMA_TILE_X + laneIdx];
            tBreg = tBsmem_r[stageReadIterator*NUM_VEC_PER_TILE*InputWVec::BitsPackPerVec + k*MMA_TILE_X + laneIdx];
            // bmma_m8n8k128_and_popc(cntp1, tBreg, tBreg, cntp1);
            bmma_m8n8k128_xor_popc(cntp2, tAreg.x, tBreg, cntp2);
            bmma_m8n8k128_xor_popc(cntp3, tAreg.y, tBreg, cntp3);
            

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
    OutputVec tCreg;
    tCreg.StoreByScalar(cntp1[0] - 2*cntp2[0] + cntp3[0], 0);
    tCreg.StoreByScalar(cntp1[1] - 2*cntp2[1] + cntp3[1], 1);
    tCgmem_w[0] = tCreg;
}

// TBN host functions

void tbn_gemm_baseline(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    tbn_gemm_baseline_kernel<<<dimGrid, dimBlock>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}

void sm80_tbn_gemm_m8n8k128(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K
) {
    assert((M%16 == 0) && "M-dim has to be multiple of 16");
    assert((N%16 == 0) && "N-dim has to be multiple of 16");
    assert((K%32 == 0) && "K-dim has to be multiple of 32");
    dim3 dimGrid(ceil(1.*N/16), ceil(1.*M/16));
    sm80_tbn_gemm_multi_stage_m8n8k128_kernel<2><<<dimGrid, 128>>>(
        a, b, c, M, N, K
    );
    CUDA_ERR_CHECK();
}


} // namespace sm80_tbn

#endif /* SM80_TBN_GEMM_CUH */