#ifndef SM80_BNN_GEMM_CUH
#define SM80_BNN_GEMM_CUH

#include <cuda.h>
// #include <assert.h>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "utils.cuh"
#include "ptx.cuh"

namespace sm80_bnn {

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

////// Device kernels for BNN

/**
 * Refered from TAB
*/
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

template<int STAGE>
__global__ void sm80_bnn_gemm_multi_stage_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K,       // num of bits pack
    int NUM
) {
    static_assert(STAGE >= 2);
    constexpr int num_scalar_per_vec = 4;
    using InputVec = CustomVec<int32_t, num_scalar_per_vec, int32_t>; // 128bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _K = K/InputVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_K/(TILE_K)); // num of tiles on K dim
    int kTileCnt = kTileNum;
    constexpr int kBlockNum = TILE_K;   // num of blocks each tile
    
    const int tIdx = threadIdx.x;
    const int tIdy = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* cast and slice gmem */
    const InputVec* tAgmem_r = partition_gmem_thread<InputVec>(reinterpret_cast<const InputVec*>(a), row, _K, tIdx);
    const InputVec* tBgmem_r = partition_gmem_thread<InputVec>(reinterpret_cast<const InputVec*>(b), col, _K, tIdy);

    /* double smem buffer */
    __shared__ InputVec smem_a_vec[STAGE*TILE_K_PADDED*TILE_K];
    __shared__ InputVec smem_b_vec[STAGE*TILE_K_PADDED*TILE_K];
    InputVec* tAsmem_w = partition_smem_thread<InputVec, TILE_K_PADDED, false>(smem_a_vec, tIdx, tIdy);
    InputVec* tBsmem_w = partition_smem_thread<InputVec, TILE_K_PADDED, true>(smem_b_vec, tIdx, tIdy);

    /* stage ptr */
    int stageWriteIterator = 0;
    int stageReadIterator = 0;
    int tileId = 0;

    // prefetch the tiles except for the last into buffers
    // NOTE: transpose is performed on fly
#pragma unroll
    for (; stageWriteIterator < STAGE-1; ) {
        cp_async_ca_shared_global<InputVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_K>(tAgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
        );
        cp_async_ca_shared_global<InputVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_K>(tBgmem_r, tileId)),
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
    int cntp = 0;
    InputVec tAreg;
    InputVec tBreg;
    while (kTileCnt > -(STAGE-1)) {
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            // launch from gmem to smem for next stage tile
            if (k == 0) {
                if (kTileCnt > 0) {
                    cp_async_ca_shared_global<InputVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_K>(tAgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_BUFFER)
                    );
                    cp_async_ca_shared_global<InputVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_K>(tBgmem_r, tileId)),
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
            bnn_loop_mma<InputScalar, InputVec>(tAreg, tBreg, cntp);

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
    c[row*N + col] = NUM - cntp - cntp;
}

template<int STAGE>
__global__ void sm80_bnn_gemm_multi_stage_m8n8k128_kernel(
    const int32_t* __restrict__ a, 
    const int32_t* __restrict__ b, 
    int32_t* c, 
    int M, 
    int N, 
    int K,       // num of bits pack
    int NUM
) {
    static_assert(STAGE >= 2);
    constexpr int num_scalar_per_vec = 4;
    using OutputVec = CustomVec<int32_t, 2>;
    using InputVec = CustomVec<int32_t, num_scalar_per_vec, int32_t>; // 128bit vector
    using InputScalar = int32_t;              // 32bit scalar
    const int _K = K/InputVec::BitsPackPerVec;  // num of vectors on K dim
    const int kTileNum = (int)ceil(1.*_K/(TILE_WIDTH_X)); // num of tiles on K dim
    int kTileCnt = kTileNum;
    constexpr int kBlockNum = TILE_WIDTH_X*InputVec::BitsPackPerVec/MMA_TILE_X;   // num of blocks each tile
    
    const int tId = threadIdx.x;
    const int laneId = tId%THREAD_PER_WARP;
    const int warpId = tId/THREAD_PER_WARP;

    const int tIdx = tId%BLOCK_X;
    const int tIdy = tId/BLOCK_X;
    const int laneIdx = laneId%WARP_X;
    const int laneIdy = laneId/WARP_X;
    const int warpIdx = warpId%NUM_WARP_PER_BLOCK_X;
    const int warpIdy = warpId/NUM_WARP_PER_BLOCK_X;
    const int blkOffsetA = blockIdx.y * TILE_WIDTH_Y * _K;
    const int blkOffsetB = blockIdx.x * TILE_WIDTH_Y * _K;

    /* cast and slice gmem */
    OutputVec* tCgmem_w = reinterpret_cast<OutputVec*>(
        c + blockIdx.y*TILE_WIDTH_Y*N + blockIdx.x*TILE_WIDTH_Y +
        warpIdy*TILE_WIDTH_Y/2*N + warpIdx*TILE_WIDTH_Y/2 +
        laneIdy*N + laneIdx*2
    );
    const InputVec* tAgmem_r = partition_gmem_thread<InputVec>(reinterpret_cast<const InputVec*>(a) + blkOffsetA, tIdy, _K, tIdx);
    const InputVec* tBgmem_r = partition_gmem_thread<InputVec>(reinterpret_cast<const InputVec*>(b) + blkOffsetB, tIdy, _K, tIdx);

    /* double smem buffer */
    __shared__ InputVec smem_a_vec[STAGE*TILE_WIDTH_X*TILE_WIDTH_Y];
    __shared__ InputVec smem_b_vec[STAGE*TILE_WIDTH_X*TILE_WIDTH_Y];
    InputVec* tAsmem_w = partition_smem_thread<InputVec, TILE_WIDTH_X, true>(smem_a_vec, tIdx, tIdy);
    InputVec* tBsmem_w = partition_smem_thread<InputVec, TILE_WIDTH_X, true>(smem_b_vec, tIdx, tIdy);

    /* stage ptr */
    int stageWriteIterator = 0;
    int stageReadIterator = 0;
    int tileId = 0;

    // prefetch the tiles except for the last into buffers
    // NOTE: transpose is performed on fly
#pragma unroll
    for (; stageWriteIterator < STAGE-1; ) {
        cp_async_ca_shared_global<InputVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_WIDTH_X>(tAgmem_r, tileId)),
            reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
        );
        cp_async_ca_shared_global<InputVec::BytesPerVec>(
            reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_WIDTH_X>(tBgmem_r, tileId)),
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
    int cntp[2] = {0};
    InputVec::BitsPackType tAreg;
    InputVec::BitsPackType tBreg;
    InputVec::BitsPackType* tAsmem_r = reinterpret_cast<InputVec::BitsPackType*>(smem_a_vec + warpIdy * NUM_VEC_PER_TILE/2 + laneIdy*TILE_WIDTH_X);
    InputVec::BitsPackType* tBsmem_r = reinterpret_cast<InputVec::BitsPackType*>(smem_b_vec + warpIdx * NUM_VEC_PER_TILE/2 + laneIdy*TILE_WIDTH_X);
    while (kTileCnt > -(STAGE-1)) {
#pragma unroll
        for (int k = 0; k < kBlockNum; k += 1) {
            // launch from gmem to smem for next stage tile
            if (k == 0) {
                if (kTileCnt > 0) {
                    cp_async_ca_shared_global<InputVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_WIDTH_X>(tAgmem_r, tileId)),
                        reinterpret_cast<int32_t*>(tAsmem_w + stageWriteIterator*NUM_VEC_PER_TILE)
                    );
                    cp_async_ca_shared_global<InputVec::BytesPerVec>(
                        reinterpret_cast<const int32_t*>(partition_gmem_tile<InputVec, TILE_WIDTH_X>(tBgmem_r, tileId)),
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
            tAreg = tAsmem_r[stageReadIterator*NUM_VEC_PER_TILE*InputVec::BitsPackPerVec + k*MMA_TILE_X + laneIdx];
            tBreg = tBsmem_r[stageReadIterator*NUM_VEC_PER_TILE*InputVec::BitsPackPerVec + k*MMA_TILE_X + laneIdx];
            bmma_m8n8k128_xor_popc(cntp, tAreg, tBreg, cntp);

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
    tCreg.StoreByScalar((NUM - cntp[0] - cntp[0]), 0);
    tCreg.StoreByScalar((NUM - cntp[1] - cntp[1]), 1);
    tCgmem_w[0] = tCreg;
}



// BNN host functions

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

void sm80_bnn_gemm_m8n8k128(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K, int NUM
) {
    assert((M%16 == 0) && "M-dim has to be multiple of 16");
    assert((N%16 == 0) && "N-dim has to be multiple of 16");
    assert((K%32 == 0) && "K-dim has to be multiple of 32");
    dim3 dimGrid(ceil(1.*N/TILE_K), ceil(1.*M/TILE_K));
    sm80_bnn_gemm_multi_stage_m8n8k128_kernel<2><<<dimGrid, 128>>>(
        a, b, c, M, N, K, NUM
    );
    CUDA_ERR_CHECK();
}

void sm80_bnn_gemm(
    const int32_t* a,
    const int32_t* b,
    int32_t* c, 
    int M, int N, int K, int NUM
) {
    if (M%16 != 0 || N%16 != 0 || K%32 != 0) {
        // fall back into baseline kernel
        bnn_gemm_baseline(
            a, b, c, M, N, K, NUM
        );
    } else {
        sm80_bnn_gemm_m8n8k128(
            a, b, c, M, N, K, NUM
        );
    }
}

} // namespace sm80_bnn

#endif /* SM80_BNN_GEMM_CUH */