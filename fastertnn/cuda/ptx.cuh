#ifndef PTX_CUH
#define PTX_CUH

#include <cuda.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
template<int N>
__forceinline__ __device__ void cp_async_ca_shared_global(
    const int32_t* gmem_src,
    int32_t* smem_dst
) {
    size_t cast_smem_dst = __cvta_generic_to_shared(smem_dst);
    size_t cast_gmem_src = (size_t)(gmem_src);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "l"(cast_smem_dst),    // Destination in shared memory
        "l"(cast_gmem_src),   // Source in global memory
        "n"(16)          // Number of bytes to copy (128bit)
    );
}

template<>
__forceinline__ __device__ void cp_async_ca_shared_global<4>(
    const int32_t* gmem_src,
    int32_t* smem_dst
) {
    size_t cast_smem_dst = __cvta_generic_to_shared(smem_dst);
    size_t cast_gmem_src = (size_t)(gmem_src);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "l"(cast_smem_dst),    // Destination in shared memory
        "l"(cast_gmem_src),   // Source in global memory
        "n"(4)          // Number of bytes to copy (128bit)
    );
}

template<>
__forceinline__ __device__ void cp_async_ca_shared_global<8>(
    const int32_t* gmem_src,
    int32_t* smem_dst
) {
    size_t cast_smem_dst = __cvta_generic_to_shared(smem_dst);
    size_t cast_gmem_src = (size_t)(gmem_src);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "l"(cast_smem_dst),    // Destination in shared memory
        "l"(cast_gmem_src),   // Source in global memory
        "n"(8)          // Number of bytes to copy (128bit)
    );
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group
__forceinline__ __device__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group
template <size_t W>
__forceinline__ __device__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group 0;\n");
}

template <>
__forceinline__ __device__ void cp_async_wait_group<1>() {
    asm volatile("cp.async.wait_group 1;\n");
}

template <>
__forceinline__ __device__ void cp_async_wait_group<2>() {
    asm volatile("cp.async.wait_group 2;\n");
}

template <>
__forceinline__ __device__ void cp_async_wait_group<4>() {
    asm volatile("cp.async.wait_group 4;\n");
}

template<typename Scalar, typename Vec>
__forceinline__ __device__ void tnn_loop_mma(const Vec& a, const Vec& b, int& cntp1, int& cntp2) {
#pragma unroll
    for (int i = 0; i < Vec::BitsPackPerVec; i++) {
        Scalar p2 = a.ReadAsBitsPack(i).y & b.ReadAsBitsPack(i).y;
        Scalar p1 = (a.ReadAsBitsPack(i).x ^ b.ReadAsBitsPack(i).x) & p2;
        cntp1 = cntp1 + __popc(p2);
        cntp2 = cntp2 + __popc(p1);
    }
}

template<typename Scalar, typename Vec>
__forceinline__ __device__ void tnn_loop_bmma(const Vec& a, const Vec& b, int& s0, int& s1, int& s2, int& s3) {
#pragma unroll
    for (int i = 0; i < Vec::BitsPackPerVec; i++) {
        s0 += __popc(a.ReadAsBitsPack(i).x & b.ReadAsBitsPack(i).x);
        s1 += __popc(a.ReadAsBitsPack(i).x & b.ReadAsBitsPack(i).y);
        s2 += __popc(a.ReadAsBitsPack(i).y & b.ReadAsBitsPack(i).x);
        s3 += __popc(a.ReadAsBitsPack(i).y & b.ReadAsBitsPack(i).y);
    }
}

template<typename Scalar, typename Vec>
__forceinline__ __device__ void bnn_loop_mma(const Vec& a, const Vec& b, int& cntp1) {
#pragma unroll
    for (int i = 0; i < Vec::BitsPackPerVec; i++) {
        Scalar p = a.ReadAsBitsPack(i) ^ b.ReadAsBitsPack(i);
        cntp1 = cntp1 + __popc(p);
    }
}

template<typename Scalar, typename VecA, typename VecW>
__forceinline__ __device__ void btn_loop_mma(const VecA& a, const VecW& b, int& cntp1, int& cntp2) {
#pragma unroll
    for (int i = 0; i < VecA::BitsPackPerVec; i++) {
        Scalar p1 = a.ReadAsBitsPack(i) ^ b.ReadAsBitsPack(i).x;
        Scalar p2 = b.ReadAsBitsPack(i).y;
        cntp1 += __popc(p2);
        cntp2 += __popc(p1 & p2);
    }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-88128
__forceinline__ __device__ void bmma_m8n8k128_and_popc(
    int* d,
    const unsigned int a,
    const unsigned int b,
    int* c
) {
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a), "r"(b), "r"(c[0]), "r"(c[1])
    );
}

__forceinline__ __device__ void bmma_m16n8k128_and_popc(
    int* d,
    const unsigned int a1,
    const unsigned int a2,
    const unsigned int b,
    int* c
) {
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a1), "r"(a2),"r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-88128
__forceinline__ __device__ void bmma_m8n8k128_xor_popc(
    int* d,
    const unsigned int a,
    const unsigned int b,
    int* c
) {
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a), "r"(b), "r"(c[0]), "r"(c[1])
    );
}

__forceinline__ __device__ void bmma_m16n8k128_xor_popc(
    int* d,
    const unsigned int a1,
    const unsigned int a2,
    const unsigned int b,
    int* c
) {
    asm volatile(
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a1), "r"(a2),"r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}



#endif