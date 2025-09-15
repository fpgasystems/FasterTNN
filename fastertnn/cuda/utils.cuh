#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <vector_types.h>

/**
 * macro for dispatching
*/

#define DISPATCH_BOOL(condition, kCondition, ...)\
    do {                                        \
        if (condition) {                        \
            constexpr bool kCondition = true;   \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else {                                \
            constexpr bool kCondition = false;  \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        }                                       \
    } while(0)


#define DISPATCH_M(row, stage, ...)     \
    do {                                        \
        if (row <= 512) {                     \
            constexpr int stage = 4;           \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else if (row <= 1024) {              \
            constexpr int stage = 3;           \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else {                                \
            constexpr int stage = 2;           \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        }                                       \
    } while(0)



#define CUDA_CALL_CHECK(func_call)              \
    do {                                        \
        const cudaError_t cudaerr = func_call;  \
        assert(cudaerr == cudaSuccess && __FILE__ && __LINE__);         \
    } while(0)

#define CUDA_ERR_CHECK()        \
    do {                        \
        assert(cudaGetLastError() == cudaSuccess && __FILE__ && __LINE__);  \
    } while(0)

/**
 * Memory offsets
*/
template<typename D>
__forceinline__ __device__ const D* partition_gmem_thread(
    const D* gmem,
    const int R,
    const int K,
    const int thrId
) {
    return (gmem + R*K + thrId);
}

template<typename D, int TILE_K>
__forceinline__ __device__ const D* partition_gmem_tile(
    const D* gmem,
    const int tileId
) {
    return (gmem + tileId*TILE_K);
}

template<typename D, int TILE_K, bool ROW_MAJOR>
__forceinline__ __device__ D* partition_smem_warp(
    D* smem,
    const int warpIdx,
    const int warpIdy
) {
    return;
}

template<typename D, int TILE_K, bool ROW_MAJOR>
__forceinline__ __device__ D* partition_smem_thread(
    D* smem,
    const int thrIdx,
    const int thrIdy
) {
    if constexpr (ROW_MAJOR)
        return (smem + thrIdy*TILE_K + thrIdx);
    else
        return (smem + thrIdx*TILE_K + thrIdy);
}



/**
 * Self-defined vector
*/

struct __align__(8) uchar8 {
    unsigned char x, y, z, w, a, b ,c, d;
};

struct __align__(16) short8 {
    short x, y, z, w, a, b ,c, d;
};

struct __align__ (16) float32 {
    float4 x, y, z, w, a, b ,c, d;
};

struct __align__(16) int8 {
    int32_t x, y, z, w, a, b ,c, d;
};

template <typename EleType, uint32_t EleNumPerVec>
struct ToVec {};

template <>
struct ToVec<uint8_t, 4> {
    using vtype = uchar4;
};

template <>
struct ToVec<uint8_t, 8> {
    using vtype = uchar8;
};

template <>
struct ToVec<int16_t, 4> {
    using vtype = short4;
};

template <>
struct ToVec<int16_t, 8> {
    using vtype = short8;
};

template <>
struct ToVec<int32_t, 2> {
    using vtype = int2;
};

template <>
struct ToVec<int32_t, 4> {
    using vtype = int4;
};

template <>
struct ToVec<int32_t, 8> {
    using vtype = int8;
};

template <>
struct ToVec<int2, 4> {
    using vtype = int8;
};

template <>
struct ToVec<int2, 2> {
    using vtype = int4;
};


template <>
struct ToVec<float, 4> {
    using vtype = float4;
};

template <>
struct ToVec<float, 2> {
    using vtype = float2;
};

template <>
struct ToVec<float, 32> {
    using vtype = float32;
};

template<typename EleType, uint32_t EleNumPerVec, typename PackType = int2>
struct CustomVec
{
    static constexpr int ElePerVec = EleNumPerVec;
    static constexpr int BytesPerVec = EleNumPerVec * sizeof(EleType);
    static constexpr int BitsPackPerVec = BytesPerVec/sizeof(PackType);
    using VecType = typename ToVec<EleType, EleNumPerVec>::vtype;
    using BitsPackType = PackType;

    using DataType = union {
        VecType vec;
        EleType ele[EleNumPerVec];
        BitsPackType bits[BitsPackPerVec];
    };

    DataType data;

    inline __device__ void Clear() {
#pragma unroll
        for (int i = 0; i < EleNumPerVec; i++)
            this->data.ele[i] = 0;
    }

    inline __device__ BitsPackType ReadAsBitsPack(int offset) const {
        return this->data.bits[offset];
    }

    inline __device__ EleType ReadAsScalar(int offset) const {
        return this->data.ele[offset];
    }

    inline __device__ void StoreByScalar(EleType ele, int offset) {
        this->data.ele[offset] = ele;
    }

    inline __device__ void LoadByVec(const void* ptr, int offset) {
        this->data.vec = static_cast<const VecType*>(ptr)[offset];
    }
};
#endif