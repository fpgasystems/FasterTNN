#include "tab_quan.cuh"
#include "conv_utils.cuh"
#include <random>
#include <iostream>

void benchmark_quantization(
    int N, int H, int W, int C,
    QuanType qtype,
    int run_time = 100
) {
    CUDA_CALL_CHECK(cudaSetDevice(0));
    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */
    const int packedC = ceil((float)C/tab_quan::BIT_PACK_SIZE);
    int size_a = N*H*W*C;
    int size_c = qtype == Ternary ? N*H*W*packedC*tab_quan::BIT_PACK_COUNT : N*H*W*packedC;

    // allocate on host side
    std::vector<float> host_a(size_a, 0);      // intput a
    // std::vector<float> host_ths(N, 0);      // intput ths
    std::vector<float> host_ths(1024, 0);      // intput ths
    std::vector<int64_t> host_c(size_c, 0);      // output c
    generate_random_array<float>(host_a.data(), size_a, FP32);
    generate_random_array<float>(host_ths.data(), 1024, FP32);
    for (int i = 0; i < 1024; i++)
        host_ths[i] = abs(host_ths[i]);

    // allocate on device side
    float* dev_a = 0;
    float* dev_ths = 0;
    int64_t* dev_c = 0;       // output c

    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_a), size_a*sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_ths), 1024*sizeof(float)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_c), size_c*sizeof(int64_t)));

    /* copy to device */
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_a, (void*)host_a.data(), size_a*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_ths, (void*)host_ths.data(), 1024*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // measure baseline performance
    float baseline_milliseconds = 0;
    cudaEventRecord(start);
    auto base_func = Ternary ? tab_quan::ternary_quantization_baseline : tab_quan::binary_quantization_baseline;
    for (int i = 0; i < run_time; i++)
        base_func(
            (int32_t*)dev_c, dev_a, dev_ths, N, H, W, C
        );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&baseline_milliseconds, start, stop);

    // measure optimized performance
    float optimize_milliseconds = 0;
    cudaEventRecord(start);
    auto opt_func = Ternary ? tab_quan::ternary_quantization : tab_quan::binary_quantization;
    for (int i = 0; i < run_time; i++)
        opt_func(
            (int32_t*)dev_c, dev_a, dev_ths, N, H, W, C
        );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&optimize_milliseconds, start, stop);

    // calculate throughput
    float total_memory_access_byte = N*H*W*C*sizeof(float) + N*H*W*C/tab_quan::BIT_PACK_SIZE * (Ternary ? sizeof(int64_t) : sizeof(int32_t));
    float baseline_throughput = (float)total_memory_access_byte/1024/1024/1024/(baseline_milliseconds/run_time) * 1000;
    float optimize_throughput = (float)total_memory_access_byte/1024/1024/1024/(optimize_milliseconds/run_time) * 1000;
    
    std::cout << "Total Memory Access: " << total_memory_access_byte/1024/1024/1024 << "GB\n";
    std::cout << "Baseline avg: " << (float)baseline_milliseconds/run_time << "ms " << baseline_throughput << "GB/s\n";
    std::cout << "Optimize avg: " << (float)optimize_milliseconds/run_time << "ms " << optimize_throughput << "GB/s\n";

}

int main(int argc, char *argv[]) {
    benchmark_quantization(8, 16, 16, 256, Binary);
    benchmark_quantization(8, 16, 16, 256, Ternary);
}