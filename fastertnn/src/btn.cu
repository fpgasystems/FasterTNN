#include "sm80_btn_gemm.cuh"
#include "conv_utils.cuh"
#include <random>
#include <iostream>

void benchmark_btngemm(
    int M, int N, int K,
    int run_time = 100
) {
    CUDA_CALL_CHECK(cudaSetDevice(0));
    CUDA_CALL_CHECK(cudaDeviceReset()); /* reset is needed to count overhead */

    int size_a = M*K/2;
    int size_b = K*N;
    int size_c = M*N;

    /* prepare host data */
    std::vector<int32_t> host_a(size_a, 0);    // input matrix a
    std::vector<int32_t> host_b(size_b, 0);    // input matrix b
    std::vector<int32_t> host_c_base(size_c, 0);      // output c
    std::vector<int32_t> host_c(size_c, 0);  

    /* generate host data */
    generate_random_array<int32_t>(host_a.data(), size_a, FP32);
    generate_random_array<int32_t>(host_b.data(), size_b, FP32);

    /* prepare device mem */
    int32_t* dev_a = 0;     // input matrix a
    int32_t* dev_b = 0;     // input matrix b
    int32_t* dev_c_base = 0;       // output c
    int32_t* dev_c = 0;       // output c

    /* allocate device memory */
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_a), size_a*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_b), size_b*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_c), size_c*sizeof(int32_t)));
    CUDA_CALL_CHECK(cudaMalloc((void**)(&dev_c_base), size_c*sizeof(int32_t)));

    /* copy to device */
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_a, (void*)host_a.data(), size_a*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy((void*)dev_b, (void*)host_b.data(), size_b*sizeof(int32_t), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // measure baseline performance
    float baseline_milliseconds = 0;
    cudaEventRecord(start);
    for (int i = 0; i < run_time; i++) {
        sm80_btn::btn_gemm_baseline(
            dev_a, dev_b, dev_c_base, M, N, K/2
        );
    }
    cudaEventRecord(stop);
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&baseline_milliseconds, start, stop);
    CUDA_CALL_CHECK(cudaMemcpy((void*)host_c_base.data(), (void*)dev_c_base, size_c*sizeof(int32_t), cudaMemcpyDeviceToHost));

    // measure optimized performance
    float optimize_milliseconds = 0;
    cudaEventRecord(start);
    for (int i = 0; i < run_time; i++)
        sm80_btn::sm80_btn_gemm_multi_stage(
            dev_a, dev_b, dev_c, M, N, K/2
        );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    cudaEventElapsedTime(&optimize_milliseconds, start, stop);
    CUDA_CALL_CHECK(cudaMemcpy((void*)host_c.data(), (void*)dev_c, size_c*sizeof(int32_t), cudaMemcpyDeviceToHost));

    // calculate throughput
    std::cout << "[" << M << "," << N << "," << K << "] ";
    std::cout << "Baseline:" << (float)baseline_milliseconds/run_time << "ms \t Optimize:" << (float)optimize_milliseconds/run_time << "ms\n";

    // host_cpu_c[1] = -1;
    int check = memcmp(host_c.data(), host_c_base.data(), size_c*sizeof(int32_t));
    int diff = 0;
    for (int i = 0; i < size_c; i++) {
        // std::cout << i << " " << host_c_base[i] << " " << host_c[i] << "\n";
        if (host_c_base[i] != host_c[i]) {
            // std::cout << "diff: " << i << " " << host_c_base[i] << " " << host_c[i] << "\n";
            diff++;
        }
    }
    std::cout << "Check=" << (check == 0) << " diff=" << diff << "\n";

}

int main(int argc, char *argv[]) {
    std::vector<int> mn = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (auto _mn : mn) {
        benchmark_btngemm(_mn, _mn, _mn*8, 10);
    }
}