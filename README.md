# Faster Ternary and Binary Neural Network Inference on CPU by Reducing Popcount Overhead

This repository contains an optimized implementation of Ternary Neural Networks (TNNs) and mixed ternary-binary variants (TBNs, BTNs) for two target CPU architectures: AVX2 and ARM (Neon). It is explained in the paper [Faster Ternary and Binary Neural Network Inference on CPU by Reducing Popcount Overhead]().

## Overview

The implementation consists of two main stages: data preparation and matrix multiplication. We explain the steps for TNNs; the TBN and BTN variants are analogous. 

1. **Data Preparation Stage**
   - Vectorized ternarization of input values using low and high thresholds: If the input value is higher than the high threshold, it is set to 1; if it is lower than the low threshold, it is set to -1; otherwise, it is set to 0. In the binary case, there is only one threshold. 
   - Bit-packing: Represent a ternary value (-1, 0, 1) using two bits in one of the following schemes. We pad the tails with zeros to the next multiple of the desired divisibility. For the binary case, there are only sign bits. 
     - Packing 0:  64 sign bits,  64 abs bits,  ...
     - Packing 1: 256 sign bits, 256 abs bits,  ... (AVX2); 128 sign bits, 128 abs bits,  ... (ARM)
     - Packing 2: 256 sign bits, 256 zero bits, ... (AVX2); 128 sign bits, 128 zero bits, ... (ARM)
   - Image-to-row transform: Convert the input image into a row vector of packed ternary values such that convolution can be computed as a matrix multiplication. For 1x1 convolutions, the input image is already in the row vector format.

2. **Matrix Multiplication (GEMM) Stage** 
   - Compute convolution as a matrix multiplication. The kernel is stored in column-major order for better spatial locality.
   - PReLU activation at the end, where we use the scaling parameter 0.5. 
   - For AVX2 TNN: optimized ternary matrix multiplication algorithm explained in the paper
   - For AVX2/ARM TNN/TBN/BTN: vectorized implementation following TAB baseline

## Timing and Benchmarking

The repository includes benchmarking utilities to measure performance:

1. **Layer-wise Timing and Testing (`timing_layers`)**
   - Measures execution time of single and multiple TNN/TBN/BTN layers
   - Single layer: high parameter sizes to push CPU to its limits
   - Multi-layer: more realistic parameter sizes based on real-world CNN architectures
   - Supports both convolution and fully-connected layers
   - Tests single layers for correctness
   - Configurable parameters:
     - Single layer: `shapes_layers.h`
     - Multiple layers: `shapes_e2e.h`
     - Test layers: `shapes_test.h`
     - Number of timing iterations and warm-up runs: `timing_layers.c`

2. **GEMM Timing (`timing_gemm`)**
   - (AVX2 TNN only)  Focused benchmarking of the core matrix multiplication algorithm
   - Separate variants for each optimization step for a systematic ablation study
   - Tests different matrix sizes and configurations
   - Configurable parameters:
     - Matrix sizes, number of timing iterations and warm-up runs: `timing_gemm.c`

3. **Makefile**
   - Targets:
     - `make time_layers_{all, tnn, tbn, btn}`: Build and run single-layer timing benchmark
     - `make time_e2e_{all, tnn, tbn, btn}`: Build and run end-to-end (multi-layer) timing benchmark
     - `make time_gemm_{all, tnn, tbn, btn}`: Build and run GEMM timing benchmark (AVX2 TNN only)
     - `make test_{all, tnn, tbn, btn}`: Build and run test suite 

## File Structure

The repository is organized as follows:

```
.
├── include/                      # Header files
│   ├── tnns/                       # Headers for source code
│   │   ├── common.h                  # Common definitions and utilities
│   │   ├── tnns_cpu.h                # Main functions for convolution and matrix multiplication
│   │   ├── data.h                    # Definitions and helpers for experiments, timing and testing
│   │   ├── binarize2row.h            # Binarize + Image-to-row
│   │   ├── mm.h                      # Matrix multiplication (GEMM) fused with activation function
│   │   └── ternarize2row.h           # Ternarize + Image-to-row
│   ├── timing/                     # Timing and benchmarking headers
│   │   ├── shapes_e2e.h              # End-to-end multi-layer shapes
│   │   ├── shapes_test.h             # Test case shapes
│   │   └── shapes_layers.h           # Single-layer shapes
│   └── utils.h                     # General utility functions
│    
├── src/                          # Source files
│   ├── tnns/                       # Main implementation files
│   │   ├── gemm-variants/            # Different matrix multiplication (MM) implementations
│   │   │   - mm_v1_p0_naive.c:         # Fuse activation function (PReLU) into MM
│   │   │   - mm_v2_p0_blocked.c:       # Block the computation for cache
│   │   │   - mm_v3_p0_popcnt.c:        # Scalar popcount, but rest vectorized
│   │   │   - mm_v4_p0_lib_2pc.c:       # Vectorized popcount using the library libpopcnt
│   │   │   - mm_v5_p1_lib_2pc.c:       # SIMD-optimized bitpacking
│   │   │   - mm_v6_p1_lib_1pc.c:       # One large popcount instead of two smaller ones
│   │   │   - mm_v7_p1_lib_csas.c:      # Optimize the CSAs 
│   │   │   - mm_v8_p2_lib_csas.c:      # Optimized encoding
│   │   │   - mm_v9_p2_lib_csas.c:      # Optimized encoding
│   │   ├── mm_p2_final_*.c           # Final optimized implementations for different architectures (for avx2, arm_neon)
│   │   ├── tnns_cpu_naive.c          # Baseline
│   │   ├── tnns_cpu_mm_p0.c          # Optimized implementation with bitpacking 0 (original packing)
│   │   ├── tnns_cpu_mm_p1.c          # Optimized implementation with SIMD-optimized bitpacking 1 (sign, abs)
│   │   ├── tnns_cpu_mm_p2.c          # Optimized implementation with SIMD-optimized bitpacking 2 (sign, zero)
│   │   ├── data.c                    # Utilities for running experiments, timing and testing
│   │   ├── binarize2row_*.c          # Binarize + Image-to-row implementation (for avx2, arm_neon)
│   │   └── ternarize2row_*.c         # Ternarize + Image-to-row implementation (for avx2, arm_neon)
│   ├── timing_gemm.c               # Matrix multiplication (GEMM) timing benchmarks
│   ├── timing_layers.c             # Single-layer and end-to-end (multi-layer) timing benchmarks
│   └── utils.c                     # Utility function implementations
│    
├── Makefile                      # Makefile
└── README.md                      
```

## Main Entry Points

The codebase provides several main entry points for different functionalities. Each entry point supports different neural network variants (TNN, TBN, BTN, BNN) and can be configured for different architectures (AVX2, ARM Neon).

### Ternary/Binary Neural Network Operations (`tnns_cpu_*.c`)

1. **Baseline Implementation** (`tnns_cpu_naive.c`)
   - `tnns_conv`: Entry point for TNN/TBN/BTN/BNN layer computation; calls NN convolution variant (`{tnn, tbn, btn, bnn}_conv`) based on input testcase configuration
   - `tnns_gemm`: Entry point for ternary matrix multiplication; runs MM on input testcase

2. **Optimized Implementations**
   - The optimized versions use the optimized data preparation (Ternarize/Binarize + Image-to-row) and matrix multiplication (+ PReLU) implementations for the corresponding bitpacking scheme.
   - Entry points `tnns_conv` and `tnns_gemm` as in the baseline
   - `tnns_cpu_mm_p0.c`: Original bitpacking(64 sign bits, 64 abs bits, ...)
   - `tnns_cpu_mm_p1.c`: SIMD-aligned bitpacking 1 (sign, abs)
   - `tnns_cpu_mm_p2.c`: SIMD-aligned bitpacking 2 (sign, zero)

### Benchmarking Utilities

1. **GEMM Timing** (`timing_gemm.c`)
   - `benchmark_random`: Benchmarks matrix multiplication performance on random input
   - Measures execution time for different matrix sizes for different optimized versions
   - Outputs results to CSV file with matrix dimensions and timing data

2. **Layer Timing** (`timing_layers.c`)
   - `benchmark_layers`: Benchmarks neural network layer performance
   - Supports three types of benchmarks:
     - Single layer timing
     - End-to-end (multi-layer) timing
     - Single-layer testing
   - Outputs results to CSV file with layer configuration and timing data

### Ternary/Binary CNN Parameters

The codebase uses two main test case structures to manage data for experiments and benchmarks. We w

1. **Neural Network Layer Test Case** (`testcase_t`)
   - Configuration parameters:
     - NN type: TNN, TBN, BTN, BNN
     - Batch size (N)
     - Input channels (C)
     - Input height and width (H, W)
     - Kernel count (KN)
     - Kernel height and width (KH, KW)
     - Padding height and width (H_pad, W_pad)
     - Stride height and width (H_stride, W_stride)
     - PReLU scaling factor (alpha)
   - Data arrays:
     - Input thresholds
     - Input data
     - Kernel weights
     - Quantized (ternarized or binarized) kernels (packing0, packing1, packing2)
     - Output buffer

2. **Matrix Multiplication Test Case** (`testcase_tnns_gemm_t`)
   - Configuration parameters:
     - Matrix dimensions (M, N, P)
     - Packed dimension (P_packed)
   - Data arrays:
     - Input matrix (X)
     - Weight matrix (W)
     - Packed inputs (X_packing0, X_packing1, X_packing2)
     - Packed weights (W_packing0, W_packing1, W_packing2)
     - Output buffer

