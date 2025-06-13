#pragma once

#include "tnns/common.h"

float *generate_array(int64_t size, bool ternary);

bool check_nhwc(float* X, float* X2, int N, int C, int H, int W);
bool check_nchw(float* X, float* X2, int N, int C, int H, int W);
bool check_matrices(float* X1, float* X2, int m, int n);

float *direct_pad(float* x, int padding1, int padding2, int N, int C, int H, int W, int pad_val);
float *direct_conv2d(float* x, float* w, int stride1, int stride2, int N, int C, int H, int W, int KN, int KH, int KW, float prelu_alpha);
float *direct_mmm(float* x, float* w, int m, int n, int p);

int64_t *bitpacking0(float* x, int I, int J);
int64_t *bitpacking1(float* x, int I, int J);
int64_t *bitpacking2(float* x, int I, int J);

int64_t* ternarize_naive(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad);
int64_t* ternarize_packing1(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad);
int64_t* ternarize_packing2(float* x, float* q_thresholds, int N, int C, int H, int W);
int64_t* ternarize_opt_avx2(float* x, float* q_thresholds, int N, int C, int H, int W);
int64_t* ternarize_opt_arm_neon(float* x, float* q_thresholds, int N, int C, int H, int W);

int64_t* binarize_naive(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad);
int64_t* binarize_opt_avx2(float* x, float* q_thresholds, int N, int C, int H, int W);
int64_t* binarize_opt_arm_neon(float* x, float* q_thresholds, int N, int C, int H, int W);
