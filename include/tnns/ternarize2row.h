#pragma once
#include "tnns/common.h"

int64_t *t2r_p0(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride);
int64_t *t2r_p1(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride);
int64_t *t2r_p2(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride);
int64_t *t2r_p2_opt(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride);
