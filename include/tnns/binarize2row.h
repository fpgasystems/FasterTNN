#pragma once
#include "tnns/common.h"

int64_t *b2r(float* x, float* q_thresholds, int N, int C, int H, int W, int H_pad, int W_pad, int kH, int kW, int H_stride, int W_stride);
