#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define popcnt64(a)       __builtin_popcountll(a)

// The bits of the container integer: int64_t
#define cntbits 64
// The bit width of quantized input values
#define BITS 2

enum tnns_type{TNN, TBN, BTN, BNN};

static inline const char* tnns_type_to_string(enum tnns_type t) {
    switch (t) {
        case TNN: return "TNN";
        case TBN: return "TBN";
        case BTN: return "BTN";
        case BNN: return "BNN";
        default:  return "UNKNOWN";
    }
}