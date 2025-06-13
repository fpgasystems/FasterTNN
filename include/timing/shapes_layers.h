
#ifndef PARAMCOUNT
#define PARAMCOUNT 8 //  c,  ih,  iw,   kn,  kw,  kh,  p,  s
#endif

static int shape_0[1][PARAMCOUNT]  = {{    256,    7,   7,   256,   3,   3,  1,  2 }};
static int shape_1[1][PARAMCOUNT]  = {{    512,    7,   7,   512,   3,   3,  1,  2 }};
static int shape_2[1][PARAMCOUNT]  = {{   1024,    7,   7,  1024,   3,   3,  1,  2 }};
static int shape_3[1][PARAMCOUNT]  = {{    256,   14,  14,   256,   3,   3,  1,  2 }};
static int shape_4[1][PARAMCOUNT]  = {{    512,   14,  14,   512,   3,   3,  1,  2 }};
static int shape_5[1][PARAMCOUNT]  = {{   1024,   14,  14,  1024,   3,   3,  1,  2 }};
static int shape_6[1][PARAMCOUNT]  = {{    256,   28,  28,   256,   3,   3,  1,  2 }};
static int shape_7[1][PARAMCOUNT]  = {{    512,   28,  28,   512,   3,   3,  1,  2 }};
static int shape_8[1][PARAMCOUNT]  = {{   1024,   28,  28,  1024,   3,   3,  1,  2 }};
static int shape_9[1][PARAMCOUNT]  = {{    256,   56,  56,   256,   3,   3,  1,  2 }};
static int shape_10[1][PARAMCOUNT] = {{    512,   56,  56,   512,   3,   3,  1,  2 }};
static int shape_11[1][PARAMCOUNT] = {{   1024,   56,  56,  1024,   3,   3,  1,  2 }};
static int shape_12[1][PARAMCOUNT] = {{   1024,    1,   1,  1024,   1,   1,  0,  1 }};
static int shape_13[1][PARAMCOUNT] = {{   2048,    1,   1,  2048,   1,   1,  0,  1 }};
static int shape_14[1][PARAMCOUNT] = {{   4096,    1,   1,  4096,   1,   1,  0,  1 }};
static int shape_15[1][PARAMCOUNT] = {{   8192,    1,   1,  8192,   1,   1,  0,  1 }};
static int shape_16[1][PARAMCOUNT] = {{  16384,    1,   1, 16384,   1,   1,  0,  1 }};

#define SHAPECOUNT_LAYERS 17
static int shapecount_layers = SHAPECOUNT_LAYERS;
static int layercounts_layers[SHAPECOUNT_LAYERS] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static int (*shapes_layers[SHAPECOUNT_LAYERS])[PARAMCOUNT] = {
    shape_0,
    shape_1,
    shape_2,
    shape_3,
    shape_4,
    shape_5,
    shape_6,
    shape_7,
    shape_8,
    shape_9,
    shape_10,
    shape_11,
    shape_12,
    shape_13,
    shape_14,
    shape_15,
    shape_16
};
