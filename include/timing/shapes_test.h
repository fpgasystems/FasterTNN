
#ifndef PARAMCOUNT
#define PARAMCOUNT 8 //  c,  ih,  iw,   kn,  kw,  kh,  p,  s
#endif

static int test_shape_0[1][PARAMCOUNT]  = {{     32,    7,   7,    32,   3,   3,  0,  1 }};
static int test_shape_1[1][PARAMCOUNT]  = {{     32,    7,   7,    32,   3,   3,  1,  2 }};
static int test_shape_2[1][PARAMCOUNT]  = {{    127,    7,   7,    32,   3,   3,  1,  1 }};
static int test_shape_3[1][PARAMCOUNT]  = {{    127,    7,   7,    32,   3,   3,  1,  1 }};
static int test_shape_4[1][PARAMCOUNT]  = {{    128,    7,   7,    32,   3,   3,  1,  2 }};
static int test_shape_5[1][PARAMCOUNT]  = {{    193,    7,   7,    32,   3,   3,  1,  2 }};
static int test_shape_6[1][PARAMCOUNT]  = {{    256,   14,  14,   256,   3,   5,  1,  2 }};
static int test_shape_7[1][PARAMCOUNT]  = {{    256,   14,  14,   256,   3,   3,  1,  2 }};
static int test_shape_8[1][PARAMCOUNT]  = {{    257,   14,  14,   257,   3,   3,  1,  2 }};
static int test_shape_9[1][PARAMCOUNT]  = {{    320,   15,  23,   320,   3,   3,  1,  2 }};
static int test_shape_10[1][PARAMCOUNT] = {{    512,    1,   1,   512,   1,   1,  0,  1 }};
static int test_shape_11[1][PARAMCOUNT] = {{   1024,   14,  14,  1024,   5,   5,  1,  2 }};

#define SHAPECOUNT_TEST 12
static int shapecount_test = SHAPECOUNT_TEST;
static int layercounts_test[SHAPECOUNT_TEST] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static int (*shapes_test[SHAPECOUNT_TEST])[PARAMCOUNT] = {
    test_shape_0,
    test_shape_1,
    test_shape_2,
    test_shape_3,
    test_shape_4,
    test_shape_5,
    test_shape_6,
    test_shape_7,
    test_shape_8,
    test_shape_9,
    test_shape_10,
    test_shape_11
};