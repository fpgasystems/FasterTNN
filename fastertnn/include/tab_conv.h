#ifndef TAB_CONV_H
#define TAB_CONV_H
#include "conv_utils.cuh"
void pre_conv(testcase_t* data);
void tab_conv(testcase_t* data, bool baseline);
void tnn_conv(testcase_t* data, bool baseline);
void tbn_conv(testcase_t* data, bool baseline);
void bnn_conv(testcase_t* data, bool baseline);
void btn_conv(testcase_t* data, bool baseline);

#endif // TAB_CONV_H