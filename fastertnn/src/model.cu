#include "shapes/shape_models.h"
#include "tab_conv.h"
#include <iostream>
#include <cassert>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "[ERROR]: " << "./model <baseline: 0/1>" << std::endl;
        return 0;
    }
    int baseline = std::stoi(std::string(argv[1]));
    int shapecount = shapecount_e2e;
    int* layercounts = layercounts_e2e;
    int (*(*shapes))[PARAMCOUNT] = shapes_e2e;
    int batch_size = 8;
    
    cudaEvent_t s;
    cudaEvent_t e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int shape = 0; shape < shapecount; shape++) {
        int layercount = layercounts[shape];
        float inference_time_ms = 0;
        for (int layerid = 1; layerid < layercount; ++layerid) {
            float curr_layer_inference_time = 0;
            testcase_t* conv_data = create_testcase();
            setup_conv_data(
                conv_data,
                ConvType::TAB_BTN,
                batch_size, 
                shapes[shape][layerid]
            );
            pre_conv(conv_data);

            cudaEventRecord(s);

            tab_conv(conv_data, baseline);

            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&curr_layer_inference_time, s, e);
            inference_time_ms += curr_layer_inference_time;

            free_testcase(conv_data);
            cudaDeviceSynchronize();
        }

        std::cout << "shape " << shape << " inference time: " << inference_time_ms << "ms\n"; 
    }
}