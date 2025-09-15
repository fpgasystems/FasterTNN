#include "shapes/shape_layers.h"
#include "tab_conv.h"
#include <iostream>
#include <cassert>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "[ERROR]: " << "./conv [layerid 1-16]" << std::endl;
        return 0;
    }
    int layerid = std::stoi(std::string(argv[1]));
    testcase_t* conv_data = create_testcase();
    int (*(*shapes))[PARAMCOUNT] = shapes_layers;
    setup_conv_data(
        conv_data, 
        ConvType::TAB_BTN,
        8, 
        shapes[layerid][0]
    );

    pre_conv(conv_data);

    for (int i = 0; i < 10; i++) {
        tab_conv(conv_data, false);
        tab_conv(conv_data, true);
    }
    
    tab_conv(conv_data, false);
    tab_conv(conv_data, true);
}