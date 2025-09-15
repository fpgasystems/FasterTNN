cmake_minimum_required(VERSION 3.20)
project(tab_inference LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)    # Enable CUDA separable compilation globally

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS "Use Generator: " ${CMAKE_GENERATOR})
else()
    message(STATUS "Use  Generator:" ${CMAKE_GENERATOR})
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

# include dir
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda)

# convolution functions
message(STATUS "Add CUDA Convolution Code")
add_library(conv_obj OBJECT ${CMAKE_SOURCE_DIR}/lib/tab_conv.cu ${CMAKE_SOURCE_DIR}/lib/conv_utils.cu)
set_target_properties(conv_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create a static library from object files
add_library(conv_lib STATIC
    $<TARGET_OBJECTS:conv_obj>
)

# src
add_executable(quan ${CMAKE_SOURCE_DIR}/src/quan.cu)
add_executable(tnn ${CMAKE_SOURCE_DIR}/src/tnn.cu)
add_executable(tbn ${CMAKE_SOURCE_DIR}/src/tbn.cu)
add_executable(bnn ${CMAKE_SOURCE_DIR}/src/bnn.cu)
add_executable(btn ${CMAKE_SOURCE_DIR}/src/btn.cu)
add_executable(conv ${CMAKE_SOURCE_DIR}/src/conv.cu)
target_link_libraries(conv PRIVATE conv_lib)
add_executable(model ${CMAKE_SOURCE_DIR}/src/model.cu)
target_link_libraries(model PRIVATE conv_lib)
