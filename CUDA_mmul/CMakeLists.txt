cmake_minimum_required(VERSION 3.0)

project(MyProject CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
add_executable(program2 src/kernel.cu)

set_property(TARGET program2 PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(program2 PRIVATE cublas)
