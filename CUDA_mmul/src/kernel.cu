//---------------------------------------------------------------------------
// kernel.cu
// Authors: Margaret Lanphere, Nick Posey, Breanna Powell, Frank Sossi,
//          Pragati Dode, Amalaye Oyake
//
// This program implements vector addition using CUDA. The program will
// use a template function as a reference and then compare the results
// to the CUDA implementation.
//
// To compile: nvcc -arch=sm_86 -lcublas kernel.cu -o lab3
// From command line
// Program compiled with Nvidia CUDA Compiler (NVCC) on Windows 11 or Ubuntu 20.10
// Required: CUDA Toolkit v12[1] To compile:
// 1. From base Folder $ cd src
// 2. No debugging symbols: nvcc -arch=sm_86 -lcublas kernel.cu -o lab3
// 3. With debugging symbols: nvcc -g -G -arch=sm_86 -lcublas kernel.cu -o lab3
// 4. To run: ./lab3
//
// 5.2 CMake instructions
// 1. mkdir build
// 2. cd build
// 3. cmake ..
// 4. make
//---------------------------------------------------------------------------
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "mmul.h"

#define TESTDATA
//#define RANDOMDATA

int main()
{

    int n{10};
    int THREAD_PER_BLOCK{32};

    // Vectors to hold timing data
    std::vector<long long> execution_w_memory;
    std::vector<long long> execution_wo_memory;

    // Allocate memory for each vector on host
    double* matrix_a = (double*)malloc(n * n * sizeof(double));
    double* matrix_b = (double*)malloc(n * n * sizeof(double));
    double* matrix_c = (double*)malloc(n * n * sizeof(double));
    double* matrix_ref = (double*)malloc(n * n * sizeof(double));

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 2);

#ifdef TESTDATA
    // Initialize matrix_a and matrix_b with test data 2 for all values
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix_a[i * n + j] = 2;
            matrix_b[i * n + j] = 2;
        }
    }
#endif

#ifdef RANDOMDATA


    // Initialize matrix_a and matrix_b with random numbers
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix_a[i * n + j] = dis(gen);
            matrix_b[i * n + j] = dis(gen);
        }
    }
#endif

    // Allocate pointers to GPU memory
    double* device_matrix_a = nullptr;
    double* device_matrix_b = nullptr;
    double* device_matrix_ref = nullptr;

    cudaMalloc((void**)&device_matrix_a, n * n * sizeof(double));
    cudaMalloc((void**)&device_matrix_b, n * n * sizeof(double));
    cudaMalloc((void**)&device_matrix_ref, n * n * sizeof(double));

    // Copy input data to GPU memory
    cudaMemcpy(device_matrix_a, matrix_a, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, matrix_b, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_ref, matrix_ref, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // calculate mmul on gpu with cuBlas
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                device_matrix_a, n, device_matrix_b, n, &beta, device_matrix_ref, n);

    // Copy output data to host memory
    cudaMemcpy(matrix_ref, device_matrix_ref, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    //print matrix_ref
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << matrix_ref[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free GPU memory
    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_ref);



    


  

    return 0;
}
