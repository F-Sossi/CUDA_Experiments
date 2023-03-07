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
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "mmul.h"



int main()
{
    // make vectors to hold timing data
    std::vector<long long> naive_w_memory;  
    std::vector<long long> naive_wo_memory;
    std::vector<long long> tiled_w_memory;
    std::vector<long long> tiled_wo_memory;
    std::vector<long long> cublas_w_memory;
    std::vector<long long> cublas_wo_memory;



    // make vectors to hold test configurtions
    std::vector<int> n_values{2047, 5045, 8066, 9546, 10240};

    // for each test configuration
    for (int i = 0; i < n_values.size(); i++){

        int n = n_values[i];
        //int thread_per_block = thread_per_block_values[i];

        //print test configuration
        std::cout << "Test Configuration: " << std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << std::endl;

        // Allocate memory for each vector on host
        double* matrix_a = (double*)malloc(n * n * sizeof(double));
        double* matrix_b = (double*)malloc(n * n * sizeof(double));
        double* matrix_naive = (double*)malloc(n * n * sizeof(double));
        double* matrix_ref = (double*)malloc(n * n * sizeof(double));
        double* matrix_tiled = (double*)malloc(n * n * sizeof(double));



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
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

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
        double* device_matrix_naive = nullptr;
        double* device_matrix_tiled = nullptr;

        //------------------------------------------------------------------------------------------
        // Compute the matrix multiplication using cuBlas
        //------------------------------------------------------------------------------------------

        std::cout << "Matrix Multiplication using cuBlas" << std::endl;

        auto cublas_w_mem_start = get_time();

        cudaMalloc((void**)&device_matrix_a, n * n * sizeof(double));
        cudaMalloc((void**)&device_matrix_b, n * n * sizeof(double));
        cudaMalloc((void**)&device_matrix_ref, n * n * sizeof(double));
        //cudaMalloc((void**)&device_matrix_naive, n * n * sizeof(double));

        // Copy input data to GPU memory
        cudaMemcpy(device_matrix_a, matrix_a, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_matrix_b, matrix_b, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_matrix_ref, matrix_ref, n * n * sizeof(double), cudaMemcpyHostToDevice);

        // calculate mmul on gpu with cuBlas
        cublasHandle_t handle;
        cublasCreate(&handle);

        const double alpha = 1.0;
        const double beta = 0.0;

        auto cublas_wo_mem_start = get_time();

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                    device_matrix_a, n, device_matrix_b, n, &beta, device_matrix_ref, n);

        auto cublas_wo_mem_end = get_time();

        // Copy output data to host memory
        cudaMemcpy(matrix_ref, device_matrix_ref, n * n * sizeof(double), cudaMemcpyDeviceToHost);

        auto cublas_w_mem_end = get_time();

        // calcualte time for cublas with memory
        auto cublas_w_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(cublas_w_mem_end - cublas_w_mem_start).count();

        // calculate time for cublas without memory
        auto cublas_wo_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(cublas_wo_mem_end - cublas_wo_mem_start).count();

        // add time to vector
        cublas_w_memory.push_back(cublas_w_mem_time);
        cublas_wo_memory.push_back(cublas_wo_mem_time);

        // print time for cublas with memory
        std::cout << "Time for cublas with memory: " << cublas_w_mem_time << " ns" << std::endl;

        // print time for cublas without memory
        std::cout << "Time for cublas without memory: " << cublas_wo_mem_time << " ns" << std::endl;
        std::cout << std::endl;

    #ifdef DEBUG
        std::cout << "Reference" << std::endl;

        //print matrix_ref
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << matrix_ref[i * n + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    #endif
        //free device memory
        cudaFree(device_matrix_a);
        cudaFree(device_matrix_b);
        cudaFree(device_matrix_ref);

        //------------------------------------------------------------------------------------------
        // Compute the matrix multiplication using naive GEMM
        //------------------------------------------------------------------------------------------
    #ifdef NAIVE

        std::cout << "Matrix Multiplication using naive GEMM" << std::endl;

        auto naive_w_mem_start = get_time();

        // Allocate pointers to GPU memory
        cudaMalloc((void **)&device_matrix_a, n * n * sizeof(double));
        cudaMalloc((void **)&device_matrix_b, n * n * sizeof(double));
        cudaMalloc((void **)&device_matrix_naive, n * n * sizeof(double));

        // Copy input data to GPU memory
        cudaMemcpy(device_matrix_a, matrix_a, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_matrix_b, matrix_b, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(device_matrix_naive, 0, n * n * sizeof(double));

        // Compute the matrix multiplication using naive GEMM block set to 32x32 as this is optimal for my GPU
        dim3 block(TILE_SIZE,TILE_SIZE);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        //dim3 block(thread_per_block, thread_per_block);

        auto naive_wo_mem_start = get_time();

        gemm_naive<<<grid, block>>>(device_matrix_a, device_matrix_b, device_matrix_naive, n, n, n);
        cudaDeviceSynchronize();

        auto naive_wo_mem_end = get_time();

        // Copy output data to host memory
        cudaMemcpy(matrix_naive, device_matrix_naive, n * n * sizeof(double), cudaMemcpyDeviceToHost);

        auto naive_w_mem_end = get_time();

        // calcualte time for naive with memory
        auto naive_w_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(naive_w_mem_end - naive_w_mem_start).count();

        // calculate time for naive without memory
        auto naive_wo_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(naive_wo_mem_end - naive_wo_mem_start).count();

        // add time to vector
        naive_w_memory.push_back(naive_w_mem_time);
        naive_wo_memory.push_back(naive_wo_mem_time);

        // print time for naive with memory
        std::cout << "Time for naive with memory: " << naive_w_mem_time << " ns" << std::endl;

        // print time for naive without memory
        std::cout << "Time for naive without memory: " << naive_wo_mem_time << " ns" << std::endl;
        std::cout << std::endl;

    #ifdef DEBUG
        std::cout << "Naive GEMM" << std::endl;

        //print matrix_naive
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << matrix_naive[i * n + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    #endif

        // Free GPU memory
        cudaFree(device_matrix_a);
        cudaFree(device_matrix_b);
        cudaFree(device_matrix_naive);
    #endif

        //------------------------------------------------------------------------------------------
        // Compute the matrix multiplication using shared memory
        //------------------------------------------------------------------------------------------
    #ifdef SHARED
        std::cout << "Matrix Multiplication using shared memory" << std::endl;

        auto tiled_w_mem_start = get_time();

        // Allocate GPU memory
        cudaMalloc((void **)&device_matrix_a, n * n * sizeof(double));
        cudaMalloc((void **)&device_matrix_b, n * n * sizeof(double));
        cudaMalloc((void **)&device_matrix_tiled, n * n * sizeof(double));
        //cudaMemset(device_matrix_tiled, 0, n * n * sizeof(double));

        // Copy input data to GPU memory
        cudaMemcpy(device_matrix_a, matrix_a, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_matrix_b, matrix_b, n * n * sizeof(double), cudaMemcpyHostToDevice);

        // Compute the matrix multiplication using shared memory
        dim3 block_tiled(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid_tiled((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE, 1);

        auto tiled_wo_mem_start = get_time();

        gemm_tiled_v2<<<grid_tiled, block_tiled>>>(device_matrix_a, device_matrix_b,
                                                                device_matrix_tiled, n, n, n);

        auto tiled_wo_mem_end = get_time();

        // Copy output data to host memory
        cudaMemcpy(matrix_tiled, device_matrix_tiled, n * n * sizeof(double), cudaMemcpyDeviceToHost);

        auto tiled_w_mem_end = get_time();

        // calcualte time for tiled with memory
        auto tiled_w_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(tiled_w_mem_end - tiled_w_mem_start).count();

        // calculate time for tiled without memory
        auto tiled_wo_mem_time = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(tiled_wo_mem_end - tiled_wo_mem_start).count();

        // add time to vector
        tiled_w_memory.push_back(tiled_w_mem_time);
        tiled_wo_memory.push_back(tiled_wo_mem_time);

        // print time for tiled with memory
        std::cout << "Time for tiled with memory: " << tiled_w_mem_time << " ns" << std::endl;

        // print time for tiled without memory
        std::cout << "Time for tiled without memory: " << tiled_wo_mem_time << " ns" << std::endl;
        std::cout << std::endl;
        
        #ifdef DEBUG
        std::cout << "matrix_tiled" << std::endl;
        
        //print matrix_tiled
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                std::cout << matrix_tiled[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        #endif
        
        // Free GPU memory
        cudaFree(device_matrix_a);
        cudaFree(device_matrix_b);
        cudaFree(device_matrix_tiled);
    #endif
        
        // calcualte the average error between the reference and the tiled matrix and the naive matrix
        double error_naive = 0.0;
        double error_tiled = 0.0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                error_naive += fabs(matrix_ref[i * n + j] - matrix_naive[i * n + j]);
                error_tiled += fabs(matrix_ref[i * n + j] - matrix_tiled[i * n + j]);
            }
        }

        std::cout << "Average error naive: " << error_naive / (n * n) << std::endl;
        std::cout << "Average error tiled: " << error_tiled / (n * n) << std::endl;
        std::cout << std::endl;

        // Free CPU memory
        free(matrix_a);
        free(matrix_b);
        free(matrix_ref);
        free(matrix_naive);
        free(matrix_tiled);
    }

#ifdef WRITE
    // Write timing data to a CSV file
    std::string filename = "results.csv";
    print_to_csv(naive_w_memory, naive_wo_memory, tiled_w_memory, tiled_wo_memory,
                 cublas_w_memory, cublas_wo_memory, n_values,filename); 
#endif  


    return 0;
}
