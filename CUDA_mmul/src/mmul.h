//---------------------------------------------------------------------------
// mmul.h
// Author: Frank Sossi,
//
//
//---------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <fstream>
#include <cmath>
#include <random>
#include <cstdio>

//#define TESTDATA
#define RANDOMDATA
//#define DEBUG

// This is the size of the block
constexpr int TILE_SIZE = 32;


//---------------------------------------------------------------------------
// Function for Naive Matrix Matrix Multiplication
// Input: pointers to matrix_a, matrix_b, and result matrix_c
//        matrix dimensions
// Output: none
//--------------------------------------------------------------------------
template <typename T>
__global__ void gemm_naive(const T *matrix_a, const T *matrix_b, T *matrix_c, int rows, int cols, int width)
{
    // Calculate the global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the dot product of the row of matrix_a and the column of matrix_b
    if (row < rows && col < cols) {
        T temp = 0.0;
        for (int k = 0; k < width; k++) {
            temp += matrix_a[row * width + k] * matrix_b[k * cols + col];
        }
        matrix_c[row * cols + col] = temp;
    }
}

//--------------------------------------------------------------------------

template <typename T>
__global__ void gemm_kahan(const T *matrix_a, const T *matrix_b, T *matrix_c, int rows, int cols, int width)
{
    // Calculate the global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the dot product of the row of matrix_a and the column of matrix_b using Kahan summation
    if (row < rows && col < cols) {
        T sum = 0.0;
        T c = 0.0; // carry term
        for (int k = 0; k < width; k++) {
            T y = matrix_a[row * width + k] * matrix_b[k * cols + col] - c;
            T t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        matrix_c[row * cols + col] = sum;
    }
}


//---------------------------------------------------------------------------
// Function for Naive Matrix Matrix Multiplication Tiling only
// Input: pointers to matrix_a, matrix_b, and result matrix_c
//        matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template <typename T>
__global__ void gemm_tiled(const T *matrix_a, const T *matrix_b, T *matrix_c, int rows, int cols, int width)
{
    // Calculate the global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Allocate shared memory for the tiles
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    // Initialize the output element to 0
    T temp = 0.0;

    // Loop over the tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load the tiles from global memory into shared memory
        if (row < rows && t * TILE_SIZE + threadIdx.x < width) {
            As[threadIdx.y][threadIdx.x] = matrix_a[row * width + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < cols && t * TILE_SIZE + threadIdx.y < width) {
            Bs[threadIdx.y][threadIdx.x] = matrix_b[(t * TILE_SIZE + threadIdx.y) + (col * width)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize threads to ensure all tiles are loaded
        __syncthreads();

        // Compute the dot product of the tiles
        for (int i = 0; i < TILE_SIZE; i++) {
            temp += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize threads to ensure all dot products are computed
        __syncthreads();
    }

    // Write the output element to global memory
    if (row < rows && col < cols) {
        matrix_c[col * rows + row] = temp;
    }
}

//---------------------------------------------------------------------------
// Function for Naive Matrix Matrix Multiplication
// Input: pointers to matrix_a, matrix_b, and result matrix_c
//        matrix dimensions
// Output: none
//---------------------------------------------------------------------------
template <typename T>
__global__ void gemm_tiled_v2(const T *matrix_a, const T *matrix_b, T *matrix_c, int rows, int cols, int width)
{
    // Calculate the global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Allocate shared memory for the tiles
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    // Initialize the output element to 0
    T temp = 0.0;

    // Loop over the tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load the tiles from global memory into shared memory
        if (row < rows && t * TILE_SIZE + threadIdx.x < width) {
            As[threadIdx.y][threadIdx.x] = matrix_a[row * width + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < cols && t * TILE_SIZE + threadIdx.y < width) {
            Bs[threadIdx.y][threadIdx.x] = matrix_b[(t * TILE_SIZE + threadIdx.y) + (col * width)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize threads to ensure all tiles are loaded
        __syncthreads();

        // Compute the dot product of the tiles
        for (int i = 0; i < TILE_SIZE; i += 2) {
            // calc temp using fma
            temp = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], temp);
            //temp += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            if (i+1 < TILE_SIZE) {
                // calc temp using fma
                temp = fma(As[threadIdx.y][i+1], Bs[i+1][threadIdx.x], temp);
                //temp += As[threadIdx.y][i+1] * Bs[i+1][threadIdx.x];
            }
        }


        // Synchronize threads to ensure all dot products are computed
        __syncthreads();
    }

    // Write the output element to global memory
    if (row < rows && col < cols) {
        matrix_c[col * rows + row] = temp;
    }
}

//---------------------------------------------------------------------------
// Function to return time
// Input: none
// Output: returns time in nanoseconds
//---------------------------------------------------------------------------
std::chrono::high_resolution_clock::time_point
get_time()
{
    return std::chrono::high_resolution_clock::now();
}

//---------------------------------------------------------------------------
// Function write timing data to a file
// Input: vectors to hold timing data
// Output: results.csv
//---------------------------------------------------------------------------

