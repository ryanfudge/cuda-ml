#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        int inIdx = row * cols + col;
        int outIdx = col * rows + row;
        output[outIdx] = input[inIdx];
        return;
    }
}