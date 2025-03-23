#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }   
}