#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    float sum = 0.0f;
    if (i < output_size) {
        for (int j = 0; j < kernel_size; j++) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
}



