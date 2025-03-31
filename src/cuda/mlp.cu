#include "transformer_model.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GELU activation function
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Kernel for first linear layer
__global__ void mlp_fc1_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int ffn_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * ffn_dim) {
        int batch_idx = idx / (seq_len * ffn_dim);
        int seq_idx = (idx % (seq_len * ffn_dim)) / ffn_dim;
        int ffn_idx = idx % ffn_dim;
        
        float val = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            val += input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + i] *
                   weight[i * ffn_dim + ffn_idx];
        }
        output[idx] = val + bias[ffn_idx];
    }
}

// Kernel for GELU activation
__global__ void gelu_kernel(
    float* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = gelu(input[idx]);
    }
}

// Kernel for second linear layer
__global__ void mlp_fc2_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int ffn_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * hidden_dim) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int seq_idx = (idx % (seq_len * hidden_dim)) / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        float val = 0.0f;
        for (int i = 0; i < ffn_dim; i++) {
            val += input[batch_idx * seq_len * ffn_dim + seq_idx * ffn_dim + i] *
                   weight[i * hidden_dim + hidden_idx];
        }
        output[idx] = val + bias[hidden_idx];
    }
}

void TransformerModel::mlp_forward(
    const float* input,
    float* output,
    const float* fc1_weight,
    const float* fc1_bias,
    const float* fc2_weight,
    const float* fc2_bias,
    int batch_size,
    int seq_len
) {
    // Calculate dimensions
    int ffn_dim = config_.hidden_dim * 4;  // Standard MLP dimension
    
    // Allocate intermediate buffer
    float* intermediate;
    size_t intermediate_size = batch_size * seq_len * ffn_dim;
    CUDA_CHECK(cudaMalloc(&intermediate, intermediate_size * sizeof(float)));
    
    // 1. First linear layer
    mlp_fc1_kernel<<<(batch_size * seq_len * ffn_dim + 255) / 256, 256, 0, stream_>>>(
        input,
        fc1_weight,
        fc1_bias,
        intermediate,
        batch_size,
        seq_len,
        config_.hidden_dim,
        ffn_dim
    );
    
    // 2. GELU activation
    gelu_kernel<<<(intermediate_size + 255) / 256, 256, 0, stream_>>>(
        intermediate,
        intermediate_size
    );
    
    // 3. Second linear layer
    mlp_fc2_kernel<<<(batch_size * seq_len * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
        intermediate,
        fc2_weight,
        fc2_bias,
        output,
        batch_size,
        seq_len,
        config_.hidden_dim,
        ffn_dim
    );
    
    // Free intermediate buffer
    CUDA_CHECK(cudaFree(intermediate));
} 