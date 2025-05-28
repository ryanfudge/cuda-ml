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
    int chunk_size,
    int hidden_dim,
    int intermediate_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;

    // Each thread computes one element of the intermediate output
    float sum = 0.0f;
    for (int h = 0; h < hidden_dim; h++) {
        sum += input[h] * weight[h * intermediate_dim + idx];
    }
    output[idx] = sum + bias[idx];
}

// Kernel for GELU activation
__global__ void gelu_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x * 0.5f * (1.0f + tanhf(0.797884560802865f * x * (1.0f + 0.044715f * x * x)));
}

// Kernel for second linear layer
__global__ void mlp_fc2_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int chunk_size,
    int intermediate_dim,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;

    // Each thread computes one element of the output
    float sum = 0.0f;
    for (int h = 0; h < intermediate_dim; h++) {
        sum += input[h] * weight[h * hidden_dim + idx];
    }
    output[idx] = sum + bias[idx];
}

cudaError_t mlp_forward(
    const float* input,
    const float* fc1_weight,
    const float* fc1_bias, 
    const float* fc2_weight,
    const float* fc2_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream_
) {
    printf("MLP forward: batch_size=%d, seq_len=%d, hidden_dim=%d\n", batch_size, seq_len, hidden_dim);
    
    // Calculate total elements to process
    size_t total_elements = batch_size * seq_len * hidden_dim;
    
    // Use a very small chunk size of 64 elements
    const size_t chunk_size = 64;
    const size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    
    printf("Total elements: %zu, Chunk size: %zu, Num chunks: %zu\n", 
           total_elements, chunk_size, num_chunks);
    
    // Allocate intermediate buffer for chunk processing
    float* intermediate_buffer = nullptr;
    cudaError_t alloc_status = cudaMalloc(&intermediate_buffer, chunk_size * sizeof(float));
    if (alloc_status != cudaSuccess) {
        printf("Failed to allocate intermediate buffer: %s\n", cudaGetErrorString(alloc_status));
        return alloc_status;
    }
    printf("Intermediate buffer allocated successfully with size %zu\n", chunk_size);

    // Process each chunk
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        size_t start_idx = chunk * chunk_size;
        size_t end_idx = min(start_idx + chunk_size, total_elements);
        size_t current_chunk_size = end_idx - start_idx;

        printf("Processing chunk %zu/%zu: elements %zu to %zu (size: %zu)\n", 
               chunk + 1, num_chunks, start_idx, end_idx, current_chunk_size);

        // FC1: input -> intermediate (4x expansion)
        dim3 block_size(64);  // Match block size to chunk size
        dim3 grid_size((current_chunk_size + block_size.x - 1) / block_size.x);
        
        printf("Launching FC1: blocks=%d, threads=%d\n", grid_size.x, block_size.x);
        mlp_fc1_kernel<<<grid_size, block_size, 0, stream_>>>(
            input + start_idx,
            fc1_weight,
            fc1_bias,
            intermediate_buffer,
            current_chunk_size,
            hidden_dim,
            intermediate_dim
        );
        if (cudaGetLastError() != cudaSuccess) {
            printf("FC1 kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            cudaFree(intermediate_buffer);
            return cudaGetLastError();
        }
        printf("FC1 completed\n");

        // GELU activation
        printf("Launching GELU: blocks=%d, threads=%d\n", grid_size.x, block_size.x);
        gelu_kernel<<<grid_size, block_size, 0, stream_>>>(
            intermediate_buffer,
            intermediate_buffer,
            current_chunk_size
        );
        if (cudaGetLastError() != cudaSuccess) {
            printf("GELU kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            cudaFree(intermediate_buffer);
            return cudaGetLastError();
        }
        printf("GELU completed\n");

        // FC2: intermediate -> output
        printf("Launching FC2: blocks=%d, threads=%d\n", grid_size.x, block_size.x);
        mlp_fc2_kernel<<<grid_size, block_size, 0, stream_>>>(
            intermediate_buffer,
            fc2_weight,
            fc2_bias,
            output + start_idx,
            current_chunk_size,
            intermediate_dim,
            hidden_dim
        );
        if (cudaGetLastError() != cudaSuccess) {
            printf("FC2 kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            cudaFree(intermediate_buffer);
            return cudaGetLastError();
        }
        printf("FC2 completed\n");

        // Synchronize after each chunk to ensure proper execution
        cudaError_t sync_status = cudaStreamSynchronize(stream_);
        if (sync_status != cudaSuccess) {
            printf("Stream synchronization failed: %s\n", cudaGetErrorString(sync_status));
            cudaFree(intermediate_buffer);
            return sync_status;
        }
    }

    printf("Freeing intermediate buffer\n");
    cudaFree(intermediate_buffer);
    printf("Memory freed successfully\n");

    printf("MLP forward completed successfully\n");
    return cudaSuccess;
} 