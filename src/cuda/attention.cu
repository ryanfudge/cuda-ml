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

// Kernel for QKV projection
__global__ void qkv_projection_kernel(
    const float* input,
    const float* qkv_weight,
    const float* qkv_bias,
    float* q,
    float* k,
    float* v,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * hidden_dim) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int seq_idx = (idx % (seq_len * hidden_dim)) / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        float val = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            val += input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + i] *
                   qkv_weight[i * hidden_dim * 3 + hidden_idx];
        }
        val += qkv_bias[hidden_idx];
        
        // Split into Q, K, V
        int head_idx = hidden_idx / head_dim;
        int head_offset = hidden_idx % head_dim;
        
        if (head_idx < num_heads) {
            q[batch_idx * seq_len * num_heads * head_dim + seq_idx * num_heads * head_dim + head_idx * head_dim + head_offset] = val;
        } else if (head_idx < 2 * num_heads) {
            k[batch_idx * seq_len * num_heads * head_dim + seq_idx * num_heads * head_dim + (head_idx - num_heads) * head_dim + head_offset] = val;
        } else {
            v[batch_idx * seq_len * num_heads * head_dim + seq_idx * num_heads * head_dim + (head_idx - 2 * num_heads) * head_dim + head_offset] = val;
        }
    }
}

// Kernel for attention scores computation
__global__ void attention_scores_kernel(
    const float* q,
    const float* k,
    float* scores,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_heads * seq_len * seq_len) {
        int batch_idx = idx / (num_heads * seq_len * seq_len);
        int head_idx = (idx % (num_heads * seq_len * seq_len)) / (seq_len * seq_len);
        int q_idx = (idx % (seq_len * seq_len)) / seq_len;
        int k_idx = idx % seq_len;
        
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q[batch_idx * seq_len * num_heads * head_dim + q_idx * num_heads * head_dim + head_idx * head_dim + i] *
                    k[batch_idx * seq_len * num_heads * head_dim + k_idx * num_heads * head_dim + head_idx * head_dim + i];
        }
        scores[idx] = score * scale;
    }
}

// Kernel for attention output computation
__global__ void attention_output_kernel(
    const float* scores,
    const float* v,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * num_heads * head_dim) {
        int batch_idx = idx / (seq_len * num_heads * head_dim);
        int seq_idx = (idx % (seq_len * num_heads * head_dim)) / (num_heads * head_dim);
        int head_idx = (idx % (num_heads * head_dim)) / head_dim;
        int head_offset = idx % head_dim;
        
        float val = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            val += scores[batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + seq_idx * seq_len + i] *
                   v[batch_idx * seq_len * num_heads * head_dim + i * num_heads * head_dim + head_idx * head_dim + head_offset];
        }
        output[idx] = val;
    }
}

// Kernel for output projection
__global__ void output_projection_kernel(
    const float* input,
    const float* proj_weight,
    const float* proj_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * hidden_dim) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int seq_idx = (idx % (seq_len * hidden_dim)) / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        float val = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            val += input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + i] *
                   proj_weight[i * hidden_dim + hidden_idx];
        }
        output[idx] = val + proj_bias[hidden_idx];
    }
}

void TransformerModel::attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int head_dim,
    bool use_cache
) {
    // Allocate temporary buffers
    float *q, *k, *v, *scores, *attn_output;
    size_t qkv_size = batch_size * seq_len * config_.num_heads * head_dim;
    size_t scores_size = batch_size * config_.num_heads * seq_len * seq_len;
    
    CUDA_CHECK(cudaMalloc(&q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scores, scores_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attn_output, qkv_size * sizeof(float)));
    
    // 1. QKV projection
    qkv_projection_kernel<<<(batch_size * seq_len * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
        query,
        weights_.layers[0].qkv_weight,
        weights_.layers[0].qkv_bias,
        q, k, v,
        batch_size, seq_len,
        config_.hidden_dim,
        config_.num_heads,
        head_dim
    );
    
    // 2. Compute attention scores
    float scale = 1.0f / sqrtf(head_dim);
    attention_scores_kernel<<<(batch_size * config_.num_heads * seq_len * seq_len + 255) / 256, 256, 0, stream_>>>(
        q, k, scores,
        batch_size, seq_len,
        config_.num_heads,
        head_dim,
        scale
    );
    
    // 3. Apply softmax
    softmax_kernel<<<(batch_size * config_.num_heads * seq_len * seq_len + 255) / 256, 256, 0, stream_>>>(
        scores, scores,
        seq_len * seq_len
    );
    
    // 4. Compute attention output
    attention_output_kernel<<<(batch_size * seq_len * config_.num_heads * head_dim + 255) / 256, 256, 0, stream_>>>(
        scores, v, attn_output,
        batch_size, seq_len,
        config_.num_heads,
        head_dim
    );
    
    // 5. Output projection
    output_projection_kernel<<<(batch_size * seq_len * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
        attn_output,
        weights_.layers[0].proj_weight,
        weights_.layers[0].proj_bias,
        output,
        batch_size, seq_len,
        config_.hidden_dim
    );
    
    // Free temporary buffers
    CUDA_CHECK(cudaFree(q));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(scores));
    CUDA_CHECK(cudaFree(attn_output));
} 