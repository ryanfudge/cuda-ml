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

// Kernel for softmax
__global__ void softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    float temperature = 1.0f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = batch_size * num_heads * seq_len;
    if (idx < total_rows) {
        int batch_head_seq_idx = idx;  // Each thread processes one row of seq_len elements
        
        // Find max value for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            float val = input[batch_head_seq_idx * seq_len + i];
            max_val = max(max_val, val);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float val = expf((input[batch_head_seq_idx * seq_len + i] - max_val) / temperature);
            output[batch_head_seq_idx * seq_len + i] = val;
            sum += val;
        }
        
        // Normalize
        for (int i = 0; i < seq_len; i++) {
            output[batch_head_seq_idx * seq_len + i] /= sum;
        }
    }
}

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
        
        // Compute Q, K, V projections
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        
        // Each weight matrix is hidden_dim x hidden_dim
        const float* q_weight = qkv_weight;
        const float* k_weight = qkv_weight + hidden_dim * hidden_dim;
        const float* v_weight = qkv_weight + 2 * hidden_dim * hidden_dim;
        
        // Input index for the current token
        int input_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
        
        // Matrix multiplication for Q, K, V
        for (int i = 0; i < hidden_dim; i++) {
            float input_val = input[input_offset + i];
            q_val += input_val * q_weight[i * hidden_dim + hidden_idx];
            k_val += input_val * k_weight[i * hidden_dim + hidden_idx];
            v_val += input_val * v_weight[i * hidden_dim + hidden_idx];
        }
        
        // Add biases
        q_val += qkv_bias[hidden_idx];
        k_val += qkv_bias[hidden_dim + hidden_idx];
        v_val += qkv_bias[2 * hidden_dim + hidden_idx];
        
        // Split into heads and write to output
        int head_idx = hidden_idx / head_dim;
        int head_offset = hidden_idx % head_dim;
        
        if (head_idx < num_heads) {
            // Output index for the current position
            int out_offset = batch_idx * seq_len * num_heads * head_dim + 
                           seq_idx * num_heads * head_dim +
                           head_idx * head_dim +
                           head_offset;
            
            q[out_offset] = q_val;
            k[out_offset] = k_val;
            v[out_offset] = v_val;
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
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    
    if (idx < total_elements) {
        // Calculate indices
        int batch_idx = idx / (num_heads * seq_len * seq_len);
        int head_idx = (idx % (num_heads * seq_len * seq_len)) / (seq_len * seq_len);
        int q_idx = (idx % (seq_len * seq_len)) / seq_len;
        int k_idx = idx % seq_len;
        
        // Base indices for q and k
        int q_base = batch_idx * seq_len * num_heads * head_dim + 
                    q_idx * num_heads * head_dim +
                    head_idx * head_dim;
        
        int k_base = batch_idx * seq_len * num_heads * head_dim +
                    k_idx * num_heads * head_dim +
                    head_idx * head_dim;
        
        // Compute dot product
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q[q_base + i] * k[k_base + i];
        }
        
        // Apply scaling and write to output
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
    int total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (idx < total_elements) {
        // Calculate indices
        int batch_idx = idx / (seq_len * num_heads * head_dim);
        int seq_idx = (idx % (seq_len * num_heads * head_dim)) / (num_heads * head_dim);
        int head_idx = (idx % (num_heads * head_dim)) / head_dim;
        int head_offset = idx % head_dim;
        
        // Base indices
        int scores_base = batch_idx * num_heads * seq_len * seq_len + 
                         head_idx * seq_len * seq_len +
                         seq_idx * seq_len;
                         
        int v_base = batch_idx * seq_len * num_heads * head_dim;
        
        // Compute weighted sum
        float val = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float score = scores[scores_base + i];
            float v_val = v[v_base + i * num_heads * head_dim + head_idx * head_dim + head_offset];
            val += score * v_val;
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
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;
    
    if (idx < total_elements) {
        // Calculate indices
        int batch_idx = idx / (seq_len * hidden_dim);
        int seq_idx = (idx % (seq_len * hidden_dim)) / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        // Base index for input
        int input_base = batch_idx * seq_len * num_heads * head_dim + 
                        seq_idx * num_heads * head_dim;
        
        // Compute output
        float val = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            int head_idx = i / head_dim;
            int head_offset = i % head_dim;
            if (head_idx < num_heads) {
                float input_val = input[input_base + head_idx * head_dim + head_offset];
                val += input_val * proj_weight[i * hidden_dim + hidden_idx];
            }
        }
        
        // Add bias and write output
        output[idx] = val + proj_bias[hidden_idx];
    }
}

struct AttentionBuffers {
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *scores = nullptr;
    float *attn_output = nullptr;
    
    ~AttentionBuffers() {
        if (q) cudaFree(q);
        if (k) cudaFree(k);
        if (v) cudaFree(v);
        if (scores) cudaFree(scores);
        if (attn_output) cudaFree(attn_output);
    }
};

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
    cudaError_t err = cudaSuccess;
    printf("Attention forward: batch_size=%d, seq_len=%d, head_dim=%d, num_heads=%d\n",
           batch_size, seq_len, head_dim, config_.num_heads);
    
    // Allocate temporary buffers
    AttentionBuffers bufs;
    size_t qkv_size = batch_size * seq_len * config_.num_heads * head_dim;
    size_t scores_size = batch_size * config_.num_heads * seq_len * seq_len;
    
    printf("Allocating memory: qkv_size=%zu, scores_size=%zu\n", qkv_size, scores_size);
    
    // Allocate all memory first
    if ((err = cudaMalloc(&bufs.q, qkv_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate q: %s\n", cudaGetErrorString(err));
        return;
    }
    if ((err = cudaMalloc(&bufs.k, qkv_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate k: %s\n", cudaGetErrorString(err));
        return;
    }
    if ((err = cudaMalloc(&bufs.v, qkv_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate v: %s\n", cudaGetErrorString(err));
        return;
    }
    if ((err = cudaMalloc(&bufs.scores, scores_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate scores: %s\n", cudaGetErrorString(err));
        return;
    }
    if ((err = cudaMalloc(&bufs.attn_output, qkv_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate attn_output: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("Memory allocated successfully\n");
    
    // Launch kernels
    int block_size = 256;
    int num_blocks;
    
    // QKV projection
    num_blocks = (batch_size * seq_len * config_.hidden_dim + block_size - 1) / block_size;
    printf("Launching QKV projection: blocks=%d, threads=%d\n", num_blocks, block_size);
    
    qkv_projection_kernel<<<num_blocks, block_size, 0, stream_>>>(
        query,
        weights_.layers[0].qkv_weight,
        weights_.layers[0].qkv_bias,
        bufs.q, bufs.k, bufs.v,
        batch_size,
        seq_len,
        config_.hidden_dim,
        config_.num_heads,
        head_dim
    );
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("QKV projection failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("QKV projection completed\n");
    
    // Attention scores
    float scale = 1.0f / sqrtf(head_dim);
    num_blocks = (batch_size * config_.num_heads * seq_len * seq_len + block_size - 1) / block_size;
    printf("Launching attention scores: blocks=%d, threads=%d, scale=%f\n", num_blocks, block_size, scale);
    
    attention_scores_kernel<<<num_blocks, block_size, 0, stream_>>>(
        bufs.q, bufs.k, bufs.scores,
        batch_size,
        seq_len,
        config_.num_heads,
        head_dim,
        scale
    );
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("Attention scores failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Attention scores completed\n");
    
    // Softmax
    num_blocks = (batch_size * config_.num_heads * seq_len + block_size - 1) / block_size;
    printf("Launching softmax: blocks=%d, threads=%d\n", num_blocks, block_size);
    
    softmax_kernel<<<num_blocks, block_size, 0, stream_>>>(
        bufs.scores,
        bufs.scores,  // In-place softmax
        batch_size,
        config_.num_heads,
        seq_len
    );
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("Softmax failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Softmax completed\n");
    
    // Attention output
    num_blocks = (batch_size * seq_len * config_.num_heads * head_dim + block_size - 1) / block_size;
    printf("Launching attention output: blocks=%d, threads=%d\n", num_blocks, block_size);
    
    attention_output_kernel<<<num_blocks, block_size, 0, stream_>>>(
        bufs.scores,
        bufs.v,
        bufs.attn_output,
        batch_size,
        seq_len,
        config_.num_heads,
        head_dim
    );
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("Attention output failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Attention output completed\n");
    
    // Output projection
    num_blocks = (batch_size * seq_len * config_.hidden_dim + block_size - 1) / block_size;
    printf("Launching output projection: blocks=%d, threads=%d\n", num_blocks, block_size);
    
    output_projection_kernel<<<num_blocks, block_size, 0, stream_>>>(
        bufs.attn_output,
        weights_.layers[0].proj_weight,
        weights_.layers[0].proj_bias,
        output,
        batch_size,
        seq_len,
        config_.hidden_dim,
        config_.num_heads,
        head_dim
    );
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("Output projection failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("Output projection completed\n");
    
    // Memory will be freed by AttentionBuffers destructor
    printf("Memory freed successfully\n");
    
    if (err != cudaSuccess) {
        printf("Attention forward failed with error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
} 