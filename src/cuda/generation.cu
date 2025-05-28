#include "transformer_model.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <random>

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

// Kernel for token embedding lookup
__global__ void token_embedding_kernel(
    const int* input_ids,
    const float* embedding_table,
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
        
        int token_id = input_ids[batch_idx * seq_len + seq_idx];
        output[idx] = embedding_table[token_id * hidden_dim + hidden_idx];
    }
}

// Kernel for positional embedding addition
__global__ void add_positional_embedding_kernel(
    float* input,
    const float* pos_embedding,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * hidden_dim) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int seq_idx = (idx % (seq_len * hidden_dim)) / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        input[idx] += pos_embedding[seq_idx * hidden_dim + hidden_idx];
    }
}

// Kernel for logits to probabilities
__global__ void logits_to_probs_kernel(
    const float* logits,
    float* probs,
    int vocab_size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        probs[idx] = expf(logits[idx] / temperature);
    }
}

// Kernel for top-k sampling
__global__ void top_k_sampling_kernel(
    const float* probs,
    int* output,
    int vocab_size,
    int k,
    float random_value
) {
    __shared__ float normalized_probs[1024];  // Assuming vocab size <= 1024
    
    // This is a simplified version. In practice, you'd want to use a more efficient
    // implementation with parallel reduction and sorting
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        
        // Compute sum and copy probabilities
        for (int i = 0; i < vocab_size; i++) {
            normalized_probs[i] = probs[i];
            sum += probs[i];
        }
        
        // Normalize probabilities
        for (int i = 0; i < vocab_size; i++) {
            normalized_probs[i] /= sum;
        }
        
        // Simple sampling (not efficient for large vocabularies)
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += normalized_probs[i];
            if (cumsum >= random_value) {
                *output = i;
                break;
            }
        }
    }
}

// Kernel for logits projection
__global__ void logits_projection_kernel(
    const float* hidden_states,
    const float* final_ln_weight,
    const float* final_ln_bias,
    const float* lm_head,
    float* logits,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * vocab_size) {
        int batch_idx = idx / vocab_size;
        int vocab_idx = idx % vocab_size;
        
        // Apply final layer norm
        float normed = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float val = hidden_states[batch_idx * hidden_dim + i];
            normed += val * final_ln_weight[i] + final_ln_bias[i];
        }
        
        // Project to vocabulary
        float logit = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            logit += normed * lm_head[i * vocab_size + vocab_idx];
        }
        logits[idx] = logit;
    }
}

void TransformerModel::generate(
    const int* input_ids,
    int batch_size,
    int seq_len,
    int max_new_tokens,
    int* output_ids,
    float temperature,
    int top_k
) {
    // Allocate buffers
    float *hidden_states, *logits, *probs;
    size_t hidden_size = batch_size * (seq_len + max_new_tokens) * config_.hidden_dim;
    size_t logits_size = batch_size * config_.vocab_size;
    
    CUDA_CHECK(cudaMalloc(&hidden_states, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&logits, logits_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&probs, logits_size * sizeof(float)));
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Process input tokens
    token_embedding_kernel<<<(batch_size * seq_len * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
        input_ids,
        weights_.token_embedding,
        hidden_states,
        batch_size,
        seq_len,
        config_.hidden_dim
    );
    
    add_positional_embedding_kernel<<<(batch_size * seq_len * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
        hidden_states,
        weights_.position_embedding,
        batch_size,
        seq_len,
        config_.hidden_dim
    );
    
    // Generate new tokens
    for (int i = 0; i < max_new_tokens; i++) {
        // Forward pass through transformer layers
        for (int layer = 0; layer < config_.num_layers; layer++) {
            forward_layer(
                hidden_states + layer * batch_size * (seq_len + i) * config_.hidden_dim,
                hidden_states + (layer + 1) * batch_size * (seq_len + i) * config_.hidden_dim,
                weights_.layers[layer],
                batch_size,
                seq_len + i,
                i > 0  // Use KV cache for subsequent tokens
            );
        }
        
        // Project to vocabulary
        logits_projection_kernel<<<(batch_size * config_.vocab_size + 255) / 256, 256, 0, stream_>>>(
            hidden_states + config_.num_layers * batch_size * (seq_len + i) * config_.hidden_dim,
            weights_.final_ln_weight,
            weights_.final_ln_bias,
            weights_.token_embedding,  // Using token embedding as language model head
            logits,
            batch_size,
            config_.hidden_dim,
            config_.vocab_size
        );
        
        // Convert logits to probabilities
        logits_to_probs_kernel<<<(config_.vocab_size + 255) / 256, 256, 0, stream_>>>(
            logits,
            probs,
            config_.vocab_size,
            temperature
        );
        
        // Sample next token
        float random_value = dis(gen);
        top_k_sampling_kernel<<<1, 256, 0, stream_>>>(
            probs,
            output_ids + i,
            config_.vocab_size,
            top_k,
            random_value
        );
        
        // Update hidden states for next token
        token_embedding_kernel<<<(batch_size * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
            output_ids + i,
            weights_.token_embedding,
            hidden_states + (seq_len + i) * config_.hidden_dim,
            batch_size,
            1,
            config_.hidden_dim
        );
        
        add_positional_embedding_kernel<<<(batch_size * config_.hidden_dim + 255) / 256, 256, 0, stream_>>>(
            hidden_states + (seq_len + i) * config_.hidden_dim,
            weights_.position_embedding + (seq_len + i) * config_.hidden_dim,
            batch_size,
            1,
            config_.hidden_dim
        );
    }
    
    // Free buffers
    CUDA_CHECK(cudaFree(hidden_states));
    CUDA_CHECK(cudaFree(logits));
    CUDA_CHECK(cudaFree(probs));
} 