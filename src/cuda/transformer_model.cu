#include "transformer_model.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

// Helper function to initialize weights with random values
void initialize_random_weights(float* d_weights, size_t size, float scale = 0.02f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    float* h_weights = new float[size];
    for (size_t i = 0; i < size; i++) {
        h_weights[i] = dist(gen);
    }
    
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_weights;
}

TransformerModel::TransformerModel(const TransformerConfig& config) : config_(config) {
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize weights
    initialize_weights();

    // Initialize KV cache
    kv_cache_.key_cache = nullptr;
    kv_cache_.value_cache = nullptr;
    kv_cache_.cache_lengths = nullptr;
    kv_cache_.max_batch_size = 0;
    kv_cache_.max_seq_len = config.max_seq_len;
    kv_cache_.head_dim = config.hidden_dim / config.num_heads;
    kv_cache_.num_heads = config.num_heads;
}

TransformerModel::~TransformerModel() {
    // Free CUDA stream
    if (stream_) {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    // Free weights
    if (weights_.token_embedding) {
        CUDA_CHECK(cudaFree(weights_.token_embedding));
    }
    if (weights_.position_embedding) {
        CUDA_CHECK(cudaFree(weights_.position_embedding));
    }
    if (weights_.final_ln_weight) {
        CUDA_CHECK(cudaFree(weights_.final_ln_weight));
    }
    if (weights_.final_ln_bias) {
        CUDA_CHECK(cudaFree(weights_.final_ln_bias));
    }

    // Free layer weights
    for (auto& layer : weights_.layers) {
        if (layer.qkv_weight) CUDA_CHECK(cudaFree(layer.qkv_weight));
        if (layer.qkv_bias) CUDA_CHECK(cudaFree(layer.qkv_bias));
        if (layer.proj_weight) CUDA_CHECK(cudaFree(layer.proj_weight));
        if (layer.proj_bias) CUDA_CHECK(cudaFree(layer.proj_bias));
        if (layer.mlp_fc1_weight) CUDA_CHECK(cudaFree(layer.mlp_fc1_weight));
        if (layer.mlp_fc1_bias) CUDA_CHECK(cudaFree(layer.mlp_fc1_bias));
        if (layer.mlp_fc2_weight) CUDA_CHECK(cudaFree(layer.mlp_fc2_weight));
        if (layer.mlp_fc2_bias) CUDA_CHECK(cudaFree(layer.mlp_fc2_bias));
        if (layer.ln1_weight) CUDA_CHECK(cudaFree(layer.ln1_weight));
        if (layer.ln1_bias) CUDA_CHECK(cudaFree(layer.ln1_bias));
        if (layer.ln2_weight) CUDA_CHECK(cudaFree(layer.ln2_weight));
        if (layer.ln2_bias) CUDA_CHECK(cudaFree(layer.ln2_bias));
    }

    // Free KV cache
    free_kv_cache();
}

void TransformerModel::initialize_weights() {
    // Allocate memory for embeddings
    size_t embedding_size = config_.vocab_size * config_.hidden_dim;
    CUDA_CHECK(cudaMalloc(&weights_.token_embedding, embedding_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights_.position_embedding, config_.max_seq_len * config_.hidden_dim * sizeof(float)));

    // Initialize embeddings with random values
    initialize_random_weights(weights_.token_embedding, embedding_size);
    initialize_random_weights(weights_.position_embedding, config_.max_seq_len * config_.hidden_dim);

    // Allocate and initialize final layer norm
    CUDA_CHECK(cudaMalloc(&weights_.final_ln_weight, config_.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weights_.final_ln_bias, config_.hidden_dim * sizeof(float)));
    initialize_random_weights(weights_.final_ln_weight, config_.hidden_dim, 1.0f);
    initialize_random_weights(weights_.final_ln_bias, config_.hidden_dim, 0.0f);

    // Initialize layer weights
    weights_.layers.resize(config_.num_layers);
    for (int i = 0; i < config_.num_layers; i++) {
        auto& layer = weights_.layers[i];
        
        // Attention weights
        size_t qkv_size = 3 * config_.hidden_dim * config_.hidden_dim;
        CUDA_CHECK(cudaMalloc(&layer.qkv_weight, qkv_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.qkv_bias, 3 * config_.hidden_dim * sizeof(float)));
        initialize_random_weights(layer.qkv_weight, qkv_size);
        initialize_random_weights(layer.qkv_bias, 3 * config_.hidden_dim, 0.0f);

        CUDA_CHECK(cudaMalloc(&layer.proj_weight, config_.hidden_dim * config_.hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.proj_bias, config_.hidden_dim * sizeof(float)));
        initialize_random_weights(layer.proj_weight, config_.hidden_dim * config_.hidden_dim);
        initialize_random_weights(layer.proj_bias, config_.hidden_dim, 0.0f);

        // MLP weights
        size_t mlp_hidden_dim = 4 * config_.hidden_dim;  // Common practice to use 4x hidden dim
        CUDA_CHECK(cudaMalloc(&layer.mlp_fc1_weight, config_.hidden_dim * mlp_hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.mlp_fc1_bias, mlp_hidden_dim * sizeof(float)));
        initialize_random_weights(layer.mlp_fc1_weight, config_.hidden_dim * mlp_hidden_dim);
        initialize_random_weights(layer.mlp_fc1_bias, mlp_hidden_dim, 0.0f);

        CUDA_CHECK(cudaMalloc(&layer.mlp_fc2_weight, mlp_hidden_dim * config_.hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.mlp_fc2_bias, config_.hidden_dim * sizeof(float)));
        initialize_random_weights(layer.mlp_fc2_weight, mlp_hidden_dim * config_.hidden_dim);
        initialize_random_weights(layer.mlp_fc2_bias, config_.hidden_dim, 0.0f);

        // Layer norm weights
        CUDA_CHECK(cudaMalloc(&layer.ln1_weight, config_.hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.ln1_bias, config_.hidden_dim * sizeof(float)));
        initialize_random_weights(layer.ln1_weight, config_.hidden_dim, 1.0f);
        initialize_random_weights(layer.ln1_bias, config_.hidden_dim, 0.0f);

        CUDA_CHECK(cudaMalloc(&layer.ln2_weight, config_.hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.ln2_bias, config_.hidden_dim * sizeof(float)));
        initialize_random_weights(layer.ln2_weight, config_.hidden_dim, 1.0f);
        initialize_random_weights(layer.ln2_bias, config_.hidden_dim, 0.0f);
    }
}

bool TransformerModel::load_weights(const char* weight_path) {
    // TODO: Implement weight loading from file
    // This would involve:
    // 1. Opening the weight file
    // 2. Reading the weights in the correct format
    // 3. Copying them to the GPU memory
    return true;
}

void TransformerModel::allocate_kv_cache(int max_batch_size) {
    // Free existing cache if any
    free_kv_cache();

    // Update cache parameters
    kv_cache_.max_batch_size = max_batch_size;

    // Allocate new cache
    size_t cache_size = max_batch_size * config_.max_seq_len * config_.num_heads * (config_.hidden_dim / config_.num_heads) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&kv_cache_.key_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&kv_cache_.value_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&kv_cache_.cache_lengths, max_batch_size * sizeof(int)));

    // Initialize cache lengths to 0
    CUDA_CHECK(cudaMemset(kv_cache_.cache_lengths, 0, max_batch_size * sizeof(int)));
}

void TransformerModel::free_kv_cache() {
    if (kv_cache_.key_cache) {
        CUDA_CHECK(cudaFree(kv_cache_.key_cache));
        kv_cache_.key_cache = nullptr;
    }
    if (kv_cache_.value_cache) {
        CUDA_CHECK(cudaFree(kv_cache_.value_cache));
        kv_cache_.value_cache = nullptr;
    }
    if (kv_cache_.cache_lengths) {
        CUDA_CHECK(cudaFree(kv_cache_.cache_lengths));
        kv_cache_.cache_lengths = nullptr;
    }
    kv_cache_.max_batch_size = 0;
}

cudaError_t TransformerModel::forward_layer(
    const float* input,
    float* output,
    const TransformerWeights::LayerWeights& layer_weights,
    int batch_size,
    int seq_len,
    bool use_cache
) {
    cudaError_t err = cudaSuccess;
    printf("Transformer forward: batch_size=%d, seq_len=%d, hidden_dim=%d\n",
           batch_size, seq_len, config_.hidden_dim);
    
    // Calculate dimensions
    size_t hidden_size = batch_size * seq_len * config_.hidden_dim;
    size_t ffn_size = batch_size * seq_len * (config_.hidden_dim * 4);
    
    // Allocate temporary buffers with proper alignment
    float *attention_output = nullptr, *mlp_output = nullptr;
    size_t alignment = 256;  // 256-byte alignment for better memory access
    
    // Align sizes
    size_t aligned_hidden_size = (hidden_size + (alignment / sizeof(float)) - 1) & ~((alignment / sizeof(float)) - 1);
    size_t aligned_ffn_size = (ffn_size + (alignment / sizeof(float)) - 1) & ~((alignment / sizeof(float)) - 1);
    
    printf("Allocating attention output: size=%zu\n", aligned_hidden_size);
    if ((err = cudaMalloc(&attention_output, aligned_hidden_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate attention output: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    printf("Allocating MLP output: size=%zu\n", aligned_hidden_size);
    if ((err = cudaMalloc(&mlp_output, aligned_hidden_size * sizeof(float))) != cudaSuccess) {
        printf("Failed to allocate MLP output: %s\n", cudaGetErrorString(err));
        goto cleanup_attention;
    }
    
    // Self-attention
    printf("Running self-attention\n");
    attention_forward(
        input,
        input,  // Key and query are the same for self-attention
        input,  // Value is also the same
        attention_output,
        batch_size,
        seq_len,
        config_.hidden_dim / config_.num_heads,
        true
    );
    
    // Run MLP
    printf("Running MLP\n");
    err = mlp_forward(
        attention_output,
        layer_weights.mlp_fc1_weight,
        layer_weights.mlp_fc1_bias,
        layer_weights.mlp_fc2_weight,
        layer_weights.mlp_fc2_bias,
        mlp_output,
        batch_size,
        seq_len,
        config_.hidden_dim,
        config_.hidden_dim * 4,  // intermediate_dim is 4x hidden_dim
        stream_
    );
    if (err != cudaSuccess) {
        printf("MLP forward failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy to output
    printf("Copying to output\n");
    if ((err = cudaMemcpy(output, mlp_output, hidden_size * sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
        printf("Failed to copy output: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
cleanup:
    // Free temporary buffers
    printf("Freeing MLP output\n");
    if (mlp_output) {
        cudaFree(mlp_output);
    }
    
cleanup_attention:
    printf("Freeing attention output\n");
    if (attention_output) {
        cudaFree(attention_output);
    }
    
    if (err != cudaSuccess) {
        printf("Transformer forward failed with error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    return err;
} 