#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

struct TransformerConfig {
    int vocab_size;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int max_seq_len;
    float dropout;
    bool use_bias;
};

struct TransformerWeights {
    // Embedding weights
    float* token_embedding;
    float* position_embedding;
    
    // Layer weights
    struct LayerWeights {
        // Self-attention
        float* qkv_weight;
        float* qkv_bias;
        float* proj_weight;
        float* proj_bias;
        
        // MLP
        float* mlp_fc1_weight;
        float* mlp_fc1_bias;
        float* mlp_fc2_weight;
        float* mlp_fc2_bias;
        
        // Layer norm
        float* ln1_weight;
        float* ln1_bias;
        float* ln2_weight;
        float* ln2_bias;
    };
    
    std::vector<LayerWeights> layers;
    
    // Final layer norm
    float* final_ln_weight;
    float* final_ln_bias;
};

struct KV_Cache {
    float* key_cache;
    float* value_cache;
    int* cache_lengths;
    int max_batch_size;
    int max_seq_len;
    int head_dim;
    int num_heads;
};

class TransformerModel {
public:
    TransformerModel(const TransformerConfig& config);
    ~TransformerModel();
    
    // Initialize model weights from file
    bool load_weights(const char* weight_path);
    
    // Inference functions
    void generate(
        const int* input_ids,
        int batch_size,
        int seq_len,
        int max_new_tokens,
        int* output_ids,
        float temperature = 1.0f,
        int top_k = 50
    );
    
    // Memory management
    void allocate_kv_cache(int max_batch_size);
    void free_kv_cache();

    // Attention forward pass (made public for testing)
    void attention_forward(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int batch_size,
        int seq_len,
        int head_dim,
        bool use_cache = false
    );
    
private:
    TransformerConfig config_;
    TransformerWeights weights_;
    KV_Cache kv_cache_;
    
    // CUDA streams
    cudaStream_t stream_;
    
    // Helper functions
    void initialize_weights();
    cudaError_t forward_layer(
        const float* input,
        float* output,
        const TransformerWeights::LayerWeights& layer_weights,
        int batch_size,
        int seq_len,
        bool use_cache
    );
    
    void mlp_forward(
        const float* input,
        float* output,
        const float* fc1_weight,
        const float* fc1_bias,
        const float* fc2_weight,
        const float* fc2_bias,
        int batch_size,
        int seq_len
    );
}; 