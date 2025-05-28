#include "transformer_model.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initialize model config
    TransformerConfig config = {
        .vocab_size = 50257,  // GPT-2 vocabulary size
        .hidden_dim = 768,    // GPT-2 small
        .num_layers = 12,
        .num_heads = 12,
        .max_seq_len = 1024,
        .dropout = 0.1f,
        .use_bias = true
    };
    
    // Create model
    TransformerModel model(config);
    
    // Test input
    int input_ids[] = {50256, 50256, 50256};  // Example input
    int output_ids[100];  // Buffer for generated tokens
    
    // Generate text
    model.generate(input_ids, 1, 3, 100, output_ids, 0.7f, 50);
    
    // Print generated tokens
    printf("Generated tokens: ");
    for (int i = 0; i < 100; i++) {
        printf("%d ", output_ids[i]);
    }
    printf("\n");
    
    return 0;
} 