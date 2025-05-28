#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../src/cuda/transformer_model.h"
#include <random>

// Helper macro for CUDA error checking
#define CUDA_CHECK_TEST(call) do {                                 \
    cudaError_t error = call;                                     \
    if (error != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__,                               \
                cudaGetErrorString(error));                       \
        FAIL() << "CUDA error";                                   \
    }                                                            \
} while(0)

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set dimensions
        batch_size = 2;
        seq_len = 8;
        head_dim = 32;  // Increased for more realistic testing
        num_heads = 4;  // Increased for multi-head testing

        // Initialize transformer model
        TransformerConfig config;
        config.hidden_dim = head_dim * num_heads;
        config.num_heads = num_heads;
        config.num_layers = 1;
        config.vocab_size = 1000;
        config.max_seq_len = 512;
        model = new TransformerModel(config);

        // Allocate test data
        size_t size = batch_size * seq_len * config.hidden_dim * sizeof(float);
        
        CUDA_CHECK_TEST(cudaMallocHost(&h_query, size));
        CUDA_CHECK_TEST(cudaMallocHost(&h_key, size));
        CUDA_CHECK_TEST(cudaMallocHost(&h_value, size));
        CUDA_CHECK_TEST(cudaMallocHost(&h_output, size));
        
        CUDA_CHECK_TEST(cudaMalloc(&d_query, size));
        CUDA_CHECK_TEST(cudaMalloc(&d_key, size));
        CUDA_CHECK_TEST(cudaMalloc(&d_value, size));
        CUDA_CHECK_TEST(cudaMalloc(&d_output, size));
        
        // Initialize test data with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);  // Normal distribution for better numerical stability
        
        for (size_t i = 0; i < batch_size * seq_len * config.hidden_dim; i++) {
            h_query[i] = dist(gen);
            h_key[i] = dist(gen);
            h_value[i] = dist(gen);
        }
        
        // Copy data to device
        CUDA_CHECK_TEST(cudaMemcpy(d_query, h_query, size, cudaMemcpyHostToDevice));
        CUDA_CHECK_TEST(cudaMemcpy(d_key, h_key, size, cudaMemcpyHostToDevice));
        CUDA_CHECK_TEST(cudaMemcpy(d_value, h_value, size, cudaMemcpyHostToDevice));
    }
    
    void TearDown() override {
        // Free memory with error checking
        CUDA_CHECK_TEST(cudaFreeHost(h_query));
        CUDA_CHECK_TEST(cudaFreeHost(h_key));
        CUDA_CHECK_TEST(cudaFreeHost(h_value));
        CUDA_CHECK_TEST(cudaFreeHost(h_output));
        
        CUDA_CHECK_TEST(cudaFree(d_query));
        CUDA_CHECK_TEST(cudaFree(d_key));
        CUDA_CHECK_TEST(cudaFree(d_value));
        CUDA_CHECK_TEST(cudaFree(d_output));

        delete model;
    }
    
    int batch_size, seq_len, head_dim, num_heads;
    float *h_query, *h_key, *h_value, *h_output;
    float *d_query, *d_key, *d_value, *d_output;
    TransformerModel* model;
};

TEST_F(AttentionTest, BasicFunctionality) {
    // Run the attention forward pass
    CUDA_CHECK_TEST(cudaDeviceSynchronize());  // Ensure previous operations are complete
    model->attention_forward(d_query, d_key, d_value, d_output,
                           batch_size, seq_len, head_dim, false);
    CUDA_CHECK_TEST(cudaDeviceSynchronize());  // Ensure attention computation is complete
    
    // Copy result back to host
    size_t size = batch_size * seq_len * (head_dim * num_heads) * sizeof(float);
    CUDA_CHECK_TEST(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify output properties
    bool all_zeros = true;
    bool all_finite = true;
    float sum = 0.0f;
    
    for (size_t i = 0; i < batch_size * seq_len * (head_dim * num_heads); i++) {
        if (h_output[i] != 0.0f) all_zeros = false;
        if (std::isfinite(h_output[i])) {
            sum += h_output[i];
        } else {
            all_finite = false;
        }
    }
    
    EXPECT_FALSE(all_zeros) << "Output should not be all zeros";
    EXPECT_TRUE(all_finite) << "Output should contain only finite values";
    EXPECT_FALSE(std::isnan(sum)) << "Output sum should not be NaN";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 