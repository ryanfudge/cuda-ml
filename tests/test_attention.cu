#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../src/cuda/attention.cuh"

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate test data
        batch_size = 2;
        seq_len = 4;
        head_dim = 8;
        size_t size = batch_size * seq_len * head_dim * sizeof(float);
        
        cudaMallocHost(&h_query, size);
        cudaMallocHost(&h_key, size);
        cudaMallocHost(&h_value, size);
        cudaMallocHost(&h_output, size);
        
        cudaMalloc(&d_query, size);
        cudaMalloc(&d_key, size);
        cudaMalloc(&d_value, size);
        cudaMalloc(&d_output, size);
        
        // Initialize test data
        for (size_t i = 0; i < batch_size * seq_len * head_dim; i++) {
            h_query[i] = 1.0f;
            h_key[i] = 2.0f;
            h_value[i] = 3.0f;
        }
        
        // Copy data to device
        cudaMemcpy(d_query, h_query, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_key, h_key, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, h_value, size, cudaMemcpyHostToDevice);
    }
    
    void TearDown() override {
        // Free memory
        cudaFreeHost(h_query);
        cudaFreeHost(h_key);
        cudaFreeHost(h_value);
        cudaFreeHost(h_output);
        
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_output);
    }
    
    int batch_size, seq_len, head_dim;
    float *h_query, *h_key, *h_value, *h_output;
    float *d_query, *d_key, *d_value, *d_output;
};

TEST_F(AttentionTest, BasicFunctionality) {
    // Run the attention kernel
    compute_attention(d_query, d_key, d_value, d_output,
                     batch_size, seq_len, head_dim);
    
    // Copy result back to host
    size_t size = batch_size * seq_len * head_dim * sizeof(float);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Verify output (basic test)
    for (size_t i = 0; i < batch_size * seq_len * head_dim; i++) {
        EXPECT_EQ(h_output[i], 0.0f);  // Currently just checking placeholder implementation
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 