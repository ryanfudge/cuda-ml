#pragma once

#include <cuda_runtime.h>

// C++ wrapper function declaration
extern "C" void compute_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int head_dim
); 