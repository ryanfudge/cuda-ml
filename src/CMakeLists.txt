# Add CUDA subdirectory
add_subdirectory(cuda)

# Create library target
add_library(cuda_ml_kernels STATIC
    cuda/1dConvolution.cu
    cuda/vectorAddition.cu
    cuda/matrixMultiplication.cu
    cuda/matrixTranspose.cu
    cuda/attention.cu
    cuda/transformer.cu
    cuda/transformer_model.cu
    cuda/mlp.cu
    cuda/generation.cu
)

# Set include directories
target_include_directories(cuda_ml_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA specific flags
set_target_properties(cuda_ml_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA specific flags
target_compile_options(cuda_ml_kernels PRIVATE
    -O3
    --use_fast_math
    -std=c++14
)

# Link against CUDA libraries
target_link_libraries(cuda_ml_kernels
    cudart
    cublas
    cublasLt
) 