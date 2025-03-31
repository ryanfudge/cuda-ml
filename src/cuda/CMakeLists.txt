set(CUDA_SOURCES
    vectorAddition.cu
    matrixMultiplication.cu
    matrixTranspose.cu
    attention.cu
    transformer.cu
    transformer_model.cu
    mlp.cu
    generation.cu
)

set(CUDA_HEADERS
    transformer_model.h
)

# Create CUDA library
cuda_add_library(cuda_ml ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set include directories
target_include_directories(cuda_ml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR}
)

# Set CUDA specific flags
set_target_properties(cuda_ml PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA specific flags
target_compile_options(cuda_ml PRIVATE
    -O3
    -arch=sm_60
    --use_fast_math
    -std=c++14
)

# Link against CUDA libraries
target_link_libraries(cuda_ml
    cudart
    cublas
    cublasLt
) 