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

# Set CUDA architecture before enabling CUDA
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable CUDA
enable_language(CUDA)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Create CUDA library
add_library(cuda_ml SHARED ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set include directories
target_include_directories(cuda_ml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA specific flags
set_target_properties(cuda_ml PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA specific flags
target_compile_options(cuda_ml PRIVATE
    -O3
    --use_fast_math
    -std=c++14
)

# Link against CUDA libraries
target_link_libraries(cuda_ml
    cudart
    cublas
    cublasLt
) 