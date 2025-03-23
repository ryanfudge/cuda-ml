# Add CUDA source files
add_library(cuda_ml_kernels
    cuda/attention.cu
    cuda/transformer.cu
)

# Set include directories
target_include_directories(cuda_ml_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Set CUDA properties
set_target_properties(cuda_ml_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA libraries
target_link_libraries(cuda_ml_kernels PUBLIC
    ${CUDA_LIBRARIES}
) 