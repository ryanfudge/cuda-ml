# Find GTest
find_package(GTest REQUIRED)

# Add test executable
add_executable(attention_test
    test_attention.cu
)

# Set CUDA properties
set_target_properties(attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against GTest and our CUDA kernels
target_link_libraries(attention_test
    PRIVATE
    cuda_ml_kernels
    GTest::GTest
    GTest::Main
)

# Add test to CTest
add_test(NAME attention_test COMMAND attention_test) 