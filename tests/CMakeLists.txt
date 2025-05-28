# Find GTest package
find_package(GTest REQUIRED)

# Add test executables
add_executable(attention_test test_attention.cu)
add_executable(test_transformer test_transformer.cu)

# Set include directories
target_include_directories(attention_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

target_include_directories(test_transformer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

# Link against libraries
target_link_libraries(attention_test
    cuda_ml_kernels
    GTest::GTest
    GTest::Main
    cudart
    cublas
    cublasLt
)

target_link_libraries(test_transformer
    cuda_ml
    cuda_ml_kernels
    GTest::GTest
    GTest::Main
    cudart
    cublas
    cublasLt
)

# Add tests to CTest
add_test(NAME attention_test COMMAND attention_test)
add_test(NAME test_transformer COMMAND test_transformer) 