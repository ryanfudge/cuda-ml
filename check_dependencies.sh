#!/bin/bash

echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) found"
    nvcc --version
else
    echo "ERROR: CUDA compiler (nvcc) not found"
    exit 1
fi

echo -e "\nChecking CUDA runtime..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver found"
    nvidia-smi
else
    echo "ERROR: NVIDIA driver not found"
    exit 1
fi

echo -e "\nChecking CMake..."
if command -v cmake &> /dev/null; then
    echo "CMake found"
    cmake --version
else
    echo "ERROR: CMake not found"
    exit 1
fi

echo -e "\nChecking Python dependencies..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "ERROR: PyTorch not found"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "ERROR: NumPy not found"
python3 -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || echo "ERROR: SciPy not found"

echo -e "\nChecking CUDA architecture..."
if [ -f /usr/local/cuda/include/cuda_runtime.h ]; then
    echo "CUDA headers found"
else
    echo "ERROR: CUDA headers not found"
    exit 1
fi

echo -e "\nChecking GTest..."
if [ -f /usr/include/gtest/gtest.h ]; then
    echo "GTest found"
else
    echo "ERROR: GTest not found"
    exit 1
fi

echo -e "\nAll checks completed!" 