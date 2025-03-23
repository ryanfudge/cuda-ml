# CUDA-ML Transformer Optimizer

This project focuses on optimizing transformer inference using custom CUDA kernels. It provides implementations and optimizations for transformer models using CUDA 12.4.

## Prerequisites

- CUDA Toolkit 12.4
- CMake (version 3.18 or higher)
- Python 3.8 or higher
- PyTorch (with CUDA support)
- CUDA-enabled GPU

## Project Structure

```
.
├── CMakeLists.txt          # Main CMake configuration
├── requirements.txt        # Python dependencies
├── src/                   # Source code directory
│   ├── cuda/             # CUDA kernel implementations
│   └── python/           # Python bindings and utilities
└── tests/                # Test files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

## Development

This project is under active development. The main focus areas are:
- Custom CUDA kernels for transformer operations
- Memory optimization techniques
- Performance benchmarking
- Integration with PyTorch

## License

MIT License 