# VanzyCNN: High-Performance C++ Inference Engine

VanzyCNN is a high-performance Convolutional Neural Network (CNN) inference engine written from scratch in C++. It is optimized for **AVX2/FMA** SIMD instructions and **OpenMP** multi-threading to achieve extreme latency performance on CPU.

## 🚀 Performance Metrics
- **Latency:** ~2.24 ms per image (64x64x3 RGB) on a laptop CPU (Victus).
- **Throughput:** 440+ FPS.
- **Computation:** Custom Matrix Multiplication (Matmul) engine capable of **131 GFLOPS**.

## 🛠️ Key Engineering Features
- **SIMD Optimization:** Hand-written AVX2 and FMA kernels for low-level performance.
- **Memory Management:** Aligned memory allocation (`posix_memalign`) and **CHW** (Channel, Height, Width) tensor format for cache-friendly access.
- **Inference Pipeline:** Includes Conv2D, ReLU, MaxPool, and Dense layers.
- **Python Integration:** Weights are trained in PyTorch and exported as binary files for native C++ inference.

## 🏗️ Architecture
The project follows a decoupled architecture where the **VanzyEngine** handles the heavy mathematical computations, and the **Inference Pipeline** manages the layer logic.
