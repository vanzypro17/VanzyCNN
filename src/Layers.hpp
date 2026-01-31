#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "Tensor.hpp"
#include "Ops.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstdlib> // posix_memalign ve free için gerekli
#include <limits>

// --- MaxPool Katmanı ---
class MaxPool {
public:
    int pool_size;
    int stride;

    MaxPool(int size, int s) : pool_size(size), stride(s) {}

    // Layers.hpp içindeki forward fonksiyonunu şu şekilde güncelle:
    void forward(const Tensor& input, Tensor& output) {
        int out_h = (input.h - pool_size) / stride + 1;
        int out_w = (input.w - pool_size) / stride + 1;

        #pragma omp parallel for collapse(3)
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < input.h && iw < input.w) {
                                max_val = std::max(max_val, input(c, ih, iw));
                            }
                        }
                    }
                    output(c, oh, ow) = max_val;
                }
            }
        }
    }
};

// --- Conv2D Katmanı ---
class Conv2D {
public:
    int in_c, out_c, ksize, stride, pad;
    Tensor weights;
    std::vector<float> bias;

    Conv2D(int in_channels, int out_channels, int kernel_size, int s = 1, int p = 0) 
        : in_c(in_channels), out_c(out_channels), ksize(kernel_size), stride(s), pad(p),
          weights(out_channels, in_channels * kernel_size, kernel_size), 
          bias(out_channels, 0.0f) {
        
        float stddev = std::sqrt(2.0f / (in_c * ksize * ksize));
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, stddev);

        for(int i = 0; i < out_c * in_c * ksize * ksize; ++i) {
            weights.data[i] = distribution(generator);
        }
    }

    void forward(const Tensor& input, Tensor& output) {
        int out_h = (input.h + 2 * pad - ksize) / stride + 1;
        int out_w = (input.w + 2 * pad - ksize) / stride + 1;

        int col_rows = in_c * ksize * ksize;
        int col_cols = out_h * out_w;

        // Linux uyumlu hizalı bellek ayırma
        float* data_col = nullptr;
        if (posix_memalign((void**)&data_col, 32, col_rows * col_cols * sizeof(float)) != 0) {
            throw std::runtime_error("Layers: Bellek ayirma hatasi!");
        }

        // Ops.hpp içindeki fonksiyonları çağırıyoruz
        im2col(input.data, in_c, input.h, input.w, ksize, stride, pad, data_col);
        matmul_avx_omp(out_c, col_cols, col_rows, weights.data, data_col, output.data);

        
        free(data_col);
    }
};

// --- ReLU Katmanı ---
class ReLU {
public:
    void forward(Tensor& t) {
        #pragma omp parallel for
        for(int i = 0; i < t.c * t.h * t.w; ++i) {
            if(t.data[i] < 0) t.data[i] = 0;
        }
    }
};

#endif
