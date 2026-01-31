#ifndef OPS_HPP
#define OPS_HPP

#include <immintrin.h>
#include <omp.h>
#include <iostream>

// --- 1. im2col: Resmi Matrise Dönüştürme ---
// Bu fonksiyon görüntüyü parçalara ayırıp matris çarpımına hazır hale getirir.
inline void im2col(const float* data_im, int channels, int height, int width,
                   int ksize, int stride, int pad, float* data_col) {
    
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    #pragma omp parallel for
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / (ksize * ksize);

        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_index = (c * out_h + h) * out_w + w;

                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0f; // Padding
                }
            }
        }
    }
}

// --- 2. matmul_avx_omp
inline void matmul_avx_omp(int M, int N, int K, const float* A, const float* B, float* C) {
    
    #pragma omp parallel for
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
            for (int j = 0; j < N; j += 8) {
                if (j + 8 <= N) {
                    __m256 b_vec = _mm256_load_ps(&B[k * N + j]);
                    __m256 c_vec = _mm256_load_ps(&C[i * N + j]);
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_store_ps(&C[i * N + j], c_vec);
                } else {
                    for (int jj = j; jj < N; ++jj) {
                        C[i * N + jj] += A[i * K + k] * B[k * N + jj];
                    }
                }
            }
        }
    }
}

#endif