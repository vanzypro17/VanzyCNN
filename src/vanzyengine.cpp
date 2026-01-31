#include <iostream>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cmath>

#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc_float(size) (float*)_aligned_malloc((size) * sizeof(float), 32)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    float* aligned_alloc_float(size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 32, size * sizeof(float)) != 0) return nullptr;
        return (float*)ptr;
    }
    #define aligned_free(ptr) free(ptr)
#endif

// AVX2 + OpenMP Matmul
void matmul_avx_omp(int M, int N, int K, const float* A, const float* B, float* C) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
            for (int j = 0; j < N; j += 8) {
                __m256 b_vec = _mm256_load_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_load_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_store_ps(&C[i * N + j], c_vec);
            }
        }
    }
}

