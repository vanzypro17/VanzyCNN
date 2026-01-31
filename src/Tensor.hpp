#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>
#include <immintrin.h>
#include <cstdlib> // posix_memalign ve free için gerekli

class Tensor {
public:
    int c, h, w; // Kanallar, Yükseklik, Genişlik
    float* data;

    Tensor(int channels, int height, int width) : c(channels), h(height), w(width) {
        // Bellek miktarını hesapla
        size_t total_size = static_cast<size_t>(c) * h * w * sizeof(float);

        // 32-byte hizalı bellek ayırıyoruz (AVX2 komutları için şart)
        #ifdef _WIN32
            data = (float*)_aligned_malloc(total_size, 32);
            if (data == nullptr) {
                std::cerr << "Hata: Windows uzerinde bellek ayrilamadi!" << std::endl;
                exit(1);
            }
        #else
            // Linux uyarısını (warn_unused_result) bu kontrolle çözüyoruz
            int status = posix_memalign((void**)&data, 32, total_size);
            if (status != 0) {
                std::cerr << "Kritik Hata: Linux uzerinde hizali bellek ayrilamadi! Kod: " << status << std::endl;
                exit(1);
            }
        #endif

        // İlk değerleri sıfırla (Temiz bir başlangıç için)
        for(int i = 0; i < c * h * w; ++i) data[i] = 0.0f;
    }

    // Destructor: Bellek sızıntısını (memory leak) önler
    ~Tensor() {
        if (data != nullptr) {
            #ifdef _WIN32
                _aligned_free(data);
            #else
                free(data);
            #endif
        }
    }

    // Veriye (Kanal, Yükseklik, Genişlik) formatında erişim
    // Formül: ci * (H * W) + hi * W + wi
    inline float& operator()(int ci, int hi, int wi) {
        return data[ci * (h * w) + hi * w + wi];
    }

    // Sadece okuma amaçlı (const) erişim
    inline const float& operator()(int ci, int hi, int wi) const {
        return data[ci * (h * w) + hi * w + wi];
    }
};

#endif