#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib> 
#include <cstdio>
#include <fstream> 

// stb_image entegrasyonu
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Tensor.hpp"
#include "Ops.hpp"
#include "Layers.hpp"

// --- YARDIMCI FONKSİYONLAR ---

// Binary ağırlık dosyalarını yükleyen fonksiyon
bool load_weights(const std::string& filename, float* target_ptr, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[HATA] Agirlik dosyasi bulunamadi: " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(target_ptr), num_elements * sizeof(float));
    if (!file) {
        std::cerr << "[HATA] Dosya okunurken boyut uyumsuzlugu: " << filename << std::endl;
        return false;
    }
    file.close();
    return true;
}

// Görüntüyü yükleyen, HWC'den CHW'ye çeviren ve Normalize eden fonksiyon
Tensor load_image_to_tensor(const char* filename, int target_c, int target_h, int target_w) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(filename, &width, &height, &channels, 3);

    if (img_data == nullptr) {
        std::cerr << "[HATA] " << filename << " yuklenemedi! Dosya yolunu kontrol edin." << std::endl;
        exit(1);
    }

    Tensor t(target_c, target_h, target_w);
    for (int c = 0; c < target_c; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                int src_h = h * height / target_h;
                int src_w = w * width / target_w;
                float pixel_val = static_cast<float>(img_data[(src_h * width + src_w) * 3 + c]);
                t(c, h, w) = pixel_val / 255.0f; 
            }
        }
    }
    stbi_image_free(img_data);
    return t;
}

// Softmax: Çıkış değerlerini olasılığa (%0-100) çevirir
void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) input[i] /= sum;
}

// --- ANA PROGRAM ---

int main() {
    std::cout << "--- VanzyCNN IHA Nesne Tanimlama Motoru (Final Release) ---" << std::endl;

    // 1. GERÇEK GÖRÜNTÜ YÜKLEME
    Tensor input_img = load_image_to_tensor("test_iha.jpg", 3, 64, 64);
    std::cout << "[INFO] Resim yuklendi: 64x64x3 (CHW & Normalized)" << std::endl;

    // 2. KATMANLARIN TANIMLANMASI
    Conv2D conv1(3, 16, 3, 1, 1); 
    ReLU relu1;
    MaxPool pool1(2, 2); 

    Tensor feat_map(16, 64, 64);      
    Tensor pooled_map(16, 32, 32);    

    // 3. AGIRLIKLARI YUKLEME
    std::cout << "[INFO] Egitilmis agirliklar yukleniyor..." << std::endl;
    if (!load_weights("conv1_weights.bin", conv1.weights.data, 16 * 3 * 3 * 3)) return 1;

    // 4. SINIFLANDIRICI BELLEK AYIRMA (HATA KONTROLLU)
    int flatten_size = 16 * 32 * 32; 
    int num_classes = 10;           
    float* dense_weights = nullptr;
    float* results = nullptr;

    // posix_memalign donus degerlerini kontrol ederek uyarilari (warnings) engelliyoruz
    if (posix_memalign((void**)&dense_weights, 32, num_classes * flatten_size * sizeof(float)) != 0) {
        std::cerr << "[HATA] Dense agirliklari icin bellek ayrilamadi!" << std::endl;
        return 1;
    }
    if (posix_memalign((void**)&results, 32, num_classes * sizeof(float)) != 0) {
        free(dense_weights);
        std::cerr << "[HATA] Sonuclar icin bellek ayrilamadi!" << std::endl;
        return 1;
    }

    // Ağırlıkları dosyadan yükle (Eger dosya yoksa rastgele kalsin yerine hata verip cikiyoruz)
    if (!load_weights("dense_weights.bin", dense_weights, num_classes * flatten_size)) {
        free(dense_weights);
        free(results);
        return 1;
    }

    // 5. İLERİ BESLEME (INFERENCE)
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "[1/4] Conv2D..." << std::endl;
    conv1.forward(input_img, feat_map);

    std::cout << "[2/4] ReLU..." << std::endl;
    relu1.forward(feat_map);

    std::cout << "[3/4] MaxPool..." << std::endl;
    pool1.forward(feat_map, pooled_map);

    std::cout << "[4/4] Dense (131 GFLOPS Matmul)..." << std::endl;
    matmul_avx_omp(num_classes, 1, flatten_size, dense_weights, pooled_map.data, results);

    softmax(results, num_classes);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 6. SONUÇLAR
    std::cout << "\n--- ANALIZ TAMAMLANDI ---" << std::endl;
    std::cout << "Islem Suresi: " << diff.count() * 1000 << " ms" << std::endl;
    
    const char* labels[] = {
    "Ucak",      // 0
    "Otomobil",  // 1
    "Kus",       // 2
    "Kedi",      // 3
    "Geyik",     // 4
    "Kopek",     // 5
    "Kurbaga",   // 6
    "At",        // 7
    "Gemi",      // 8
    "Kamyon"     // 9
    };
    std::cout << "\nTahmin Olasiliklari:" << std::endl;
    for(int i = 0; i < 10; i++) {
        printf(" - %-12s: %.2f%%\n", labels[i], results[i] * 100.0f);
    }

    free(dense_weights);
    free(results);

    return 0;
}