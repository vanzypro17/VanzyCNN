#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Tensor.hpp"
#include <fstream>
#include <stdexcept>

Tensor load_image_to_tensor(const char* filename, int target_c, int target_h, int target_w) {
    int width, height, channels;
    // Resmi yükle (unsigned char* formatında 0-255 arası değerler)
    unsigned char* img_data = stbi_load(filename, &width, &height, &channels, target_c);

    if (img_data == nullptr) {
        throw std::runtime_error("Resim yuklenemedi! Dosya yolunu kontrol et.");
    }

    Tensor t(target_c, target_h, target_w);

    // HWC (0,1,2, 0,1,2...) formatından CHW (000..., 111..., 222...) formatına dönüşüm
    // Aynı zamanda 0-255 arası pikselleri 0.0f - 1.0f arasına normalize ediyoruz
    for (int c = 0; c < target_c; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                // Eğer resim boyutu hedef boyuttan farklıysa basit bir ölçekleme mantığı:
                int src_h = h * height / target_h;
                int src_w = w * width / target_w;
                
                float pixel_val = static_cast<float>(img_data[(src_h * width + src_w) * target_c + c]);
                float normalized_val = pixel_val / 255.0f; 
                t(c, h, w) = (normalized_val - 0.5f) / 0.5f;
            }
        }
    }

    stbi_image_free(img_data);
    return t;
}

bool load_weights(const std::string& filename, float* target_ptr, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Hata: Agirlik dosyasi bulunamadi: " << filename << std::endl;
        return false;
    }

    // Dosyayi tek seferde bellege oku
    file.read(reinterpret_cast<char*>(target_ptr), num_elements * sizeof(float));
    
    if (!file) {
        std::cerr << "Hata: Dosya okunurken beklenmedik son (Boyut uyumsuzlugu?)" << std::endl;
        return false;
    }

    file.close();
    return true;
}