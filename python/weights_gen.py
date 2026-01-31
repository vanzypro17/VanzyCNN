import numpy as np

# Conv1 agirliklari: 16 filtre, 3 kanal, 3x3 boyut
conv1 = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.01
conv1.tofile("conv1_weights.bin")

# Dense agirliklari: 10 sinif, 16384 (16*32*32) giris
dense = np.random.randn(10, 16384).astype(np.float32) * 0.01
dense.tofile("dense_weights.bin")

print("Test agirliklari basariyla olusturuldu!")