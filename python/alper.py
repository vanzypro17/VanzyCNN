import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 1. Senini C++ mimarinin aynısını PyTorch'ta kuruyoruz
class VanzyCNN(nn.Module):
    def __init__(self):
        super(VanzyCNN, self).__init__()
        # Conv1: 3 giriş, 16 çıkış, 3x3 kernel, padding 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Dense: 16 * 32 * 32 (pooling sonrası) -> 10 sınıf
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc(x)
        return x

# 2. Veri Setini Hazırla (Resimleri 64x64'e boyutlandırıyoruz)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# 3. Hızlıca Eğit (Sadece 1-2 epoch, "akıllanması" için yeterli)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = VanzyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

print("Eğitim başlıyor... Bu yaklaşık 1-2 dakika sürecek.")
for epoch in range(50): # Hız için sadece 1 tur
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 99: break 

print("Eğitim bitti! Ağırlıklar senin motorun için dışa aktarılıyor...")

# 4. Ağırlıkları BINARY olarak kaydet
# Conv1 Weights: [Out, In, K, K] -> [16, 3, 3, 3]
conv_w = net.conv1.weight.detach().cpu().numpy().astype(np.float32)
conv_w.tofile("conv1_weights.bin")

# Dense Weights: [Out, In] -> [10, 16384]
fc_w = net.fc.weight.detach().cpu().numpy().astype(np.float32)
fc_w.tofile("dense_weights.bin")

print("Dosyalar hazır: conv1_weights.bin ve dense_weights.bin")