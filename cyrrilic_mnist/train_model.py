import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt

save_path = Path(__file__).parent

class LeNet5(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # вход 1 канал (альфа), выход 6 каналов, ядро 5на5
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = self.tanh(self.conv3(x))
        x = x.view(x.size(0), -1) # распрямляем перед fc слоями
        x = self.tanh(self.fc1(x))
        x = self.fc2(x) # выход вероятности для каждой буквы
        return x

class CyrillicMNIST(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = [] # путь к изображению, label
        self.classes = sorted(os.listdir(root)) # отсорт папки букв
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)} # словарь буква:число
        for c in self.classes:
            c_path = os.path.join(root, c)
            for img_name in os.listdir(c_path):
                self.samples.append((os.path.join(c_path, img_name), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples) # общ кол-во примеров

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGBA") # помним про альфа канал 
        alpha = img.getchannel('A') # берем только альфа канал, тк буква там
        if self.transform:
            alpha = self.transform(alpha)
        return alpha, label # возв тензор, метка

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10), # +- 10 градуслв
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

if __name__ == "__main__":

    dataset = CyrillicMNIST("Cyrillic", transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LeNet5()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_list, acc_list = [], [] # loss, acc для построения графиков

    for epoch in range(9):
        model.train() # обучение 
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data) # forward 
            loss = criterion(output, target)
            loss.backward() # backward (градиентыы считаем)
            optimizer.step() # обновление весов

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                pred = output.argmax(dim=1) # номер класса с макс вероятностью
                correct += (pred == target).sum().item()
        acc = 100.0 * correct / len(dataset)
        print(f"epoch {epoch+1}, acc={acc:.2f}") # acc текущ эпохи
        loss_list.append(loss.item())
        acc_list.append(acc)

    torch.save(model.state_dict(), save_path / "model.pth") # сохр веса

    # строим график
    plt.plot(loss_list, label="Loss")
    plt.plot(acc_list, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(save_path / "train.png")