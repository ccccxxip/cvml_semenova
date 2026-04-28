import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# аугментация - обучения
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224), # случайное вырезание куска изображения
    transforms.RandomHorizontalFlip(), # отражение
    transforms.RandomRotation(15), # += 15 градусов
    transforms.ColorJitter(0.3, 0.3, 0.2), # изменение яркости
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# аугментация - валидации 
val_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = ImageFolder(
    "spheres_and_cubes_new/images/train",
    transform=train_transform
)
val_ds = ImageFolder(
    "spheres_and_cubes_new/images/val",
    transform=val_transform
)
print("classes:", train_ds.classes) # выводим классы 

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4) # батчи по 16 картинок
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4) 

# проверка трех моделей (b0, b1, b2)
def build_model(version="b0"):
    if version == "b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif version == "b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif version == "b2":
        weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b2(weights=weights)

    for p in model.features.parameters():
        p.requires_grad = False # обучаем только классификатор 

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 3) # меняем выход на 3 класса 
    )

    return model

def run(model, loader, criterion, optimizer=None):
    if optimizer is not None: # если есть optimizer - обучаем 
        model.train()
        train_mode = True
    else:
        model.eval() # иначе - инференс 
        train_mode = False

    total_loss, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            logits = model(images) # forward pass 
            loss = criterion(logits, labels) # ошибка 

            if train_mode: # backward pass 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1) # класс с максимальной вероятностью 
            correct += (preds == labels).sum().item() # правильные ответы 
            total += images.size(0)

    return total_loss / total, correct / total

def train_model(version):
    print(f"\nmodel:  {version.upper()}")

    model = build_model(version)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters())
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0.0

    for epoch in range(10):  # одинаковое число эпох!
        train_loss, train_acc = run(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run(model, val_loader, criterion)

        scheduler.step()

        print(f"{version}, epoch {epoch}, acc={val_acc:.4f}")

        if val_acc > best_acc: 
            best_acc = val_acc
            torch.save(model.state_dict(), f"{version}.pth") # сохр лучшую модель

    return model

def confusion_matrix(model, loader):
    model.eval()
    num_classes = 3
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(1)

            for t, p in zip(labels, preds):
                matrix[t][p] += 1

    return matrix

def plot_cm(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    model_b0 = train_model("b0")
    model_b1 = train_model("b1")
    model_b2 = train_model("b2")

    cm_b0 = confusion_matrix(model_b0, val_loader)
    cm_b1 = confusion_matrix(model_b1, val_loader)
    cm_b2 = confusion_matrix(model_b2, val_loader)

    plot_cm(cm_b0, "EfficientNet B0")
    plot_cm(cm_b1, "EfficientNet B1")
    plot_cm(cm_b2, "EfficientNet B2")