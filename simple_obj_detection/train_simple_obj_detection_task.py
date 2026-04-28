import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["square", "circle", "triangle"]

class ShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform # аугментация 
        self.samples = [] # картинка + label

        for cls_name in classes:
            img_dir = root / cls_name / "images" # картинка 
            label_dir = root / cls_name / "labels" # метка 

            for img_path in sorted(img_dir.glob("*.png")):
                label_path = label_dir / (img_path.stem + ".txt")
                self.samples.append((img_path, label_path))

    def __len__(self): # кол-во картинок 
        return len(self.samples)

    def __getitem__(self, idx): # элемент по индексу 
        img_path, label_path = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB) # в RGB

        if self.transform:
            img = self.transform(Image.fromarray(img)) # если есть аугмент, то применяем
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # иначе нормализуем 0-1 и (HWC -> CHW)

        cls, x, y, w, h = map(float, label_path.read_text().split()) # класс x_center, y_center, ширина, высота
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32) # в тензор

        return img, int(cls), bbox # картинка, класс, ббокс


class SimpleDetector(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = nn.Sequential( # извлекаем признаки 
            nn.Conv2d(3, 32, 3, padding=1), # 3 канала, 32 фильтра 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(2), # к размеру 2на2
        )

        self.flatten = nn.Flatten() # тензор в вектор (256 * 2 * 2 = 1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.cls_head = nn.Linear(128, num_classes) # pred class (3 класса)

        self.bbox_head = nn.Sequential( # pred bbox 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # (x, y, w, h)
        )

    def forward(self, x):
        x = self.backbone(x) # (batch, 3, h, w)
        x = self.flatten(x) # (batch, 256, 2, 2)
        x = self.fc(x) # (batch, 128)

        cls = self.cls_head(x) # (batch, 3) - логиты для классов 
        bbox = torch.sigmoid(self.bbox_head(x)) # (batch, 4) - знач в диапазоне [0, 1]

        return cls, bbox

# пересечения ббоксов (метрика=пересечен/объединение)
def compute_iou(pred, target):
    
    # перевод в углы
    p_x1 = pred[:, 0] - pred[:, 2] / 2 # левая граница pred
    p_y1 = pred[:, 1] - pred[:, 3] / 2 # верхняя граница pred
    p_x2 = pred[:, 0] + pred[:, 2] / 2 # правая граница pred
    p_y2 = pred[:, 1] + pred[:, 3] / 2 # нижняя граница pred

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    # ширина пересечения * высота пересечения = S пересечения 
    inter = (
        (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) # ширина пересечения
        * (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0) # высота пересечения
    )

    union = ((p_x2 - p_x1) * (p_y2 - p_y1)) + ((t_x2 - t_x1) * (t_y2 - t_y1)) - inter # S объединения = S1 + S2 - пересечение
    return inter / (union + 1e-7) # IoU 

def giou_loss(pred, target):
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1) # левая граница 
    inter_y1 = torch.max(p_y1, t_y1) 
    inter_x2 = torch.min(p_x2, t_x2) # правая граница 
    inter_y2 = torch.min(p_y2, t_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0) # ширина 
    inter_h = (inter_y2 - inter_y1).clamp(min=0) # высота 
    inter = inter_w * inter_h # S пересечения

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0) # S pred
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0) # S targeta 

    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)

    # самый маленький прямоугольник содерж оба bbox
    c_x1 = torch.min(p_x1, t_x1) # самый левый край двух bbox
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2) # самый правый край двух bbox
    c_y2 = torch.max(p_y2, t_y2)
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    # GIoU = IoU - (area_c - union)/area_c
    # area_c - минимальный enclosing box
    giou = iou - (area_c - union) / (area_c + 1e-7) # штраф если bbox далеко

    return (1 - giou).mean()

def detection_loss(cls_pred, bbox_pred, cls_t, bbox_t, lambda_bbox=6.0):
    loss_cls = F.cross_entropy(cls_pred, cls_t) # лосс класификации 

    loss_xy = F.mse_loss(bbox_pred[:, :2], bbox_t[:, :2]) # лосс для центра
    loss_wh = F.mse_loss(bbox_pred[:, 2:], bbox_t[:, 2:]) # лосс для ширины/вфсоты

    loss_iou = giou_loss(bbox_pred, bbox_t) # лосс GIoU

    loss_bbox = loss_xy * 5 + loss_wh * 3 + loss_iou * 2 # важность частей bbox 

    return loss_cls + lambda_bbox * loss_bbox, loss_cls, loss_bbox

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

root = Path("shapes_dataset")
train_ds = ShapesDataset(root / "train", transform=transform)
val_ds = ShapesDataset(root / "val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = SimpleDetector(len(classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5) # уменьшает lr каждые 8 эпох

epochs = 20
save_path = root / "best.pt"
no_improve = 0
patience = 7 # останавливаем если 7 эпох без улучшений
history = defaultdict(list)
best_acc = 0.0

for epoch in range(1, epochs + 1):

    model.train() # режим обучения
    train_loss = train_cls = train_box = 0.0

    for images, cls_t, bbox_t in train_loader:
        images, cls_t, bbox_t = images.to(device), cls_t.to(device), bbox_t.to(device)

        optimizer.zero_grad() # обнуляем градиенты
        cls_pred, bbox_pred = model(images) # forward pass 

        loss, lc, lb = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)
        loss.backward() # backward pass (считаем градиенты)
        optimizer.step() # update weight

        train_loss += loss.item()
        train_cls += lc.item()
        train_box += lb.item()

    n = len(train_loader)
    history["train_loss"].append(train_loss / n)

    model.eval() # режим оценки
    val_loss = correct = total = iou_sum = 0.0

    with torch.no_grad(): # откл градиенты
        for images, cls_t, bbox_t in val_loader:
            images, cls_t, bbox_t = images.to(device), cls_t.to(device), bbox_t.to(device)

            cls_pred, bbox_pred = model(images)

            loss, _, _ = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)
            val_loss += loss.item()

            # точность классификации
            correct += (cls_pred.argmax(1) == cls_t).sum().item()
            total += cls_t.size(0)

            # точность bbox 
            iou_sum += compute_iou(bbox_pred, bbox_t).mean().item()

    acc = correct / total
    mean_iou = iou_sum / len(val_loader)

    history["val_loss"].append(val_loss / len(val_loader))
    history["val_acc"].append(acc)
    history["val_iou"].append(mean_iou)

    scheduler.step() # update lr

    print(f"Epoch {epoch:2d}, loss={history['train_loss'][-1]:.4f}, acc={acc:.3f}, IoU={mean_iou:.3f}")

    # save bets model 
    if acc > best_acc:
        best_acc = acc
        no_improve = 0
        torch.save(model.state_dict(), save_path)
    else:
        no_improve += 1

    # early stoppping 
    if no_improve >= patience:
        break

print("best acc: ", best_acc)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("loss")
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")

plt.subplot(1, 3, 2)
plt.title("acc")
plt.plot(history["val_acc"], label="acc")

plt.subplot(1, 3, 3)
plt.title("IoU")
plt.plot(history["val_iou"], label="IoU")

plt.legend()
plt.tight_layout()
plt.show()

def show_predictions(loader, model, n=8):
    model.eval()
    images, cls_t, bbox_t = next(iter(loader)) # один батч
    images = images.to(device)

    with torch.no_grad():
        cls_pred, bbox_pred = model(images)

    preds = cls_pred.argmax(1).cpu() # индексы предск классов

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        # в HWC
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        H, W = img.shape[:2]

        # истинный bbox 
        cx, cy, bw, bh = bbox_t[i].numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
        )
        # pred bbox 
        cx, cy, bw, bh = bbox_pred[i].cpu().numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
        )

        gt_name = classes[cls_t[i]]
        pr_name = classes[preds[i]]
        color = "green" if preds[i] == cls_t[i] else "red"
        ax.set_title(f"Real:{gt_name}  Predicted:{pr_name}", color=color, fontsize=9)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# на валидационной выборке 
show_predictions(val_loader, model)