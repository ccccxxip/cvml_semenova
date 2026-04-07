import torch
import cv2
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import time 
from collections import deque
import torch.nn as nn
from pathlib import Path

save_path = Path(__file__).parent

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 # предобученные веса
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters(): # берем только последний слой - классификатор
        param.requires_grad = False
    
    features = model.classifier[1].in_features # размер входа последнего слоя классификтора 
    model.classifier[1] = nn.Linear(features, 1) # делаем один выход (бинарная классификация)
    return model 

model = build_model()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, 
           model.parameters()), # обучаем не замороженные слои
    lr = 0.0001
)

# аугментация 
transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

# обучение
def train(buffer):
    if len(buffer) < 10:
        return None # если данных мало - не обучаем 
    model.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = model(images).squeeze() # forward проход
    loss = criterion(predictions, labels)
    loss.backward() # градиенты
    optimizer.step() # обновление весов 
    return loss.item()


def predict(frame):
    model.eval() # режим инференса
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # подготовка изображения
    tensor = tensor.unsqueeze(0) 
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item() # вероятность
    label = "person" if prob > 0.5 else "no person" 
    return label, prob

# для размеченных данных
class Buffer():
    def __init__(self, maxsize = 16): # очередь фиксированного размера 
        self.frames = deque(maxlen = maxsize)
        self.labels = deque(maxlen = maxsize)

    def append(self, tensor, label): # добавление примера 
        self.frames.append(tensor)
        self.labels.append(label)
    
    def __len__(self): 
        return len(self.frames)
    
    def get_batch(self): # батч
        images = torch.stack(list(self.frames)) # объединяем изображения в тензор
        labels = torch.tensor(list(self.labels), dtype = torch.float32) # тензор меток 
        return images, labels

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

buffer = Buffer() # изображения, метки
count_labeled = 0 # кол-во размеченных изображений

while True:
    _, frame = cap.read() # читаем кадр 
    cv2.imshow("Camera", frame) # показываем 

    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # для модели в RGB

    if key == ord("q"):
        break

    elif key == ord("1"): # person
        tensor = transform(image)
        buffer.append(tensor, 1.0) # добавляем картинку, label
        count_labeled += 1

    elif key == ord("2"): # no person
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1

    elif key == ord("p"): # predict
        t = time.perf_counter()
        label, confidence = predict(frame) 
        print(f"elapsed time {time.perf_counter() - t}")
        print(label, confidence)

    elif key == ord("s"): # save model
        torch.save(model.state_dict(), save_path / "model.pth") 

    if count_labeled >= buffer.frames.maxlen: #  если достаточно разметили
        loss = train(buffer)
        if loss:
            print(f"loss = {loss}")
        count_labeled = 0