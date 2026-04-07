import torch
import cv2
import torchvision
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

def build_model():
    model = torchvision.models.efficientnet_b0(weights=None) # не занружаем веса
    features = model.classifier[1].in_features # кол-во входов последнего слоя классификатора
    model.classifier[1] = nn.Linear(features, 1) # делаем выход последнего слоя = 1
    return model 

model = build_model()

model.load_state_dict(
    torch.load(Path(__file__).parent / "model.pth", map_location="cpu") # занружаем веса нашей модели 
)

model.eval() # режим pred  

transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

def predict(frame):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # transform + rgb 
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        predicted = model(tensor).squeeze() # forward + логит 
        prob = torch.sigmoid(predicted).item() # prob

    label = "person" if prob > 0.5 else "no person"
    return label, prob

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    label, prob = predict(frame) # pred 

    cv2.putText(frame, f"{label} {prob:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, # размер текста
                (147, 20, 255), # розовый
                2) # толщина линии

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()