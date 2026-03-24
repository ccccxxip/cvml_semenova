import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
import os
from train_model import LeNet5 

model_path = Path(__file__).parent / "model.pth"
if not model_path.exists():
    raise RuntimeError("not trained :( ") 

model = LeNet5()
model.load_state_dict(torch.load(model_path, map_location="cpu")) 
model.eval()

classes = sorted(os.listdir("Cyrillic"))  # список букв в порядке индексов

transform = transforms.Compose([
    transforms.ToPILImage(),  # делаем в пил
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

canvas = np.zeros((256, 256), dtype="uint8")
cv2.namedWindow("Canvas", cv2.WINDOW_GUI_NORMAL)
position = []  # тек поз мыши
draw = False

def on_mouse(event, x, y, flags, param):
    global draw, position
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
    elif event == cv2.EVENT_MOUSEMOVE and draw:
        position = [y, x]  # обновляем коорд

cv2.setMouseCallback("Canvas", on_mouse)

while True:
    if position:
        cv2.circle(canvas, (position[1], position[0]), 5, 255, -1) # кисть (кружок)

    key = cv2.waitKey(10) & 0xFF

    match key:
        case 27:
            break
        case 99:
            canvas[:] = 0
            position = []
        case 112:
            with torch.no_grad():
                tensor = transform(canvas)
                batch = tensor.unsqueeze(0) # для батча
                output = model(batch)
                prediction = output.argmax(dim=1).item()
                probability = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                print(f"pred: {classes[prediction]}")

    cv2.imshow("Canvas", canvas)

cv2.destroyAllWindows()