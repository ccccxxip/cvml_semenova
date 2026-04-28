from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

classes = {0: "cube", 1: "neither", 2: "sphere"}

root = Path("yolo/spheres_and_cubes_new")

config = {
    "path": str(root.absolute()),
    "train": str((root / "images/train").absolute()),
    "val": str((root / "images/val").absolute()),
    "nc": len(classes), # кол-во классов 
    "names": list(classes.values()) # имена классов 
}

with open(root / "dataset.yaml", "w") as f:
    yaml.dump(config, f, allow_unicode=True)

size = "n" # быстрая 
model = YOLO(f"yolo26{size}.pt")

result = model.train(
    data=str(root / "dataset.yaml"),

    imgsz=640, # размер изображения 
    batch=16, # кол-во в батче 
    workers=4, # кол-во потоков

    epochs=10,
    patience=5, # early stoppping 

    optimizer="AdamW",
    lr0=0.01,
    warmup_epochs=3, # первые 3 эпохи - плавный старт 
    cos_lr=True, # lr уменьшается по косинусу 

    dropout=0.2,

    # угментация

    # изменение цвета 
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # отражения 
    flipud=0.0, # по вертикали (0)
    fliplr=0.5, # по горизонтали (50)

    mosaic=1.0, # склеивает 4 картинки 
    degrees=5.0, # поворот +- 5 градусов 
    scale=0.5, # машстаб 
    translate=0.1, # сдвиг изображения 

    conf=0.001,
    iou=0.7, # threshold bbox 

    project="figures",
    name="yolo",

    save=True, # save weights 
    save_period=5, # save weights кадлые 5 эпох

    device=0 if torch.cuda.is_available() else "cpu",

    verbose=True,
    plots=True,
    val=True,

    close_mosaic=8, # выкл мозайки после 8 эпох 
    amp=True # ускоряет обучение 
)

print(result.save_dir)