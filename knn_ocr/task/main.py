import cv2
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.measure import label, regionprops

BASE_DIR = Path("knn_ocr/task")
TRAIN_DIR = BASE_DIR / "train"

symbols = [] # список символов
X_train = [] # пизнаки каждого изображения
y_train = [] # классы соотв символам 

# обучение
cls_index = -1 # уникальный номер класса 
for cls in sorted(TRAIN_DIR.iterdir()): # отсорт список папок внутри трейна
    cls_index += 1 # присваиваем класс 
    symbols.append(cls.name[-1]) # берем только последний символ названия папки 

    for img_path in sorted(cls.glob("*.png")):
        img = imread(img_path) # читаем изображение в массив np

        # извлечение признаков
        if img.ndim == 3: # цветное
            gray = np.mean(img, axis=2).astype(np.uint8) # делаем серым
        else:
            gray = img.astype(np.uint8)  

        bin_img = gray > 0 # бинаризация (если не 0(не черный), тогда возвр тру)

        regions = regionprops(label(bin_img)) # каждому объекту присваиваем уникальный номер
                                                #  вычисляем св-ва каждого объекта 

        # r.extent - S объекта к S bbox
        filtered = [r for r in regions if r.extent <= 0.81] # исключаем слишком квадратные области (шум)
        if not filtered:
            feat = np.zeros(5, dtype=np.float32)
        else:
            r = filtered[0]
            feat = [
                r.eccentricity, # вытянутость объекта (0 = круг, 1 = линия)
                r.solidity, # плотность объекта
                r.extent, # S объекта к S bbox
                r.perimeter / r.area if r.area != 0 else 0, # плотность контура
                r.area_convex / r.area if r.area != 0 else 0 # насколько объект близок к выпуклому
            ]
            feat = np.array(feat, dtype=np.float32) # вектор признаков длиной

        X_train.append(feat) # массив признаков 
        y_train.append(cls_index) # номер класса

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1,1)

# создаем и обучаем KNN
knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train) # каждая строка массива - один объект

# распознаем тестовые изображения
for i in range(7):
    img = imread(BASE_DIR / f"{i}.png")
    gray = np.mean(img, axis=2).astype(np.uint8) # преобразование серого
    bin_img = gray > 0
    regions = regionprops(label(bin_img.T)) # транспонируем изображение, чтобы буквы шли по строкам
    
    test_features = []
    for r in regions:
        if r.extent < 0.7:
            # извлечение признаков
            feat = [
                r.eccentricity,
                r.solidity,
                r.extent,
                r.perimeter / r.area if r.area != 0 else 0,
                r.area_convex / r.area if r.area != 0 else 0
            ]
            test_features.append(np.array(feat, dtype=np.float32))

    if test_features:
        test_features = np.array(test_features, dtype=np.float32).reshape(-1,5) # в массив 
        ret, result, neighbours, dist = knn.findNearest(test_features, k=3)
        for r in result:
            print(symbols[int(r.item())], end="") # обратно номер в букву 
    print("")