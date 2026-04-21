from ultralytics import YOLO
import cv2
import time

classes = {0: "cube", 1: "neither", 2: "sphere"}

model = YOLO("/Users/alenasemenova/semenova_cv/runs/detect/figures/yolo-5/weights/best.pt") # занрузка весов обученной модели 

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    ret, frame = cap.read() # кадр, тек изображение 
    if not ret:
        break

    t = time.perf_counter()

    result = model.predict(
        source=frame, # кадр с камеры 
        conf=0.25, # мин уверенность 
        iou=0.7,
        imgsz=640, # размер 
        verbose=False
    )[0]

    boxes = result.boxes.xyxy.cpu().numpy() # коорд рамок 
    cls = result.boxes.cls.cpu().numpy() # классы 
    confs = result.boxes.conf.cpu().numpy() # уверенность (0-1)

    for box, c, conf in zip(boxes, cls, confs):
        x1, y1, x2, y2 = map(int, box) # все в int 

        label = classes[int(c)] # число - текст (куб или сфера)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # прямоугольник зеленый толщина 2

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10), # текст немного выше рамки 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    fps = 1 / (time.perf_counter() - t) # кол-во кадров в сек 
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()