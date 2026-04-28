import cv2 
import time 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

# считаем угол abc (локоть)
def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0]) # локоть -> кисть
    ab = np.atan2(a[1] - b[1], a[0] - b[0]) # локоть -> плечо
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle # нормализация, чтобы угол был всегла полож
    return 360 - angle if angle > 180 else angle # от 0 до 180 


def detect_pushups(annotated, keypoints, counter, stage, last_rep_time):

    left_shoulder = keypoints[5] # плечо
    left_elbow = keypoints[7] # локоть 
    left_wrist = keypoints[9] # кисть 

    if left_shoulder[0] > 0 and left_elbow[0] > 0 and left_wrist[0] > 0: # если коорд > 0, то они сущ

        angle = get_angle(left_shoulder, left_elbow, left_wrist) # расчет угла 

        if angle < 90: # рука согнута 
            stage = "down"

        if angle > 160 and stage == "down": # рука прямая 
            if time.time() - last_rep_time > 0.5: # защита от двойного засчитывания (между повторениями мин 0.5 сек)
                counter += 1 # засчитываем одно отжимание 
                last_rep_time = time.time()
            stage = "up" # обновляем состояние 

        # вывод угла на экран 
        cv2.putText(annotated, 
                    f"angle {angle:.1f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2) 

        return True, counter, stage, last_rep_time # тру - человек найден 

    return None, counter, stage, last_rep_time # не найден


model = YOLO("yolo26n-pose.pt")

camera = cv2.VideoCapture(0)

counter = 0 # счетчик 
stage = None # up/down 
last_rep_time = 0 # чтобы двойного засчитывания не было 

last_seen_time = time.time()
reset_timeout = 3 # если 3 сек нет в кадре - обнуляем счетчик 


while camera.isOpened():
    ret, frame = camera.read()
    cv2.imshow("camera", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model(frame)
    print(f"fps {1 / (time.perf_counter() - t):.1f}")

    # сброс 
    if time.time() - last_seen_time > reset_timeout: # если прошло больше 3 сек после последн обнаружения человека - нет человека 
        counter = 0
        stage = None

    if not results: # если ничего не обнаружили - пропускаем 
        cv2.imshow("pose", frame)
        continue

    result = results[0] # первый результат 
    keypoints = result.keypoints.xy.tolist() # коорд всех точек 

    if not keypoints: # если точек нет
        cv2.imshow("pose", frame)
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], 
                   result.orig_shape, 5, True)
    annotated = annotator.result()

    # считаем отжимания 
    detected, counter, stage, last_rep_time = detect_pushups(
        annotated, keypoints[0], counter, stage, last_rep_time
    )

    # если человек найжен 
    if detected:
        last_seen_time = time.time()

    # вывод счетчика на экран 
    cv2.putText(annotated, 
                f"pushups: {counter}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 255), 3)

    cv2.imshow("pose", annotated)


camera.release()
cv2.destroyAllWindows()