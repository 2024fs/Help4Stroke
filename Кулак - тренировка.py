from cv2 import cv2
import mediapipe as mp
import numpy as np
import random
import time

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

#Создание детектора
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
counter = 0
prev_fist = False
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)

    # Перевести в RGB
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Распознавание
    results = handsDetector.process(flippedRGB)

    # Отрисовка поля для вывода
    cv2.rectangle(flippedRGB, (0, 0), (100, 80), (0, 150, 0), -1)

    # Отрисовка распознанного
    if results.multi_hand_landmarks is not None:
         (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
         ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
         if 2 * r / ws > 1.3:
             prev_fist = False
         else:
             if not prev_fist:
                 counter += 1
                 prev_fist = True

    # Отображение результата
    cv2.putText(flippedRGB, str(counter), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

    #Индикатор завершения упражнения
    niceimages = ['1.jpg', '2.png', '3.jpg']
    if counter == 20:
        img = cv2.imread(random.choice(niceimages))
        cv2.imshow('Image', img)
        time.sleep(3)
        break

    # Перевести в BGR и отобразить
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Кулак", res_image)

handsDetector.close()
