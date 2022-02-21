import cv2
import mediapipe as mp
import numpy as np
import random
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # Первая точка
    b = np.array(b)  # Вторая точка
    c = np.array(c)  # Третья точка

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

side = input()
cap = cv2.VideoCapture(0)

# Переменные счёта шага и этапа
counter = 0
stage = None

## Настройка экземпляра mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Перевести изображение в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Обнаружение
        results = pose.process(image)

        # Перевести изображение в BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Извлечение опорных точек
        try:
            landmarks = results.pose_landmarks.landmark

            # Определить координаты
            if side == 'лево':
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            if side == 'право':
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Подсчёт угла
            angle = calculate_angle(shoulder, elbow, wrist)

            # Процесс подсчёта шага и этапа
            if angle > 160:
                stage = "x"
            if angle < 30 and stage == 'x':
                stage = "v"
                counter += 1

        except:
            pass

        # Отрисовка поля для вывода
        cv2.rectangle(image, (0, 0), (225, 100), (0, 150, 0), -1)

        # Вывод шага (номер упражнения)
        cv2.putText(image, 'HOMEP', (15, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Вывод этапа (вверх/вниз)
        cv2.putText(image, 'BBEPX', (100, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (110, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Индикатор завершения упражнения
        niceimages = ['1.jpg', '2.png', '3.jpg']
        if counter == 30:
            img = cv2.imread(random.choice(niceimages))
            cv2.imshow('Image', img)
            time.sleep(3)
            break

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
