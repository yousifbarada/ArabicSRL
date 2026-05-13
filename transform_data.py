import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
data_dir = './data'
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.2,max_num_hands=1)
data = []
labels = []

for dir_ in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(data_dir, dir_, file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            # collect coordinates first
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # normalize + flatten
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        data.append(data_aux)
        labels.append(dir_)
f= open('data.pkl','wb')
pickle.dump((data,labels),f)
f.close()        