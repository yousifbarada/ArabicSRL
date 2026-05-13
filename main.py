import pickle

import cv2
import mediapipe as mp
import numpy as np

model = pickle.load(open('./model.pkl', 'rb'))['model']
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands (static_image_mode=False, min_detection_confidence=0.2, max_num_hands=1)
labels_Dic={0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح',}
while True:
    # handel space (camre see 2 sec wihtout any hand to handle the space between 2 sentences)

    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    word = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            data_aux = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data_aux = np.array(data_aux).reshape(1, -1)
            pred = int(model.predict(data_aux)[0])
            print('Predicted class: {}'.format(labels_Dic[pred]))
            # word += labels_Dic[str(pred)]   
            # print('Predicted class: {}'.format(labels_Dic[str(pred)]))
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            #                           mp_drawing_styles.get_default_hand_landmarks_style(),
            #                           mp_drawing_styles.get_default_hand_connections_style())
            # cv2.putText(frame, 'Predicted class: {}'.format(labels_Dic[str(pred)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break