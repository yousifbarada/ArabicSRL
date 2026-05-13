import os
import random
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
from torch import nn

# =========================
# CONFIG
# =========================

DATASET_PATH = "data"

# =========================
# LABELS
# =========================

# labels_Dic = {
#     0: 'ا',
#     1: 'ب',
#     2: 'ت',
#     3: 'ث',
#     4: 'ج',
#     5: 'ح',
#     6: 'خ',
#     7: 'د',
#     8: 'ر',
#     9: 'ز',
#     10: 'س',
#     11: 'ش',
#     12: 'ص',
#     13: 'ض',
#     14: 'ط',
#     15: 'ظ',
#     16: 'ع',
#     17: 'غ',
#     18: 'ف',
#     19: 'ق',
#     20: 'ك',
#     21: 'ل',
#     22: 'م',
#     23: 'ن',
#     24: 'ه',
#     25: 'و',
#     26: 'ي',
#     27: 'ة'
# }

# =========================
# MODEL
# =========================

NUM_CLASSES = 28

model = nn.Sequential(
    nn.Linear(42, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, NUM_CLASSES)
)

model.load_state_dict(
    torch.load("model.pth", map_location=torch.device("cpu"))
)

model.eval()

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5,
    max_num_hands=1
)

# =========================
# LOOP OVER FOLDERS
# =========================

for folder_name in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        continue

    print(f"\n========== {folder_name} ==========")

    images = os.listdir(folder_path)

    if len(images) == 0:
        continue

    # choose 5 random images
    selected_images = random.sample(
        images,
        min(5, len(images))
    )

    for image_name in selected_images:

        image_path = os.path.join(
            folder_path,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:

            print(f"{image_name} -> No hand detected")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        x_ = []
        y_ = []

        for i in range(21):

            x_.append(hand_landmarks.landmark[i].x)
            y_.append(hand_landmarks.landmark[i].y)

        data_aux = []

        for i in range(21):

            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            # normalization
            data_aux.append(
                (x - min(x_)) / (max(x_) - min(x_) + 1e-6)
            )

            data_aux.append(
                (y - min(y_)) / (max(y_) - min(y_) + 1e-6)
            )

        data_aux = np.array(
            data_aux,
            dtype=np.float32
        ).reshape(1, -1)

        input_tensor = torch.tensor(
            data_aux,
            dtype=torch.float32
        )

        # =========================
        # Prediction
        # =========================

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence = torch.max(
                probabilities
            ).item()

            pred = torch.argmax(
                output,
                dim=1
            ).item()

        print(pred)

        print(
            f"{image_name} -> "
            f"Predicted: {pred} | "
            f"Confidence: {confidence:.2f}"
        )

        # =========================
        # SHOW IMAGE
        # =========================

        cv2.putText(
            img,
            f"{pred} ({confidence:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Prediction", img)

        key = cv2.waitKey(0)

        # press q to stop
        if key == ord('q'):
            exit()

cv2.destroyAllWindows()