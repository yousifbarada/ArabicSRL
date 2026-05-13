import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
import time
import pickle
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# =========================
# Model
# =========================

model = nn.Sequential(
    nn.Linear(42, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 28)
)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =========================
# Camera & MediaPipe
# =========================

cap = cv2.VideoCapture(0)

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    max_num_hands=1
)

# =========================
# Labels
# =========================

labels_Dic = {
    0:'ا', 1:'ب', 2:'ت', 3:'ث', 4:'ج', 5:'ح', 6:'خ',
    7:'د', 8:'ر', 9:'ز', 10:'س', 11:'ش', 12:'ص', 13:'ض',
    14:'ط', 15:'ظ', 16:'ع', 17:'غ', 18:'ف', 19:'ق',
    20:'ك', 21:'ل', 22:'م', 23:'ن', 24:'ه', 25:'و',
    26:'ي', 27:' '
}

# =========================
# Control Gestures
# Change START_PRED and CLEAR_PRED to any label index
# START_PRED : hold for 2 seconds to start writing
# CLEAR_PRED : hold for 2 seconds to clear sentence
# =========================

START_PRED     = 11   # 'sh'
CLEAR_PRED     = 0    # 'a'
START_HOLD_SEC = 2.0
CLEAR_HOLD_SEC = 2.0

# =========================
# Arabic Drawer
# =========================

def draw_arabic(img, text, pos=(10, 100), font_size=40):
    reshaped  = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil   = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(img_pil)
    font      = ImageFont.truetype("arial.ttf", font_size)
    draw.text(pos, bidi_text, font=font, fill=(255, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================
# Variables
# =========================

sentence           = ""
is_writing         = False

last_pred          = None
last_hand_time     = time.time()
last_char_time     = time.time()
gesture_hold_start = None

# How long the same prediction must be stable before a letter is added.
# Lower value = faster reading. Raise if getting too many wrong letters.
char_delay  = 2.5

space_delay = 5.0
clear_delay = 8.0

# Smoothing buffer — majority vote over last N frames
BUFFER_SIZE = 6
pred_buffer = []

# =========================
# Main Loop
# =========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img_rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results      = hands.process(img_rgb)
    current_time = time.time()

    # =========================
    # HAND DETECTED
    # =========================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]
        last_hand_time = current_time

        x_ = [hand_landmarks.landmark[i].x for i in range(21)]
        y_ = [hand_landmarks.landmark[i].y for i in range(21)]

        data_aux = []
        for i in range(21):
            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        input_tensor = torch.tensor(
            np.array(data_aux, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32
        )

        with torch.no_grad():
            output = model(input_tensor)
            pred   = torch.argmax(output, dim=1).item()

        # --- Smoothing: majority vote over buffer ---
        pred_buffer.append(pred)
        if len(pred_buffer) > BUFFER_SIZE:
            pred_buffer.pop(0)

        # Use the most common prediction in the buffer
        stable_pred = max(set(pred_buffer), key=pred_buffer.count)

        pred_encoded = label_encoder.inverse_transform([stable_pred])[0]
        label        = labels_Dic[int(pred_encoded)]

        # =========================
        # Hold Timer
        # Tracks how long the same stable prediction has been held
        # =========================

        if stable_pred == last_pred:
            if gesture_hold_start is None:
                gesture_hold_start = current_time
        else:
            gesture_hold_start = None

        hold_duration = (
            current_time - gesture_hold_start
            if gesture_hold_start else 0.0
        )
        last_pred = stable_pred

        # =========================
        # START
        # =========================

        if int(pred_encoded) == START_PRED and not is_writing:

            progress = min(hold_duration / START_HOLD_SEC, 1.0)
            bar_w    = int(300 * progress)

            cv2.rectangle(frame, (10, 58), (310, 83), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, 58), (10 + bar_w, 83), (0, 220, 0), -1)
            cv2.putText(frame, 'Hold to START',
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

            if hold_duration >= START_HOLD_SEC:
                is_writing         = True
                gesture_hold_start = None
                last_char_time     = current_time
                pred_buffer.clear()
                print("[START]")

        # =========================
        # CLEAR
        # =========================

        elif int(pred_encoded) == CLEAR_PRED:

            progress = min(hold_duration / CLEAR_HOLD_SEC, 1.0)
            bar_w    = int(300 * progress)

            cv2.rectangle(frame, (10, 58), (310, 83), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, 58), (10 + bar_w, 83), (0, 0, 255), -1)
            cv2.putText(frame, 'Hold to CLEAR',
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            if hold_duration >= CLEAR_HOLD_SEC:
                sentence           = ""
                is_writing         = False
                gesture_hold_start = None
                pred_buffer.clear()
                print("[CLEAR]")

        # =========================
        # Writing
        # A letter is added every char_delay seconds as long as
        # the same stable prediction is held. Change your hand
        # position to type a different letter.
        # =========================

        elif is_writing:

            if (
                current_time - last_char_time > char_delay
                and int(pred_encoded) != 27
            ):
                sentence      += label
                last_char_time = current_time
                print(f"[CHAR] {label}")

            cv2.putText(frame, f'Predicted: {label}',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'Show START gesture (2s)',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        mp_drawing.draw_landmarks(
            frame, hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # =========================
    # NO HAND
    # =========================

    else:

        gesture_hold_start = None
        last_pred          = None
        pred_buffer.clear()
        elapsed            = current_time - last_hand_time

        if is_writing:
            if elapsed > space_delay and not sentence.endswith(" "):
                sentence += " "
                print("[SPACE]")

            if elapsed > clear_delay:
                sentence   = ""
                is_writing = False
                print("[AUTO-CLEAR]")

        cv2.putText(frame, 'No hand detected',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # =========================
    # HUD
    # =========================

    status = "WRITING" if is_writing else "PAUSED"
    color  = (0, 255, 0) if is_writing else (0, 0, 255)
    cv2.putText(frame, f'[ {status} ]',
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame = draw_arabic(frame, sentence)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()