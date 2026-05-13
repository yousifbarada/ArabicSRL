# Arabic Sign Language (ArSL) Real-Time Recognition

## Overview
This project is a real-time computer vision system designed to detect and translate the Arabic Sign Language (ArSL) alphabet into readable Arabic text. The system utilizes MediaPipe for extracting hand landmarks and a custom PyTorch feedforward neural network for classification. It is built to operate entirely via webcam without the need for a keyboard.

## Features
- **Real-Time Translation:** Translates 28 Arabic alphabet letters instantly using webcam input.
- **Gesture Controls:** Full control over the writing process using specific hand gestures (e.g., hold a designated sign for 2 seconds to start writing, or another to clear the text).
- **Smart Smoothing:** Implements a temporal smoothing buffer (majority voting) over recent frames to ensure stable and accurate character prediction.
- **Auto-Spacing and Auto-Clearing:** Automatically inserts spaces between words if the hand is removed from the frame for a specific duration, and clears the screen if removed for a longer period.
- **Arabic Text Rendering:** Utilizes `arabic_reshaper` and `python-bidi` to correctly render connected right-to-left Arabic text directly on the OpenCV video feed.

-  <img width="426" height="240" alt="Video Project 20" src="https://github.com/user-attachments/assets/93f86bb0-eca0-4aaf-9014-9e1d9f3d7642" />

## Project Structure
- `make_data.py`: A script to collect image data for each class (letter) using the webcam.
- `transform_data.py`: Processes the collected images, extracts 21 hand landmarks using MediaPipe, normalizes the coordinates, and saves the dataset as a pickle file (`data.pkl`).
- `clf.py`: The PyTorch training script. It loads the dataset, trains a multi-layer perceptron (MLP) neural network, and saves the trained model weights (`model.pth`) and label encoder (`label_encoder.pkl`).
- `test.py`: The main application script. It runs the real-time inference, handles the gesture controls, applies smoothing, and renders the user interface and Arabic text.
- `try.py`: An evaluation script to test the trained model on random static images from the dataset and display prediction confidence scores.
- `main.py`: An alternative/legacy inference script for basic testing.

## Requirements
To run this project, you need Python 3.x and the following dependencies:

```bash
pip install opencv-python mediapipe numpy torch torchvision scikit-learn pillow arabic-reshaper python-bidi
How to Use
1. Data Collection
If you want to build your own dataset, run the data collection script. It will create folders for each class and capture 350 images per class.

Bash
python make_data.py
2. Data Processing
Extract hand landmarks from the captured images and save them into a serialized format.

Bash
python transform_data.py
3. Model Training
Train the PyTorch neural network on the extracted features. This will generate model.pth and label_encoder.pkl.

Bash
python clf.py
4. Real-Time Inference
Run the main script to start translating Arabic sign language in real-time.

Bash
python test.py
Usage Instructions within test.py:
Show the START gesture (configured to class 11 / 'ش') and hold it for 2 seconds to begin writing.

Change your hand pose to input different letters. The system will append a letter once the prediction stabilizes.

Hide your hand for 5 seconds to add a space.

Hide your hand for 8 seconds, or hold the CLEAR gesture (configured to class 0 / 'ا') for 2 seconds, to clear the current sentence.

Model Architecture
The classification model is a lightweight Feedforward Neural Network built with PyTorch:

Input Layer: 42 features (21 landmarks * 2 coordinates [x, y])

Hidden Layer 1: 128 neurons, ReLU activation

Hidden Layer 2: 64 neurons, ReLU activation

Output Layer: 28 neurons (corresponding to the ArSL alphabet classes)
