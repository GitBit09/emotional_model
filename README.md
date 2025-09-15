# Multimodal Emotion Recognition

This project implements real-time multimodal emotion recognition by combining facial expression analysis and audio emotion classification.

## Features

- Facial emotion recognition using CNN on live webcam feed
- Voice emotion recognition using SVM trained on RAVDESS dataset
- Real-time fusion of audio and video predictions (average and weighted)
- Live display with 4 quadrant view of all modes

## Usage

1. Activate your Python environment:

2. Run the live recognition script:


3. Press `q` to quit.

## Training Voice Model

Use the script `train_voice_svm_ravdess.py` to extract features and train the voice emotion classifier on the RAVDESS dataset.

## Dependencies

- Python 3.8+
- TensorFlow
- scikit-learn
- librosa
- OpenCV
- Sounddevice

---

Feel free to modify and extend the project!
