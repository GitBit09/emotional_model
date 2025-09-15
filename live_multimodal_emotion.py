import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import joblib
import librosa

# Load facial expression CNN model
facial_model = tf.keras.models.load_model('facial_expression_cnn_realdata.h5')
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

# Load trained voice emotion SVM
svm_model = joblib.load('voice_emotion_svm_model.pkl')

def preprocess_face(frame, face_ratio=0.3):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(min(w, h) * face_ratio)
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)
    face_region = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 48, 48, 1), (x1, y1, x2, y2)

def record_audio(duration=2, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_vocal_features_from_audio(audio, sr=16000):
    audio = audio.astype(np.float64)
    rmse = librosa.feature.rms(y=audio).mean()
    zcr = librosa.feature.zero_crossing_rate(y=audio).mean()
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    return np.array([rmse, zcr, spec_centroid])

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Facial emotion prediction
        face_input, bbox = preprocess_face(frame)
        preds = facial_model.predict(face_input)
        emotion_idx = np.argmax(preds)
        facial_emotion = emotion_classes[emotion_idx]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Face: " + facial_emotion, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Voice emotion prediction (record and predict)
        audio_data = record_audio()
        audio_features = extract_vocal_features_from_audio(audio_data).reshape(1, -1)
        voice_emotion = svm_model.predict(audio_features)[0]
        cv2.putText(frame, "Voice: " + voice_emotion, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Multimodal Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
