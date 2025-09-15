import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import joblib
import librosa
import threading
import queue

# Emotion classes for face model (unchanged)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

# Load facial CNN model
facial_model = tf.keras.models.load_model('facial_expression_cnn_realdata.h5')

# Load trained voice model and preprocessing objects
svm_model = joblib.load('voice_emotion_svm_ravdess.pkl')
label_encoder = joblib.load('label_encoder_ravdess.pkl')
scaler = joblib.load('scaler_ravdess.pkl')

audio_queue = queue.Queue()

def extract_vocal_features_from_audio(audio, sr=16000):
    audio = audio.astype(np.float64)
    rmse = librosa.feature.rms(y=audio).mean()
    zcr = librosa.feature.zero_crossing_rate(y=audio).mean()
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1)
    features = np.hstack([rmse, zcr, spec_centroid, mfccs])
    return features

def audio_recording_thread(duration=1, fs=16000):
    while True:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio_queue.put(audio.flatten())

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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    threading.Thread(target=audio_recording_thread, daemon=True).start()

    latest_voice_pred = "N/A"
    latest_voice_probs = np.zeros(len(emotion_classes))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        # Facial prediction
        face_input, bbox = preprocess_face(frame)
        face_probs = facial_model.predict(face_input)[0]
        face_pred = np.argmax(face_probs)
        face_emotion = emotion_classes[face_pred]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, "Video: " + face_emotion, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Voice prediction from latest audio available
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            features = extract_vocal_features_from_audio(audio_data).reshape(1, -1)
            features_scaled = scaler.transform(features)
            voice_pred_label = svm_model.predict(features_scaled)[0]
            voice_pred = label_encoder.inverse_transform([voice_pred_label])[0]

            voice_probs_raw = svm_model.predict_proba(features_scaled)[0]

            # Map voice_probs_raw to common emotion_classes order with zeros for missing
            voice_probs = np.zeros(len(emotion_classes))
            for i, emo in enumerate(label_encoder.classes_):
                if emo in emotion_classes:
                    idx = emotion_classes.index(emo)
                    voice_probs[idx] = voice_probs_raw[i]

            latest_voice_pred = voice_pred
            latest_voice_probs = voice_probs

        cv2.putText(frame, "Audio-only: " + latest_voice_pred, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Average fusion
        combined_probs = (face_probs + latest_voice_probs) / 2
        combined_pred_avg = np.argmax(combined_probs)
        combined_emotion_avg = emotion_classes[combined_pred_avg]
        cv2.putText(frame, "Average Fusion: " + combined_emotion_avg, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Weighted fusion
        w_audio, w_video = 0.6, 0.4
        fusion_probs = w_audio * latest_voice_probs + w_video * face_probs
        fusion_pred = np.argmax(fusion_probs)
        combined_emotion_fusion = emotion_classes[fusion_pred]
        cv2.putText(frame, "Weighted Fusion: " + combined_emotion_fusion, (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow('Multimodal Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
