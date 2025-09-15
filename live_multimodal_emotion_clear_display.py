import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import joblib
import librosa

# All emotion classes for facial model
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
# SVM model classes (make sure these match your training)
svm_emotions = ['angry', 'happy', 'sad', 'surprised']

# Load models
facial_model = tf.keras.models.load_model('facial_expression_cnn_realdata.h5')
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

        # Audio prediction
        audio_data = record_audio()
        audio_features = extract_vocal_features_from_audio(audio_data).reshape(1, -1)
        voice_pred = svm_model.predict(audio_features)[0]
        voice_probs_raw = svm_model.predict_proba(audio_features)[0]

        # Zero-pad SVM probs to 7-class order
        voice_probs = np.zeros(len(emotion_classes))
        for i, emo in enumerate(svm_emotions):
            idx = emotion_classes.index(emo)
            voice_probs[idx] = voice_probs_raw[i]

        # Combined average probabilities
        combined_probs = (face_probs + voice_probs) / 2
        combined_pred_avg = np.argmax(combined_probs)
        combined_emotion_avg = emotion_classes[combined_pred_avg]

        # Weighted fusion example
        w_audio, w_video = 0.6, 0.4
        fusion_probs = w_audio * voice_probs + w_video * face_probs
        fusion_pred = np.argmax(fusion_probs)
        combined_emotion_fusion = emotion_classes[fusion_pred]

        # Draw semi-transparent rectangles as text backgrounds
        overlay = frame.copy()
        alpha = 0.6  # Transparency factor

        cv2.rectangle(overlay, (10, 10), (frame_w - 10, 70), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 80), (frame_w - 10, 130), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 140), (frame_w - 10, 190), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 200), (frame_w - 10, 250), (0, 0, 0), -1)

        # Blend with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display all four modes with colors
        cv2.putText(frame, "Audio-only: " + voice_pred, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Video-only: " + face_emotion, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Average Fusion: " + combined_emotion_avg, (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Weighted Fusion: " + combined_emotion_fusion, (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow('Multimodal Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
