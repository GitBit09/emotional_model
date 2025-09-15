import cv2
import numpy as np
import tensorflow as tf

# Load your trained facial expression CNN model
model = tf.keras.models.load_model('facial_expression_cnn_realdata.h5')

# Class names must match your dataset’s emotion labels
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        face_input, bbox = preprocess_face(frame)
        preds = model.predict(face_input)
        emotion_idx = np.argmax(preds)
        emotion_label = emotion_classes[emotion_idx]

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame, "Face: " + emotion_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # TODO: Integrate voice emotion prediction here
        
        cv2.imshow('Live Multimodal Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
