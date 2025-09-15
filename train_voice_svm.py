import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

from audio_feature_extraction import extract_vocal_features

# Map of emotion labels to audio file names
emotion_files = {
    'angry': 'angry.wav',
    'happy': 'happy.wav',
    'sad': 'sad.wav',
    'surprised': 'surprised.wav'
}

def build_dataset(emotion_files):
    features_list = []
    labels_list = []
    for emotion, filename in emotion_files.items():
        features = extract_vocal_features(filename)
        features_list.append(features)
        labels_list.append(emotion)
    return np.array(features_list), np.array(labels_list)

def train_svm(features, labels):
    # Create a pipeline with StandardScaler and SVM with RBF kernel
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    clf.fit(features, labels)
    return clf

if __name__ == "__main__":
    print("Building dataset...")
    features, labels = build_dataset(emotion_files)
    print("Feature vectors:", features)
    print("Labels:", labels)
    
    print("Training SVM classifier...")
    svm_model = train_svm(features, labels)
    
    # Save the trained model
    joblib.dump(svm_model, 'voice_emotion_svm_model.pkl')
    print("Model trained and saved as 'voice_emotion_svm_model.pkl'.")
