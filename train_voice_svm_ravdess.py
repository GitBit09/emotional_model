import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Map RAVDESS emotion codes to emotion labels
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    rmse = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    features = np.hstack([rmse, zcr, spec_centroid, mfccs])
    return features

def get_all_wav_files(folder):
    wav_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def load_data(dataset_path):
    features_list = []
    labels_list = []
    wav_files = get_all_wav_files(dataset_path)
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        emotion_code = filename.split('-')[2]
        emotion = emotion_map.get(emotion_code)
        if emotion:
            features = extract_features(file_path)
            features_list.append(features)
            labels_list.append(emotion)
    
    return np.array(features_list), np.array(labels_list)

if __name__ == "__main__":
    dataset_path = './audio_RAVDESS'  # Update if needed
    
    print("Loading dataset and extracting features...")
    X, y = load_data(dataset_path)
    
    print(f"Extracted features from {len(X)} audio files.")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVM classifier...")
    svm_clf = SVC(probability=True, kernel='rbf', random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    
    y_pred = svm_clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.2f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("Saving trained model and preprocessing objects...")
    joblib.dump(svm_clf, 'voice_emotion_svm_ravdess.pkl')
    joblib.dump(label_encoder, 'label_encoder_ravdess.pkl')
    joblib.dump(scaler, 'scaler_ravdess.pkl')
    
    print("All done! Models and preprocessing files saved.")
