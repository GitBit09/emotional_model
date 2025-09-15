import librosa
import numpy as np

def extract_vocal_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    rmse = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    return np.array([rmse, zcr, spec_centroid])

if __name__ == "__main__":
    audio_file = 'test1.wav'  # Replace this with your audio file path
    features = extract_vocal_features(audio_file)
    print("Extracted features:", features)
