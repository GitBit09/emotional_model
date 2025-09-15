import joblib
from audio_feature_extraction import extract_vocal_features

# Load the trained model
svm_model = joblib.load('voice_emotion_svm_model.pkl')

# Extract features from new audio file to test
test_file = 'test1.wav'  # Replace with your actual test audio file
features = extract_vocal_features(test_file).reshape(1, -1)

# Predict emotion
predicted_emotion = svm_model.predict(features)[0]
probabilities = svm_model.predict_proba(features)[0]

print(f"Predicted emotion: {predicted_emotion}")
print("Probabilities:", dict(zip(svm_model.classes_, probabilities)))
