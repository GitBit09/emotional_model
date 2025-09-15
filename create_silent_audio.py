import numpy as np
from scipy.io.wavfile import write

fs = 16000  # Sample rate (16 kHz)
duration = 1  # seconds

# Create array of zeros for silent audio
silent_audio = np.zeros(fs * duration, dtype=np.int16)

# Save as WAV file in your project folder
write('C:\\train_py\\test_audio.wav', fs, silent_audio)

print("Silent test audio file 'test_audio.wav' created successfully.")
