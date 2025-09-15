# Initialize git repo if not already initialized
if (!(Test-Path ".git")) {
    git init
}

# Create .gitignore file if it doesn't exist
if (!(Test-Path ".gitignore")) {
@"
# Python cache
__pycache__/
*.py[cod]

# Virtual environments
env/
venv/

# Data and models (optional)
audio_RAVDESS/
*.h5
*.pkl

# VS Code settings
.vscode/
"@ | Out-File -Encoding utf8 .gitignore
    Write-Host ".gitignore created."
}

# Create README.md file if it doesn't exist
if (!(Test-Path "README.md")) {
@"
# Multimodal Emotion Recognition

This project implements real-time multimodal emotion recognition by combining facial expression analysis and audio emotion classification.

## Features

- Facial emotion recognition using CNN on live webcam feed
- Voice emotion recognition using SVM trained on RAVDESS dataset
- Real-time fusion of audio and video predictions (average and weighted)
- Live display with 4 quadrant view of all modes

## Usage

1. Activate your Python environment:

