import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_facial_cnn():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),  # 48x48 grayscale images
        layers.Conv2D(32, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(7, activation='softmax')  # 7 emotion classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Creating Facial Expression CNN...")
    model = create_facial_cnn()

    # Create dummy data: 70 images (10 per emotion), 48x48 grayscale
    X_dummy = np.random.rand(70, 48, 48, 1).astype(np.float32)
    y_dummy = np.repeat(np.arange(7), 10)  # Labels 0 to 6 for 7 classes
    
    print("Training model on dummy data (quick init)...")
    model.fit(X_dummy, y_dummy, epochs=3, batch_size=10, verbose=2)
    
    # Save the model
    model.save('facial_expression_cnn.h5')
    print("Model trained and saved as 'facial_expression_cnn.h5'.")
