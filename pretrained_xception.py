import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2

# Load the pre-trained Xception model
def build_pretrained_model(input_shape=(224, 224, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add custom layers on top of the base model for fine-tuning
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Dropout for regularization
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification (deepfake or not)
    
    # Model to be trained
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False  # We freeze the base Xception layers to use pre-trained weights

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to preprocess video frames
def extract_frames(video_path, frame_rate=1):
    """Extract frames from video at the given frame rate."""
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            # Resize frame to match the input shape of the model
            resized_frame = cv2.resize(image, (224, 224))
            frames.append(resized_frame)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return np.array(frames)

# Function to detect deepfake using pre-trained model
def detect_deepfake(video_path, model):
    frames = extract_frames(video_path)
    frames = frames / 255.0

    predictions = model.predict(frames)
    
    # Calculate average prediction
    avg_prediction = np.mean(predictions)
    
    return avg_prediction