# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Define a function to preprocess frames for the model
def preprocess_frame(frame, target_size):
    """
    Preprocess the frame to fit the model input.
    - Resize to the model's input size
    - Normalize pixel values
    - Convert BGR to RGB (as OpenCV loads in BGR format)
    """
    resized_frame = cv2.resize(frame, target_size)  # Resize to target size
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    normalized_frame = rgb_frame / 255.0  # Normalize pixel values
    return normalized_frame.reshape(1, target_size[0], target_size[1], 3)  # Reshape for the model

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be accessed!")
    exit()

# Define the target size for your model's input
input_shape = (model.input_shape[1], model.input_shape[2])  # Assuming channels_last format

# Real-time video capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    # Preprocess the captured frame
    preprocessed_frame = preprocess_frame(frame, input_shape)

    # Predict using the model
    prediction = model.predict(preprocessed_frame)
    status = "Open" if prediction[0][0] > 0.5 else "Closed"

    # Display the result on the video feed
    cv2.putText(frame, f"Eye: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Eye Status Detection', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
