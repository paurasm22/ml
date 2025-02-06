import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def recognize_eye_status(model_path, image_size=(64, 64)):
    """
    Function to recognize whether eyes are open or closed using a webcam feed.
    
    Args:
        model_path (str): Path to the pre-trained model file (.h5).
        image_size (tuple): The target image size for the model (default: (64, 64)).

    Usage:
        recognize_eye_status('eye_status_model.h5')
    """
    # Load the pre-trained model
    model = load_model(model_path)

    # Load Haar Cascade for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale (required for Haar cascades)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
        
        for (x, y, w, h) in faces:
            # Extract the region of interest (face)
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY))

            for (ex, ey, ew, eh) in eyes:
                # Extract each eye region
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                eye_roi_resized = cv2.resize(eye_roi, image_size)  # Resize to model's input size
                eye_roi_array = np.expand_dims(img_to_array(eye_roi_resized) / 255.0, axis=0)

                # Make a prediction
                prediction = model.predict(eye_roi_array)
                label = "Open" if prediction > 0.5 else "Closed"
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)

                # Draw rectangle around the eye and label it
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        cv2.imshow("Eye Status Recognition", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example Usage
recognize_eye_status('model.h5')
