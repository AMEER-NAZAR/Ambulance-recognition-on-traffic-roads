import cv2
import numpy as np
from roboflow import Roboflow

# Replace with your actual Roboflow API key
api_key = "c6sQHHveoGkXViQSwtHv"

# Initialize Roboflow client
rf = Roboflow(api_key=api_key)

# Replace with your project name and model version (adjust if needed)
project_name = "ambulance-regocnition"
model_version = "2"

try:
  # Get project and model details
  project = rf.workspace().project(project_name)
  model = project.version(model_version).model
except Exception as e:
  print("Error: Could not access project or model. Check project name, version, and API key.")
  print(e)
  exit()

# Function to detect objects and draw bounding boxes
def detect_objects(frame, confidence=0.4, overlap=0.3):
  # Preprocess the frame if required by your model (e.g., resize)
  # ... your preprocessing logic here ...

  # Convert frame to RGB format (assuming model expects it)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Perform inference using Roboflow model
  predictiondict = model.predict(frame_rgb, confidence=confidence, overlap=overlap)
  print(predictiondict)
  # Draw bounding boxes for detected objects
  try:
    for prediction in predictiondict["predictions"]:
        print(type(prediction))
        x, y, width, height = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green for detected ambulance

    return frame
  except:
    pass

# Open webcam capture
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame capture was successful
  if not ret:
    print("Error: Failed to capture frame")
    break

  # Detect objects and draw bounding boxes
  processed_frame = detect_objects(frame.copy())  # Avoid modifying the original frame

  # Display the resulting frame
  cv2.imshow('ambulance detection', processed_frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) == ord('q'):
    break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
