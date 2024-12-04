import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="c6sQHHveoGkXViQSwtHv")

# Load the model
project = rf.workspace().project("ambulance-regocnition")
model = project.version("2").model




# # Function to draw bounding boxes
# def draw_boxes(image, predictions):
#     for prediction in predictions:
#         x = int(prediction['x'])
#         y = int(prediction['y'])
#         width = int(prediction['width'])
#         height = int(prediction['height'])
#         class_name = prediction['class']
#         confidence = prediction['confidence']
#         print(f"{x},{y},{width},{height},{class_name},{confidence}")
#     return image

# Capture video from webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    # Predict on the frame
    predictions = model.predict(frame, confidence=60, overlap=30).json()['predictions']
    print(predictions)
    
    # Draw bounding boxes
    #sannotated_frame = draw_boxes(frame, predictions)

    # # Show the annotated frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
q