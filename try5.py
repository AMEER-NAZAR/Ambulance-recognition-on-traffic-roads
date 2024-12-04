import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="c6sQHHveoGkXViQSwtHv")

# Load the model
project = rf.workspace().project("ambulance-regocnition")
model = project.version("2").model

def convert_to_opencv_coordinates(x, y, width, height, image_height):
    # Convert y-coordinate from bottom to top
    y_opencv = image_height - y - height

    # Return OpenCV-compatible coordinates
    return (x, y_opencv, width, height)

# Function to draw bounding boxes
def draw_boxes(image, predictions):
    for prediction in predictions:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        class_name = prediction['class']
        confidence = prediction['confidence']
        print(f"{x},{y},{width},{height},{class_name},{confidence}")
        x, y, width, height = convert_to_opencv_coordinates(x, y, width, height, image.shape[0])  # Assuming image.shape[0] gives the height

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Path to the image file
image_path = "./2.jpg"

# Read the image
frame = cv2.imread(image_path)

# Predict on the frame
predictions = model.predict(frame, confidence=40, overlap=30).json()['predictions']
print(predictions)

# Draw bounding boxes
annotated_frame = draw_boxes(frame, predictions)

# Show the annotated frame
cv2.imshow('Object Detection', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
q