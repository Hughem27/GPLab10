import cv2
import numpy as np
from ultralytics import YOLO

# Function to detect specific colored objects in a frame
def detect_objects(frame):
    # Convert frame to HSV color space and define blue color range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue color and find contours
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify and return coordinates of detected objects
    detected_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))
    return detected_objects

# Load YOLOv8 model and open video file
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("GPCars.mp4")

# Setup for video writing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_GPCars.mp4.mp4', fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame and detect objects
    frame = cv2.resize(frame, (800, 600))  
    detected_objects = detect_objects(frame)

    # Apply YOLOv8 tracking and annotate frame
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    # Highlight detected objects
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Resize annotated frame and write to output file
    annotated_frame = cv2.resize(annotated_frame, (800, 600))
    out.write(annotated_frame)

    # Display frames
    cv2.imshow('Frame', frame)
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
