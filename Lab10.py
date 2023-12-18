import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Optionally, filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            detected_objects.append((x, y, w, h))
    return detected_objects

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("traffic3.mp4")

# Looping
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #Resizing the frame 
    frame = cv2.resize(frame, (800, 600))  
    detected_objects = detect_objects(frame)

    # Run YOLOv8 persistant tracking
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    for (x, y, w, h) in detected_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    annotated_frame = cv2.resize(annotated_frame, (800, 600))

    # Display
    cv2.imshow('Frame', frame)
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
