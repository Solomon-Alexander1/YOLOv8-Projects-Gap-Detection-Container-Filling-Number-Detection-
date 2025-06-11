import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("C:/yolov8_project/runs/detect/train2/weights/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Constants for pixel-to-cm conversion (adjust as needed)
KNOWN_WIDTH_CM = 10  # Approximate width of the book (change if needed)
FOCAL_LENGTH = 600  # Adjust this value for better accuracy

def calculate_distance(x1, x2):
    """Calculate the distance between two detected objects in cm."""
    if x1 is None or x2 is None:
        return None  # Distance can't be calculated if one object is missing
    
    # Absolute difference between x-coordinates
    distance_px = abs(x1 - x2)
    
    # Convert pixels to cm (Adjust scaling factor as needed)
    distance_cm = (distance_px * KNOWN_WIDTH_CM) / FOCAL_LENGTH

    return round(distance_cm, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Initialize object positions
    book_x, bottle_x = None, None

    # Process detected objects
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])  # Class ID
            x, y, w, h = map(int, box.xywh[0])  # Bounding box center x, y, width, height

            if cls_id == 0:  # Book
                book_x = x  # Center of the book
                color = (0, 255, 0)  # Green
                label = "Book"

            elif cls_id == 1:  # Bottle
                bottle_x = x  # Center of the bottle
                color = (255, 0, 0)  # Blue
                label = "Bottle"

            # Draw bounding boxes
            x_min = x - w // 2
            x_max = x + w // 2
            cv2.rectangle(frame, (x_min, y - h // 2), (x_max, y + h // 2), color, 2)
            cv2.putText(frame, label, (x_min, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Check if both book and bottle are detected
    if book_x is not None and bottle_x is not None:
        gap_distance = calculate_distance(book_x, bottle_x)

        # Define "Gap" condition
        gap_label = "No Gap" if gap_distance < 1 else "Gap"

        # Display results
        cv2.putText(frame, f"Distance: {gap_distance} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, gap_label, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Detecting...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
