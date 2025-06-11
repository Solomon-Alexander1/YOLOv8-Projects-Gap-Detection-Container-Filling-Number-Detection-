from ultralytics import YOLO
import cv2
import easyocr

# Load YOLOv8 model
model = YOLO("runs/detect/train5/weights/best.pt")

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Open camera with DirectShow backend (avoids MSMF error)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for performance

    # Run YOLOv8 on the frame
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"  # Label with confidence

            # Extract the detected number using OCR
            roi = frame[y1:y2, x1:x2]  # Crop the detected region
            ocr_results = reader.readtext(roi, detail=0)  # Get detected text

            if ocr_results:
                detected_number = ocr_results[0]  # Take the first detected number
            else:
                detected_number = "Unknown"

            # Display bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} - {detected_number}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Number Detection with OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
