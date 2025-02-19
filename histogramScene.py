import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0) 

# Save previous frame variable
prev_hist = None
THRESHOLD = 0.6  # Scene change detection threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale --> speed of calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 정규화

    # No value to compare in first frame, so just save
    if prev_hist is not None:
        # Compare histogram
        similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

        # Detect scene change if similarity is below a certain threshold
        if similarity < THRESHOLD:
            print("Scene Change Detected!")

    # Save current histogram as previous value
    prev_hist = hist

    # Output to screen
    cv2.imshow('Webcam Scene Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
