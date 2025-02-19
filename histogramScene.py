import cv2
import numpy as np
import time
from detect_AU import detect_AU  # AU 감지 코드 불러오기

cap = cv2.VideoCapture(0)  # 웹캠
prev_hist = None
prev_ratios = []
prev_rotations = []
THRESHOLD = 0.6  # 씬 변경 감지 임계값
FRAME_MEMORY = 15  # AU 비교 프레임

# FPS 측정
prev_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # AU 감지 실행 (비율 기반, 정면 벡터 보정 추가)
    frame, au_changed = detect_AU(frame)

    if au_changed:
        print("🤖 표정 변화 감지!")

    # 씬 변경 감지 (히스토그램 방식)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    if prev_hist is not None:
        similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if similarity < THRESHOLD:
            print("🎬 Scene Change Detected!")

    prev_hist = hist

    # FPS 계산
    frame_count += 1
    if frame_count >= 10:  # 10프레임마다 FPS 갱신
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0

    # FPS 화면 출력
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Scene & AU Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
