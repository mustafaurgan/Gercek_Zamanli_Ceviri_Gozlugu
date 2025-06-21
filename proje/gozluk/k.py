import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MJPEG formatını zorla
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame alınamadı.")
        continue

    print("Frame shape:", frame.shape)
    cv2.imshow("Kamera Testi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
