import cv2

# V4L2 backend ile kamerayı başlat
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MJPEG formatını zorla
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Pencereyi önceden oluştur (sadece 1 kez)
cv2.namedWindow("Ters Kamera Görüntüsü", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame alınamadı.")
        continue

    # Görüntüyü 180 derece döndür (ters çevirme)
    rotated = cv2.rotate(frame, cv2.ROTATE_180)

    # Tek pencere üzerinden göster
    cv2.imshow("Ters Kamera Görüntüsü", rotated)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
