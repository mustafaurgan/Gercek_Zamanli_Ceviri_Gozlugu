import cv2
import pytesseract
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from transformers import pipeline

# Tesseract yolu
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# NLTK stopwords indir
nltk.download('stopwords')

# Hugging Face çeviri modeli: İngilizce → Türkçe
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")

# Kamera başlat (V4L2 backend ile)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# MJPEG formatını zorla
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Çözünürlük ayarla
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def translate_text(text):
    translated = translation_pipeline(text, max_length=400)
    return translated[0]['translation_text']

def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

frame_rate = 20
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame alınamadı.")
        continue

    if len(frame.shape) < 3 or frame.shape[2] != 3:
        print(f"Geçersiz görüntü formatı: {frame.shape}")
        continue

    frame_count += 1
    if frame_count % frame_rate != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    
    
    boxes = pytesseract.image_to_boxes(threshold_image)
    h, w, _ = frame.shape
    for b in boxes.splitlines():
        b = b.split()
        if len(b) >= 5:
            x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(frame, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

    text = pytesseract.image_to_string(threshold_image)
    print(f"Taranan Metin: {text}")

    cleaned_text = text.strip()
    translated_text = translate_text(cleaned_text)

    print(f"Okunan Metin: {cleaned_text}")
    print(f"Çevirilen Metin: {translated_text}")

    with open("islemler.txt", "w") as file:
        file.write(f"Okunan Metin: {cleaned_text}\n")
        file.write(f"Cevirilen Metin: {translated_text}\n")

    cv2.putText(frame, cleaned_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, translated_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Real-time Metin Ceviri", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
