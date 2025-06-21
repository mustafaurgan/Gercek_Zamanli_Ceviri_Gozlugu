import cv2
import pytesseract
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading
import queue
from transformers import pipeline

# Tesseract yolu
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

class ImprovedTranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Geliştirilmiş Real-time Çeviri")
        self.root.geometry("1000x800")
        
        # Çeviri modeli (daha büyük ve doğru model)
        self.translation_pipeline = pipeline("translation", 
                                          model="Helsinki-NLP/opus-mt-tc-big-en-tr",
                                          device="cpu")
        
        # Kamera ayarları
        self.cap = None
        self.camera_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.text_queue = queue.Queue(maxsize=1)
        
        # GUI elemanları
        self.create_widgets()
        
        # Varsayılan ayarlar
        self.frame_rate = 10  # Daha yavaş hızda daha iyi sonuç için
        self.frame_count = 0
        
    def create_widgets(self):
        # Ana çerçeveler
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Kamera ve kontrol paneli
        camera_frame = ttk.LabelFrame(main_frame, text="Kamera Görüntüsü")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack()
        
        # Kontrol butonları
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Başlat", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Durdur", command=self.stop_capture, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Kalite ayarları
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(settings_frame, text="Hassasiyet:").pack(side=tk.LEFT)
        self.sensitivity = tk.IntVar(value=3)  # 1-5 arası
        ttk.Scale(settings_frame, from_=1, to=5, variable=self.sensitivity, 
                 command=self.update_sensitivity).pack(side=tk.LEFT, padx=5)
        
        # Metin alanları
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Okunan metin
        original_frame = ttk.LabelFrame(text_frame, text="Okunan Metin (İngilizce)")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_text = scrolledtext.ScrolledText(original_frame, height=10, wrap=tk.WORD)
        self.original_text.pack(fill=tk.BOTH, expand=True)
        
        # Çevrilen metin
        translated_frame = ttk.LabelFrame(text_frame, text="Çevrilen Metin (Türkçe)")
        translated_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.translated_text = scrolledtext.ScrolledText(translated_frame, height=10, wrap=tk.WORD)
        self.translated_text.pack(fill=tk.BOTH, expand=True)
        
        # Geçmiş kayıtları
        history_frame = ttk.LabelFrame(main_frame, text="Geçmiş Çeviriler")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=10, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # Çıkış butonu
        exit_btn = ttk.Button(main_frame, text="Çıkış", command=self.on_close)
        exit_btn.pack(pady=5)
        
    def update_sensitivity(self, value):
        # Hassasiyet ayarını güncelle
        self.frame_rate = 20 - (int(float(value)) * 2)
        
    def start_capture(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Kamera açılamadı!")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.camera_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Kamera görüntüsünü işleme thread'i
            self.camera_thread = threading.Thread(target=self.process_camera, daemon=True)
            self.camera_thread.start()
            
            # GUI güncelleme
            self.update_camera_view()
            
    def stop_capture(self):
        if self.camera_running:
            self.camera_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
    def process_camera(self):
        while self.camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
              
            frame = cv2.rotate(frame, cv2.ROTATE_180) #kamera 180 derece dondu
                
            self.frame_count += 1
            if self.frame_count % self.frame_rate != 0:
                continue
                
            # Görüntü işleme (daha iyi OCR için)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            _, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Metin tanıma (daha iyi config)
            custom_config = r'--oem 3 --psm 6 -l eng'
            boxes = pytesseract.image_to_boxes(threshold_image, config=custom_config)
            
            # Metin kutularını çiz
            h, w, _ = frame.shape
            for b in boxes.splitlines():
                b = b.split()
                if len(b) >= 5:
                    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                    cv2.rectangle(frame, (x, h - y), (x2, h - y2), (0, 255, 0), 2)
                    
            # Tüm metni oku (daha iyi sonuç için)
            text = pytesseract.image_to_string(threshold_image, config=custom_config)
            cleaned_text = ' '.join(text.strip().split())  # Fazla boşlukları kaldır
            
            if cleaned_text and len(cleaned_text.split()) > 2:  # En az 3 kelime
                # Çeviri yap
                translated_text = self.translate_text(cleaned_text)
                
                # Kuyruğa ekle
                if self.text_queue.empty():
                    self.text_queue.put((cleaned_text, translated_text))
            
            # Görüntüyü kuyruğa ekle
            if self.frame_queue.empty():
                self.frame_queue.put(frame)
                
    def translate_text(self, text):
        try:
            # Daha uzun metinler için chunklara böl
            if len(text) > 400:
                chunks = [text[i:i+400] for i in range(0, len(text), 400)]
                translated_chunks = []
                for chunk in chunks:
                    translated = self.translation_pipeline(chunk, max_length=512)
                    translated_chunks.append(translated[0]['translation_text'])
                return ' '.join(translated_chunks)
            else:
                translated = self.translation_pipeline(text, max_length=512)
                return translated[0]['translation_text']
        except Exception as e:
            print(f"Çeviri hatası: {e}")
            return "Çeviri hatası oluştu"
        
    def update_camera_view(self):
        if not self.camera_running:
            return
            
        # Görüntüyü güncelle
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
        # Metinleri güncelle
        if not self.text_queue.empty():
            original, translated = self.text_queue.get()
            self.original_text.delete(1.0, tk.END)
            self.original_text.insert(tk.END, original)
            
            self.translated_text.delete(1.0, tk.END)
            self.translated_text.insert(tk.END, translated)
            
            self.history_text.insert(tk.END, f"İngilizce: {original}\n")
            self.history_text.insert(tk.END, f"Türkçe: {translated}\n")
            self.history_text.insert(tk.END, "-"*50 + "\n")
            
        # 50ms sonra tekrar çağır
        self.root.after(50, self.update_camera_view)
        
    def on_close(self):
        self.stop_capture()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedTranslationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()