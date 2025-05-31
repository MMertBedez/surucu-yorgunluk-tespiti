import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf
from gaze_tracking import GazeTracking

class DriverMonitoring:
    def __init__(self):
        # MediaPipe Face Mesh başlatma
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # GazeTracking başlatma
        self.gaze = GazeTracking()
        
        # Göz noktaları
        self.LEFT_EYE = [386, 374]  # [üst nokta, alt nokta]
        self.RIGHT_EYE = [159, 145]  # [üst nokta, alt nokta]
        
        # Ağız noktaları
        self.MOUTH_POINTS = [
            61,  # Sol ağız köşesi
            291, # Sağ ağız köşesi
            0,   # Çene
            17,  # Sol çene
            291, # Sağ çene
            405, # Üst dudak
            314, # Alt dudak
            13,  # Üst dudak orta
            14,  # Alt dudak orta
        ]
        
        # Esneme modeli yükleme
        self.yawn_model = tf.keras.models.load_model('model_yawn_best.h5')
        
        # Değişkenler
        self.eyes_closed_start_time = None
        self.WARNING_THRESHOLD = 10  # 10 saniye
        self.EAR_THRESHOLD = 0.015  # Göz açıklık eşiği
        self.YAWN_THRESHOLD = 0.5  # Esneme eşiği
        
        # Esneme sayacı
        self.yawn_count = 0
        self.last_yawn_state = False
        
    def calculate_ear(self, eye_points, landmarks):
        """Göz açıklık oranını hesapla - sadece dikey mesafeyi kullan"""
        vertical_dist = np.linalg.norm(np.array([landmarks.landmark[eye_points[0]].x, landmarks.landmark[eye_points[0]].y]) - 
                                     np.array([landmarks.landmark[eye_points[1]].x, landmarks.landmark[eye_points[1]].y]))
        return vertical_dist
    
    def get_mouth_roi(self, frame, landmarks):
        """Ağız bölgesini tespit et ve kırp"""
        h, w = frame.shape[:2]
        
        # Ağız noktalarının koordinatlarını al
        mouth_points = []
        for point_idx in self.MOUTH_POINTS:
            landmark = landmarks.landmark[point_idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_points.append((x, y))
        
        # Ağız bölgesinin sınırlarını bul
        x_coords = [p[0] for p in mouth_points]
        y_coords = [p[1] for p in mouth_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Bölgeyi biraz genişlet
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Ağız bölgesini kes
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        
        return mouth_roi, (x_min, y_min, x_max, y_max)
    
    def preprocess_mouth(self, mouth_roi):
        """Ağız görüntüsünü model için hazırla"""
        if mouth_roi is None or mouth_roi.size == 0:
            return None
            
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        # 224x224 boyutuna getir
        resized = cv2.resize(gray, (224, 224))
        
        # Normalize et
        normalized = resized / 255.0
        
        # Model giriş formatına getir
        input_data = np.expand_dims(normalized, axis=-1)  # Kanal boyutu ekle
        input_data = np.expand_dims(input_data, axis=0)   # Batch boyutu ekle
        
        return input_data
    
    def process_frame(self, frame):
        """Frame'i işle ve tüm analizleri yap"""
        # Frame'i RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # GazeTracking'i güncelle
        self.gaze.refresh(frame)
        
        # Sonuçları tutacak değişkenler
        eye_status = "Normal"
        yawn_status = "Esnemiyor"
        gaze_status = "Yola Bakiyor"
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # 1. Göz Analizi
            left_ear = self.calculate_ear(self.LEFT_EYE, landmarks)
            right_ear = self.calculate_ear(self.RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2
            
            eyes_closed = avg_ear < self.EAR_THRESHOLD
            if eyes_closed:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = time.time()
                else:
                    closed_duration = time.time() - self.eyes_closed_start_time
                    if closed_duration >= self.WARNING_THRESHOLD:
                        eye_status = "SURUCU UYUYOR!"
            else:
                self.eyes_closed_start_time = None
            
            # 2. Esneme Analizi
            mouth_roi, (x1, y1, x2, y2) = self.get_mouth_roi(frame, landmarks)
            if mouth_roi is not None:
                processed_mouth = self.preprocess_mouth(mouth_roi)
                if processed_mouth is not None:
                    yawn_prob = self.yawn_model.predict(processed_mouth, verbose=0)[0][0]
                    is_yawn = yawn_prob > self.YAWN_THRESHOLD
                    
                    # Esneme sayacını güncelle
                    if is_yawn and not self.last_yawn_state:
                        self.yawn_count += 1
                    self.last_yawn_state = is_yawn
                    
                    yawn_status = "Esnemiyor" if is_yawn else "Esniyor"
                    
                    # Ağız bölgesini yeşil çerçeve ile işaretle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Ağız noktalarını çiz
                    for point_idx in self.MOUTH_POINTS:
                        landmark = landmarks.landmark[point_idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # 3. Bakış Yönü Analizi
            if self.gaze.pupils_located:
                if self.gaze.is_right():
                    gaze_status = "Saga Bakiyor"
                elif self.gaze.is_left():
                    gaze_status = "Sola Bakiyor"
                else:
                    gaze_status = "Yola Bakiyor"
                
                if self.gaze.is_blinking():
                    gaze_status = "Gozler Kapali"
            
            # Göz noktalarını çiz
            for eye_points in [self.LEFT_EYE, self.RIGHT_EYE]:
                for point in eye_points:
                    x = int(landmarks.landmark[point].x * frame.shape[1])
                    y = int(landmarks.landmark[point].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        # Sonuçları ekrana yaz
        y_position = 30
        cv2.putText(frame, f"Goz Durumu: {eye_status}", (10, y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Esneme: {yawn_status}", (10, y_position + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Esneme Sayisi: {self.yawn_count}", (10, y_position + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Bakis: {gaze_status}", (10, y_position + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gözler kapalıysa süreyi göster
        if self.eyes_closed_start_time is not None:
            closed_duration = time.time() - self.eyes_closed_start_time
            cv2.putText(frame, f"Kapali Kalma Suresi: {closed_duration:.1f}s", (10, y_position + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Uyarı durumunda kırmızı çerçeve
        if eye_status == "SURUCU UYUYOR!":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        
        return frame

def main():
    # Sürücü izleme sistemini başlat
    monitor = DriverMonitoring()
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame'i işle
        processed_frame = monitor.process_frame(frame)
        
        # Sonucu göster
        cv2.imshow('Surucu Izleme Sistemi', processed_frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 