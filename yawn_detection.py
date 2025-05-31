import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# MediaPipe Face Mesh'i başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Modeli yükle
try:
    model = load_model("model_yawn_best.h5")
except:
    print("Hata: model_yawn_best.h5 dosyası bulunamadı!")
    exit()

# Ağız noktalarının indeksleri (MediaPipe Face Mesh'te)
MOUTH_POINTS = [
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

# Esneme sayacı
yawn_count = 0
# Son esneme durumu
last_yawn_state = False
# Esneme tespiti için eşik değeri
YAWN_THRESHOLD = 0.5

def get_mouth_roi(frame, face_landmarks):
    """Ağız bölgesini tespit et ve kes"""
    h, w = frame.shape[:2]
    
    # Ağız noktalarının koordinatlarını al
    mouth_points = []
    for point_idx in MOUTH_POINTS:
        landmark = face_landmarks.landmark[point_idx]
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

def preprocess_frame(frame, face_landmarks):
    """Ağız bölgesini kesip model için hazırla"""
    # Ağız bölgesini al
    mouth_roi, (x_min, y_min, x_max, y_max) = get_mouth_roi(frame, face_landmarks)
    
    # Gri tonlamaya çevir
    mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    
    # Model için boyutlandır (224x224)
    mouth_roi = cv2.resize(mouth_roi, (224, 224))
    
    # Normalize et
    mouth_roi = mouth_roi / 255.0
    
    # Model için kanal boyutu ekle (batch_size, height, width, channels)
    mouth_roi = np.expand_dims(mouth_roi, axis=-1)  # Kanal boyutu ekle
    mouth_roi = np.expand_dims(mouth_roi, axis=0)   # Batch boyutu ekle
    
    return mouth_roi, (x_min, y_min, x_max, y_max)

def main():
    global yawn_count, last_yawn_state
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Görüntüyü RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Ağız bölgesini hazırla
                mouth_roi, (x_min, y_min, x_max, y_max) = preprocess_frame(frame, face_landmarks)
                
                # Esneme tahmini yap
                yawn_prob = model.predict(mouth_roi, verbose=0)[0][0]
                
                # Esneme durumunu belirle
                is_yawn = yawn_prob > YAWN_THRESHOLD
                
                # Esneme sayacını güncelle
                if is_yawn and not last_yawn_state:
                    yawn_count += 1
                last_yawn_state = is_yawn
                
                # Ağız bölgesini çerçevele
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Ağız noktalarını çiz
                for point_idx in MOUTH_POINTS:
                    landmark = face_landmarks.landmark[point_idx]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
                # Esneme durumunu ve sayacı ekrana yazdır
                status = "Esnemiyor" if is_yawn else "Esniyor"
                cv2.putText(frame, f"Durum: {status}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Esneme Sayisi: {yawn_count}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Esneme Olasiligi: {yawn_prob:.2f}", (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Görüntüyü göster
        cv2.imshow('Esneme Tespiti', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 