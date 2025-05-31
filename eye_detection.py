import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe Face Mesh'i başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Göz noktalarının indeksleri - sadece üst ve alt noktaları kullanacağız
# Sol göz için noktalar (üst ve alt)
LEFT_EYE = [386, 374]  # [üst nokta, alt nokta]
# Sağ göz için noktalar (üst ve alt)
RIGHT_EYE = [159, 145]  # [üst nokta, alt nokta]

def calculate_ear(eye_points, landmarks):
    """Göz açıklık oranını hesapla - sadece dikey mesafeyi kullan"""
    # Üst ve alt noktalar arasındaki dikey mesafeyi hesapla
    vertical_dist = np.linalg.norm(np.array([landmarks.landmark[eye_points[0]].x, landmarks.landmark[eye_points[0]].y]) - 
                                 np.array([landmarks.landmark[eye_points[1]].x, landmarks.landmark[eye_points[1]].y]))
    
    return vertical_dist

def main():
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    # Göz kapalı kalma süresi için değişkenler
    eyes_closed_start_time = None
    WARNING_THRESHOLD = 10  # 10 saniye
    EAR_THRESHOLD = 0.015  # Göz açıklık eşiği (dikey mesafe için)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Görüntüyü RGB'ye çevir
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Sol göz EAR hesapla
                left_ear = calculate_ear(LEFT_EYE, face_landmarks)
                # Sağ göz EAR hesapla
                right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
                
                # Ortalama EAR değeri
                avg_ear = (left_ear + right_ear) / 2
                
                # Gözlerin durumunu kontrol et
                eyes_closed = avg_ear < EAR_THRESHOLD
                
                # Gözler kapalıysa süreyi başlat veya güncelle
                if eyes_closed:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    else:
                        closed_duration = time.time() - eyes_closed_start_time
                        if closed_duration >= WARNING_THRESHOLD:
                            # Uyarı mesajını ekrana yaz
                            cv2.putText(frame, "SURUCU UYUYOR!", (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            # Ekrana kırmızı uyarı çerçevesi ekle
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                else:
                    # Gözler açıksa süreyi sıfırla
                    eyes_closed_start_time = None
                
                # Göz açıklık oranlarını ekrana yaz
                cv2.putText(frame, f"Sol Goz Mesafe: {left_ear:.4f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Sag Goz Mesafe: {right_ear:.4f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Gözler kapalıysa süreyi göster
                if eyes_closed_start_time is not None:
                    closed_duration = time.time() - eyes_closed_start_time
                    cv2.putText(frame, f"Kapali Kalma Suresi: {closed_duration:.1f}s", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Göz noktalarını çiz
                for eye_points in [LEFT_EYE, RIGHT_EYE]:
                    for point in eye_points:
                        x = int(face_landmarks.landmark[point].x * frame.shape[1])
                        y = int(face_landmarks.landmark[point].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Kırmızı noktalar

        # Görüntüyü göster
        cv2.imshow('Goz Takibi', frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()