import cv2
import numpy as np
from gaze_tracking import GazeTracking

def main():
    # Kamerayı başlat
    webcam = cv2.VideoCapture(0)
    
    # GazeTracking nesnesini oluştur
    gaze = GazeTracking()
    
    while True:
        # Kameradan frame al
        _, frame = webcam.read()
        
        # GazeTracking'i yenile
        gaze.refresh(frame)
        
        # Frame'i göz takibi için işle
        frame = gaze.annotated_frame()
        
        # Göz yönünü belirle
        if gaze.pupils_located:
            if gaze.is_right():
                text = "Saga Bakiyor"
            elif gaze.is_left():
                text = "Sola Bakiyor"
            else:
                text = "Yola Bakiyor"
                
            # Göz açıklık oranlarını al
            left_ratio = gaze.eye_left.blinking if gaze.eye_left else 0
            right_ratio = gaze.eye_right.blinking if gaze.eye_right else 0
            
            # Gözler kapalıysa
            if gaze.is_blinking():
                text = "Gozler Kapali"
        else:
            text = "Yuz Tespit Edilemedi"
            left_ratio = 0
            right_ratio = 0
        
        # Sonuçları ekrana yazdır
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.putText(frame, f"Sol Goz Orani: {left_ratio:.2f}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 2)
        cv2.putText(frame, f"Sag Goz Orani: {right_ratio:.2f}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 2)
        
        # Frame'i göster
        cv2.imshow("Goz Takibi", frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 