import sys
import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from PyQt5.QtMultimedia import QSound
from gaze_tracking import GazeTracking

class WarningLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                background-color: #e74c3c;
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.hide()
        self.animation = QPropertyAnimation(self, b"styleSheet")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
    def start_warning(self):
        self.show()
        self.animation.setStartValue("""
            QLabel {
                background-color: #e74c3c;
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.animation.setEndValue("""
            QLabel {
                background-color: #c0392b;
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.animation.start()

class StatusLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                color: #2c3e50;
                border-radius: 10px;
                padding: 12px;
                margin: 5px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #34495e;
            }
        """)

class DriverMonitoringUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sürücü İzleme Sistemi")
        self.setMinimumSize(1400, 800)  # Minimum pencere boyutunu artırdık
        
        # Ses dosyası için değişken
        self.warning_sound = QSound("warning.mp3")  # Ses dosyasını projenize eklemeyi unutmayın
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QFrame {
                background-color: #f0f0f0;
                border-radius: 15px;
                padding: 15px;
                border: 2px solid #34495e;
            }
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                color: #2c3e50;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                border-radius: 3px;
            }
        """)
        
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
            61, 291, 0, 17, 291, 405, 314, 13, 14
        ]
        
        # Esneme modeli yükleme
        try:
            self.yawn_model = tf.keras.models.load_model('model_yawn_best.h5')
        except:
            print("Esneme modeli yüklenemedi!")
            self.yawn_model = None
        
        # Değişkenler
        self.eyes_closed_start_time = None
        self.WARNING_THRESHOLD = 10  # 10 saniye
        self.EAR_THRESHOLD = 0.015  # Göz açıklık eşiği
        self.YAWN_THRESHOLD = 0.5  # Esneme eşiği
        self.yawn_count = 0
        self.last_yawn_state = False
        
        # Yeni değişkenler ekle
        self.gaze_direction_start_time = None
        self.last_gaze_direction = None
        self.GAZE_WARNING_THRESHOLD = 10  # 10 saniye
        self.ROAD_GAZE_WARNING_THRESHOLD = 3  # 3 saniye
        self.last_road_gaze_time = None
        
        # Yorgunluk eşikleri
        self.FATIGUE_THRESHOLDS = {
            'yawn_count': {
                'low': 3,    # 3 esneme = yorgun
                'high': 6    # 6 esneme = çok yorgun
            },
            'eye_closed_time': {
                'low': 5,    # 5 saniye = yorgun
                'high': 8    # 8 saniye = çok yorgun
            }
        }
        
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title_label = QLabel("SÜRÜCÜ İZLEME SİSTEMİ")
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Ana içerik alanı
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Sol panel (kamera görüntüsü)
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # Kamera görüntüsü için frame
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border-radius: 10px;
                border: 2px solid #34495e;
            }
        """)
        camera_layout = QVBoxLayout(camera_frame)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)  # Kamera görüntüsünü büyüttük
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_label)
        left_layout.addWidget(camera_frame)
        
        # İlerleme çubuğu
        self.warning_progress = QProgressBar()
        self.warning_progress.setMinimum(0)
        self.warning_progress.setMaximum(100)
        self.warning_progress.setValue(0)
        self.warning_progress.setFormat("Uyarı Süresi: %v%")
        self.warning_progress.hide()
        left_layout.addWidget(self.warning_progress)
        
        # Kamera kontrol butonları
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.start_button = QPushButton("🎥 Kamerayı Başlat")
        self.stop_button = QPushButton("⏹ Kamerayı Durdur")
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)
        
        # Sağ panel (durum bilgileri)
        right_panel = QFrame()
        right_panel.setFixedWidth(450)  # Sağ panel genişliğini artırdık
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)  # Boşlukları artırdık
        
        # Uyarı etiketi
        self.warning_label = WarningLabel("DİKKAT! SÜRÜCÜ UYUYOR!")
        right_layout.addWidget(self.warning_label)
        
        # Durum göstergeleri
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-radius: 10px;
                border: 2px solid #34495e;
                padding: 15px;
            }
            QLabel {
                font-size: 15px;  # Yazı boyutunu artırdık
                padding: 8px;     # İç boşluğu artırdık
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        status_layout.setSpacing(12)  # Etiketler arası boşluğu artırdık
        
        self.status_labels = {}
        status_items = [
            ("fatigue_status", "👤 Yorgunluk Durumu: Normal"),
            ("eye_status", "👁 Göz Durumu: Normal"),
            ("yawn_status", "🥱 Esneme Durumu: Normal"),
            ("gaze_status", "👀 Bakış Yönü: Yola Bakıyor"),
            ("warning_status", "⚠️ Uyarı Durumu: Yok"),
            ("closed_time", "⏱ Kapalı Kalma Süresi: 0.0s"),
            ("yawn_count", "📊 Esneme Sayısı: 0")
        ]
        
        for key, text in status_items:
            label = StatusLabel(text)
            self.status_labels[key] = label
            status_layout.addWidget(label)
        
        right_layout.addWidget(status_frame)
        right_layout.addStretch()
        
        # Panelleri ana layout'a ekle
        content_layout.addWidget(left_panel, 2)
        content_layout.addWidget(right_panel, 1)
        layout.addLayout(content_layout)
        
        # Alt bilgi paneli
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        bottom_frame.setFixedHeight(60)
        bottom_layout = QVBoxLayout(bottom_frame)
        
        self.info_label = QLabel("🟡 Sistem hazır. Kamerayı başlatmak için 'Kamerayı Başlat' butonuna tıklayın.")
        self.info_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.info_label.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(self.info_label)
        
        layout.addWidget(bottom_frame)
        
        # Kamera ve timer değişkenleri
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Buton bağlantıları
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        
        # Pencere boyutunu ayarla
        self.setMinimumSize(1200, 700)
    
    def calculate_ear(self, eye_points, landmarks):
        """Göz açıklık oranını hesapla"""
        vertical_dist = np.linalg.norm(np.array([landmarks.landmark[eye_points[0]].x, landmarks.landmark[eye_points[0]].y]) - 
                                     np.array([landmarks.landmark[eye_points[1]].x, landmarks.landmark[eye_points[1]].y]))
        return vertical_dist
    
    def get_mouth_roi(self, frame, landmarks):
        """Ağız bölgesini tespit et ve kırp"""
        h, w = frame.shape[:2]
        mouth_points = []
        for point_idx in self.MOUTH_POINTS:
            landmark = landmarks.landmark[point_idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_points.append((x, y))
        
        x_coords = [p[0] for p in mouth_points]
        y_coords = [p[1] for p in mouth_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        return mouth_roi, (x_min, y_min, x_max, y_max)
    
    def preprocess_mouth(self, mouth_roi):
        """Ağız görüntüsünü model için hazırla"""
        if mouth_roi is None or mouth_roi.size == 0:
            return None
            
        gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)
        return input_data
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)  # 30ms = ~33 FPS
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
    
    def stop_camera(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_label.clear()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def analyze_fatigue(self, yawn_count, max_eye_closed_time):
        """Tüm metrikleri kullanarak yorgunluk durumunu analiz et"""
        fatigue_score = 0
        
        # Esneme sayısına göre puan
        if yawn_count >= self.FATIGUE_THRESHOLDS['yawn_count']['high']:
            fatigue_score += 2
        elif yawn_count >= self.FATIGUE_THRESHOLDS['yawn_count']['low']:
            fatigue_score += 1
        
        # Göz kapalı kalma süresine göre puan
        if max_eye_closed_time >= self.FATIGUE_THRESHOLDS['eye_closed_time']['high']:
            fatigue_score += 2
        elif max_eye_closed_time >= self.FATIGUE_THRESHOLDS['eye_closed_time']['low']:
            fatigue_score += 1
        
        # Yorgunluk durumunu belirle
        if fatigue_score >= 3:
            return "ÇOK YORGUN"
        elif fatigue_score >= 1:
            return "YORGUN"
        else:
            return "NORMAL"
    
    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Frame'i BGR'den RGB'ye çevir
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame)
                
                # GazeTracking'i güncelle
                self.gaze.refresh(frame)
                
                # Varsayılan durumlar
                eye_status = "Normal"
                yawn_status = "Esnemiyor"
                gaze_status = "Yola Bakiyor"
                warning_status = "Yok"
                max_eye_closed_time = 0
                
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
                            self.warning_progress.setValue(0)
                            self.warning_progress.show()
                        else:
                            closed_duration = time.time() - self.eyes_closed_start_time
                            max_eye_closed_time = closed_duration
                            progress = min(int((closed_duration / self.WARNING_THRESHOLD) * 100), 100)
                            self.warning_progress.setValue(progress)
                            
                            if closed_duration >= self.WARNING_THRESHOLD:
                                eye_status = "SURUCU UYUYOR!"
                                warning_status = "DİKKAT!"
                                self.warning_label.start_warning()
                                self.warning_sound.play()  # Sesli uyarı çal
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)
                                self.info_label.setText("🔴 UYARI: Sürücü uyuyor! Lütfen mola verin!")
                    else:
                        if self.eyes_closed_start_time is not None:
                            self.eyes_closed_start_time = None
                            self.warning_progress.hide()
                            self.warning_label.hide()
                            self.info_label.setText("🟢 Sistem aktif. Her şey normal görünüyor.")
                    
                    # 2. Bakış Yönü Analizi
                    if self.gaze.pupils_located:
                        current_gaze = None
                        if self.gaze.is_right():
                            gaze_status = "Saga Bakiyor"
                            current_gaze = "right"
                            self.last_road_gaze_time = None
                        elif self.gaze.is_left():
                            gaze_status = "Sola Bakiyor"
                            current_gaze = "left"
                            self.last_road_gaze_time = None
                        else:
                            gaze_status = "Yola Bakiyor"
                            current_gaze = "center"
                            if self.last_road_gaze_time is None:
                                self.last_road_gaze_time = time.time()
                        
                        # Yola bakmama uyarısı
                        if current_gaze != "center":
                            if self.last_road_gaze_time is not None:
                                time_not_looking_road = time.time() - self.last_road_gaze_time
                                if time_not_looking_road >= self.ROAD_GAZE_WARNING_THRESHOLD:
                                    warning_status = "DİKKAT! YOLA BAKMIYORSUNUZ!"
                                    self.warning_label.start_warning()
                                    self.warning_sound.play()
                                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)
                                    self.info_label.setText("🔴 UYARI: Yola bakmıyorsunuz! Lütfen yola odaklanın!")
                        
                        # Bakış yönü uyarı sistemi
                        if current_gaze in ["left", "right"]:
                            if self.gaze_direction_start_time is None:
                                self.gaze_direction_start_time = time.time()
                                self.last_gaze_direction = current_gaze
                            elif self.last_gaze_direction == current_gaze:
                                gaze_duration = time.time() - self.gaze_direction_start_time
                                if gaze_duration >= self.GAZE_WARNING_THRESHOLD:
                                    warning_status = f"DİKKAT! {current_gaze.upper()} TARAFTA ÇOK UZUN SÜREDİR BAKIYOR!"
                                    self.warning_label.start_warning()
                                    self.warning_sound.play()  # Sesli uyarı çal
                                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)
                                    self.info_label.setText(f"🔴 UYARI: {current_gaze.upper()} tarafa çok uzun süredir bakıyorsunuz!")
                        else:
                            self.gaze_direction_start_time = None
                            self.last_gaze_direction = None
                            self.info_label.setText("🟢 Sistem aktif. Her şey normal görünüyor.")
                        
                        if self.gaze.is_blinking():
                            gaze_status = "Gozler Kapali"
                    
                    # 3. Esneme Analizi
                    if self.yawn_model is not None:
                        mouth_roi, (x1, y1, x2, y2) = self.get_mouth_roi(frame, landmarks)
                        if mouth_roi is not None:
                            processed_mouth = self.preprocess_mouth(mouth_roi)
                            if processed_mouth is not None:
                                yawn_prob = self.yawn_model.predict(processed_mouth, verbose=0)[0][0]
                                is_yawn = yawn_prob > self.YAWN_THRESHOLD
                                
                                if is_yawn and not self.last_yawn_state:
                                    self.yawn_count += 1
                                self.last_yawn_state = is_yawn
                                
                                yawn_status = "Esniyor" if is_yawn else "Esnemiyor"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Göz noktalarını çiz
                    for eye_points in [self.LEFT_EYE, self.RIGHT_EYE]:
                        for point in eye_points:
                            x = int(landmarks.landmark[point].x * frame.shape[1])
                            y = int(landmarks.landmark[point].y * frame.shape[0])
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                
                # Yorgunluk analizi
                fatigue_status = self.analyze_fatigue(self.yawn_count, max_eye_closed_time)
                
                # Durumları güncelle
                self.status_labels["fatigue_status"].setText(f"👤 Yorgunluk Durumu: {fatigue_status}")
                self.status_labels["eye_status"].setText(f"👁 Göz Durumu: {eye_status}")
                self.status_labels["yawn_status"].setText(f"🥱 Esneme Durumu: {yawn_status}")
                self.status_labels["gaze_status"].setText(f"👀 Bakış Yönü: {gaze_status}")
                self.status_labels["warning_status"].setText(f"⚠️ Uyarı Durumu: {warning_status}")
                self.status_labels["yawn_count"].setText(f"📊 Esneme Sayısı: {self.yawn_count}")
                
                if self.eyes_closed_start_time is not None:
                    closed_duration = time.time() - self.eyes_closed_start_time
                    self.status_labels["closed_time"].setText(f"⏱ Kapalı Kalma Süresi: {closed_duration:.1f}s")
                else:
                    self.status_labels["closed_time"].setText("⏱ Kapalı Kalma Süresi: 0.0s")
                
                # Frame'i göster
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DriverMonitoringUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 