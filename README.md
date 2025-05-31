# Sürücü İzleme Sistemi (Driver Monitoring System)

Bu proje, sürücülerin yorgunluk ve dikkat durumlarını gerçek zamanlı olarak izleyen bir yapay zeka tabanlı sistemdir. Sistem, bilgisayar görüşü ve makine öğrenmesi tekniklerini kullanarak sürücünün göz hareketlerini, esneme durumunu ve bakış yönünü analiz eder.

## Özellikler

- 🔍 **Göz Takibi**: Sürücünün gözlerinin açık/kapalı durumunu tespit eder
- ⏱ **Uyku Tespiti**: Gözlerin 10 saniyeden fazla kapalı kalması durumunda uyarı verir
- 🥱 **Esneme Tespiti**: Sürücünün esneme durumunu tespit eder ve sayar
- 👀 **Bakış Yönü Analizi**: Sürücünün yola bakıp bakmadığını kontrol eder
- 🔊 **Sesli Uyarılar**: Tehlikeli durumlarda sesli uyarı verir
- 🎥 **Gerçek Zamanlı İzleme**: Kamera görüntüsü üzerinden anlık analiz yapar
- 📊 **Durum Göstergeleri**: Tüm metrikleri kullanıcı dostu bir arayüzde gösterir

## Gereksinimler

- Python 3.10.0
- Webcam
- Ses çıkışı (uyarılar için)

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/surucu-izleme-sistemi.git
cd surucu-izleme-sistemi
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Aşağıdaki Drive linkinden modeli indirip çalıştıracağınız python dosyayı ile aynı dizine yükleyin:
   https://drive.google.com/file/d/1BkuZJfxNAEtxISvZiTTGC7oFMAxmoE9N/view?usp=sharing
   
5. Ses dosyasını ekleyin:
- `warning.mp3` dosyasını projenin ana dizinine ekleyin


## Kullanım

Sistemi başlatmak için:

```bash
python driver_ui_kopya2.py
```

### Kontroller

- 🎥 **Kamerayı Başlat**: Kamera görüntüsünü başlatır
- ⏹ **Kamerayı Durdur**: Kamera görüntüsünü durdurur
- ❌ **Çıkış**: Programı kapatır (pencereyi kapatarak)

### Uyarı Durumları

- 🔴 **Kırmızı Uyarı**: Sürücü uyuyor veya yola bakmıyor
- 🟡 **Sarı Uyarı**: Yorgunluk belirtileri tespit edildi
- 🟢 **Yeşil**: Her şey normal

## Proje Yapısı

- `driver_ui_kopya2.py`: Ana uygulama ve kullanıcı arayüzü
- `driver_monitoring.py`: Temel izleme ve analiz fonksiyonları
- `model_yawn_best.h5`: Esneme tespiti için eğitilmiş model
- `warning.mp3`: Uyarı sesi
- `requirements.txt`: Gerekli Python paketleri
- `README.md`: Proje dokümantasyonu

## Teknik Detaylar

- **Göz Tespiti**: MediaPipe Face Mesh kullanılarak yapılır
- **Esneme Tespiti**: Özel eğitilmiş TensorFlow modeli kullanılır
- **Bakış Yönü**: GazeTracking kütüphanesi ile analiz edilir
- **Arayüz**: PyQt5 ile geliştirilmiş modern ve kullanıcı dostu

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: Açıklama'`)
4. Dalınıza push yapın (`git push origin yeni-ozellik`)
5. Bir Pull Request oluşturun

## İletişim

Sorularınız veya önerileriniz için bir issue açabilirsiniz.

