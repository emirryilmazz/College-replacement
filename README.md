# College Student Placement Predictor

Basit ve modern bir Streamlit arayüzü ile `model/pipe.pkl` içindeki eğitilmiş pipeline kullanarak üniversite öğrencilerinin işe yerleştirilme durumunu tahmin eder.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## Kullanım
- Öğrenci bilgilerini girin (IQ, CGPA, akademik performans vb.).
- Staj deneyimi ve proje sayısını belirtin.
- "Yerleştirme Durumunu Tahmin Et" butonuna basın.

## Özellikler
- **IQ**: Zeka seviyesi (40-158)
- **Prev_Sem_Result**: Önceki dönem sonucu (5.0-10.0)
- **CGPA**: Genel not ortalaması (5.0-10.0)
- **Academic_Performance**: Akademik performans (1-10)
- **Internship_Experience**: Staj deneyimi (Yes/No)
- **Extra_Curricular_Score**: Ders dışı etkinlik skoru (1-10)
- **Communication_Skills**: İletişim becerileri (1-10)
- **Projects_Completed**: Tamamlanan proje sayısı (0-10)

Uygulama, modelin beklediği özellik formatını otomatik olarak oluşturur ve öğrencinin işe yerleştirilip yerleştirilmeyeceğini tahmin eder.
