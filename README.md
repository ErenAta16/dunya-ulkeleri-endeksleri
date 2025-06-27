# 🌍 HDI Tahmin Modeli - Dünya Ülkeleri Endeksleri 2025

Bu proje, **2025 yılı dünya ülkeleri verilerini** kullanarak **İnsani Gelişme Endeksi (HDI)** tahmin eden kapsamlı bir makine öğrenmesi projesidir.

## 📊 Proje Özeti

- **Dataset**: 194 ülke, 8 özellik
- **Model**: Random Forest Regressor
- **Doğruluk**: **%82.9** (R² = 0.829)
- **Özellik Sayısı**: 11 (6 orijinal + 5 türetilmiş)
- **Görselleştirme**: 6 farklı kapsamlı analiz

## 🎯 Model Performansı

| Metrik | Değer |
|--------|-------|
| **Test R²** | 0.829 (%82.9) |
| **Test RMSE** | 0.084 |
| **Test MAE** | 0.043 |
| **Test MSE** | 0.007 |

## 📈 En Önemli Özellikler

1. **GDP Per capita PPP**: %78.9
2. **Nominal GDP Per capita**: %14.2
3. **GINI**: %2.0

## 🗂️ Dosya Yapısı

```
📁 BM Ülke Ölçümleri 2025/
├── 🐍 enhanced_model.py              # Ana model eğitim kodu
├── 🐍 model_usage_example.py         # Model kullanım örneği
├── 🐍 Veri_seti_analizi.py          # Veri seti keşif analizi
├── 🐍 Veri_seti_istatistikleri.py   # Temel istatistikler
├── 🤖 enhanced_hdi_model_*.pkl       # Eğitilmiş model dosyası
├── 📊 enhanced_correlation_heatmap.png
├── 📊 enhanced_histograms.png
├── 📊 enhanced_top_hdi_countries.png
├── 📊 enhanced_feature_importance.png
├── 📊 enhanced_actual_vs_predicted.png
└── 📊 enhanced_model_dashboard.png
```

## 🚀 Kullanım

### 1. Model Eğitimi
```bash
python enhanced_model.py
```

### 2. Model Kullanımı
```python
import joblib
import numpy as np

# Model yükleme
model_data = joblib.load('enhanced_hdi_model_*.pkl')
model = model_data['model']
features = model_data['features']

# Örnek tahmin
example_country = {
    'Population (in millions)': 84.0,
    'GDP Per capita PPP (in USD)': 28000.0,
    'Nominal GDP Per capita (in USD)': 9000.0,
    'GINI': 41.0,
    # ... diğer özellikler
}

predicted_hdi = model.predict([feature_values])[0]
print(f"Tahmin edilen HDI: {predicted_hdi:.3f}")
```

## 📊 Görselleştirmeler

### 1. 🔥 Korelasyon Haritası
- Özellikler arası ilişkilerin ısı haritası
- HDI ile en yüksek korelasyonlu özellikler

### 2. 📈 Histogramlar
- Tüm sayısal özelliklerin dağılım grafikleri
- Veri kalitesi ve dağılım analizi

### 3. 🏆 Top HDI Ülkeleri
- En yüksek HDI'ya sahip 20 ülke
- Detaylı HDI değerleri

### 4. ⚡ Özellik Önemleri
- Random Forest model özellik önemleri
- GDP Per capita PPP dominansı

### 5. 🎯 Gerçek vs Tahmin
- Model tahmin performansı
- R², RMSE, MAE metrikleri

### 6. 📋 Kapsamlı Dashboard
- 4 panelli özet görünüm
- Model performansı ve veri dağılımı

## 🔬 Feature Engineering

Orijinal 6 özellikten **11 özellik** türetildi:

**Orijinal Özellikler:**
- Population (in millions)
- Nominal GDP Per capita (in USD)
- GDP Per capita PPP (in USD)
- GINI
- AREA (in Sq km)

**Türetilmiş Özellikler:**
- `GDP_PPP_Ratio`: Nominal/PPP oranı
- `Population_Density`: Nüfus yoğunluğu
- `GDP_per_Area`: Alan başına GDP
- `Log_Population`: Log dönüşümü
- `Log_GDP`: Log dönüşümü
- `Log_Area`: Log dönüşümü

## 📋 Gereksinimler

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

## 🔧 Kurulum

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## 📊 Veri Seti

- **Kaynak**: Kaggle - Global Country Metrics 2025
- **Ülke Sayısı**: 194 (193 BM üyesi + Vatikan + Filistin)
- **Özellikler**: HDI, GDP, Nüfus, Alan, GINI
- **Eksik Veri**: %10.3 (GINI), %1.0 (HDI)

## 🎯 Model Başarısı

**En İyi Tahminler:**
- Uruguay: Hata 0.006
- Danimarka: Hata 0.008
- İsveç: Hata 0.012

**En Zor Tahminler:**
- Niger: Hata 0.438
- Çad: Hata 0.284
- Mali: Hata 0.201

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Yazar

**Eren Ata** - [ErenAta16](https://github.com/ErenAta16)

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
