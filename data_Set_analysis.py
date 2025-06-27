import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=== BM Ülke Ölçümleri 2025 Veri Seti Analizi ===\n")

# Veri setini indir
print("Veri seti indiriliyor...")
path = kagglehub.dataset_download("prashantdhanuk/global-country-metrics-2025-hdi-gdp-pop-area")
print(f"İndirilen dosya yolu: {path}")

# Dosyaları listele
print("\nVeri setindeki dosyalar:")
files = os.listdir(path)
for file in files:
    print(f"- {file}")

# CSV dosyasını yükle
csv_files = [f for f in files if f.endswith('.csv')]
if csv_files:
    print(f"\n'{csv_files[0]}' dosyası yükleniyor...")
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    print(f"\n📊 VERİ SETİ GENEL BİLGİLERİ:")
    print(f"- Toplam ülke sayısı: {len(df)}")
    print(f"- Sütun sayısı: {len(df.columns)}")
    print(f"- Veri seti boyutu: {df.shape}")
    
    print(f"\n📋 SÜTUN ADLARI:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\n🔍 İLK 5 SATIR:")
    print(df.head())
    
    print(f"\n📈 VERİ TİPLERİ VE EKSİK DEĞERLER:")
    print(df.info())
    
    print(f"\n📊 SAYISAL VERİLER İÇİN İSTATİSTİKLER:")
    print(df.describe())
    
    # Eksik değer analizi
    print(f"\n❌ EKSİK DEĞER ANALİZİ:")
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_table = pd.DataFrame({
        'Eksik Değer Sayısı': missing_data,
        'Eksik Değer Yüzdesi': missing_percent
    })
    missing_table = missing_table[missing_table['Eksik Değer Sayısı'] > 0].sort_values('Eksik Değer Sayısı', ascending=False)
    if not missing_table.empty:
        print(missing_table)
    else:
        print("Hiç eksik değer bulunamadı!")
    
    # Veri setini global değişken olarak sakla
    globals()['country_data'] = df
    
    print(f"\n✅ Veri seti başarıyla yüklendi ve 'country_data' değişkenine atandı.")
    print(f"📝 Analiz için kullanabileceğiniz komutlar:")
    print(f"   - country_data.head() : İlk 5 satırı göster")
    print(f"   - country_data.info() : Veri tipi bilgileri")
    print(f"   - country_data.describe() : İstatistiksel özet")
    print(f"   - country_data.columns : Sütun adları")

else:
    print("❌ CSV dosyası bulunamadı!") 