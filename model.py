import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("=== HDI TAHMİN MODELİ ===")
print("Kapsamlı Görselleştirmelerle HDI Tahmin Modeli\n")

# Veri yükleme
cache_path = r"C:\Users\erena\.cache\kagglehub\datasets\prashantdhanuk\global-country-metrics-2025-hdi-gdp-pop-area\versions\1\countries_metric - Sheet1.csv"

try:
    df = pd.read_csv(cache_path, encoding='utf-8')
    print(f"✅ Veri yüklendi: {df.shape}")
except:
    print("❌ Veri yüklenemedi!")
    exit()

def convert_numeric(x):
    try:
        if isinstance(x, str):
            x = x.replace('$', '').replace(',', '')
            if 'trillion' in x.lower():
                num = float(x.lower().replace('trillion', '').strip()) * 1e12
            elif 'billion' in x.lower():
                num = float(x.lower().replace('billion', '').strip()) * 1e9
            elif 'million' in x.lower():
                num = float(x.lower().replace('million', '').strip()) * 1e6
            else:
                num = float(x.strip())
            return num
        return x
    except Exception:
        return np.nan

# Veri ön işleme
print("🔧 Veri ön işleme...")
cols_to_convert = ['Population (in millions)',
                   'Nominal Gross Domestic Product (in USD)',
                   'Nominal GDP Per capita (in USD)',
                   'GDP Per capita PPP (in USD)',
                   'AREA (in Sq km)']

for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].apply(convert_numeric)

# Feature Engineering
print("🔧 Feature Engineering...")

def safe_divide(a, b):
    return np.where((b != 0) & (~np.isnan(b)), a / b, np.nan)

df['GDP_PPP_Ratio'] = safe_divide(df['Nominal GDP Per capita (in USD)'], df['GDP Per capita PPP (in USD)'])
df['Population_Density'] = safe_divide(df['Population (in millions)'] * 1e6, df['AREA (in Sq km)'])
df['GDP_per_Area'] = safe_divide(df['Nominal Gross Domestic Product (in USD)'], df['AREA (in Sq km)'])
df['Log_Population'] = np.log1p(df['Population (in millions)'].fillna(0))
df['Log_GDP'] = np.log1p(df['Nominal Gross Domestic Product (in USD)'].fillna(0))
df['Log_Area'] = np.log1p(df['AREA (in Sq km)'].fillna(0))

features = [
    'Population (in millions)', 'Log_Population',
    'Nominal GDP Per capita (in USD)', 'GDP Per capita PPP (in USD)',
    'GDP_PPP_Ratio', 'Population_Density', 'GDP_per_Area',
    'Log_GDP', 'Log_Area', 'GINI', 'AREA (in Sq km)'
]

available_features = [f for f in features if f in df.columns]
print(f"✅ Kullanılacak özellik sayısı: {len(available_features)}")

# Veri temizleme
data = df[available_features + ['Human Development Index (HDI)', 'country_name']].copy()
data = data.replace([np.inf, -np.inf], np.nan)
data_clean = data.dropna()

print(f"📊 Final veri seti: {len(data_clean)} ülke")

# VİZÜALİZASYONLAR
print("\n🎨 KAPSAMLI VİZÜALİZASYONLAR OLUŞTURULUYOR...")

# 1. Korelasyon Haritası
numeric_df = data_clean.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Özellikler Arası Korelasyon Haritası', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Korelasyon haritası: correlation_heatmap.png")

# 2. Histogramlar
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 16))
axes = axes.ravel()

for i, col in enumerate(numeric_df.columns):
    if i < len(axes):
        axes[i].hist(numeric_df[col].dropna(), bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel('Frekans', fontsize=8)
        axes[i].grid(True, alpha=0.3)

for i in range(len(numeric_df.columns), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Tüm Sayısal Özelliklerin Dağılımları', fontsize=16)
plt.tight_layout()
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Histogramlar: histograms.png")

# 3. En Yüksek HDI'lı 20 Ülke
top_hdi = data_clean.nlargest(20, 'Human Development Index (HDI)')
plt.figure(figsize=(12, 10))
sns.barplot(data=top_hdi, y='country_name', x='Human Development Index (HDI)', 
            palette='viridis', orient='h')
plt.title('En Yüksek İnsani Gelişme Endeksi (HDI) - İlk 20 Ülke', fontsize=14, pad=20)
plt.xlabel('HDI Değeri', fontsize=12)
plt.ylabel('Ülke', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top_hdi_countries.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Top HDI ülkeleri: top_hdi_countries.png")

# MODEL EĞİTİMİ
X = data_clean[available_features]
y = data_clean['Human Development Index (HDI)']
countries = data_clean['country_name']

X_train, X_test, y_train, y_test, countries_train, countries_test = train_test_split(
    X, y, countries, test_size=0.2, random_state=42
)

print(f"\n🎯 MODEL EĞİTİMİ:")
print(f"  Eğitim seti: {len(X_train)} ülke")
print(f"  Test seti: {len(X_test)} ülke")

final_model = RandomForestRegressor(random_state=42, n_estimators=100)
final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*50)
print("🏆 MODEL PERFORMANSI")
print("="*50)
print(f"Eğitim R²: {train_r2:.6f} ({train_r2*100:.2f}%)")
print(f"Test R²: {test_r2:.6f} ({test_r2*100:.2f}%)")
print(f"Test MSE: {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# 4. Feature Importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Özellik Önemleri - Random Forest Model', fontsize=14, pad=20)
plt.xlabel('Önem Derecesi', fontsize=12)
plt.ylabel('Özellikler', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Özellik önemleri: feature_importance.png")

# 5. Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.7, s=60, color='darkblue', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Mükemmel Tahmin')

plt.text(0.05, 0.95, f'R² = {test_r2:.3f}\nRMSE = {test_rmse:.3f}\nMAE = {test_mae:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.xlabel('Gerçek HDI Değerleri', fontsize=12)
plt.ylabel('Tahmin Edilen HDI Değerleri', fontsize=12)
plt.title(f'Gerçek vs Tahmin Edilen HDI Değerleri\nModel Performansı: {test_r2*100:.1f}%', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gerçek vs Tahmin: actual_vs_predicted.png")

# 6. Kapsamlı Dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Feature Importance (Top 8)
top_features = feature_importance.head(8)
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis', ax=axes[0,0])
axes[0,0].set_title('En Önemli 8 Özellik')
axes[0,0].set_xlabel('Önem Derecesi')

# Actual vs Predicted
axes[0,1].scatter(y_test, y_pred_test, alpha=0.7, color='blue')
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,1].set_xlabel('Gerçek HDI')
axes[0,1].set_ylabel('Tahmin HDI')
axes[0,1].set_title(f'Gerçek vs Tahmin (R²={test_r2:.3f})')
axes[0,1].grid(True, alpha=0.3)

# HDI Distribution
axes[1,0].hist(y, bins=20, alpha=0.7, color='green', edgecolor='black')
axes[1,0].set_xlabel('HDI Değerleri')
axes[1,0].set_ylabel('Frekans')
axes[1,0].set_title('HDI Değerlerinin Dağılımı')
axes[1,0].grid(True, alpha=0.3)

# Performance Metrics
metrics_text = f"""MODEL PERFORMANSI

Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)
Test RMSE: {test_rmse:.4f}
Test MAE: {test_mae:.4f}
Test MSE: {test_mse:.6f}

Overfitting: {train_r2 - test_r2:.4f}

Eğitim Seti: {len(X_train)} ülke
Test Seti: {len(X_test)} ülke
Toplam: {len(data_clean)} ülke"""

axes[1,1].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[1,1].set_xlim(0, 1)
axes[1,1].set_ylim(0, 1)
axes[1,1].axis('off')
axes[1,1].set_title('Model Özet İstatistikleri')

plt.suptitle('HDI Tahmin Modeli - Kapsamlı Dashboard', fontsize=16)
plt.tight_layout()
plt.savefig('model_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Model dashboard: model_dashboard.png")

# Model kaydetme
model_filename = f"hdi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
model_package = {
    'model': final_model,
    'features': available_features,
    'feature_importance': feature_importance,
    'metrics': {
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }
}

joblib.dump(model_package, model_filename)
print(f"\n💾 Model kaydedildi: {model_filename}")

print("\n" + "="*60)
print("🎉 HDI TAHMİN MODELİ HAZIR!")
print("="*60)
print(f"📁 Oluşturulan görselleştirmeler:")
print(f"  1. correlation_heatmap.png")
print(f"  2. histograms.png")
print(f"  3. top_hdi_countries.png")
print(f"  4. feature_importance.png")
print(f"  5. actual_vs_predicted.png")
print(f"  6. model_dashboard.png")
print(f"📊 Model performansı: {test_r2*100:.1f}% doğruluk") 