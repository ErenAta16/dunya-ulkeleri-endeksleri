import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("prashantdhanuk/global-country-metrics-2025-hdi-gdp-pop-area")

print("Path to dataset files:", path)

# List files in the dataset directory
print("\nFiles in the dataset:")
for file in os.listdir(path):
    print(f"- {file}")

# Try to load and preview the data
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    print(f"\nLoading {csv_files[0]}...")
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())
    
    print(f"\nBasic statistics:")
    print(df.describe())
else:
    print("No CSV files found in the dataset.") 