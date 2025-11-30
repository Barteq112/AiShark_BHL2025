import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OZE_COLUMNS = [
    'Biomass',
    'Hydro Run-of-river and poundage',
    'Hydro Water Reservoir',
    'Solar',
    'Wind Onshore'
]
ENERGY_TIME_COL = 'start_time'
WEATHER_TIME_COL = 'czas'

HORYZONT_PRZYSZLOSC = 72  # Target: 72h w przód
HISTORIA_LAGI = 24        # Input: 24h w tył

# --- 2. WCZYTYWANIE DANYCH ---
print("Wczytywanie danych...")
df_energy = pd.read_csv('Kraje_2022_2025/dane_energia_Polska.csv')
df_weather = pd.read_csv('Kraje_2022_2025/polska_pogoda_godzinowa.csv')

# --- 3. PREPROCESSING ---
df_energy[ENERGY_TIME_COL] = pd.to_datetime(df_energy[ENERGY_TIME_COL])
df_weather[WEATHER_TIME_COL] = pd.to_datetime(df_weather[WEATHER_TIME_COL])

df_energy = df_energy.sort_values(ENERGY_TIME_COL)
df_weather = df_weather.sort_values(WEATHER_TIME_COL)

# Obliczenie OZE_Share
dostepne_oze = [c for c in OZE_COLUMNS if c in df_energy.columns]
df_energy['OZE_Share'] = df_energy[dostepne_oze].sum(axis=1).clip(0, 1)
df_energy = df_energy.dropna(subset=['OZE_Share'])

# --- 4. MERGE ---
df = pd.merge(
    df_energy,
    df_weather,
    left_on=ENERGY_TIME_COL,
    right_on=WEATHER_TIME_COL,
    how='inner',
    suffixes=('_energy', '_weather')
)

# --- 5. FEATURE ENGINEERING ---
print("Generowanie cech...")
# Czas
df['hour'] = df[ENERGY_TIME_COL].dt.hour
df['month'] = df[ENERGY_TIME_COL].dt.month
df['dayofyear'] = df[ENERGY_TIME_COL].dt.dayofyear

def add_cyclical_features(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(df[col_name] * (2. * np.pi / period))
    df[f'{col_name}_cos'] = np.cos(df[col_name] * (2. * np.pi / period))
    return df

df = add_cyclical_features(df, 'hour', 24)
df = add_cyclical_features(df, 'month', 12)
df = add_cyclical_features(df, 'dayofyear', 365.25)

print("Przykład danych: ")
print(df.head())
print("Info:")
print(df.info())

# --- 6. KONFIGURACJA ANALIZY ---

# TUTAJ ZMIENIASZ PRZESUNIĘCIE (w godzinach)
TARGET_SHIFT = 1  # Np. 1, 24, 48 - o ile godzin w przód chcemy przewidywać
TARGET_COL_BASE = 'OZE_Share'
TARGET_NAME = f'Target_{TARGET_COL_BASE}_{TARGET_SHIFT}h'

# Tworzenie folderu na wyniki
OUTPUT_DIR = 'analiza'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Wyniki zostaną zapisane w folderze: {OUTPUT_DIR}/")

# --- 7. PRZYGOTOWANIE DANYCH ---

# Tworzenie Targetu: OZE_Share przesunięte o TARGET_SHIFT godzin w przód
# shift(-N) bierze wartość z wiersza o N pozycji niżej
df[TARGET_NAME] = df[TARGET_COL_BASE].shift(-TARGET_SHIFT)

# Usunięcie wierszy z NaN (ostatnie wiersze nie mają przyszłości do przewidzenia)
df_analysis = df.dropna().copy()

# Wybór tylko kolumn numerycznych
numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns

# --- 8. OBLICZANIE KORELACJI ---

print(f"Obliczanie korelacji dla przesunięcia: {TARGET_SHIFT}h...")

# Macierz korelacji Pearsona (liniowa)
corr_matrix_pearson = df_analysis[numeric_cols].corr(method='pearson')
# Macierz korelacji Spearmana (monotoniczna)
corr_matrix_spearman = df_analysis[numeric_cols].corr(method='spearman')

# Wyciągnięcie korelacji z Targetem i posortowanie
# drop(TARGET_NAME) usuwa autokorelację (korelację targetu samego ze sobą równą 1)
target_corr_pearson = corr_matrix_pearson[TARGET_NAME].drop(TARGET_NAME).sort_values()
target_corr_spearman = corr_matrix_spearman[TARGET_NAME].drop(TARGET_NAME).sort_values()


# --- 9. WIZUALIZACJA I ZAPISYWANIE ---

def save_and_show_barplot(corr_series, title, filename, color):
    plt.figure(figsize=(12, 8))
    corr_series.plot(kind='barh', color=color, edgecolor='black')
    plt.title(title, fontsize=14)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlabel('Współczynnik korelacji')
    plt.tight_layout()

    # Zapis do pliku
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f"Zapisano wykres: {save_path}")
    plt.show()


# 1. Wykres Pearsona
save_and_show_barplot(
    target_corr_pearson,
    f'Korelacja Pearsona z {TARGET_NAME} (Liniowa)',
    f'Pearson_{TARGET_NAME}.png',
    'skyblue'
)

# 2. Wykres Spearmana
save_and_show_barplot(
    target_corr_spearman,
    f'Korelacja Spearmana z {TARGET_NAME} (Monotoniczna)',
    f'Spearman_{TARGET_NAME}.png',
    'lightgreen'
)

# 3. Macierz Korelacji (Heatmap)
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix_pearson, dtype=bool))

sns.heatmap(corr_matrix_pearson,
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            square=True)

plt.title(f'Macierz Korelacji Pearsona (Shift {TARGET_SHIFT}h)', fontsize=16)
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, f'Matrix_Pearson_{TARGET_NAME}.png')
plt.savefig(heatmap_path)
print(f"Zapisano macierz: {heatmap_path}")
plt.show()

# --- 10. PODSUMOWANIE TEKSTOWE ---
print(f"\n--- Top 5 korelacji (Pearson) dla przesunięcia {TARGET_SHIFT}h ---")
print(pd.concat([target_corr_pearson.head(5), target_corr_pearson.tail(5)]))