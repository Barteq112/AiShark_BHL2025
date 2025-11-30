import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # Do zapisu modelu
import time
import warnings
import os

# --- 0. KONFIGURACJA ŚRODOWISKA ---
warnings.filterwarnings("ignore")
os.makedirs('model_output', exist_ok=True)
os.makedirs('wykresy/xgboost', exist_ok=True)

# --- 1. KONFIGURACJA ---
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
HISTORIA_LAGI = 24       # Input: 24h w tył

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

# B. Definicja zmiennych do modelu (STRICT)
# Lista z Twojej analizy korelacji
selected_features_names = [
    'OZE_Share',
    'Fossil Hard coal',
    'Fossil Brown coal/Lignite',
    'Solar',
    'wiatr',
    'wilgotność',
    'Tsun_min',
    'temperatura',
    'Fossil Gas',
    'hour_cos'
]

# Sprawdzenie dostępności kolumn
features_to_use = [c for c in selected_features_names if c in df.columns]
print(f"Zmienne bazowe wybrane do nauki ({len(features_to_use)}):")
print(features_to_use)

# --- 6. TWORZENIE LAGÓW I TARGETÓW ---
print(f"Generowanie lagów dla wybranych zmiennych...")

# Targety (72h w przód)
target_dfs = [df['OZE_Share'].shift(-h).rename(f'target_{h}h') for h in range(1, HORYZONT_PRZYSZLOSC + 1)]
target_names = [f'target_{h}h' for h in range(1, HORYZONT_PRZYSZLOSC + 1)]

# Lagi (tylko dla features_to_use)
lag_dfs = []
lag_names = []
for h in range(1, HISTORIA_LAGI + 1):
    tmp = df[features_to_use].shift(h).add_suffix(f'_lag_{h}h')
    lag_dfs.append(tmp)
    lag_names.extend(tmp.columns.tolist())

# Złączenie
# df zawiera zmienne bazowe, target_dfs to cele, lag_dfs to historia
df_final = pd.concat([df] + target_dfs + lag_dfs, axis=1)
df_final = df_final.dropna()

# --- DEFINICJA WEJŚCIA (X) - STRICT ---
# Bierzemy TYLKO: bieżące wartości wybranych cech + ich lagi
# IGNORUJEMY: hour_sin, hour_cos, month, time itd.
input_features = features_to_use + lag_names

print(f"\nLiczba cech wejściowych (X): {len(input_features)}")
print("Przykładowe cechy wejściowe:", input_features[:5], "...", input_features[-5:])
X = df_final[input_features]
y = df_final[target_names]
timestamps = df_final[ENERGY_TIME_COL] # Zachowujemy czas dla analizy

# --- 7. PODZIAŁ DANYCH (Train / Val / Test) ---
# Podział chronologiczny: 70% Trening, 15% Walidacja, 15% Test
n = len(df_final)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
time_test = timestamps.iloc[val_end:] # Czasy dla zbioru testowego

print(f"Liczba próbek - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- 8. NORMALIZACJA ---
scaler = MinMaxScaler()
# Fit tylko na treningu!
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 9. TRENING ---
print("\nRozpoczynam trening (XGBoost MultiOutput)...")
start_time = time.time()

xgb_params = {
    'n_estimators': 800,
    'learning_rate': 0.03,
    'max_depth': 6,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda', # Zmień na 'cpu' jeśli nie masz GPU NVIDIA
    'random_state': 42,
    'n_jobs': 1
}

model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params), n_jobs=1)
model.fit(X_train_scaled, y_train)

print(f"Czas treningu: {time.time() - start_time:.2f} s")

# Zapis modelu
joblib.dump(model, 'model_output/xgboost_multioutput.pkl')
joblib.dump(scaler, 'model_output/scaler.pkl')
print("Model i scaler zapisane w folderze 'model_output'.")

# --- 10. PREDYKCJA I EWALUACJA (TEST SET) ---
print("\nGenerowanie predykcji dla zbioru testowego...")
# Predykcja
preds_test = model.predict(X_test_scaled)
preds_test = np.clip(preds_test, 0, 1)
preds_df = pd.DataFrame(preds_test, columns=target_names, index=y_test.index)

# --- 11. ANALIZA WYNIKÓW ---

# A. Metryki ogólne
mae = mean_absolute_error(y_test, preds_df)
rmse = np.sqrt(mean_squared_error(y_test, preds_df))
r2 = r2_score(y_test, preds_df)

print("\n--- WYNIKI DLA ZBIORU TESTOWEGO ---")
print(f"MAE (średni błąd absolutny): {mae:.4f}")
print(f"RMSE (błąd średniokwadratowy): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# B. Analiza średnich (Agregacja 72h)
# Obliczamy średnią wartość OZE w horyzoncie 72h dla każdego momentu startowego
# To mówi nam: "Ile średnio energii OZE będzie przez najbliższe 3 dni?"
y_test_mean_72h = y_test.mean(axis=1)
preds_mean_72h = preds_df.mean(axis=1)

analysis_df = pd.DataFrame({
    'czas_startu': time_test,
    'Rzeczywista_Srednia_72h': y_test_mean_72h.values,
    'Prognoza_Srednia_72h': preds_mean_72h.values
})
analysis_df['Blad_Sredniej'] = analysis_df['Prognoza_Srednia_72h'] - analysis_df['Rzeczywista_Srednia_72h']

# Zapis wyników do CSV
analysis_df.to_csv('model_output/analiza_testowa_srednie.csv', index=False)
preds_df.to_csv('model_output/pelne_predykcje_test.csv') # Uwaga: duży plik

# --- 12. GENEROWANIE WYKRESÓW ---
print("\nGenerowanie wykresów...")
plt.style.use('bmh') # Ładniejszy styl wykresów

# Wykres 1: Rzeczywista vs Prognozowana średnia (scatter plot)
plt.figure(figsize=(10, 10))
plt.scatter(analysis_df['Rzeczywista_Srednia_72h'], analysis_df['Prognoza_Srednia_72h'], alpha=0.3, s=10)
plt.plot([0, 1], [0, 1], 'r--') # Linia idealna
plt.xlabel('Rzeczywista Średnia OZE (najbliższe 72h)')
plt.ylabel('Prognozowana Średnia OZE (najbliższe 72h)')
plt.title(f'Korelacja prognozy 3-dniowej (R2={r2:.3f})')
plt.savefig('wykresy/xgboost/1_scatter_srednia.png')
plt.close()

# Wykres 2: Szereg czasowy (ostatnie 500 punktów testowych)
plt.figure(figsize=(15, 6))
subset = 500
plt.plot(analysis_df['czas_startu'].tail(subset), analysis_df['Rzeczywista_Srednia_72h'].tail(subset), label='Realna Średnia 72h', color='black', alpha=0.7)
plt.plot(analysis_df['czas_startu'].tail(subset), analysis_df['Prognoza_Srednia_72h'].tail(subset), label='Prognoza Modelu 72h', color='blue', alpha=0.8)
plt.title('Porównanie średniej generacji OZE w oknie 72h (Szereg Czasowy)')
plt.legend()
plt.savefig('wykresy/xgboost/2_szereg_czasowy.png')
plt.close()

# Wykres 3: Błąd w zależności od horyzontu (1h vs 72h)
mae_per_hour = []
for h in range(1, HORYZONT_PRZYSZLOSC + 1):
    col = f'target_{h}h'
    mae_per_hour.append(mean_absolute_error(y_test[col], preds_df[col]))

plt.figure(figsize=(12, 6))
plt.plot(range(1, HORYZONT_PRZYSZLOSC + 1), mae_per_hour, marker='o')
plt.title('Jak błąd rośnie wraz z horyzontem czasowym?')
plt.xlabel('Godzina w przyszłość (Forecast Horizon)')
plt.ylabel('MAE (Błąd średni)')
plt.grid(True)
plt.savefig('wykresy/xgboost/3_blad_vs_horyzont.png')
plt.close()

# Wykres 4: Przykładowa pojedyncza prognoza (ostatnia ze zbioru)
last_idx = -1
y_sample_real = y_test.iloc[last_idx].values
y_sample_pred = preds_df.iloc[last_idx].values
hours = range(1, 73)

plt.figure(figsize=(12, 6))
plt.plot(hours, y_sample_real, label='Rzeczywistość', marker='o')
plt.plot(hours, y_sample_pred, label='Prognoza', marker='x')
plt.title(f'Przykładowa prognoza na 72h (Start: {time_test.iloc[last_idx]})')
plt.xlabel('Godziny naprzód')
plt.ylabel('OZE Share')
plt.legend()
plt.savefig('wykresy/xgboost/4_pojedyncza_prognoza.png')
plt.close()

print("\nZakończono! Wyniki w folderach 'model_output' oraz 'wykresy/xgboost'.")