import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import time
import warnings
import os

# --- 0. KONFIGURACJA ŚRODOWISKA ---
warnings.filterwarnings("ignore")
os.makedirs('model_output/ridge', exist_ok=True)
os.makedirs('wykresy/ridge', exist_ok=True)

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
HISTORIA_LAGI = 24        # Input: 24h w tył

# --- 2. WCZYTYWANIE DANYCH ---
print("Wczytywanie danych...")
# Upewnij się, że ścieżki są poprawne
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
print("Generowanie cech (identycznie jak w XGBoost)...")
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

# B. Wybór zmiennych do modelu (NA PODSTAWIE KORELACJI)
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

# Sprawdzenie dostępności
features_to_use = [c for c in selected_features_names if c in df.columns]
print(f"Zmienne bazowe wybrane do nauki ({len(features_to_use)}):")
print(features_to_use)

# --- 6. TWORZENIE LAGÓW I TARGETÓW ---
print(f"Generowanie lagów i targetów...")

# Targety
target_dfs = [df['OZE_Share'].shift(-h).rename(f'target_{h}h') for h in range(1, HORYZONT_PRZYSZLOSC + 1)]
target_names = [f'target_{h}h' for h in range(1, HORYZONT_PRZYSZLOSC + 1)]

# Lagi (Tylko dla features_to_use)
lag_dfs = []
lag_names = []
for h in range(1, HISTORIA_LAGI + 1):
    tmp = df[features_to_use].shift(h).add_suffix(f'_lag_{h}h')
    lag_dfs.append(tmp)
    lag_names.extend(tmp.columns.tolist())

# Złączenie
df_final = pd.concat([df] + target_dfs + lag_dfs, axis=1)
df_final = df_final.dropna()

# --- DEFINICJA X i y (STRICT) ---
# Tylko wybrane cechy + ich lagi. BEZ cech czasowych (sin/cos/hour).
input_features = features_to_use + lag_names

print(f"\nLiczba cech wejściowych (X): {len(input_features)}")
print("Przykładowe cechy:", input_features[:5], "...", input_features[-5:])

X = df_final[input_features]
y = df_final[target_names]
timestamps = df_final[ENERGY_TIME_COL]

# --- 7. PODZIAŁ DANYCH ---
n = len(df_final)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
time_test = timestamps.iloc[val_end:]

print(f"Liczba próbek - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- 8. NORMALIZACJA ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 9. TRENING (RIDGE REGRESSION) ---
print("\nRozpoczynam trening (Ridge Regression MultiOutput)...")
start_time = time.time()

# Ridge to liniowy model z regularyzacją (lepiej radzi sobie z lagami niż zwykła regresja liniowa)
# alpha=1.0 to standardowa siła regularyzacji
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train_scaled, y_train)

print(f"Czas treningu: {time.time() - start_time:.4f} s") # To będzie bardzo szybkie!

# Zapis modelu
joblib.dump(model, 'model_output/ridge/ridge_multioutput.pkl')
joblib.dump(scaler, 'model_output/ridge/scaler.pkl')
print("Model i scaler zapisane w folderze 'model_output/ridge'.")

# --- 10. PREDYKCJA I EWALUACJA (TEST SET) ---
print("\nGenerowanie predykcji dla zbioru testowego...")
preds_test = model.predict(X_test_scaled)

# WAŻNE: Modele liniowe mogą zwrócić wartości <0 lub >1, więc musimy je przyciąć (clip)
preds_test = np.clip(preds_test, 0, 1)

preds_df = pd.DataFrame(preds_test, columns=target_names, index=y_test.index)

# --- 11. ANALIZA WYNIKÓW ---

mae = mean_absolute_error(y_test, preds_df)
rmse = np.sqrt(mean_squared_error(y_test, preds_df))
r2 = r2_score(y_test, preds_df)

print("\n--- WYNIKI DLA ZBIORU TESTOWEGO (RIDGE) ---")
print(f"MAE (średni błąd absolutny): {mae:.4f}")
print(f"RMSE (błąd średniokwadratowy): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Analiza średnich
y_test_mean_72h = y_test.mean(axis=1)
preds_mean_72h = preds_df.mean(axis=1)

analysis_df = pd.DataFrame({
    'czas_startu': time_test,
    'Rzeczywista_Srednia_72h': y_test_mean_72h.values,
    'Prognoza_Srednia_72h': preds_mean_72h.values
})
analysis_df['Blad_Sredniej'] = analysis_df['Prognoza_Srednia_72h'] - analysis_df['Rzeczywista_Srednia_72h']

# Zapis wyników
analysis_df.to_csv('model_output/ridge/analiza_testowa_srednie.csv', index=False)

# --- 12. GENEROWANIE WYKRESÓW ---
print("\nGenerowanie wykresów...")
plt.style.use('bmh')

# Wykres 1: Scatter
plt.figure(figsize=(10, 10))
plt.scatter(analysis_df['Rzeczywista_Srednia_72h'], analysis_df['Prognoza_Srednia_72h'], alpha=0.3, s=10, color='orange')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Rzeczywista Średnia OZE (72h)')
plt.ylabel('Prognozowana Średnia OZE (72h)')
plt.title(f'RIDGE: Korelacja prognozy 3-dniowej (R2={r2:.3f})')
plt.savefig('wykresy/ridge/1_scatter_srednia.png')
plt.close()

# Wykres 2: Szereg czasowy
plt.figure(figsize=(15, 6))
subset = 500
plt.plot(analysis_df['czas_startu'].tail(subset), analysis_df['Rzeczywista_Srednia_72h'].tail(subset), label='Realna Średnia', color='black', alpha=0.7)
plt.plot(analysis_df['czas_startu'].tail(subset), analysis_df['Prognoza_Srednia_72h'].tail(subset), label='Ridge Prognoza', color='orange', alpha=0.8)
plt.title('RIDGE: Porównanie średniej generacji OZE (Szereg Czasowy)')
plt.legend()
plt.savefig('wykresy/ridge/2_szereg_czasowy.png')
plt.close()

# Wykres 3: Błąd vs Horyzont
mae_per_hour = []
for h in range(1, HORYZONT_PRZYSZLOSC + 1):
    col = f'target_{h}h'
    mae_per_hour.append(mean_absolute_error(y_test[col], preds_df[col]))

plt.figure(figsize=(12, 6))
plt.plot(range(1, HORYZONT_PRZYSZLOSC + 1), mae_per_hour, marker='o', color='orange')
plt.title('RIDGE: Jak błąd rośnie wraz z horyzontem czasowym?')
plt.xlabel('Godzina w przyszłość')
plt.ylabel('MAE')
plt.grid(True)
plt.savefig('wykresy/ridge/3_blad_vs_horyzont.png')
plt.close()

# Wykres 4: Pojedyncza prognoza
last_idx = -1
y_sample_real = y_test.iloc[last_idx].values
y_sample_pred = preds_df.iloc[last_idx].values
hours = range(1, 73)

plt.figure(figsize=(12, 6))
plt.plot(hours, y_sample_real, label='Rzeczywistość', marker='o', color='black')
plt.plot(hours, y_sample_pred, label='Ridge Prognoza', marker='x', color='orange')
plt.title(f'RIDGE: Przykładowa prognoza na 72h (Start: {time_test.iloc[last_idx]})')
plt.xlabel('Godziny naprzód')
plt.ylabel('OZE Share')
plt.legend()
plt.savefig('wykresy/ridge/4_pojedyncza_prognoza.png')
plt.close()

print("\nZakończono! Wyniki w folderach 'model_output/ridge' oraz 'wykresy/ridge'.")

# --- 13. ANALIZA WAG (FEATURE IMPORTANCE) ---
print("\n--- ANALIZA WPŁYWU ZMIENNYCH ---")

# Pobieramy nazwy cech z DataFrame'a treningowego
feature_names = X.columns.tolist()

# Mamy 72 modele (jeden na każdą godzinę horyzontu).
# Musimy uśrednić ich wagi, żeby zobaczyć ogólny wpływ cechy na całe 3 dni.
all_coefs = []

# Iterujemy przez każdy z 72 modeli wewnątrz MultiOutputRegressor
for estimator in model.estimators_:
    all_coefs.append(estimator.coef_)

# Zamieniamy na macierz numpy
all_coefs = np.array(all_coefs) # Kształt: (72, liczba_cech)

# Obliczamy średnią WARTOŚĆ BEZWZGLĘDNĄ (interesuje nas siła wpływu, niezależnie czy na plus czy minus)
mean_abs_coefs = np.mean(np.abs(all_coefs), axis=0)

# Obliczamy też średnią RZECZYWISTĄ (żeby wiedzieć czy cecha podnosi (+) czy obniża (-) produkcję)
mean_raw_coefs = np.mean(all_coefs, axis=0)

# Tworzymy DataFrame do wyświetlenia
importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Sila_Wplywu': mean_abs_coefs,   # To służy do sortowania
    'Kierunek_Wplywu': mean_raw_coefs # To mówi czy + czy -
})

# Sortujemy od najważniejszych
importance_df = importance_df.sort_values(by='Sila_Wplywu', ascending=False).reset_index(drop=True)

# Wyświetlamy TOP 20 w konsoli
print("\nTOP 20 Najważniejszych zmiennych:")
print(importance_df[['Cecha', 'Sila_Wplywu', 'Kierunek_Wplywu']].head(20))

# Zapis do CSV (wszystkie cechy)
importance_df.to_csv('model_output/ridge/wagi_zmiennych.csv', index=False)

# --- 14. WYKRES WAG ---
plt.figure(figsize=(12, 8))

# Bierzemy TOP 20 cech do wykresu
top_n = 20
top_features = importance_df.head(top_n).sort_values(by='Sila_Wplywu', ascending=True) # Sortujemy rosnąco dla wykresu poziomego

# Kolory słupków: Zielony jeśli wpływ dodatni, Czerwony jeśli ujemny
colors = ['green' if x > 0 else 'red' for x in top_features['Kierunek_Wplywu']]

plt.barh(top_features['Cecha'], top_features['Sila_Wplywu'], color=colors)
plt.xlabel('Średnia siła wpływu (Absolutna wartość wagi)')
plt.title(f'TOP {top_n} Cech wpływających na generację OZE (Model Liniowy)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Dodajemy legendę kolorów
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Wpływ Dodatni (Zwiększa OZE)'),
                   Patch(facecolor='red', label='Wpływ Ujemny (Zmniejsza OZE)')]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('wykresy/ridge/5_wagi_zmiennych.png')
plt.show()