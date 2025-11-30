import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import pickle
import warnings

# Wyciszenie ostrzeżeń
warnings.filterwarnings("ignore")

# --- 1. KONFIGURACJA ---
OZE_COLUMNS = [
    'Biomass - Actual Aggregated [MW]',
    'Hydro Run-of-river and poundage - Actual Aggregated [MW]',
    'Hydro Water Reservoir - Actual Aggregated [MW]',
    'Solar - Actual Aggregated [MW]',
    'Wind Onshore - Actual Aggregated [MW]'
]
ENERGY_TIME_COL = 'start_time'
WEATHER_TIME_COL = 'czas'
MODEL_FILENAME = 'model_oze_polska_final.json'
SCALER_FILENAME = 'scaler_pogoda_final.pkl'

# --- 2. WCZYTYWANIE DANYCH ---
print("Wczytywanie danych...")
df_energy = pd.read_csv('data/dane_energia_Polska.csv')
df_weather = pd.read_csv('data/polska_pogoda_godzinowa.csv')

# --- 3. PREPROCESSING ---
df_energy[ENERGY_TIME_COL] = pd.to_datetime(df_energy[ENERGY_TIME_COL])
df_weather[WEATHER_TIME_COL] = pd.to_datetime(df_weather[WEATHER_TIME_COL])

df_energy = df_energy.sort_values(ENERGY_TIME_COL)
df_weather = df_weather.sort_values(WEATHER_TIME_COL)

# Target
df_energy['OZE_Share'] = df_energy[OZE_COLUMNS].sum(axis=1).clip(0, 1)
df_energy = df_energy.dropna(subset=['OZE_Share'])

# --- 4. ŁĄCZENIE DANYCH ---
df = pd.merge(
    df_energy[[ENERGY_TIME_COL, 'OZE_Share']],
    df_weather,
    left_on=ENERGY_TIME_COL,
    right_on=WEATHER_TIME_COL,
    how='inner'
)


# --- 5. INŻYNIERIA CECH ---
def add_time_features(dataframe, time_col):
    df_temp = dataframe.copy()
    hours = df_temp[time_col].dt.hour
    months = df_temp[time_col].dt.month
    days = df_temp[time_col].dt.dayofyear

    # Czas cykliczny
    df_temp['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df_temp['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df_temp['month_sin'] = np.sin(2 * np.pi * months / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * months / 12)
    df_temp['day_sin'] = np.sin(2 * np.pi * days / 366)
    df_temp['day_cos'] = np.cos(2 * np.pi * days / 366)
    return df_temp


df = add_time_features(df, ENERGY_TIME_COL)

# --- HISTORIA (LAGS) ---
print("Generowanie historii (Lags)...")
df['lag_72h'] = df['OZE_Share'].shift(72)
df['lag_168h'] = df['OZE_Share'].shift(168)
df = df.dropna()

# --- DEFINICJE KOLUMN ---
WEATHER_COLS = ['temperatura', 'wilgotność', 'wiatr', 'opady', 'ciśnienie', 'Tsun_min']
TIME_COLS = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
LAG_COLS = ['lag_72h', 'lag_168h']

FEATURES = WEATHER_COLS + TIME_COLS + LAG_COLS
TARGET = 'OZE_Share'

# --- 6. PODZIAŁ DANYCH ---
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# --- 6.1 ZAAWANSOWANY SZUM GAUSSOWSKI ---
print(f"\n--- APLIKOWANIE REALISTYCZNEGO BŁĘDU PROGNOZY (GAUSS) ---")
np.random.seed(42)

NOISE_CONFIG = {
    'temperatura': 2.2,  # +/- 2.2 stopnia
    'wiatr': 2.5,  # +/- 2.5 m/s
    'ciśnienie': 2.5,  # +/- 2.5 hPa
    'wilgotność': 12.0  # +/- 12%
}

for col, sigma in NOISE_CONFIG.items():
    if col in train_df.columns:
        # Generowanie szumu addytywnego
        noise = np.random.normal(loc=0.0, scale=sigma, size=len(train_df))
        train_df[col] += noise

        # Ograniczenia fizyczne
        if col == 'wiatr':
            train_df[col] = train_df[col].clip(lower=0)
        elif col == 'wilgotność':
            train_df[col] = train_df[col].clip(lower=0, upper=100)

# --- 6.2 NORMALIZACJA ---
print("\nNormalizacja pogody...")
scaler = StandardScaler()
scaler.fit(train_df[WEATHER_COLS])  # Uczymy na zaszumionych

train_df[WEATHER_COLS] = scaler.transform(train_df[WEATHER_COLS])
test_df[WEATHER_COLS] = scaler.transform(test_df[WEATHER_COLS])

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# --- 7. TRENING ---
print(f"\nTrenowanie modelu na {len(X_train)} próbkach...")
start_time = time.time()

model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=7,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)
print(f"Czas treningu: {time.time() - start_time:.2f} s")

# --- 8. EWALUACJA ---
preds = np.clip(model.predict(X_test), 0, 1)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- WYNIKI NA ZBIORZE TESTOWYM ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# --- WYKRESY (STARA KONFIGURACJA) ---
plt.style.use('ggplot')

# Wykres 1: Liniowy - Zoom na 200h
plt.figure(figsize=(15, 5))
subset_n = 200
plt.plot(y_test.values[:subset_n], label='Rzeczywiste (Ground Truth)', alpha=0.8, linewidth=2)
plt.plot(preds[:subset_n], label='Prognoza (Pred)', alpha=0.8, linestyle='--', linewidth=2)
plt.title(f'Sprawdzenie modelu: Pierwsze {subset_n} godzin ze zbioru testowego')
plt.ylabel('Udział OZE (0-1)')
plt.legend()
plt.tight_layout()
plt.show()

# Wykres 2: Feature Importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(model, max_num_features=15, importance_type='weight', title='Wpływ cech na prognozę',
                    show_values=False)
plt.tight_layout()
plt.show()

# Wykres 3: Scatter Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds, alpha=0.1, s=2, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
plt.xlabel('Wartości Rzeczywiste')
plt.ylabel('Wartości Prognozowane')
plt.title('Scatter Plot (Rozrzut błędu)')
plt.tight_layout()
plt.show()

# Zapis modelu
model.save_model(MODEL_FILENAME)
with open(SCALER_FILENAME, 'wb') as f:
    pickle.dump(scaler, f)

# --- 9. PROGNOZA 72H ---
print("\nGenerowanie prognozy na 72h...")

last_timestamp = df[ENERGY_TIME_COL].max()
future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=72, freq='h')

future_weather = df_weather.iloc[-72:].copy()
future_weather[WEATHER_TIME_COL] = future_dates

# A. Czas i Normalizacja (na czysto)
future_weather = add_time_features(future_weather, WEATHER_TIME_COL)
future_weather[WEATHER_COLS] = scaler.transform(future_weather[WEATHER_COLS])

# B. Historie (Lookup)
history_map = df.set_index(ENERGY_TIME_COL)['OZE_Share'].to_dict()
lag_72_vals = []
lag_168_vals = []

for date in future_dates:
    val_72 = history_map.get(date - pd.Timedelta(hours=72), 0.0)
    val_168 = history_map.get(date - pd.Timedelta(hours=168), 0.0)
    lag_72_vals.append(val_72)
    lag_168_vals.append(val_168)

future_weather['lag_72h'] = lag_72_vals
future_weather['lag_168h'] = lag_168_vals

# C. Predykcja
future_preds = np.clip(model.predict(future_weather[FEATURES]), 0, 1)

forecast_df = pd.DataFrame({'czas': future_dates, 'prognoza_OZE': future_preds})

# Wykres łączony
plt.figure(figsize=(12, 5))
last_history = df.iloc[-72:]
plt.plot(last_history[ENERGY_TIME_COL], last_history['OZE_Share'], label='Historia')
plt.plot(forecast_df['czas'], forecast_df['prognoza_OZE'], label='Prognoza (+72h)', color='green', linestyle='--')
plt.title("Historia vs Prognoza")
plt.legend()
plt.grid(True)
plt.show()

print(forecast_df.head())