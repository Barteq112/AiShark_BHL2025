import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import warnings

# Wyciszenie ostrzeżeń
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

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
MODEL_FILENAME = 'model_oze_polska.json'

# --- 2. WCZYTYWANIE DANYCH ---
print("Wczytywanie danych...")
df_energy = pd.read_csv('data/dane_energia_Polska.csv')
df_weather = pd.read_csv('data/polska_pogoda_godzinowa.csv')

# --- 3. PREPROCESSING ---
df_energy[ENERGY_TIME_COL] = pd.to_datetime(df_energy[ENERGY_TIME_COL])
df_weather[WEATHER_TIME_COL] = pd.to_datetime(df_weather[WEATHER_TIME_COL])

df_energy = df_energy.sort_values(ENERGY_TIME_COL)
df_weather = df_weather.sort_values(WEATHER_TIME_COL)

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
df['hour'] = df[ENERGY_TIME_COL].dt.hour
df['month'] = df[ENERGY_TIME_COL].dt.month
df['dayofyear'] = df[ENERGY_TIME_COL].dt.dayofyear
df['dayofweek'] = df[ENERGY_TIME_COL].dt.dayofweek

FEATURES = ['temperatura', 'wilgotność', 'wiatr', 'opady', 'ciśnienie', 'Tsun_min',
            'hour', 'month']
TARGET = 'OZE_Share'

# --- 6. PODZIAŁ DANYCH ---
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# --- 7. TRENING NA GPU ---
print("\nRozpoczynam trening na GPU...")
start_time = time.time()

model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    objective='reg:squarederror',
    device='cuda',
    tree_method='hist',
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)
print(f"Czas treningu: {time.time() - start_time:.2f} s")

# --- 8. ZAAWANSOWANE SPRAWDZENIE MODELU ---
print("\n--- EWALUACJA MODELU ---")
preds = model.predict(X_test)
preds = np.clip(preds, 0, 1)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"MAE (Średni błąd): {mae:.4f}")
print(f"RMSE (Błąd kwadratowy): {rmse:.4f}")
print(f"R2 Score (Dopasowanie): {r2:.4f} (im bliżej 1.0 tym lepiej)")

# --- 8.1 WIZUALIZACJA ---
# Ustawienia wykresów
plt.style.use('ggplot')

# Wykres 1: Rzeczywiste vs Prognozowane (wycinek 200 godzin dla czytelności)
plt.figure(figsize=(15, 5))
subset_n = 200
plt.plot(y_test.values[:subset_n], label='Rzeczywiste', alpha=0.7)
plt.plot(preds[:subset_n], label='Prognoza', alpha=0.7)
plt.title(f'Porównanie Rzeczywistość vs Prognoza (pierwsze {subset_n} godzin testowych)')
plt.ylabel('Udział OZE (0-1)')
plt.legend()
plt.show()



# Wykres 2: Ważność cech (Feature Importance)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10, importance_type='weight', title='Wpływ pogody na OZE', show_values=False)
plt.show()



# Wykres 3: Scatter Plot (Idealny model to linia prosta po przekątnej)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds, alpha=0.1, s=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Linia idealna
plt.xlabel('Wartości Rzeczywiste')
plt.ylabel('Wartości Prognozowane')
plt.title('Wykres rozrzutu błędów')
plt.show()

# --- 9. ZAPIS I PROGNOZA 72H ---
model.save_model(MODEL_FILENAME)
print(f"\nModel zapisany jako {MODEL_FILENAME}")

# Prognoza przyszłości
last_timestamp = df[ENERGY_TIME_COL].max()
future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=72, freq='h')

future_weather = df_weather.iloc[-72:].copy()
future_weather[WEATHER_TIME_COL] = future_dates
future_weather['hour'] = future_weather[WEATHER_TIME_COL].dt.hour
future_weather['month'] = future_weather[WEATHER_TIME_COL].dt.month
future_weather['dayofyear'] = future_weather[WEATHER_TIME_COL].dt.dayofyear
future_weather['dayofweek'] = future_weather[WEATHER_TIME_COL].dt.dayofweek

future_preds = np.clip(model.predict(future_weather[FEATURES]), 0, 1)

forecast_df = pd.DataFrame({'czas': future_dates, 'prognoza_OZE': future_preds})
print("\nPrognoza na 72h (head):")
print(forecast_df.head())