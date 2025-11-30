import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import os

# Importy do sieci neuronowych
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Wyciszenie ostrzeżeń
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mniej logów z TensorFlow

# --- 0. KONFIGURACJA GPU (DODATEK) ---
# Sprawdzenie i konfiguracja GPU (jeśli dostępne)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Wykryto GPU: {gpus}")
    except RuntimeError as e:
        print(e)

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
MODEL_FILENAME = 'model_oze_lstm.h5'

# PARAMETRY LSTM
LOOKBACK = 48  # Ile godzin wstecz patrzy model
FORECAST = 72  # Ile godzin w przód przewidujemy

Kraj = 'Polska'

# --- 2. WCZYTYWANIE DANYCH ---
print("Wczytywanie danych...")
df_energy = pd.read_csv(f'Kraje_2022_2025/dane_energia_{Kraj}.csv')
df_weather = pd.read_csv(f'Kraje_2022_2025/{Kraj.lower()}_pogoda_godzinowa.csv')

# --- 3. PREPROCESSING ---
df_energy[ENERGY_TIME_COL] = pd.to_datetime(df_energy[ENERGY_TIME_COL])
df_weather[WEATHER_TIME_COL] = pd.to_datetime(df_weather[WEATHER_TIME_COL])

df_energy = df_energy.sort_values(ENERGY_TIME_COL)
df_weather = df_weather.sort_values(WEATHER_TIME_COL)

dostepne_oze = [c for c in OZE_COLUMNS if c in df_energy.columns]
df_energy['OZE_Share'] = df_energy[dostepne_oze].sum(axis=1).clip(0, 1)
df_energy = df_energy.dropna(subset=['OZE_Share'])

# --- 4. ŁĄCZENIE DANYCH ---
df = pd.merge(
    df_energy,  # Bierzemy cały df_energy, żeby mieć dostęp do wszystkich kolumn OZE (Biomass, Wind itp.)
    df_weather,
    left_on=ENERGY_TIME_COL,
    right_on=WEATHER_TIME_COL,
    how='inner',
    suffixes=('_energy', '_weather')
)

# --- 5. INŻYNIERIA CECH ---

# Czas
df['hour'] = df[ENERGY_TIME_COL].dt.hour
df['month'] = df[ENERGY_TIME_COL].dt.month
df['dayofyear'] = df[ENERGY_TIME_COL].dt.dayofyear

# Transformacja cykliczna
def add_cyclical_features(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(df[col_name] * (2. * np.pi / period))
    df[f'{col_name}_cos'] = np.cos(df[col_name] * (2. * np.pi / period))
    return df

df = add_cyclical_features(df, 'hour', 24)
df = add_cyclical_features(df, 'month', 12)
df = add_cyclical_features(df, 'dayofyear', 365.25)

print(f"\n--- SYMULACJA PROGNOZ POGODY (DODAWANIE REALISTYCZNEGO BŁĘDU) ---")
print("Konwersja idealnych danych historycznych na 'symulowane prognozy'.")
print("W rzeczywistości nie znamy przyszłej pogody idealnie - model musi radzić sobie z błędem prognozy.")

np.random.seed(42)

# Konfiguracja
SHIFT_FORECAST = 72

# Konfiguracja typowego błędu prognozy meteorologicznej (Odchylenie standardowe)
# Wartości te symulują, jak bardzo prognoza pogody myli się względem rzeczywistości
FORECAST_ERROR_CONFIG = {
    'temperatura': 2.0,  # Prognoza temperatury myli się średnio o +/- 2.0 stopnie
    'wiatr': 1.5,  # Prognoza prędkości wiatru myli się o +/- 1.5 m/s
    'ciśnienie': 2.0,  # Prognoza ciśnienia +/- 2 hPa
    'wilgotność': 10.0  # Wilgotność jest trudna do trafienia (+/- 10%)
}

# Lista kolumn pogodowych do przetworzenia
weather_cols = list(FORECAST_ERROR_CONFIG.keys())
print(weather_cols)

# Iterujemy po kolumnach
for col in weather_cols:
    if col in df.columns:
        print(f" -> Generowanie prognozy dla: {col} (Dane z t+{SHIFT_FORECAST}h + Szum)")

        # KROK 1: Pobranie "prawdziwej" pogody z przyszłości
        # .shift(-24) oznacza: weź wartość z wiersza, który jest 24 godziny "niżej" (w przyszłości)
        # i wstaw ją do bieżącego wiersza.
        future_weather = df[col].shift(-SHIFT_FORECAST)

        # KROK 2: Wygenerowanie błędu prognozy (Szum Gaussa)
        noise = np.random.normal(loc=0.0, scale=FORECAST_ERROR_CONFIG[col], size=len(df))

        # KROK 3: Nadpisanie kolumny (Teraz kolumna 'wiatr' to 'prognoza wiatru na jutro')
        df[f'{col}_prognoza'] = future_weather + noise

        # KROK 4: Korekta fizyczna (clip)
        if col == 'wiatr':
            df[f'{col}_prognoza'] = df[f'{col}_prognoza'].clip(lower=0)
        elif col == 'wilgotność':
            df[f'{col}_prognoza'] = df[f'{col}_prognoza'].clip(lower=0, upper=100)

df = df.dropna()

# B. Wybór zmiennych do modelu (Te same co w Ridge/XGBoost)
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
    'wiatr_prognoza',
    'wilgotnosc_prognoza',
    # 'cisniene_prognoza',
    'temperatura_prognoza'
]

# Cechy czasowe, które chcemy w modelu
time_features = ['hour_cos']

# Łączna lista kolumn wejściowych
# Najpierw sprawdzamy, co jest dostępne w DF
available_features = [c for c in selected_features_names if c in df.columns]
FINAL_FEATURES = available_features + time_features

print(f"Liczba cech wejściowych: {len(FINAL_FEATURES)}")
print(f"Cechy: {FINAL_FEATURES}")

# --- 6. PRZYGOTOWANIE DANYCH DLA LSTM (SLIDING WINDOW) ---
print("Przygotowanie sekwencji danych...")

# 1. Ustalenie indeksów podziału na surowych danych
n = len(df)
train_end = int(n * 0.70)
# val_end użyjemy później do podziału sekwencji, tu potrzebujemy tylko granicy treningu do scalera

# 2. Skalowanie (POPRAWIONE)
scaler = MinMaxScaler()

# WAŻNE: fit() robimy TYLKO na danych treningowych!
# Model poznaje min/max tylko z przeszłości.
scaler.fit(df[FINAL_FEATURES].iloc[:train_end])

# Transformujemy całość (żeby zachować ciągłość dla okien czasowych)
# Dane testowe zostaną przeskalowane według parametrów z treningu (tak jak w produkcji)
data_scaled = scaler.transform(df[FINAL_FEATURES])

# 3. Tworzenie sekwencji
# Znalezienie indeksu kolumny docelowej (OZE_Share) w macierzy
target_col_idx = FINAL_FEATURES.index('OZE_Share')

def create_sequences(dataset, lookback, forecast, target_idx):
    X, y = [], []
    # Iterujemy po danych
    for i in range(len(dataset) - lookback - forecast + 1):
        X.append(dataset[i : i + lookback, :])
        y.append(dataset[i + lookback : i + lookback + forecast, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, LOOKBACK, FORECAST, target_col_idx)

# 4. Podział gotowych sekwencji (X, y) na zbiory
# Musimy przeliczyć indeksy, bo create_sequences "zjada" początkowe wiersze (lookback)
# Długość X jest mniejsza od len(df) o (lookback + forecast - 1)


# Ponieważ transformowaliśmy całość, teraz po prostu dzielimy X i y proporcjonalnie
# Używamy tych samych proporcji co przy ustalaniu train_end
n_samples = len(X)
train_split_idx = int(n_samples * 0.70)
val_split_idx = int(n_samples * 0.85)

X_train, y_train = X[:train_split_idx], y[:train_split_idx]
X_val, y_val = X[train_split_idx:val_split_idx], y[train_split_idx:val_split_idx]
X_test, y_test = X[val_split_idx:], y[val_split_idx:]

# Przygotowanie osi czasu dla testu (do wykresów)
# Bierzemy znaczniki czasu odpowiadające momentowi STARTU prognozy w zbiorze testowym
# Indeksy w df przesuwają się o lookback względem indeksów w X
test_start_data_idx = val_split_idx + LOOKBACK - 1
test_end_data_idx = test_start_data_idx + len(X_test)
time_test = df[ENERGY_TIME_COL].iloc[test_start_data_idx : test_end_data_idx].values

print(f"Kształt wejścia X: {X.shape}")
print(f"Liczba próbek - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- 7. BUDOWA I TRENING MODELU LSTM ---
print("\nBudowanie modelu LSTM...")

model = Sequential()
# Warstwa 1
model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Warstwa 2
model.add(LSTM(16, return_sequences=False))
model.add(Dropout(0.2))

# Warstwa wyjściowa (72 wyjścia)
model.add(Dense(FORECAST))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

print("Rozpoczynam trening...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

print(f"Czas treningu: {time.time() - start_time:.2f} s")

# Wykres straty
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Przebieg uczenia (Loss)')
plt.legend()
plt.show()

# --- 8. EWALUACJA ---
print("\n--- EWALUACJA ---")
y_pred_scaled = model.predict(X_test)
y_pred = np.clip(y_pred_scaled, 0, 1)
y_true = y_test

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"Średnie MAE (cały horyzont): {mae:.4f}")
print(f"Średnie RMSE: {rmse:.4f}")
print(f"Średni R2 Score (Dopasowanie): {r2:.4f}")

# --- 9. WIZUALIZACJA PORÓWNAWCZA ---
print("\n--- PORÓWNANIE: OSTATNIA PRÓBKA TESTOWA ---")

X_sample = X_test[-1].reshape(1, LOOKBACK, len(FINAL_FEATURES))
y_sample_true = y_test[-1]

y_sample_pred = model.predict(X_sample)[0]
y_sample_pred = np.clip(y_sample_pred, 0, 1)

horyzont_godziny = range(1, FORECAST + 1)
srednia_real = np.mean(y_sample_true)
srednia_pred = np.mean(y_sample_pred)

plt.figure(figsize=(14, 7))
plt.plot(horyzont_godziny, y_sample_true, label=f'Rzeczywistość (Śr: {srednia_real:.1%})', color='black', linewidth=2)
plt.plot(horyzont_godziny, y_sample_pred, label=f'LSTM Prognoza (Śr: {srednia_pred:.1%})', color='red', linestyle='--')
plt.fill_between(horyzont_godziny, y_sample_true, y_sample_pred, color='red', alpha=0.1)
plt.title(f'LSTM: Prognoza na 72h (Input: {LOOKBACK}h)')
plt.legend()
plt.show()

# --- 10. PROGNOZA PRZYSZŁOŚCI ---
print("\n--- GENEROWANIE PROGNOZY NA KOLEJNE 3 DNI ---")

last_sequence = data_scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(FINAL_FEATURES))
future_pred = model.predict(last_sequence)[0]
future_pred = np.clip(future_pred, 0, 1)

last_timestamp = df[ENERGY_TIME_COL].max()
future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=FORECAST, freq='h')

forecast_df = pd.DataFrame({'czas': future_dates, 'prognoza_LSTM': future_pred})
forecast_df.to_csv('model_output/lstm/prognoza_lstm_72h.csv', index=False)

plt.figure(figsize=(10, 5))
plt.plot(forecast_df['czas'], forecast_df['prognoza_LSTM'], marker='.')
plt.title(f'LSTM: Prognoza generacji OZE na przyszłość (Start: {last_timestamp})')
plt.grid(True)
plt.show()

print("Zapisano prognozę LSTM.")

import joblib

# --- ZAPISYWANIE MODELU I SCALERA ---
print("\nZapisywanie modelu i narzędzi...")

# 1. Zapis modelu (architektura + wagi + stan treningu)
model_filename = f'model_output/lstm/final_model_{Kraj}.keras'
model.save(model_filename)
print(f"Zapisano model sieci: {model_filename}")

# 2. Zapis Scalera (niezbędny do predykcji na nowych danych!)
scaler_filename = f'model_output/lstm/scaler_{Kraj}.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Zapisano scaler: {scaler_filename}")