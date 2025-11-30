import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
CUSTOM_KERAS_OBJECTS = {
   
    'Orthogonal': tf.keras.initializers.Orthogonal
}
# --- FUNKCJE POMOCNICZE ---
def add_cyclical_features(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(df[col_name] * (2. * np.pi / period))
    df[f'{col_name}_cos'] = np.cos(df[col_name] * (2. * np.pi / period))
    return df


def prognozuj_oze_dla_kraju(kraj, target_date):
    """
    Wczytuje model LSTM i dane, generuje symulowane prognozy pogody (z szumem),
    a następnie zwraca predykcję OZE.
    """

    # --- 1. KONFIGURACJA ŚCIEŻEK ---
    scaler_filename = BASE_DIR / 'model_output' / 'lstm' / f'scaler_{kraj}.pkl'
    model_filename = BASE_DIR / 'model_output' / 'lstm' / f'final_model_{kraj}.keras'

    data_energy_path = BASE_DIR / 'Kraje_2022_2025' / f'dane_energia_{kraj}.csv'
    data_weather_path = BASE_DIR / 'Kraje_2022_2025' / f'{kraj.lower()}_pogoda_godzinowa.csv'

    # Parametry modelu
    LOOKBACK = 48
    FORECAST = 72

    # --- NOWA LISTA CECH (Zgodna z Twoim kodem treningowym) ---
    # UWAGA: Używamy nazw kolumn takich, jakie wygeneruje pętla (z polskimi znakami jeśli były w kluczach)
    FINAL_FEATURES = [
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
        # 'wilgotnosc_prognoza',
        'temperatura_prognoza',
        'hour_cos'
    ]

    print(f"\n--- PRZETWARZANIE DLA KRAJU: {kraj.upper()} ---")

    # --- 2. WALIDACJA PLIKÓW ---
    required_files = [scaler_filename, model_filename, data_energy_path, data_weather_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"BŁĄD: Nie znaleziono pliku: {file_path}")
            return None

    # --- 3. WCZYTYWANIE MODELU I SCALERA ---
    try:
        model = tf.keras.models.load_model(model_filename, custom_objects=CUSTOM_KERAS_OBJECTS)
        scaler = joblib.load(scaler_filename)
    except Exception as e:
        print(f"Błąd podczas wczytywania modelu/scalera: {e}")
        return None

    # --- 4. WCZYTYWANIE I OBRÓBKA DANYCH ---
    df_energy = pd.read_csv(data_energy_path)
    df_weather = pd.read_csv(data_weather_path)

    df_energy['start_time'] = pd.to_datetime(df_energy['start_time'])
    df_weather['czas'] = pd.to_datetime(df_weather['czas'])

    df_energy = df_energy.sort_values('start_time')
    df_weather = df_weather.sort_values('czas')

    # Obliczenie OZE Share
    oze_cols = ['Biomass', 'Hydro Run-of-river and poundage', 'Hydro Water Reservoir', 'Solar', 'Wind Onshore']
    available_oze = [c for c in oze_cols if c in df_energy.columns]

    if not available_oze:
        df_energy['OZE_Share'] = 0
    else:
        df_energy['OZE_Share'] = df_energy[available_oze].sum(axis=1).clip(0, 1)

    # Merge
    df = pd.merge(df_energy, df_weather, left_on='start_time', right_on='czas', how='inner')

    # Feature Engineering (Czas)
    df['hour'] = df['start_time'].dt.hour
    df = add_cyclical_features(df, 'hour', 24)

    # --- 5. SYMULACJA PROGNOZ POGODY (Kod dodany) ---
    print("--- Generowanie symulowanych prognoz pogody ---")
    np.random.seed(42)  # Dla powtarzalności wyników inferencji

    SHIFT_FORECAST = 72

    FORECAST_ERROR_CONFIG = {
        'temperatura': 2.0,
        'wiatr': 1.5,
        'ciśnienie': 2.0,
        'wilgotność': 10.0
    }

    weather_cols = list(FORECAST_ERROR_CONFIG.keys())

    for col in weather_cols:
        if col in df.columns:
            # Shift (-72h) - bierzemy "prawdziwą przyszłość" z pliku
            future_weather = df[col].shift(-SHIFT_FORECAST)

            # Dodajemy szum
            noise = np.random.normal(loc=0.0, scale=FORECAST_ERROR_CONFIG[col], size=len(df))

            # Tworzymy kolumnę _prognoza
            df[f'{col}_prognoza'] = future_weather + noise

            # Clip (korekta fizyczna)
            if col == 'wiatr':
                df[f'{col}_prognoza'] = df[f'{col}_prognoza'].clip(lower=0)
            elif col == 'wilgotność':
                df[f'{col}_prognoza'] = df[f'{col}_prognoza'].clip(lower=0, upper=100)

    # Usuwamy wiersze, które po shifcie mają NaN (czyli ostatnie 72h datasetu)
    df = df.dropna().reset_index(drop=True)

    # Uzupełnianie brakujących kolumn technicznych (np. węgiel) zerami, jeśli ich nie ma w danym kraju
    for col in FINAL_FEATURES:
        if col not in df.columns:
            # Jeśli brakuje kolumny 'prognoza', to znaczy że nie było też kolumny bazowej pogody -> błąd danych
            if 'prognoza' in col:
                print(f"Ostrzeżenie: Brak kolumny pogodowej niezbędnej do modelu: {col}")
            df[col] = 0

    # --- 6. PRZYGOTOWANIE SEKWENCJI ---
    target_ts = pd.Timestamp(target_date)
    rows_at_date = df[df['start_time'] == target_ts]

    if rows_at_date.empty:
        # Ponieważ robimy dropna(), zakres danych się skrócił o 72h.
        print(f"BŁĄD: Data {target_date} nie jest dostępna (może być w strefie uciętej przez shift prognozy).")
        print(f"Dostępne dane kończą się: {df['start_time'].max()}")
        return None

    target_idx = rows_at_date.index[0]

    if target_idx < LOOKBACK:
        print("BŁĄD: Za mało danych historycznych przed wybraną datą.")
        return None

    # Pobranie okna [t-48 : t]
    # Teraz input_data_raw zawiera już kolumny '_prognoza' wygenerowane wyżej
    input_data_raw = df.loc[target_idx - LOOKBACK: target_idx - 1, FINAL_FEATURES]

    # Skalowanie
    try:
        input_data_scaled = scaler.transform(input_data_raw)
    except ValueError as e:
        print(f"BŁĄD Skalera: {e}")
        print("Upewnij się, że lista FINAL_FEATURES jest identyczna jak podczas treningu.")
        return None

    X_input = input_data_scaled.reshape(1, LOOKBACK, len(FINAL_FEATURES))

    # --- 7. PREDYKCJA ---
    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = np.clip(y_pred_scaled[0], 0, 1)

    # --- 8. WYNIKI ---
    avg_24 = np.mean(y_pred[:24])
    avg_48 = np.mean(y_pred[:48])
    avg_72 = np.mean(y_pred)

    if target_idx + FORECAST > len(df):
        print("Ostrzeżenie: Brak pełnych danych rzeczywistych dla całego horyzontu 72h.")
        # Pobieramy tyle ile jest
        y_true = df.loc[target_idx:, 'OZE_Share'].values
    else:
        y_true = df.loc[target_idx: target_idx + FORECAST - 1, 'OZE_Share'].values
    # Rzeczywistość (obsługa krótszych danych jeśli koniec pliku)
    real_24 = np.mean(y_true[:24]) if len(y_true) >= 24 else np.nan
    real_48 = np.mean(y_true[:48]) if len(y_true) >= 48 else np.nan
    real_72 = np.mean(y_true) if len(y_true) > 0 else np.nan

    print(f"Sukces. Wygenerowano prognozę dla: {target_date}")

    return {
        "kraj": kraj,
        "data": target_date,
        # Prognozy
        "pred_24h": avg_24,
        "pred_48h": avg_48,
        "pred_calosc": avg_72,
        # Rzeczywistość
        "real_24h": real_24,
        "real_48h": real_48,
        "real_calosc": real_72,
        # Surowe dane
        "prognoza_raw": y_pred,
        "rzeczywistosc_raw": y_true
    }


# --- PRZYKŁAD UŻYCIA ---
KRAJ = 'Polska'
DATA_DOCELOWA = '2024-12-25 12:00:00'

# wynik = prognozuj_oze_dla_kraju(KRAJ, DATA_DOCELOWA)

# if wynik:
#     # Obliczanie błędu (w punktach procentowych)
#     diff_24 = (wynik['pred_24h'] - wynik['real_24h']) * 100
#     diff_48 = (wynik['pred_48h'] - wynik['real_48h']) * 100
#     diff_72 = (wynik['pred_calosc'] - wynik['real_calosc']) * 100

#     print("\n" + "=" * 75)
#     print(f"RAPORT PORÓWNAWCZY DLA: {wynik['kraj']} (Start: {wynik['data']})")
#     print("=" * 75)
#     print(f"{'HORYZONT':<10} | {'PROGNOZA':<15} | {'RZECZYWISTOŚĆ':<15} | {'RÓŻNICA (pp)':<15}")
#     print("-" * 75)
#     print(f"{'24h':<10} | {wynik['pred_24h']:<15.2%} | {wynik['real_24h']:<15.2%} | {diff_24:+5.2f} p.p.")
#     print(f"{'48h':<10} | {wynik['pred_48h']:<15.2%} | {wynik['real_48h']:<15.2%} | {diff_48:+5.2f} p.p.")
#     print(f"{'72h':<10} | {wynik['pred_calosc']:<15.2%} | {wynik['real_calosc']:<15.2%} | {diff_72:+5.2f} p.p.")
#     print("=" * 75)