from datetime import datetime
from meteostat import Stations, Hourly  # Zmiana z Daily na Hourly
import pandas as pd

# 1. Konfiguracja zakresu dat
# Uwaga: Pobieranie danych godzinowych dla 5 lat dla 20 stacji to dużo danych.
# Może to chwilę potrwać.
start = datetime(2020, 1, 1)
end = datetime(2025, 12, 31)

# 2. Pobranie i filtracja stacji
stations = Stations()
stations = stations.region('PL')

# Kluczowa zmiana: sprawdzamy inventory 'hourly'
stations = stations.inventory('hourly', True)
stations_df = stations.fetch()

# FILTRACJA:
# a) Tylko stacje z kodem WMO
stations_df = stations_df[stations_df['wmo'].notnull()]

# b) Tylko stacje, które mają dane godzinowe po 2024 roku
# ZMIANA: używamy kolumny 'hourly_end', a nie 'daily_end'
stations_df = stations_df[stations_df['hourly_end'] > datetime(2024, 1, 1)]

# ZAPEWNIENIE RÓŻNORODNOŚCI MIEJSC:
# Sortujemy od Północy (morze) na Południe (góry)
stations_df = stations_df.sort_values('latitude', ascending=False)

# Wybieramy równo 20 stacji rozłożonych geograficznie
if len(stations_df) > 20:
    step = len(stations_df) // 20
    stations_subset = stations_df.iloc[::step].head(20)
else:
    stations_subset = stations_df

print(f"Wybrano {len(stations_subset)} stacji z danymi godzinowymi:")
print(stations_subset[['name', 'region', 'latitude']].to_string())

# 3. Pobranie danych GODZINOWYCH dla wybranej grupy
print("\nPobieranie danych godzinowych (to może chwilę potrwać)...")
# ZMIANA: Klasa Hourly
data = Hourly(stations_subset, start, end)
data = data.fetch()

# 4. Sprawdzenie i agregacja
if data.empty:
    print("Brak danych pomiarowych.")
else:
    # Uśrednianie przestrzenne (dla całej Polski) po indeksie czasu
    poland_avg = data.groupby('time').mean()

    print("\n--- Wynik: Uśrednione dane godzinowe (2020-2025) ---")

    # ZMIANA: W danych godzinowych kolumny nazywają się inaczej:
    # temp (temperatura), dwpt (punkt rosy), rhum (wilgotność), prcp (opady), wspd (wiatr)
    available_cols = [c for c in ['temp', 'rhum', 'wspd', 'prcp'] if c in poland_avg.columns]
    print(poland_avg[available_cols].tail())

    # Zapis do pliku
    filename = 'polska_pogoda_godzinowa.csv'
    poland_avg.to_csv(filename)
    print(f"\nDane zapisano do pliku: {filename}")