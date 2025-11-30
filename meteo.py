
from datetime import datetime
from meteostat import Stations, Hourly
import pandas as pd

# 1. Konfiguracja zakresu dat
start = datetime(2022, 10, 10)
end = datetime(2025, 10, 22)

# 2. Pobranie i filtracja stacji
stations = Stations()
stations = stations.region('NO')

stations = stations.inventory('hourly', True)
stations_df = stations.fetch()

stations_df = stations_df[stations_df['wmo'].notnull()]
stations_df = stations_df[stations_df['hourly_end'] > datetime(2024, 1, 1)]
stations_df = stations_df.sort_values('latitude', ascending=False)

if len(stations_df) > 20:
    step = len(stations_df) // 20
    stations_subset = stations_df.iloc[::step].head(20)
else:
    stations_subset = stations_df

print(f"Wybrano {len(stations_subset)} stacji z danymi godzinowymi:")
print(stations_subset[['name', 'region', 'latitude']].to_string())

# 3. Pobranie danych GODZINOWYCH z meteostat
print("\nPobieranie danych godzinowych Meteostat...")
data = Hourly(stations_subset.index.tolist(), start, end).fetch()

if data.empty:
    print("Brak danych pomiarowych.")
    exit()

# 4. Agregacja
poland_avg = data.groupby('time').mean()

# --- Usuwamy niechciane kolumny (TSUN ZOSTAJE!)
to_remove = ['dwpt', 'snow', 'wdir', 'wpgt', 'coco']
for col in to_remove:
    if col in poland_avg.columns:
        poland_avg = poland_avg.drop(columns=col)

# --- Tylko wybrane kolumny — BEZ sun_percent
wanted = ['temp', 'rhum', 'wspd', 'prcp', 'pres', 'tsun']
final_cols = [c for c in wanted if c in poland_avg.columns]
poland_avg = poland_avg[final_cols]

print("\nDane Meteostat OK.")

# --- Zaokrąglenie + nazwy PL
poland_avg = poland_avg.round(2)

poland_avg = poland_avg.rename(columns={
    "temp": "temperatura",
    "rhum": "wilgotność",
    "wspd": "wiatr",
    "prcp": "opady",
    "pres": "ciśnienie",
    "tsun": "Tsun_min"   # minuty słońca w ciągu godziny
})

# --- Uzupełnienie braków
poland_avg = poland_avg.fillna(0.0)

# --- Dodanie czytelnej kolumny czasu
poland_avg.insert(0, "czas", poland_avg.index)
poland_avg["czas"] = poland_avg["czas"].dt.strftime("%Y-%m-%d %H:%M")

# =====================================================================
#                        ZAPIS DO CSV
# =====================================================================

filename = "Norway_pogoda_godzinowa.csv"
poland_avg.to_csv(filename, index=False)
print(poland_avg.info())
print(f"\n✔ Zapisano plik: {filename}")
print(poland_avg.head())

