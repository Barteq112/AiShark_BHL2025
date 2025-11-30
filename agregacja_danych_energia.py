import pandas as pd
import numpy as np
import os

kraj = "Norway_energy"
# Definicja ścieżki do folderu
base_path = os.path.join('data', kraj)

# Lista lat do przetworzenia (od 2022 do 2025 włącznie)
lata = range(2022, 2025)

dataframes = []

for rok in lata:
    # Tworzenie nazwy pliku zgodnie z podanym wzorcem
    # Zakładam, że pliki mają rozszerzenie .csv
    nazwa_pliku = f"Actual Generation per Production Type_{rok}01010000-{rok + 1}01010000.csv"
    sciezka_pliku = os.path.join(base_path, nazwa_pliku)

    # Sprawdzenie, czy plik istnieje
    if os.path.exists(sciezka_pliku):
        try:
            print(f"Wczytuję plik: {nazwa_pliku}")
            # Wczytanie pliku CSV.
            # Jeśli Twoje pliki mają inny separator (np. średnik), dodaj argument sep=';'
            df = pd.read_csv(sciezka_pliku)
            dataframes.append(df)
        except Exception as e:
            print(f"Błąd przy wczytywaniu {nazwa_pliku}: {e}")
    else:
        print(f"Ostrzeżenie: Plik nie istnieje -> {sciezka_pliku}")

# Łączenie danych
if dataframes:
    full_df = pd.concat(dataframes, ignore_index=True)

    # --- CZYSZCZENIE DATY ---
    full_df['start_time'] = full_df['MTU'].str.split(' - ').str[0]
    full_df['start_time'] = pd.to_datetime(full_df['start_time'], dayfirst=True)
    full_df = full_df[full_df['start_time'].dt.minute == 0]
    full_df = full_df.drop(columns=['MTU'])

    # --- USUWANIE ZBĘDNYCH KOLUMN ---
    kolumny_do_usuniecia = [
        'Energy storage - Actual Aggregated [MW]',
        'Fossil Oil shale - Actual Aggregated [MW]',
        'Fossil Peat - Actual Aggregated [MW]',
        'Geothermal - Actual Aggregated [MW]',
        'Hydro Pumped Storage - Actual Consumption [MW]',
        'Marine - Actual Aggregated [MW]',
        'Nuclear - Actual Aggregated [MW]',
        'Other - Actual Aggregated [MW]',
        'Other renewable - Actual Aggregated [MW]',
        'Waste - Actual Aggregated [MW]',
        'Wind Offshore - Actual Aggregated [MW]',
        'Area'
    ]
    full_df = full_df.drop(columns=kolumny_do_usuniecia, errors='ignore')



    # 2. Konwersja na typ datetime (ważne: dayfirst=True dla formatu polskiego 01.01.2022)
    full_df['start_time'] = pd.to_datetime(full_df['start_time'], dayfirst=True)

    # 3. Filtrowanie: zostawiamy tylko wiersze, gdzie minuta wynosi 0
    full_df = full_df[full_df['start_time'].dt.minute == 0]

    # Sprawdzenie wyniku
    print(f"Liczba wierszy po filtracji: {len(full_df)}")
    print(full_df['start_time'].head())

    # Lista kolumn produkcyjnych (wszystko poza start_time)
    cols_to_convert = [col for col in full_df.columns if col != 'start_time']

    # 1. Najpierw wymuszamy konwersję na liczby.
    # 'coerce' zamieni wszystkie "n/e", "-", spacje i śmieci na NaN (Not a Number)
    for col in cols_to_convert:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Diagnostyka: Zobaczmy ile mamy braków w każdej kolumnie
    print("\n--- DIAGNOSTYKA BRAKÓW DANYCH ---")
    print(full_df.isnull().sum())

    # 2
    full_df = full_df.fillna(0)

    # Resetowanie indeksu
    full_df = full_df.reset_index(drop=True)

    # Sprawdzenie wyniku
    print(f"Liczba wierszy po usunięciu nulli: {len(full_df)}")

    # Pobieramy listę kolumn, które powinny być liczbowe (wszystkie poza Area i start_time)
    cols_to_convert = [col for col in full_df.columns if col not in ['Area', 'start_time']]

    # Konwersja na liczby
    for col in cols_to_convert:
        full_df[col] = pd.to_numeric(full_df[col])

    print(full_df.info())
    # --- ZMIANA NAZW KOLUMN (NOWY FRAGMENT) ---
    # 1. Usuwamy " [MW]"
    full_df.columns = full_df.columns.str.replace(' [MW]', '', regex=False)
    # 2. Usuwamy " - Actual Aggregated" (dla czytelności, zostaje np. samo "Biomass")
    full_df.columns = full_df.columns.str.replace(' - Actual Aggregated', '', regex=False)

    print("Zmienione nazwy kolumn:")
    print(full_df.columns.tolist())

    # 1. Zdefiniowanie kolumn z produkcją (wszystkie poza 'Area' i 'start_time')
    cols_production = [col for col in full_df.columns if col not in ['Area', 'start_time']]

    # 2. Stworzenie kolumny z sumą całkowitą
    full_df['Total Generation [MW]'] = full_df[cols_production].sum(axis=1)
    full_df = full_df[full_df['Total Generation [MW]'] > 0]
    # 3. Zamiana poszczególnych kolumn na wartości względne (0.0 - 1.0)
    # Dzielimy każdą kolumnę produkcyjną przez nowo powstałą sumę
    for col in cols_production:
        full_df[col] = full_df[col] / full_df['Total Generation [MW]']

    # Sprawdzenie wyniku
    print("Podgląd danych (wartości powinny być < 1):")
    print(full_df[cols_production].head())

    print("\nPodgląd kolumny z sumą (wartości w MW):")
    print(full_df['Total Generation [MW]'].head())

    # Weryfikacja: suma udziałów w wierszu powinna wynosić 1.0 (lub bardzo blisko przez zaokrąglenia)
    # print(full_df[cols_production].sum(axis=1).head())


    print("\nSukces! Dane połączone i posprzatane")
    print(f"Liczba wierszy: {len(full_df)}")
    print("Podgląd danych:")
    print(full_df.head())
    print(full_df.info())
    # Zapisanie do pliku 'polaczone_dane.csv'
    # index=False zapobiega zapisywaniu numerów wierszy jako osobnej kolumny
    full_df.to_csv(f'Kraje_2022_2025/dane_energia_{kraj}.csv', index=False)
else:
    print("Nie udało się wczytać żadnych danych.")