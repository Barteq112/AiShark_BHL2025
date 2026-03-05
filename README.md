# GreenOps Engine

**Projekt zrealizowany przez zespół AI Sharks w ramach hackathonu BHL 2025. Całość rozwiązania została zaprojektowana i zaimplementowana w ciągu zaledwie 24 godzin.**

## O projekcie

GreenOps Engine to innowacyjny system pozwalający na redukcję śladu węglowego infrastruktury IT poprzez inteligentne planowanie i delegowanie zadań obliczeniowych. Narzędzie decyduje, w którym centrum danych najlepiej wykonać dane obliczenia, opierając się na predykcji udziału Odnawialnych Źródeł Energii (OZE) w ogólnej generacji prądu.

Dzięki GreenOps Engine firmy mogą budować swoje rozwiązania w sposób zrównoważony, redukując emisję CO2 bez kompromisów w zakresie wydajności obliczeniowej.

## Problem Biznesowy i Motywacja

Rozwiązanie odpowiada na rosnące wymogi prawne i rynkowe związane z ekologią:
- **Obowiązek raportowania ESG i dyrektywa CSRD**: Od 2024 do 2026 roku kolejne grupy przedsiębiorstw w UE są obejmowane obowiązkiem szczegółowego raportowania niefinansowego.
- **Wymogi łańcucha dostaw**: Firmy muszą raportować emisje swoich dostawców. Brak dostarczania danych ESG może skutkować utratą kontraktów B2B.
- **Koszty finansowe**: Niski rating ESG prowadzi do droższego kredytowania i odpływu inwestorów. Dodatkowo, rosnące koszty emisji CO2 (60-80 EUR za tonę) bezpośrednio obciążają budżety operacyjne.

Przeniesienie obliczeń w czasie i przestrzeni przy użyciu naszego narzędzia pozwala zredukować koszty emisji CO2 nawet o 50%, a także pomaga ustabilizować sieci energetyczne poprzez zagospodarowanie nadmiarowej energii z OZE (np. fotowoltaiki w słoneczne dni).

## Grupa Docelowa

System jest skierowany w szczególności do:
- **Software house'ów i firm AI**, które ponoszą wysokie koszty przetwarzania w chmurze (trening modeli LLM, render farmy).
- **Dużych korporacji** objętych dyrektywą CSRD (potrzeba raportowania i audytów).
- **Instytutów badawczych i uniwersytetów**, które muszą spełniać wymogi ekologiczne w ramach grantów.
- **Mniejszych dostawców chmurowych**, chcących zyskać przewagę konkurencyjną nad gigantami technologicznymi.

## Architektura i Modele Predykcyjne

System opiera się na ciągłym pobieraniu i analizie danych z systemów energetycznych oraz pogodowych w celu wygenerowania rekomendacji dla algorytmu optymalizującego alokację zasobów (Light, Medium, Heavy Computing).

**Wykorzystane dane:**
- **ENTSO-E**: Dane o generacji energii z podziałem na źródła.
- **Open-Meteo**: Dane pogodowe (temperatura, wiatr, opady, ciśnienie, wilgotność, nasłonecznienie).

W ramach inżynierii danych zastosowano m.in. kodowanie cykliczne dla zmiennych czasowych (godziny, dni, miesiące). 

**Wytrenowane modele uczenia maszynowego:**
Podczas 24-godzinnego hackathonu przetestowaliśmy i zaimplementowaliśmy następujące modele do prognozowania udziału OZE:
- **Ridge (Regresja Grzbietowa)** - jako baseline i model o wysokiej interpretowalności do analizy istotności cech.
- **XGBoost (eXtreme Gradient Boosting)** - do wychwytywania nieliniowych zależności.
- **LSTM (Long Short-Term Memory)** - model głębokiego uczenia zoptymalizowany pod kątem predykcji szeregów czasowych.

## Plany Rozwoju (Future Work)

- Transfer rozwiązania na mniejsze, precyzyjniejsze regiony, co pozwoli na jeszcze dokładniejsze przewidywania dla konkretnych centrów obliczeniowych.
- Dalsze dopracowanie modeli predykcyjnych (np. uwzględnienie większej ilości zmiennych zewnętrznych).
- Integracja z telemetrią samych centrów danych w celu balansowania obciążenia i unikania przeciążania tych najbardziej ekologicznych serwerowni.

## Instalacja i Uruchomienie

Wszystkie wymagania systemowe (requirements) oraz szczegółowa instrukcja uruchomienia aplikacji i modeli znajdują się w dedykowanym pliku README wewnątrz katalogu aplikacji.

Przejdź do instrukcji technicznej: **[app/readme.md](app/readme.md)**
