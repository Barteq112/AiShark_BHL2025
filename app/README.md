

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
3. Run the app:
   `npm run dev`

# **GreenOps Engine: Platforma Zrównoważonej Chmury**

**GreenOps Engine** to platforma do wizualizacji wykorzystania energii odnawialnej w europejskich centrach danych, mająca na celu promowanie zrównoważonej chmury obliczeniowej.

---

## **Struktura Projektu**

Projekt jest aplikacją full-stack zbudowaną w oparciu o **React (frontend)** i **FastAPI (backend)** z wykorzystaniem **Vite** jako systemu budowania.

| Plik/Katalog | Opis |
|--------------|------|
| **App.tsx** | Główny komponent React, definiuje routing i logikę przekazywania stanu nawigacji. |
| **index.html** | Główny plik HTML, konfiguracja Tailwind CSS, importmap i ładowanie skryptów. |
| **constants.ts** | Definicje statycznych danych geograficznych i energetycznych krajów europejskich. |
| **package.json** | Konfiguracja projektu, zależności i skrypty uruchamiania. |
| **main.py** | Serwer backendowy FastAPI, obsługa endpointów API i logiki danych. |

---

## **Uruchomienie Projektu**

Projekt wykorzystuje skrypty zdefiniowane w `package.json`, aby jednocześnie uruchomić frontend i backend za pomocą **concurrently**.

### **Wymagania**

- Node.js i npm  
- Python 3.x  
- Zainstalowane pakiety Pythona: FastAPI, Uvicorn  


### **1. Instalacja zależności**

Instalacja zależności frontendu:

```bash
npm install
 
2. Uruchomienie Serwera
Użyj głównego skryptu dev, aby uruchomić zarówno backend (FastAPI na porcie 8000), jak i frontend (Vite):
Bash
npm run dev
•	Frontend (React/Vite): Domyślnie na porcie 3000 
•	Backend (FastAPI): Na porcie 8000.
________________________________________
Frontend: React i Routing
Frontend zarządza interfejsem użytkownika i przepływem aplikacji:
Routing (App.tsx)
Aplikacja używa HashRouter i definiuje dwie główne ścieżki:
•	/: Komponent Home – strona startowa z wyborem typu obliczeń.
•	/dashboard: Komponent Dashboard – strona z wizualizacją danych.
Przepływ Nawigacji i Przekazywanie Danych
1.	Użytkownik wybiera typ obliczeń (LIGHT, MEDIUM, HEAVY) w komponencie Home, co wywołuje funkcję handleStartCompute.
2.	Stan loadingType jest ustawiany, co powoduje wyświetlenie komponentu LoadingScreen.
3.	Po zakończeniu symulowanego ładowania, wywoływana jest funkcja handleLoadingComplete.
4.	Wybrany typ obliczeń jest przekazywany do komponentu Dashboard za pomocą state w obiekcie nawigacji (Maps('/dashboard', { state: { computeType: typeToSend } })).
________________________________________
Backend: FastAPI i Dane
Backend w Pythonie pełni rolę serwera API:
Konfiguracja API (main.py)
•	CORS: Umożliwia komunikację z frontendem (domyślnie z http://localhost:3000), konfiguracja jest otwarta na wszystkie metody i nagłówki.
•	Model Danych: Klasa Pydantic ComputeRequest definiuje oczekiwane dane wejściowe dla żądań POST (pole compute_type).
Endpoint API
•	POST /api/get-energy-data:
o	Wejście: Ciało żądania zawierające compute_type (np. "HEAVY").
o	Wyjście: Zwraca obiekt JSON z kluczami status, compute_type oraz data (listą dostępnych krajów).
o	Uwaga: Obecnie endpoint zwraca statyczne dane, ale jest przygotowany do implementacji logiki filtrowania w oparciu o wartość compute_type (np. sugerowanie krajów z wyższym % OZE dla ciężkich obliczeń).
________________________________________
Dane Statyczne (constants.ts)
Plik constants.ts zawiera dane o europejskich centrach danych używane w frontendzie:
Kraj	% OZE	Centra Danych	Status
Norway (no)	98%	Lefdal Mine, Bulk Campus, Green Mountain (DC1, DC2, DC3)	online
Germany (de)	52%	Equinix, Hetzner, Interxion, Global Switch, Vantage, Colt DCS (Frankfurt)	online
France (fr)	78%	Data4, OVH (Gravelines, Roubaix, Strasbourg), Global Switch, Interxion (Paris)	maintenance
Poland (pl)	25%	Atman, Equinix Warsaw, Beyond.pl, Data4 Poland, 3S Data Center, COIG / WASKO	online
Definiuje również URL do mapy GeoJSON dla Europy: https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson.





