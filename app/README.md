

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
```
### **2. Uruchomienie Serwera**
Użyj głównego skryptu dev, aby uruchomić zarówno backend (FastAPI na porcie 8000), jak i frontend (Vite):
```bash
npm run dev
```
•	Frontend (React/Vite): Domyślnie na porcie 3000 
•	Backend (FastAPI): Na porcie 8000.
________________________________________
## **Frontend: React i Routing**

Frontend zarządza interfejsem użytkownika oraz przepływem danych w aplikacji.

### **Routing (App.tsx)**

Aplikacja korzysta z `HashRouter` i definiuje dwie główne ścieżki:

- `/` — **Home**: strona startowa z wyborem typu obliczeń  
- `/dashboard` — **Dashboard**: strona z wizualizacją danych  

---

### **Przepływ nawigacji i przekazywanie danych**

1. Użytkownik wybiera typ obliczeń (`LIGHT`, `MEDIUM`, `HEAVY`) w komponencie **Home**, co wywołuje funkcję `handleStartCompute`.
2. Ustawiany jest stan `loadingType`, który powoduje wyświetlenie komponentu **LoadingScreen**.
3. Po zakończeniu symulowanego ładowania uruchamiana jest funkcja `handleLoadingComplete`.
4. Wybrany typ obliczeń zostaje przekazany do komponentu **Dashboard** poprzez mechanizm `state` w `navigate`:

```ts
navigate('/dashboard', {
  state: { computeType: typeToSend }
});
________________________________________
## 🟢 **Backend: FastAPI i Dane**

Backend w Pythonie pełni rolę serwera API, który komunikuje się z frontendem i zwraca dane o dostępnych centrach danych oraz ich parametrach energetycznych.

---

### **Konfiguracja API (`main.py`)**

- **CORS** — umożliwia komunikację z frontendem (domyślnie z `http://localhost:3000`).  
  Konfiguracja jest otwarta na wszystkie metody i nagłówki.

- **Model danych** — klasa `ComputeRequest` (Pydantic) definiuje strukturę danych wejściowych dla żądań POST:

```python
class ComputeRequest(BaseModel):
    compute_type: str

### **Endpoint zwraca obiekt JSON zawierający:**

- `status`
- `compute_type`
- `data` — listę dostępnych krajów wraz z ich parametrami energetycznymi

Przykład:

```json
{
  "status": "ok",
  "compute_type": "HEAVY",
  "data": [
    {
      "country": "Norway",
      "renewables": 98,
      "status": "online"
    },
    {
      "country": "France",
      "renewables": 78,
      "status": "maintenance"
    }
  ]
}
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






