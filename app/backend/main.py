from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.join(current_dir, '..', '..')


if root_dir not in sys.path:

    sys.path.insert(0, root_dir)

try:
    import uzycie_modelu
except ImportError as e:
    print(f"Error importing module: {e}")



# To jest ta kluczowa linijka, której szuka uvicorn:
app = FastAPI()

# Konfiguracja CORS - pozwala Twojemu Reactowi (localhost:3000) łączyć się z Pythonem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model danych, które przychodzą z Reacta (Light, Medium, Heavy)
class ComputeRequest(BaseModel):
    computeType: str

# Twoje dane o krajach (przeniesione z constants.ts)
COUNTRIES_DATA = [
    {
        "id": "pl",
        "name": "Poland",
        "lat": 52.06, # Środek Polski
        "lon": 19.48,
        "renewablePercentage": 20,
        "facilities": ["Atman", "Equinix Warsaw", "Beyond.pl", "Data4 Poland", "3S Data Center", "COIG / WASKO"]
    },
    {
        "id": "de",
        "name": "Germany",
        "lat": 51.16, # Środek Niemiec
        "lon": 10.45,
        "renewablePercentage": 52,
        "facilities": ["Equinix Frankfurt", "Hetzner Falkenstein", "Hetzner Nuremberg", "Interxion Frankfurt", "Global Switch Frankfurt", "Vantage Frankfurt", "Colt DCS Frankfurt"]
    },
    {
        "id": "no",
        "name": "Norway",
        "lat": 60.47, # Środek Norwegii
        "lon": 8.46,
        "renewablePercentage": 98,
        "facilities": ["Lefdal Mine Datacenter", "Bulk Campus N01", "Green Mountain DC1", "Green Mountain DC2", "Green Mountain DC3"]
    },
    {
        "id": "fr",
        "name": "France",
        "lat": 46.60, # Środek Francji
        "lon": 1.88,
        "renewablePercentage": 78,
        "facilities": ["Data4 Paris-Saclay", "OVH Gravelines", "OVH Roubaix", "OVH Strasbourg", "Global Switch Paris", "Interxion Paris"]
    },
    {
        "id": "nl",
        "name": "Netherlands",
        "lat": 52.13, # Środek Holandii
        "lon": 5.29,
        "renewablePercentage": 45,
        "facilities": ["Google Eemshaven", "Equinix Amsterdam", "Interxion Amsterdam"]
    },
    {
        "id": "it",
        "name": "Italy",
        "lat": 41.87, # Środek Włoch
        "lon": 12.56,
        "renewablePercentage": 42,
        "facilities": ["Aruba Global Cloud Data Center (IT3)", "Aruba IT1", "Aruba IT2", "SuperNAP Italia"]
    }
]

# Endpoint API
@app.post("/api/energy-mix")
def get_energy_data(request: ComputeRequest):

    DATA_DOCELOWA = '2024-12-25 12:00:00'

    PRED_MAPOWANIE = {
        "Light Computing": "pred_24h",
        "Medium Computing": "pred_48h",
        "Heavy Computing": "pred_calosc"
    }


    KRAJE_I_ZMIENNE = {
        "Polska": "ren_Poland",
        "Niemcy": "ren_Germany",
        "Norwegia": "ren_Norway",
        "Francja": "ren_France",
        "Holandia": "ren_Netherlands",
        "Włochy": "ren_Italy"
    }

    klucz_predykcji = PRED_MAPOWANIE.get(request.computeType, None)
    wyniki = {}
    print(f"Obliczanie prognoz dla typu zużycia: {request.computeType}")
    for nazwa_kraju, nazwa_zmiennej in KRAJE_I_ZMIENNE.items():
        if klucz_predykcji:
                # Ten blok try/except powinien być wcięty pod IF
                try:
                    wartosc = (uzycie_modelu.prognozuj_oze_dla_kraju(nazwa_kraju, DATA_DOCELOWA)[klucz_predykcji] * 100).round(2)
                except Exception as e:
                    print(f"Błąd podczas pobierania danych dla kraju {nazwa_kraju}: {e}")
                    wartosc = 0
        else:
        # Ten blok else jest poprawnie na tym samym poziomie co IF
            wartosc = 1
        
        wyniki[nazwa_zmiennej] = wartosc
    
    print("Wyniki prognoz energii odnawialnej:", wyniki)
    
    
    COUNTRIES_DATA = [
    {
        "id": "pl",
        "name": "Poland",
        "lat": 52.06, # Środek Polski
        "lon": 19.48,
        "renewablePercentage": wyniki.get("ren_Poland", 0),
        "facilities": ["Atman", "Equinix Warsaw", "Beyond.pl", "Data4 Poland", "3S Data Center", "COIG / WASKO"]
    },
    {
        "id": "de",
        "name": "Germany",
        "lat": 51.16, # Środek Niemiec
        "lon": 10.45,
        "renewablePercentage": wyniki.get("ren_Germany", 0),
        "facilities": ["Equinix Frankfurt", "Hetzner Falkenstein", "Hetzner Nuremberg", "Interxion Frankfurt", "Global Switch Frankfurt", "Vantage Frankfurt", "Colt DCS Frankfurt"]
    },
    {
        "id": "no",
        "name": "Norway",
        "lat": 60.47, # Środek Norwegii
        "lon": 8.46,
        "renewablePercentage": wyniki.get("ren_Norway", 0),
        "facilities": ["Lefdal Mine Datacenter", "Bulk Campus N01", "Green Mountain DC1", "Green Mountain DC2", "Green Mountain DC3"]
    },
    {
        "id": "fr",
        "name": "France",
        "lat": 46.60, # Środek Francji
        "lon": 1.88,
        "renewablePercentage": wyniki.get("ren_France", 0),
        "facilities": ["Data4 Paris-Saclay", "OVH Gravelines", "OVH Roubaix", "OVH Strasbourg", "Global Switch Paris", "Interxion Paris"]
    },
    {
        "id": "nl",
        "name": "Netherlands",
        "lat": 52.13, # Środek Holandii
        "lon": 5.29,
        "renewablePercentage": wyniki.get("ren_Netherlands", 0),
        "facilities": ["Google Eemshaven", "Equinix Amsterdam", "Interxion Amsterdam"]
    },
    {
        "id": "it",
        "name": "Italy",
        "lat": 41.87, # Środek Włoch
        "lon": 12.56,
        "renewablePercentage": wyniki.get("ren_Italy", 0),
        "facilities": ["Aruba Global Cloud Data Center (IT3)", "Aruba IT1", "Aruba IT2", "SuperNAP Italia"]
    }
    ]
    
    return {
        "status": "success",
        "data": COUNTRIES_DATA
    }