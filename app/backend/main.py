from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

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
    # To wyświetli Ci w terminalu backendu co kliknął użytkownik
    print(f"Otrzymano zapytanie dla: {request.computeType}") 
    match request.computeType:
        case "Light computing":
            steps = 12
        case "Medium computing":
            steps = 36
        case "Heavy computing":
            steps = 72
        case _:
            steps = 12
    
    return {
        "status": "success",
        "data": COUNTRIES_DATA
    }