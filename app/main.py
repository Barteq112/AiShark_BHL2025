# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Konfiguracja CORS (pozwala Reactowi rozmawiać z Pythonem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adres Twojego Reacta
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model danych, które przychodzą z Reacta
class ComputeRequest(BaseModel):
    compute_type: str  # "LIGHT", "MEDIUM", "HEAVY"

# Baza danych w Pythonie
COUNTRIES_DATA = [
    { "id": "pl", "name": "Poland", "renewablePercentage": 20, "lat": 52.06, "lon": 19.48, "facilities": ["Atman", "Equinix Warsaw"] },
    { "id": "no", "name": "Norway", "renewablePercentage": 98, "lat": 60.47, "lon": 8.46, "facilities": ["Lefdal Mine", "Green Mountain"] },
    # ... reszta krajów
]

@app.post("/api/get-energy-data")
def get_energy_data(request: ComputeRequest):
    print(f"Otrzymano żądanie typu: {request.compute_type}")
    
    # Tutaj możesz dodać logikę, np. jeśli HEAVY, to pokaż tylko kraje z > 80% OZE
    # Na razie zwracamy wszystko:
    return {
        "status": "success",
        "compute_type": request.compute_type,
        "data": COUNTRIES_DATA
    }