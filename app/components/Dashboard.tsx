import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom'; // <--- DODAJ TEN IMPORT
import EuropeMap from './EuropeMap';
import Sidebar from './Sidebar';
import { CountryNode } from '../types';
import { LayoutDashboard, ArrowLeft, Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';

const Dashboard: React.FC = () => {
  const [selectedCountry, setSelectedCountry] = useState<CountryNode | null>(null);
  const [countries, setCountries] = useState<CountryNode[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  
  // 1. Odbieramy dane przesłane z App.tsx
  const location = useLocation();
  // Jeśli ktoś wejdzie bezpośrednio linkiem, domyślnie ustawiamy 'LIGHT'
  const computeType = location.state?.computeType || 'LIGHT';

  useEffect(() => {
    const fetchEnergyData = async () => {
      try {
        console.log("Wysyłam do backendu typ:", computeType); // Log dla sprawdzenia

        const response = await fetch('http://localhost:8000/api/energy-mix', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          // 2. Wstawiamy dynamiczną zmienną zamiast sztywnego tekstu
          body: JSON.stringify({ computeType: computeType }), 
        });

        const result = await response.json();
        
        if (result.status === 'success') {
          setCountries(result.data);
        }
      } catch (error) {
        console.error("Błąd połączenia z backendem:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchEnergyData();
  }, [computeType]); // Dodajemy computeType do zależności

  if (loading) {
    return (
      <div className="h-screen w-screen bg-slate-950 flex items-center justify-center text-emerald-500">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 animate-spin" />
          <span className="font-mono text-sm tracking-widest">CONNECTING TO GRID...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-slate-950 flex flex-col text-slate-100 overflow-hidden">
      {/* Mini Header */}
      <header className="h-14 px-6 border-b border-slate-800 flex items-center justify-between bg-slate-950 z-20">
        <div className="flex items-center gap-4">
            <Link to="/" className="text-slate-400 hover:text-emerald-400 transition-colors">
                <ArrowLeft className="w-5 h-5" />
            </Link>
            <div className="h-6 w-px bg-slate-800"></div>
            <h1 className="font-bold text-sm tracking-wide uppercase text-slate-200 flex items-center gap-2">
                <LayoutDashboard className="w-4 h-4 text-emerald-500" />
                Network Status
            </h1>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono text-emerald-500">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            System Operational
        </div>
      </header>

      {/* Main Grid */}
      <div className="flex-grow flex flex-col md:flex-row overflow-hidden">
        
        {/* Map Area */}
        <div className="w-full md:w-[70%] h-full relative bg-slate-900 p-4">
            <EuropeMap 
                countries={countries} // Przekazujemy dane z backendu do mapy
                onCountrySelect={setSelectedCountry} 
                selectedCountry={selectedCountry}
            />
        </div>

        {/* Sidebar Area */}
        <div className="w-full md:w-[30%] h-full bg-slate-950 relative z-10 shadow-2xl shadow-black">
            <Sidebar 
                selectedCountry={selectedCountry} 
                allCountries={countries} // Przekazujemy dane z backendu do listy
                onSelectCountry={setSelectedCountry}
            />
        </div>

      </div>
    </div>
  );
};

export default Dashboard;