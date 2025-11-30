import React from 'react';
import { CountryNode } from '../types';
import { Building2, Leaf, Globe, Server, CheckCircle2, Trophy, ArrowRight, Zap } from 'lucide-react';

interface SidebarProps {
  selectedCountry: CountryNode | null;
  allCountries: CountryNode[];
  onSelectCountry: (node: CountryNode) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedCountry, allCountries, onSelectCountry }) => {
  
  if (!selectedCountry) {
    // Logic to find top 3 greenest regions
    const topRegions = [...allCountries]
        .sort((a, b) => b.renewablePercentage - a.renewablePercentage)
        .slice(0, 3);

    return (
      <div className="h-full flex flex-col border-l border-slate-800 bg-slate-900 overflow-y-auto">
        <div className="p-6 border-b border-slate-800">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Globe className="w-5 h-5 text-emerald-400" />
              Network Overview
            </h2>
            <p className="text-slate-400 text-sm mt-1">Select a region on the map or from the list below to view details.</p>
        </div>

        <div className="p-6">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <Trophy className="w-4 h-4 text-yellow-500" /> Top Green Regions
            </h3>

            <div className="space-y-4">
                {topRegions.map((country, index) => (
                    <button 
                        key={country.id}
                        onClick={() => onSelectCountry(country)}
                        className="w-full text-left group bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-emerald-500/50 p-4 rounded-xl transition-all duration-200"
                    >
                        <div className="flex justify-between items-start mb-2">
                            <div>
                                <span className="text-xs font-mono text-emerald-500 mb-1 block">Rank #{index + 1}</span>
                                <h4 className="font-bold text-slate-100 flex items-center gap-2">
                                    {country.name}
                                </h4>
                            </div>
                            <div className="bg-slate-900/80 px-2 py-1 rounded text-emerald-400 font-mono font-bold text-sm border border-slate-700 flex items-center gap-1">
                                <Zap className="w-3 h-3" fill="currentColor" />
                                {country.renewablePercentage}%
                            </div>
                        </div>
                        
                        <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden mb-3 border border-slate-800">
                            <div 
                                className="h-full bg-emerald-500 group-hover:bg-emerald-400 transition-colors shadow-[0_0_8px_rgba(16,185,129,0.5)]" 
                                style={{ width: `${country.renewablePercentage}%` }}
                            ></div>
                        </div>

                        <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>{country.facilities.length} Facilities available</span>
                            <span className="flex items-center group-hover:text-emerald-400 transition-colors">
                                Select Region <ArrowRight className="w-3 h-3 ml-1" />
                            </span>
                        </div>
                    </button>
                ))}
            </div>

            <div className="mt-8 p-4 bg-slate-800/30 rounded-lg border border-slate-800 text-center">
                <p className="text-xs text-slate-500">
                    Calculated based on real-time grid mix data and renewable power purchase agreements (PPAs).
                </p>
            </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col border-l border-slate-800 bg-slate-900 overflow-y-auto">
      <div className="p-6 border-b border-slate-800 bg-slate-900 sticky top-0 z-10">
        <button 
            onClick={() => onSelectCountry(null as any)} // A bit of a hack to clear selection if we wanted a back button, though Map handles click-off usually
            className="mb-4 text-xs text-slate-500 hover:text-emerald-400 flex items-center gap-1 transition-colors"
        >
            <ArrowRight className="w-3 h-3 rotate-180" /> Back to Overview
        </button>

        <div className="flex items-start justify-between">
            <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  {selectedCountry.name}
                  {selectedCountry.renewablePercentage >= 80 && (
                      <Leaf className="w-5 h-5 text-emerald-400" fill="currentColor" fillOpacity={0.2} />
                  )}
                </h2>
                <p className="text-slate-400 text-sm mt-1">Sovereign Zone</p>
            </div>
            <div className={`px-2 py-1 rounded text-xs font-bold uppercase tracking-wider ${selectedCountry.status === 'online' ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-800' : 'bg-amber-900/30 text-amber-400 border border-amber-800'}`}>
                {selectedCountry.status}
            </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-3">
            <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                <span className="text-xs text-slate-400 block mb-1">Renewable Usage</span>
                <span className={`text-xl font-mono font-bold ${selectedCountry.renewablePercentage > 50 ? 'text-emerald-400' : 'text-amber-500'}`}>
                    {selectedCountry.renewablePercentage}%
                </span>
            </div>
            <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                <span className="text-xs text-slate-400 block mb-1">Facilities</span>
                <span className="text-xl font-mono font-bold text-slate-200">
                    {selectedCountry.facilities.length}
                </span>
            </div>
        </div>
      </div>

      <div className="p-6">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Regional Facilities</h3>
        
        <div className="space-y-3">
          {selectedCountry.facilities.length === 0 ? (
             <p className="text-slate-500 text-sm italic">No active facilities available in this region.</p>
          ) : (
            selectedCountry.facilities.map((facility, index) => (
                <div key={index} className="flex items-center gap-3 bg-slate-800/50 p-3 rounded border border-slate-800 hover:border-emerald-500/30 transition-colors">
                    <div className="w-8 h-8 rounded bg-slate-900 flex items-center justify-center shrink-0 border border-slate-700">
                        <Building2 className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="flex-grow">
                        <span className="text-sm font-medium text-slate-200 block">{facility}</span>
                        <span className="text-[10px] text-slate-500 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3 text-emerald-500/80" /> Operational
                        </span>
                    </div>
                </div>
            ))
          )}
        </div>
        
        <div className="mt-8 p-4 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
             <div className="flex items-center gap-2 text-emerald-400 mb-2">
                 <Server className="w-4 h-4" />
                 <span className="text-xs font-bold uppercase">Grid Analysis</span>
             </div>
             <p className="text-xs text-slate-400 leading-relaxed">
                 Compute workloads in {selectedCountry.name} are optimized for {selectedCountry.renewablePercentage > 70 ? 'Hydro and Wind' : selectedCountry.renewablePercentage > 40 ? 'Mixed Renewables' : 'Grid Stability'}.
                 Allocating resources here contributes {selectedCountry.renewablePercentage}% directly to green energy initiatives.
             </p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;