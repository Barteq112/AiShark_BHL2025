import React from 'react';
import { ComputeType } from '../types';
import { Leaf, Search, ArrowRight, Wind, Zap } from 'lucide-react';

interface HomeProps {
  onStartCompute: (type: ComputeType) => void;
}

const Home: React.FC<HomeProps> = ({ onStartCompute }) => {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col text-slate-100 selection:bg-emerald-500/30">
      {/* Header */}
      <header className="p-6 flex items-center justify-between border-b border-slate-800/50 backdrop-blur-md sticky top-0 z-50 bg-slate-950/80">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(16,185,129,0.4)]">
             <Leaf className="text-slate-900 w-5 h-5" />
          </div>
          <span className="font-bold text-xl tracking-tight">GreenComputing</span>
        </div>
        <nav>
            <button className="text-sm text-slate-400 hover:text-white transition-colors">Documentation</button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex-grow flex flex-col items-center justify-center px-4 py-12 relative overflow-hidden">
        
        {/* Background Elements */}
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="text-center max-w-3xl mx-auto mb-16 relative z-10">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-teal-200 to-white tracking-tight leading-tight">
            Build your ideas <br/> sustainably.
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Access high-performance computing resources powered by renewable energy. 
            Reduce your carbon footprint without compromising on speed.
          </p>
        </div>

        {/* Action Grid */}
        <div className="grid md:grid-cols-3 gap-6 w-full max-w-5xl relative z-10">
          
          {/* Light Computing */}
          <button 
            onClick={() => onStartCompute(ComputeType.LIGHT)}
            className="group relative bg-slate-900/50 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/50 p-8 rounded-2xl transition-all duration-300 text-left hover:shadow-[0_0_30px_rgba(16,185,129,0.1)] hover:-translate-y-1"
          >
            <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mb-6 group-hover:bg-emerald-900/30 transition-colors">
                <Wind className="w-6 h-6 text-emerald-400" />
            </div>
            <h3 className="text-xl font-bold mb-2 text-white">Light Computing</h3>
            <p className="text-sm text-slate-400 mb-6">Perfect for web hosting, basic scripts, and microservices.</p>
            <div className="flex items-center text-emerald-400 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                Start Instance <ArrowRight className="w-4 h-4 ml-2" />
            </div>
          </button>

          {/* Medium Computing */}
          <button 
            onClick={() => onStartCompute(ComputeType.MEDIUM)}
            className="group relative bg-slate-900/50 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/50 p-8 rounded-2xl transition-all duration-300 text-left hover:shadow-[0_0_30px_rgba(16,185,129,0.1)] hover:-translate-y-1"
          >
             <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mb-6 group-hover:bg-emerald-900/30 transition-colors">
                <Zap className="w-6 h-6 text-emerald-400" />
            </div>
            <h3 className="text-xl font-bold mb-2 text-white">Medium Computing</h3>
            <p className="text-sm text-slate-400 mb-6">Ideal for data processing, compilation, and standard ML inference.</p>
             <div className="flex items-center text-emerald-400 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                Start Instance <ArrowRight className="w-4 h-4 ml-2" />
            </div>
          </button>

          {/* Heavy Computing */}
          <button 
             onClick={() => onStartCompute(ComputeType.HEAVY)}
             className="group relative bg-slate-900/50 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/50 p-8 rounded-2xl transition-all duration-300 text-left hover:shadow-[0_0_30px_rgba(16,185,129,0.1)] hover:-translate-y-1"
          >
             <div className="w-12 h-12 bg-slate-800 rounded-full flex items-center justify-center mb-6 group-hover:bg-emerald-900/30 transition-colors">
                <div className="flex gap-0.5">
                    <div className="w-1.5 h-4 bg-emerald-400 rounded-full"></div>
                    <div className="w-1.5 h-6 bg-emerald-400 rounded-full"></div>
                    <div className="w-1.5 h-4 bg-emerald-400 rounded-full"></div>
                </div>
            </div>
            <h3 className="text-xl font-bold mb-2 text-white">Heavy Computing</h3>
            <p className="text-sm text-slate-400 mb-6">Designed for LLM training, rendering farms, and complex simulations.</p>
             <div className="flex items-center text-emerald-400 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                Start Instance <ArrowRight className="w-4 h-4 ml-2" />
            </div>
          </button>
        </div>

        {/* Secondary Actions */}
        <div className="mt-12">
            <button className="flex items-center gap-2 px-6 py-3 rounded-full border border-slate-700 hover:border-emerald-500/50 hover:bg-slate-900 text-slate-300 transition-all">
                <Search className="w-4 h-4" />
                <span>Search</span>
            </button>
        </div>

      </main>

      {/* Footer */}
      <footer className="p-6 text-center text-slate-600 text-sm">
        <p>&copy; 2025 GreenComputing. Powered by nature.</p>
      </footer>
    </div>
  );
};

export default Home;
