import React, { useEffect, useState } from 'react';
import { ComputeType } from '../types';
import { Loader2, Server, Zap } from 'lucide-react';

interface LoadingScreenProps {
  type: ComputeType;
  onComplete: () => void;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ type, onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('Initializing Handshake...');

  useEffect(() => {
    const duration = 2500; // 2.5 seconds fake load
    const intervalTime = 50;
    const steps = duration / intervalTime;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      const newProgress = Math.min((currentStep / steps) * 100, 100);
      setProgress(newProgress);

      if (newProgress > 20 && newProgress < 50) setStage('Finding low-carbon nodes...');
      if (newProgress > 50 && newProgress < 80) setStage('Allocating sustainable resources...');
      if (newProgress > 80) setStage('Establishing secure tunnel...');

      if (currentStep >= steps) {
        clearInterval(interval);
        setTimeout(onComplete, 300);
      }
    }, intervalTime);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-emerald-900/20 via-slate-950 to-slate-950"></div>
      
      <div className="z-10 text-center max-w-md w-full px-6">
        <div className="mb-8 flex justify-center">
            <div className="relative">
                <div className="absolute inset-0 bg-emerald-500 blur-xl opacity-20 animate-pulse"></div>
                <Server className="w-16 h-16 text-emerald-400 relative z-10" />
            </div>
        </div>

        <h2 className="text-2xl font-bold mb-2 tracking-tight text-white">
          Deploying {type}
        </h2>
        <p className="text-slate-400 text-sm font-mono mb-8 h-6">{stage}</p>

        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden border border-slate-700">
          <div 
            className="h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] transition-all duration-75 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        
        <div className="flex justify-between mt-2 text-xs text-slate-500 font-mono">
           <span>0%</span>
           <span><Zap className="w-3 h-3 inline mr-1 text-yellow-500"/>Optimizing Energy</span>
           <span>100%</span>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;
