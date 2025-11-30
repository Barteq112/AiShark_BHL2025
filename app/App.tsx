import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Home from './components/Home';
import Dashboard from './components/Dashboard';
import LoadingScreen from './components/LoadingScreen';
import { ComputeType } from './types';

// Wrapper component to handle navigation logic
const AppContent: React.FC = () => {
  const [loadingType, setLoadingType] = useState<ComputeType | null>(null);
  const navigate = useNavigate();

  const handleStartCompute = (type: ComputeType) => {
    setLoadingType(type);
  };

  const handleLoadingComplete = () => {
    // 1. Zapisujemy wartość do zmiennej pomocniczej, zanim wyczyścimy stan
    const typeToSend = loadingType;
    
    // 2. Czyścimy stan, aby ukryć LoadingScreen
    setLoadingType(null);
    
    // 3. Przekazujemy zapisaną wartość do Dashboardu poprzez state nawigacji
    navigate('/dashboard', { state: { computeType: typeToSend } });
  };

  if (loadingType) {
    return <LoadingScreen type={loadingType} onComplete={handleLoadingComplete} />;
  }

  return (
    <Routes>
      <Route path="/" element={<Home onStartCompute={handleStartCompute} />} />
      <Route path="/dashboard" element={<Dashboard />} />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;