import React, { useState, useEffect } from 'react';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Prediction from './pages/Prediction';
import About from './pages/About';
import apiService from './services/api';

function App() {
  const [activePage, setActivePage] = useState('home');
  const [healthStatus, setHealthStatus] = useState('connecting'); // 'connecting' | 'online' | 'offline'

  // Poll backend health status
  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const data = await apiService.checkHealth();
        if (data.status === 'healthy') {
          setHealthStatus('online');
        } else {
          setHealthStatus('connecting'); // Initializing states
        }
      } catch (err) {
        setHealthStatus('offline');
      }
    };

    // Run immediately on mount
    checkApiStatus();

    // Check status every 10 seconds
    const interval = setInterval(checkApiStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const renderPage = () => {
    switch (activePage) {
      case 'home':
        return <Home setActivePage={setActivePage} />;
      case 'prediction':
        return <Prediction />;
      case 'about':
        return <About />;
      default:
        return <Home setActivePage={setActivePage} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-600/30 selection:text-white">
      
      {/* Navigation Header */}
      <Navigation 
        activePage={activePage} 
        setActivePage={setActivePage} 
        healthStatus={healthStatus} 
      />

      {/* Main Page View Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 transition-all duration-300">
        {renderPage()}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900/60 bg-slate-950/40 text-center py-6 text-[10px] text-slate-500 font-medium tracking-wide uppercase">
        <p>© 2026 MaintAI Predictive Diagnostics Dashboard • Built for Demo & Presentation</p>
      </footer>
    </div>
  );
}

export default App;
