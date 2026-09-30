import React from 'react';
import { Activity, Home, Sliders, Info } from 'lucide-react';

function Navigation({ activePage, setActivePage, healthStatus }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'prediction', label: 'Inference Panel', icon: Sliders },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        
        {/* Brand / Logo */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActivePage('home')}>
          <div className="bg-gradient-to-tr from-indigo-500 to-violet-600 p-2.5 rounded-xl text-white shadow-md shadow-indigo-500/20">
            <Activity size={22} className="animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-extrabold bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent tracking-wide">
              MaintAI
            </h1>
            <p className="text-[10px] font-medium tracking-widest text-indigo-400 uppercase">
              Predictive Diagnostics Engine
            </p>
          </div>
        </div>

        {/* Tab Links & Live Status */}
        <div className="flex flex-col sm:flex-row items-center gap-4">
          <nav className="flex items-center bg-slate-900/90 p-1.5 rounded-xl border border-slate-800/80">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all duration-200 ${
                    isActive 
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' 
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }`}
                >
                  <Icon size={14} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className={`flex items-center space-x-2 text-[10px] font-bold px-3 py-2 rounded-xl border transition-all duration-300 ${
            healthStatus === 'online'
              ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
              : healthStatus === 'offline'
              ? 'bg-red-500/5 border-red-500/20 text-red-400'
              : 'bg-amber-500/5 border-amber-500/20 text-amber-400'
          }`}>
            <span className={`h-1.5 w-1.5 rounded-full ${
              healthStatus === 'online'
                ? 'bg-emerald-500 animate-ping'
                : healthStatus === 'offline'
                ? 'bg-red-500'
                : 'bg-amber-500 animate-pulse'
            }`}></span>
            <span className="uppercase tracking-wider">
              API: {healthStatus === 'online' ? 'Online' : healthStatus === 'offline' ? 'Offline' : 'Connecting'}
            </span>
          </div>
        </div>

      </div>
    </header>
  );
}

export default Navigation;
