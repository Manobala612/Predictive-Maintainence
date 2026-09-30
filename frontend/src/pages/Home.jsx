import React from 'react';
import { Sliders, ShieldAlert, Cpu, Award, Zap, Thermometer, Wrench } from 'lucide-react';

function Home({ setActivePage }) {
  const datasetStats = [
    { label: "Dataset Instances", value: "10,000" },
    { label: "Numerical Sensors", value: "5 Features" },
    { label: "Target Model Accuracy", value: "98.3%" },
    { label: "Features Evaluated", value: "6 Total" },
  ];

  const failureCategories = [
    {
      title: "Tool Wear Failure (TWF)",
      description: "Triggered when the tool wear time exceeds wear bounds, prompting surface damage or breakage.",
      icon: ShieldAlert,
      color: "border-orange-500/20 text-orange-400 bg-orange-500/5",
    },
    {
      title: "Heat Dissipation Failure (HDF)",
      description: "Occurs if the process temperature difference drops below critical heat exchange values.",
      icon: Thermometer,
      color: "border-red-500/20 text-red-400 bg-red-500/5",
    },
    {
      title: "Power & Torque Failure (PWF)",
      description: "Triggered if torque and speed parameters fall outside the stable operational curves.",
      icon: Zap,
      color: "border-amber-500/20 text-amber-400 bg-amber-500/5",
    },
  ];

  return (
    <div className="space-y-10 py-6 max-w-7xl mx-auto px-6">
      
      {/* Hero Welcome Card */}
      <section className="relative overflow-hidden rounded-3xl bg-slate-900 border border-slate-800 p-8 sm:p-12 shadow-2xl flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-transparent to-transparent pointer-events-none"></div>
        
        <div className="space-y-4 max-w-xl text-center md:text-left z-10">
          <div className="inline-flex items-center space-x-2 bg-indigo-500/10 border border-indigo-500/20 px-3 py-1 rounded-full text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Cpu size={14} />
            <span>Smart Manufacturing Analytics</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
            AI-Driven Machine <br className="hidden sm:inline" />
            Predictive Maintenance
          </h2>
          <p className="text-sm text-slate-400 leading-relaxed">
            Diagnose sensor data streams, assess asset risks, and anticipate failures before they occur. Powered by a pre-trained Random Forest model trained on the AI4I 2020 dataset.
          </p>
          <div className="pt-2">
            <button
              onClick={() => setActivePage('prediction')}
              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold tracking-wide shadow-lg shadow-indigo-600/20 transition flex items-center gap-2 mx-auto md:mx-0"
            >
              <Sliders size={16} />
              Open Inference Panel
            </button>
          </div>
        </div>

        {/* Diagnostic Visual Card */}
        <div className="bg-slate-950/80 border border-slate-850 p-6 rounded-2xl w-full max-w-sm shadow-inner relative">
          <div className="flex justify-between items-center mb-4 pb-4 border-b border-slate-800/80">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Diagnosis Dashboard</span>
            <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">ONLINE</span>
          </div>
          <div className="space-y-3.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-500">Machine Failure Risk</span>
              <span className="font-semibold text-emerald-400">0.02% (Normal)</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-1.5">
              <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: '12%' }}></div>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-500">Operating Temperature</span>
              <span className="font-semibold text-slate-300">298.1 K</span>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-500">Spindle Torque Load</span>
              <span className="font-semibold text-slate-300">42.8 Nm</span>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Statistics Grid */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {datasetStats.map((stat, i) => (
          <div key={i} className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 backdrop-blur-sm hover:border-slate-700/60 transition-all duration-300">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{stat.label}</p>
            <p className="text-2xl font-black text-white mt-1.5">{stat.value}</p>
          </div>
        ))}
      </section>

      {/* Failures Breakdown */}
      <section className="space-y-6">
        <div className="flex items-center space-x-2">
          <Award size={20} className="text-indigo-400" />
          <h3 className="text-lg font-bold text-white">Failure Modes Evaluated</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {failureCategories.map((cat, i) => {
            const Icon = cat.icon;
            return (
              <div key={i} className={`border rounded-2xl p-6 flex flex-col justify-between gap-4 transition duration-300 hover:scale-[1.01] ${cat.color}`}>
                <div className="space-y-2">
                  <div className="w-10 h-10 rounded-xl bg-slate-900/60 flex items-center justify-center border border-slate-800">
                    <Icon size={18} />
                  </div>
                  <h4 className="font-bold text-white text-sm">{cat.title}</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">{cat.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

export default Home;
