import React from 'react';
import { Database, ShieldAlert, HeartHandshake, GitBranch, Layers } from 'lucide-react';

function About() {
  const techStack = [
    { name: "FastAPI", role: "REST API Inference Engine", desc: "Serves endpoints asynchronously, performing live predictions." },
    { name: "React & Vite", role: "UI & Hot Reloading SPA", desc: "Builds responsive, high-performance dashboards." },
    { name: "MLflow", role: "Telemetry Logging & Tracking", desc: "Records prediction parameters and accuracy histories." },
    { name: "DVC (Data Version Control)", role: "Dataset & Pickle tracking", desc: "Versions dataset models without bloated git trees." },
    { name: "Docker Compose", role: "Stack Containerization", desc: "Unifies UI, API, and MLflow services under one CLI launcher." }
  ];

  return (
    <div className="max-w-4xl mx-auto px-6 py-6 space-y-8">
      
      {/* Introduction */}
      <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-3">
        <div className="flex items-center space-x-2 text-indigo-400">
          <Database size={20} />
          <h3 className="text-md font-bold text-white">AI4I 2020 Dataset Context</h3>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset designed to reflect real-world predictive maintenance data points. It consists of 10,000 tool instances with features denoting quality variants (L, M, H) and operational sensor measurements (Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear).
        </p>
      </section>

      {/* Technology breakdown list */}
      <section className="space-y-4">
        <div className="flex items-center space-x-2 text-indigo-400">
          <Layers size={20} />
          <h3 className="text-md font-bold text-white">System Architecture & MLOps Stack</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {techStack.map((tech, i) => (
            <div key={i} className="bg-slate-900/30 border border-slate-800/80 p-5 rounded-xl hover:border-slate-700/60 transition-all duration-300">
              <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block">{tech.role}</span>
              <h4 className="font-extrabold text-white text-sm mt-1">{tech.name}</h4>
              <p className="text-xs text-slate-400 mt-2 leading-relaxed">{tech.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Safety guidelines banner */}
      <section className="border border-indigo-500/20 bg-indigo-500/5 rounded-2xl p-6 flex gap-4 items-start">
        <GitBranch size={28} className="text-indigo-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider">Evaluation Warning</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            This dashboard uses dynamic rule-based recommendations cross-referenced against the outputs of the Random Forest model. Ensure Docker containers are compiled (`docker compose up --build`) to bind front-end APIs with back-end models correctly.
          </p>
        </div>
      </section>

    </div>
  );
}

export default About;
