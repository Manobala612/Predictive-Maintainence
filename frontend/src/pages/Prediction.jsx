import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import { 
  Settings, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Thermometer, 
  RotateCw, 
  Gauge, 
  Wrench, 
  RefreshCw,
  History,
  Trash2
} from 'lucide-react';

function Prediction() {
  const [formData, setFormData] = useState({
    Type: 'M',
    'Air temperature [K]': 300.0,
    'Process temperature [K]': 310.0,
    'Rotational speed [rpm]': 1500.0,
    'Torque [Nm]': 40.0,
    'Tool wear [min]': 50.0
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('maint_predict_history');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error("Failed to parse prediction history", e);
      }
    }
  }, []);

  const saveToHistory = (inputData, predictionResult) => {
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      inputs: { ...inputData },
      result: { ...predictionResult }
    };
    
    // Keep last 5 runs
    const updatedHistory = [newEntry, ...history.slice(0, 4)];
    setHistory(updatedHistory);
    localStorage.setItem('maint_predict_history', JSON.stringify(updatedHistory));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('maint_predict_history');
  };

  const loadFromHistory = (entry) => {
    setFormData(entry.inputs);
    setResult(entry.result);
    setError(null);
  };

  // Quick preset loader
  const loadPreset = (preset) => {
    switch (preset) {
      case 'normal':
        setFormData({
          Type: 'M',
          'Air temperature [K]': 298.1,
          'Process temperature [K]': 308.6,
          'Rotational speed [rpm]': 1551.0,
          'Torque [Nm]': 42.8,
          'Tool wear [min]': 0.0
        });
        break;
      case 'tool_wear':
        setFormData({
          Type: 'L',
          'Air temperature [K]': 298.2,
          'Process temperature [K]': 308.7,
          'Rotational speed [rpm]': 1408.0,
          'Torque [Nm]': 46.3,
          'Tool wear [min]': 203.0
        });
        break;
      case 'heat_dissipation':
        setFormData({
          Type: 'M',
          'Air temperature [K]': 302.2,
          'Process temperature [K]': 310.1,
          'Rotational speed [rpm]': 1361.0,
          'Torque [Nm]': 53.6,
          'Tool wear [min]': 84.0
        });
        break;
      case 'overstrain':
        setFormData({
          Type: 'H',
          'Air temperature [K]': 301.9,
          'Process temperature [K]': 310.8,
          'Rotational speed [rpm]': 1421.0,
          'Torque [Nm]': 61.8,
          'Tool wear [min]': 214.0
        });
        break;
      default:
        break;
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'Type' ? value : parseFloat(value) || 0
    }));
  };

  const runDiagnosis = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await apiService.predict(formData);
      setResult(data);
      saveToHistory(formData, data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to connect to API server.');
    } finally {
      setLoading(false);
    }
  };

  // Determine Risk Category based on predictions & probabilities
  const getRiskDetails = (failure, probability) => {
    if (failure === 1) {
      return { 
        level: "CRITICAL DANGER", 
        color: "text-red-400 border-red-500/30 bg-red-500/5",
        status: "Action Required" 
      };
    }
    if (probability > 0.3) {
      return { 
        level: "ELEVATED WARNING", 
        color: "text-amber-400 border-amber-500/30 bg-amber-500/5",
        status: "Monitor System" 
      };
    }
    return { 
      level: "SAFE OPERATION", 
      color: "text-emerald-400 border-emerald-500/30 bg-emerald-500/5",
      status: "Stable" 
    };
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-6 space-y-8">
      
      {/* Grid containing Input Forms & Results View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Simulation Form Control Panel */}
        <section className="lg:col-span-7 bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-800/80">
            <div className="flex items-center space-x-2">
              <Settings className="text-indigo-400" size={20} />
              <h3 className="text-md font-bold text-white">Telemetry Configuration</h3>
            </div>
            <span className="text-[10px] text-slate-500 uppercase tracking-widest">Adjust Sensor Inputs</span>
          </div>

          {/* Demo Presets bar */}
          <div className="mb-6 bg-slate-950/60 p-4 rounded-xl border border-slate-850">
            <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider mb-2.5">Diagnostic Evaluation Presets:</p>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => loadPreset('normal')} type="button" className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-xs rounded-lg border border-slate-800 hover:border-slate-700 transition flex items-center gap-1.5 text-slate-300">
                <CheckCircle size={13} className="text-emerald-400" /> Normal Run
              </button>
              <button onClick={() => loadPreset('tool_wear')} type="button" className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-xs rounded-lg border border-slate-800 hover:border-slate-700 transition flex items-center gap-1.5 text-slate-300">
                <Wrench size={13} className="text-amber-400" /> Tool Failure
              </button>
              <button onClick={() => loadPreset('heat_dissipation')} type="button" className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-xs rounded-lg border border-slate-800 hover:border-slate-700 transition flex items-center gap-1.5 text-slate-300">
                <Thermometer size={13} className="text-red-400" /> Heat Dissipation
              </button>
              <button onClick={() => loadPreset('overstrain')} type="button" className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-xs rounded-lg border border-slate-800 hover:border-slate-700 transition flex items-center gap-1.5 text-slate-300">
                <AlertTriangle size={13} className="text-orange-400" /> Overstrain Fail
              </button>
            </div>
          </div>

          <form onSubmit={runDiagnosis} className="space-y-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Type */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5">Machine Variant Type</label>
                <select name="Type" value={formData.Type} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none">
                  <option value="L">Low (L) - Standard Quality</option>
                  <option value="M">Medium (M) - Midrange Quality</option>
                  <option value="H">High (H) - High Grade Quality</option>
                </select>
              </div>

              {/* Air Temp */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5 flex items-center gap-1">
                  <Thermometer size={13} className="text-slate-500" /> Air Temperature [280 - 320 K]
                </label>
                <input type="number" step="0.1" min="280" max="320" name="Air temperature [K]" value={formData['Air temperature [K]']} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none" required />
              </div>

              {/* Process Temp */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5 flex items-center gap-1">
                  <Thermometer size={13} className="text-rose-500" /> Process Temperature [280 - 320 K]
                </label>
                <input type="number" step="0.1" min="280" max="320" name="Process temperature [K]" value={formData['Process temperature [K]']} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none" required />
              </div>

              {/* Rotational Speed */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5 flex items-center gap-1">
                  <RotateCw size={13} className="text-sky-400" /> Rotational Speed [500 - 4000 rpm]
                </label>
                <input type="number" min="500" max="4000" name="Rotational speed [rpm]" value={formData['Rotational speed [rpm]']} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none" required />
              </div>

              {/* Torque */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5 flex items-center gap-1">
                  <Gauge size={13} className="text-amber-400" /> Torque [0 - 100 Nm]
                </label>
                <input type="number" step="0.1" min="0" max="100" name="Torque [Nm]" value={formData['Torque [Nm]']} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none" required />
              </div>

              {/* Tool Wear */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 mb-1.5 flex items-center gap-1">
                  <Wrench size={13} className="text-orange-400" /> Tool Wear [0 - 300 min]
                </label>
                <input type="number" min="0" max="300" name="Tool wear [min]" value={formData['Tool wear [min]']} onChange={handleChange} className="w-full bg-slate-950/80 border border-slate-850 rounded-lg p-2.5 text-xs text-slate-200 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none" required />
              </div>

            </div>

            <button type="submit" disabled={loading} className="w-full py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-xl text-xs font-semibold tracking-widest hover:from-indigo-500 hover:to-violet-500 transition shadow-lg shadow-indigo-600/10 flex items-center justify-center gap-2">
              {loading ? <RefreshCw className="animate-spin" size={14} /> : <Activity size={14} />}
              {loading ? 'Evaluating Model Inference...' : 'Evaluate Operational Risk'}
            </button>
          </form>
        </section>

        {/* Results View Dashboard panel */}
        <section className="lg:col-span-5 flex flex-col justify-between">
          <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm flex-1 flex flex-col justify-center min-h-[380px]">
            
            {/* Waiting view */}
            {!result && !error && !loading && (
              <div className="text-center py-10">
                <div className="h-14 w-14 bg-slate-950 border border-slate-850 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-500">
                  <Activity size={24} />
                </div>
                <h4 className="text-sm font-bold text-slate-300">Operational Diagnostics</h4>
                <p className="text-xs text-slate-500 max-w-xs mx-auto mt-2">
                  Configure sensor details or load evaluation presets, then select diagnostic analysis.
                </p>
              </div>
            )}

            {/* Loading view */}
            {loading && (
              <div className="text-center py-10 animate-pulse">
                <div className="h-14 w-14 bg-indigo-950 border border-indigo-900 text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4">
                  <RefreshCw className="animate-spin" size={24} />
                </div>
                <h4 className="text-sm font-bold text-indigo-300">Computing Tree Classifications</h4>
                <p className="text-xs text-slate-500 mt-1">Interrogating Random Forest classifier...</p>
              </div>
            )}

            {/* Error view */}
            {error && (
              <div className="p-5 bg-red-950/20 border border-red-500/20 rounded-xl text-center">
                <AlertTriangle className="mx-auto mb-2 text-red-500" size={28} />
                <h4 className="text-xs font-bold text-red-400">Diagnosis Offline</h4>
                <p className="text-xs text-red-300/70 mt-1">{error}</p>
              </div>
            )}

            {/* Diagnosis Result View */}
            {result && (() => {
              const risk = getRiskDetails(result.machine_failure, result.failure_probability);
              const RiskIcon = result.machine_failure === 1 ? AlertTriangle : CheckCircle;
              return (
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Assessment Result</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${risk.color}`}>
                      {risk.level}
                    </span>
                  </div>

                  <div className={`p-4 border rounded-xl flex gap-3 ${
                    result.machine_failure === 1 ? 'bg-red-500/5 border-red-500/20 text-red-200' : 'bg-emerald-500/5 border-emerald-500/20 text-emerald-200'
                  }`}>
                    <RiskIcon size={24} className="flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-bold">
                        {result.machine_failure === 1 ? 'Critical Failure Risk' : 'Machine Stable'}
                      </h4>
                      <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">
                        {result.machine_failure === 1 
                          ? 'Model output confirms high probability of machine failure.' 
                          : 'Operating parameters are within normal tolerances.'}
                      </p>
                    </div>
                  </div>

                  {/* Confidence progress bar */}
                  <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-850 space-y-3">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-500">Inference Confidence</span>
                      <span className={`font-bold ${result.machine_failure === 1 ? 'text-red-400' : 'text-emerald-400'}`}>
                        {(result.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          result.machine_failure === 1 ? 'bg-gradient-to-r from-red-500 to-rose-500' : 'bg-gradient-to-r from-emerald-500 to-teal-500'
                        }`}
                        style={{ width: `${result.confidence_score * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Maintenance Recommendation */}
                  <div className="space-y-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Troubleshooting Recommendations</span>
                    <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-850 text-xs text-slate-300 font-mono leading-relaxed whitespace-pre-line">
                      {result.recommendation}
                    </div>
                  </div>
                </div>
              );
            })()}

          </div>
        </section>
      </div>

      {/* History Log view bottom panel */}
      {history.length > 0 && (
        <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/80">
            <div className="flex items-center space-x-2">
              <History className="text-indigo-400" size={18} />
              <h3 className="text-sm font-bold text-white">Prediction History Log (Local Session)</h3>
            </div>
            <button 
              onClick={clearHistory}
              className="text-[10px] font-semibold text-red-400 hover:text-red-300 flex items-center gap-1 hover:bg-red-500/10 px-2 py-1 rounded transition"
            >
              <Trash2 size={11} /> Clear Log
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="text-slate-500 border-b border-slate-800">
                  <th className="pb-2 font-semibold">Timestamp</th>
                  <th className="pb-2 font-semibold">Type</th>
                  <th className="pb-2 font-semibold text-center">Telemetry Parameters</th>
                  <th className="pb-2 font-semibold text-center">Prediction Outcome</th>
                  <th className="pb-2 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {history.map((entry) => (
                  <tr key={entry.id} className="text-slate-300 hover:bg-slate-800/20 transition-colors">
                    <td className="py-2.5 text-slate-400 font-mono">{entry.timestamp}</td>
                    <td className="py-2.5 font-bold">{entry.inputs.Type}</td>
                    <td className="py-2.5 text-center text-slate-400 font-mono">
                      {entry.inputs['Air temperature [K]']}K / {entry.inputs['Rotational speed [rpm]']} rpm / {entry.inputs['Torque [Nm]']}Nm / {entry.inputs['Tool wear [min]']} min
                    </td>
                    <td className="py-2.5 text-center">
                      <span className={`px-2 py-0.5 rounded font-bold text-[10px] border ${
                        entry.result.machine_failure === 1 
                          ? 'bg-red-500/10 border-red-500/20 text-red-400' 
                          : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                      }`}>
                        {entry.result.machine_failure === 1 ? 'FAILURE' : 'STABLE'} ({(entry.result.confidence_score * 100).toFixed(0)}%)
                      </span>
                    </td>
                    <td className="py-2.5 text-right">
                      <button 
                        onClick={() => loadFromHistory(entry)}
                        className="text-[10px] font-semibold text-indigo-400 hover:text-indigo-300 underline"
                      >
                        Reload
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

    </div>
  );
}

export default Prediction;
