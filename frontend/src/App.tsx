import React, { useEffect, useState, useCallback } from 'react';
import { Header } from './components/Header';
import { StatusCard } from './components/StatusCard';
import { SystemInfo } from './components/SystemInfo';
import { AudioInspector } from './components/AudioInspector';
import { AcousticFeaturesPanel } from './components/AcousticFeaturesPanel';
import { SyntheticDetectionPanel } from './components/SyntheticDetectionPanel';
import { DetectorValidationPanel } from './components/DetectorValidationPanel';
import { SpeakerVerificationPanel } from './components/SpeakerVerificationPanel';
import { ImpersonationRiskPanel } from './components/ImpersonationRiskPanel';
import { RealtimeMonitoringPanel } from './components/RealtimeMonitoringPanel';
import { ScientificValidationPanel } from './components/ScientificValidationPanel';
import { OperationsDashboard } from './components/OperationsDashboard';
import { VoiceIntegrityConsole } from './components/VoiceIntegrityConsole';
import { FraudPreventionCommandCenter } from './components/FraudPreventionCommandCenter';
import { VoiceShieldCommandCenter } from './components/VoiceShieldCommandCenter';
import { ErrorBoundary } from './components/ErrorBoundary';
import { testConnection } from './services/api';

import { DashboardState } from './types';
import { AlertCircle, RefreshCw, Layers, Sliders, Volume2, ShieldAlert, CheckCircle2, UserCheck, ShieldCheck, Radio, Scale, Server, Sparkles, Trophy } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function App() {
  const [state, setState] = useState<DashboardState>({
    status: 'connecting',
    health: null,
    system: null,
    lastChecked: null,
    latencyMs: null,
    error: null,
  });

  const [activeTab, setActiveTab] = useState<
    | 'command-center'
    | 'prevention'
    | 'console'
    | 'step12'
    | 'step11'
    | 'streaming'
    | 'risk-fusion'
    | 'risk'
    | 'speaker'
    | 'detection'
    | 'validation'
    | 'features'
    | 'inspector'
  >('command-center');
  const [isRefreshing, setIsRefreshing] = useState(false);


  const fetchStatus = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const { health, system, latencyMs } = await testConnection();
      setState({
        status: 'online',
        health,
        system,
        lastChecked: new Date(),
        latencyMs,
        error: null,
      });
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        status: 'offline',
        lastChecked: new Date(),
        error: err.message || 'Failed to communicate with the VoiceShield backend.',
      }));
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      <Header status={state.status} />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-8">
        {/* Error Notification Banner */}
        {state.status === 'offline' && (
          <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 backdrop-blur-md shadow-lg flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
            <div className="flex items-start sm:items-center gap-3">
              <div className="p-2 rounded-xl bg-rose-500/20 text-rose-400">
                <AlertCircle className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-rose-300">Backend Connection Error</h3>
                <p className="text-xs text-rose-400/90 mt-0.5">
                  Unable to connect to VoiceShield API at{' '}
                  <span className="font-mono underline">{API_BASE_URL}</span>. Make sure the backend service is running.
                </p>
                {state.error && (
                  <p className="text-[11px] font-mono text-rose-400/70 mt-1">
                    Details: {state.error}
                  </p>
                )}
              </div>
            </div>

            <button
              onClick={fetchStatus}
              disabled={isRefreshing}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-white text-xs font-semibold shadow-md shadow-rose-500/20 transition disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Retry Connection</span>
            </button>
          </div>
        )}

        {/* Hero Section */}
        <section className="text-left space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-slate-400 text-xs font-medium">
            <Trophy className="w-3.5 h-3.5 text-amber-400" />
            <span>Step 15: Final Hackathon Product Integration &amp; Live Command Center</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            AI-Powered Real-Time Detection &amp; Prevention of Voice Cloning Attacks
          </h2>
          <p className="text-sm text-slate-400 max-w-2xl">
            VoiceShield unifies pretrained AASIST synthetic voice detection, SpeechBrain ECAPA-TDNN biometric verification, acoustic prosody dynamics, and automated zero-trust fraud prevention workflows to neutralize voice cloning before sensitive transactions execute.
          </p>
        </section>

        {/* Tab Selection */}
        <div className="flex flex-wrap items-center gap-2.5 border-b border-slate-800 pb-3">
          {/* Flagship Hackathon Tab */}
          <button
            onClick={() => setActiveTab('command-center')}
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-black tracking-wide transition transform ${
              activeTab === 'command-center'
                ? 'bg-gradient-to-r from-rose-600 via-rose-500 to-amber-600 text-white shadow-xl shadow-rose-600/30 ring-2 ring-rose-400/50 scale-105'
                : 'bg-rose-950/40 text-rose-300 border border-rose-500/30 hover:bg-rose-900/40'
            }`}
          >
            <Trophy className="w-4 h-4 text-amber-300 animate-pulse" />
            <span>Hackathon Command Center (Step 15)</span>
          </button>

          <button
            onClick={() => setActiveTab('prevention')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'prevention'
                ? 'bg-gradient-to-r from-rose-600 to-indigo-600 text-white shadow-lg shadow-rose-600/30 ring-1 ring-rose-400/50'
                : 'bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <ShieldAlert className="w-4 h-4 text-rose-300" />
            <span>Fraud Prevention (Step 14)</span>
          </button>

          <button
            onClick={() => setActiveTab('console')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'console'
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-600/30 ring-1 ring-indigo-400/50'
                : 'bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Sparkles className="w-4 h-4 text-indigo-300" />
            <span>Voice Integrity Console (Step 13)</span>
          </button>

          <button
            onClick={() => setActiveTab('step12')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'step12'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30 ring-1 ring-blue-400/50'
                : 'bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Server className="w-4 h-4 text-blue-300" />
            <span>Operations &amp; Health (Step 12)</span>
          </button>

          <button
            onClick={() => setActiveTab('step11')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'step11'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 ring-1 ring-indigo-400/50'
                : 'bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Scale className="w-4 h-4 text-indigo-300" />
            <span>Validation (Step 11)</span>
          </button>

          <button
            onClick={() => setActiveTab('streaming')}
            className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
              activeTab === 'streaming'
                ? 'bg-rose-600 text-white shadow-lg shadow-rose-600/30'
                : 'bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Radio className="w-4 h-4 text-rose-300 animate-pulse" />
            <span>Streaming (Step 10)</span>
          </button>

          <button
            onClick={() => setActiveTab('risk-fusion')}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition ${
              activeTab === 'risk-fusion' || activeTab === 'risk'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 ring-1 ring-indigo-400/50'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Risk Fusion</span>
          </button>

          <button
            onClick={() => setActiveTab('speaker')}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition ${
              activeTab === 'speaker'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <UserCheck className="w-3.5 h-3.5" />
            <span>Speaker Biometrics</span>
          </button>

          <button
            onClick={() => setActiveTab('detection')}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition ${
              activeTab === 'detection'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Synthetic Detection</span>
          </button>

          <button
            onClick={() => setActiveTab('features')}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition ${
              activeTab === 'features'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Acoustic Features</span>
          </button>

          <button
            onClick={() => setActiveTab('inspector')}
            className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition ${
              activeTab === 'inspector'
                ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-850'
            }`}
          >
            <Volume2 className="w-3.5 h-3.5" />
            <span>Audio Ingestion</span>
          </button>
        </div>

        {/* Interactive Workspace Panel */}
        {activeTab === 'command-center' && <VoiceShieldCommandCenter />}
        {activeTab === 'prevention' && <FraudPreventionCommandCenter />}
        {activeTab === 'console' && <VoiceIntegrityConsole />}
        {activeTab === 'step12' && (
          <ErrorBoundary fallbackTitle="Operations dashboard temporarily unavailable">
            <OperationsDashboard />
          </ErrorBoundary>
        )}
        {activeTab === 'step11' && <ScientificValidationPanel />}

        {activeTab === 'streaming' && <RealtimeMonitoringPanel />}
        {(activeTab === 'risk-fusion' || activeTab === 'risk') && (
          <ErrorBoundary fallbackTitle="Risk Fusion engine temporarily unavailable">
            <ImpersonationRiskPanel />
          </ErrorBoundary>
        )}
        {activeTab === 'speaker' && <SpeakerVerificationPanel />}
        {activeTab === 'detection' && <SyntheticDetectionPanel />}
        {activeTab === 'validation' && <DetectorValidationPanel />}
        {activeTab === 'features' && <AcousticFeaturesPanel />}
        {activeTab === 'inspector' && <AudioInspector />}


        {/* Status Monitoring Card */}
        <StatusCard
          status={state.status}
          health={state.health}
          latencyMs={state.latencyMs}
          lastChecked={state.lastChecked}
          isRefreshing={isRefreshing}
          onRefresh={fetchStatus}
        />

        {/* System & Endpoints Details */}
        <SystemInfo system={state.system} apiBaseUrl={API_BASE_URL} />
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 px-6 py-4 text-center text-xs text-slate-600">
        <p>VoiceShield Security Architecture • Step 15: Final Hackathon Product Integration &amp; Live Command Center</p>
      </footer>
    </div>
  );
}

export default App;


