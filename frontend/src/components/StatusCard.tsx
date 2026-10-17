import React from 'react';
import { Activity, CheckCircle2, AlertTriangle, RefreshCw, Server } from 'lucide-react';
import { ConnectionStatus, HealthResponse } from '../types';

interface StatusCardProps {
  status: ConnectionStatus;
  health: HealthResponse | null;
  latencyMs: number | null;
  lastChecked: Date | null;
  isRefreshing: boolean;
  onRefresh: () => void;
}

export const StatusCard: React.FC<StatusCardProps> = ({
  status,
  health,
  latencyMs,
  lastChecked,
  isRefreshing,
  onRefresh,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
      <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-slate-800 text-indigo-400 border border-slate-700">
            <Server className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Backend Status</h2>
            <p className="text-xs text-slate-400">FastAPI Core Service Monitor</p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium transition disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Operational Status */}
        <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">Status</span>
            {status === 'online' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : status === 'connecting' ? (
              <Activity className="w-4 h-4 text-amber-400 animate-spin" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-rose-400" />
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className={`text-lg font-bold tracking-tight ${
                status === 'online'
                  ? 'text-emerald-400'
                  : status === 'connecting'
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`}
            >
              {status === 'online' ? 'ONLINE' : status === 'connecting' ? 'CONNECTING' : 'OFFLINE'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {status === 'online' ? 'All core health checks passing' : 'Service currently unreachable'}
          </p>
        </div>

        {/* API Version */}
        <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">API Version</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-lg font-bold text-white tracking-tight">
            {health?.version ? `v${health.version}` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            {health?.service || 'VoiceShield API'}
          </p>
        </div>

        {/* Latency & Ping */}
        <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400">Latency</span>
            <span className="text-[10px] text-slate-500 font-mono">
              {lastChecked ? lastChecked.toLocaleTimeString() : '—'}
            </span>
          </div>
          <div className="text-lg font-bold text-white tracking-tight">
            {latencyMs !== null ? `${latencyMs} ms` : '—'}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Round-trip response time
          </p>
        </div>
      </div>
    </div>
  );
};
