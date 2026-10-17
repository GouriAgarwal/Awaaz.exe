import React from 'react';
import { Cpu, Terminal, ExternalLink, Info, Check } from 'lucide-react';
import { SystemStatusResponse } from '../types';

interface SystemInfoProps {
  system: SystemStatusResponse | null;
  apiBaseUrl: string;
}

export const SystemInfo: React.FC<SystemInfoProps> = ({ system, apiBaseUrl }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* System Status Details */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2.5 rounded-xl bg-slate-800 text-cyan-400 border border-slate-700">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">System Status</h2>
            <p className="text-xs text-slate-400">Runtime environment and operational configuration</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-xs text-slate-400">Backend State</span>
            <span className="text-xs font-mono font-medium text-emerald-400 uppercase">
              {system?.backend || 'offline'}
            </span>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-xs text-slate-400">Environment</span>
            <span className="text-xs font-mono font-medium text-indigo-400 uppercase">
              {system?.environment || 'unknown'}
            </span>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <span className="text-xs text-slate-400">API Version</span>
            <span className="text-xs font-mono font-medium text-slate-200">
              {system?.version ? `v${system.version}` : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Available API Endpoints */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2.5 rounded-xl bg-slate-800 text-purple-400 border border-slate-700">
            <Terminal className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">Foundation Endpoints</h2>
            <p className="text-xs text-slate-400">Verified REST endpoints in the current step</p>
          </div>
        </div>

        <div className="space-y-2.5">
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold">
                GET
              </span>
              <span className="text-xs font-mono text-slate-300">/api/health</span>
            </div>
            <a
              href={`${apiBaseUrl}/api/health`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1"
            >
              Test <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold">
                GET
              </span>
              <span className="text-xs font-mono text-slate-300">/api/system/status</span>
            </div>
            <a
              href={`${apiBaseUrl}/api/system/status`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1"
            >
              Test <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 text-[10px] font-mono font-bold">
                DOCS
              </span>
              <span className="text-xs font-mono text-slate-300">/docs (Swagger UI)</span>
            </div>
            <a
              href={`${apiBaseUrl}/docs`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-indigo-400 hover:text-indigo-300 inline-flex items-center gap-1"
            >
              Open <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};
