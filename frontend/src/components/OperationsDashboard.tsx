import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  Server,
  Shield,
  Cpu,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Radio,
  Sliders,
  Layers,
  FileText,
  Lock,
} from 'lucide-react';
import {
  getLiveness,
  getReadiness,
  getOperationalStatus,
  getApplicationMetrics,
  getMonitoredModels,
  getStreamingTelemetry,
  getOperationalErrors,
} from '../services/api';
import type {
  LivenessResponseInfo,
  ReadinessResponseInfo,
  OperationalStatusResponseInfo,
  ApplicationMetricsResponseInfo,
  ModelMonitoringInfo,
  StreamingOperationalInfo,
  OperationalErrorSummaryInfo,
} from '../types';

// ==============================================================================
// Safe Defensive Numeric Helpers (No .toFixed() on undefined/null, no NaN/Infinity)
// ==============================================================================

export const safeNumber = (value: unknown, fallback = 0): number => {
  if (value === null || value === undefined) {
    return fallback;
  }
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
};

export const formatNumber = (
  value: unknown,
  decimals = 2,
  fallback = 'N/A'
): string => {
  if (value === null || value === undefined) {
    return fallback;
  }
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) {
    return fallback;
  }
  return numberValue.toFixed(decimals);
};

export const formatPercent = (
  value: unknown,
  decimals = 1,
  fallback = 'N/A'
): string => {
  const formatted = formatNumber(value, decimals, fallback);
  return formatted === fallback ? fallback : `${formatted}%`;
};

export const formatUptime = (seconds: unknown): string => {
  if (seconds === null || seconds === undefined) return 'N/A';
  const sec = safeNumber(seconds, -1);
  if (sec < 0) return 'N/A';
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  const hours = Math.floor(sec / 3600);
  const minutes = Math.floor((sec % 3600) / 60);
  return `${hours}h ${minutes}m`;
};

export const OperationsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'models' | 'streaming' | 'errors' | 'security'>('overview');
  const [loading, setLoading] = useState<boolean>(true);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Telemetry states
  const [liveness, setLiveness] = useState<LivenessResponseInfo | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponseInfo | null>(null);
  const [opStatus, setOpStatus] = useState<OperationalStatusResponseInfo | null>(null);
  const [metrics, setMetrics] = useState<ApplicationMetricsResponseInfo | null>(null);
  const [models, setModels] = useState<ModelMonitoringInfo[]>([]);
  const [streaming, setStreaming] = useState<StreamingOperationalInfo | null>(null);
  const [errors, setErrors] = useState<OperationalErrorSummaryInfo | null>(null);

  const fetchAllData = useCallback(async () => {
    try {
      setErrorMsg(null);
      const [liveRes, readyRes, statusRes, metricsRes, modelsRes, streamRes, errsRes] = await Promise.all([
        getLiveness().catch(() => null),
        getReadiness().catch(() => null),
        getOperationalStatus().catch(() => null),
        getApplicationMetrics().catch(() => null),
        getMonitoredModels().catch(() => []),
        getStreamingTelemetry().catch(() => null),
        getOperationalErrors(20).catch(() => null),
      ]);

      if (liveRes) setLiveness(liveRes);
      if (readyRes) setReadiness(readyRes);
      if (statusRes) setOpStatus(statusRes);
      if (metricsRes) setMetrics(metricsRes);
      if (Array.isArray(modelsRes)) setModels(modelsRes);
      if (streamRes) setStreaming(streamRes);
      if (errsRes) setErrors(errsRes);
      setLastRefreshed(new Date());
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to fetch operational telemetry');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAllData();
    if (!autoRefresh) return;
    const interval = setInterval(fetchAllData, 5000);
    return () => clearInterval(interval);
  }, [fetchAllData, autoRefresh]);

  const getReadinessBadge = (state?: string) => {
    switch (state) {
      case 'READY':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
            READY
          </span>
        );
      case 'DEGRADED':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-300">
            DEGRADED
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-300">
            {state || 'NOT READY'}
          </span>
        );
    }
  };

  // Safe streaming capacity utilization calculation
  const streamActive = safeNumber(streaming?.active_sessions, 0);
  const streamMax = safeNumber(streaming?.max_active_sessions, 0);
  const streamUtil = streamMax > 0
    ? (streaming?.utilization_percentage !== undefined ? streaming.utilization_percentage : (streamActive / streamMax) * 100)
    : 0;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Header Card */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-blue-50 text-blue-700 rounded-lg">
                <Server className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  Production Operations &amp; Health (Step 12)
                </h1>
                <p className="text-sm text-slate-500">
                  Containerized deployment telemetry, health probes, model lifecycle, streaming safeguards, and structured error tracking
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 self-end md:self-auto">
            <div className="text-right text-xs text-slate-500 hidden sm:block">
              <div>Refreshed: {lastRefreshed.toLocaleTimeString()}</div>
              <div className="font-mono text-slate-400">
                Uptime: {opStatus?.uptime_seconds !== undefined ? formatUptime(opStatus.uptime_seconds) : (metrics?.uptime_seconds !== undefined ? formatUptime(metrics.uptime_seconds) : 'N/A')}
              </div>
            </div>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors flex items-center gap-1.5 ${
                autoRefresh
                  ? 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100'
                  : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
              }`}
            >
              <Activity className={`w-3.5 h-3.5 ${autoRefresh ? 'animate-pulse text-blue-600' : ''}`} />
              {autoRefresh ? 'Polling Active (5s)' : 'Polling Paused'}
            </button>
            <button
              onClick={() => {
                setLoading(true);
                fetchAllData();
              }}
              disabled={loading}
              className="p-2 text-slate-600 hover:text-slate-900 rounded-lg hover:bg-slate-100 border border-slate-200"
              title="Manual Refresh"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-600' : ''}`} />
            </button>
          </div>
        </div>

        {/* Global Alert / Status Bar */}
        {errorMsg && (
          <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-800 text-sm flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Key Operational KPI Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Readiness Status</div>
            <div className="mt-2 flex items-center gap-2">
              {getReadinessBadge(readiness?.status || opStatus?.application_status)}
            </div>
            <div className="text-xs text-slate-400 mt-1 truncate">
              {readiness?.message || (readiness?.status === 'READY' ? 'All subsystems operational' : 'Evaluating system readiness...')}
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Streaming Load</div>
            <div className="mt-1 text-2xl font-bold text-slate-800">
              {streaming ? `${streamActive} / ${streamMax || 'N/A'}` : '0 / 20'}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Util: {streaming ? formatPercent(streamUtil, 1) : '0%'}
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Avg HTTP Latency</div>
            <div className="mt-1 text-2xl font-bold text-slate-800">
              {metrics ? `${formatNumber(metrics.http?.average_latency_ms, 1)}ms` : '0.0ms'}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Req: {safeNumber(metrics?.http?.total_requests, 0)} | 2xx: {safeNumber(metrics?.http?.status_2xx, 0)}
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Recorded Failures</div>
            <div className="mt-1 text-2xl font-bold text-slate-800">
              {errors ? safeNumber(errors.total_errors_recorded, 0) : 0}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              4xx: {safeNumber(metrics?.http?.status_4xx, 0)} | 5xx: {safeNumber(metrics?.http?.status_5xx, 0)}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex border-b border-slate-200 bg-white rounded-t-xl px-4 overflow-x-auto">
        {[
          { id: 'overview', label: 'Overview & Health Probes', icon: Activity },
          { id: 'models', label: 'Model Lifecycle & Integrity', icon: Cpu },
          { id: 'streaming', label: 'Streaming Operations', icon: Radio },
          { id: 'errors', label: 'Error Taxonomy & Audits', icon: AlertTriangle },
          { id: 'security', label: 'Security & Deploy Specs', icon: Lock },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 py-3.5 px-4 font-medium text-sm border-b-2 whitespace-nowrap transition-colors ${
                isActive
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: OVERVIEW & HEALTH PROBES */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Readiness Component Checks */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-emerald-600" />
              Readiness Probe Components (/api/health/ready)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {readiness?.components && Object.keys(readiness.components).length > 0 ? (
                Object.entries(readiness.components).map(([key, comp]) => {
                  const isHealthy = comp?.healthy === true;
                  const detailMsg = comp?.message || (isHealthy ? 'Operational' : 'Subsystem reported failure');
                  return (
                    <div
                      key={key}
                      className={`p-4 rounded-lg border flex items-start justify-between ${
                        isHealthy ? 'bg-emerald-50/40 border-emerald-200' : 'bg-rose-50 border-rose-200'
                      }`}
                    >
                      <div>
                        <div className="font-semibold text-sm text-slate-800">
                          {comp?.name || key.replace(/_/g, ' ')}
                        </div>
                        <div className="text-xs text-slate-600 mt-1 font-mono">
                          Status: {comp?.status || (isHealthy ? 'READY' : 'FAILED')} — {detailMsg}
                        </div>
                        {comp?.details && Object.keys(comp.details).length > 0 && (
                          <div className="text-[11px] text-slate-500 mt-1 font-mono">
                            {JSON.stringify(comp.details)}
                          </div>
                        )}
                      </div>
                      {isHealthy ? (
                        <CheckCircle className="w-5 h-5 text-emerald-600 shrink-0" />
                      ) : (
                        <XCircle className="w-5 h-5 text-rose-600 shrink-0" />
                      )}
                    </div>
                  );
                })
              ) : (
                <div className="p-4 text-xs text-slate-400 col-span-2">
                  No component checks returned yet. Connecting to backend...
                </div>
              )}
            </div>
          </div>

          {/* System Telemetry & Disclosures */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* System Info */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-md font-bold text-slate-900 mb-3 flex items-center gap-2">
                <Server className="w-4 h-4 text-blue-600" />
                Service &amp; Runtime Configuration
              </h3>
              <div className="divide-y divide-slate-100 text-sm">
                <div className="py-2.5 flex justify-between">
                  <span className="text-slate-500">Service Name</span>
                  <span className="font-medium text-slate-900">VoiceShield API</span>
                </div>
                <div className="py-2.5 flex justify-between">
                  <span className="text-slate-500">Active Detector</span>
                  <span className="font-mono text-slate-900">{opStatus?.active_detector || 'VoiceShield-AASIST-Pretrained-v1'}</span>
                </div>
                <div className="py-2.5 flex justify-between">
                  <span className="text-slate-500">Environment</span>
                  <span className="font-mono text-slate-900 uppercase">{opStatus?.environment || 'development'}</span>
                </div>
                <div className="py-2.5 flex justify-between">
                  <span className="text-slate-500">Uptime</span>
                  <span className="font-mono text-slate-900">
                    {opStatus?.uptime_seconds !== undefined ? formatUptime(opStatus.uptime_seconds) : 'N/A'}
                  </span>
                </div>
                <div className="py-2.5 flex justify-between">
                  <span className="text-slate-500">Session Capacity</span>
                  <span className="font-mono text-slate-900">
                    {opStatus ? `${opStatus.active_sessions} / ${opStatus.max_active_sessions}` : '0 / 20'}
                  </span>
                </div>
              </div>
            </div>

            {/* Scientific Disclosures */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-md font-bold text-slate-900 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-amber-600" />
                Scientific Disclosures &amp; Honesty Notice
              </h3>
              <div className="space-y-3 text-xs text-slate-600">
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <span className="font-semibold text-amber-900">Pretrained Weights Disclosure:</span>
                  <p className="mt-1">
                    AASIST and ECAPA-TDNN operate with genuine pretrained checkpoints. Without local evaluation dataset manifests,
                    they retain provisional, uncalibrated scientific status.
                  </p>
                </div>
                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <span className="font-semibold text-blue-900">Ephemeral Privacy Guarantee:</span>
                  <p className="mt-1">
                    Raw audio buffers are held strictly in memory with a rolling 15s limit and zeroed upon session termination.
                    High-dimensional embeddings are never persisted or exposed in structured logs.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: MODEL LIFECYCLE & INTEGRITY */}
      {activeTab === 'models' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-600" />
              Pretrained Model Operational Telemetry &amp; SHA-256 Verification
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {models && models.length > 0 ? (
                models.map((m) => (
                  <div key={m.model_id} className="border border-slate-200 rounded-lg p-5 bg-slate-50/50">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900 text-base">{m.display_name || m.model_id}</span>
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-semibold ${
                          m.checkpoint_verified
                            ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                            : 'bg-rose-100 text-rose-800 border border-rose-300'
                        }`}
                      >
                        {m.checkpoint_verified ? 'SHA-256 VERIFIED' : 'INTEGRITY MISMATCH'}
                      </span>
                    </div>

                    <div className="mt-4 space-y-2 text-xs">
                      <div>
                        <span className="text-slate-500">Expected Checksum:</span>
                        <div className="font-mono text-slate-700 bg-white p-1 rounded border mt-0.5 break-all">
                          {m.expected_sha256 || 'N/A'}
                        </div>
                      </div>
                      <div>
                        <span className="text-slate-500">Actual Checksum:</span>
                        <div className="font-mono text-slate-700 bg-white p-1 rounded border mt-0.5 break-all">
                          {m.actual_sha256 || 'N/A'}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-4 pt-3 border-t border-slate-200 text-center">
                      <div>
                        <div className="text-slate-400 text-xs">Inferences</div>
                        <div className="font-bold text-slate-800 mt-0.5">{safeNumber(m.inference_count, 0)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 text-xs">Failures</div>
                        <div className="font-bold text-slate-800 mt-0.5">{safeNumber(m.inference_failure_count, 0)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 text-xs">Avg Latency</div>
                        <div className="font-bold text-slate-800 mt-0.5">{formatNumber(m.average_latency_ms, 1)}ms</div>
                      </div>
                    </div>

                    <div className="mt-3 p-2 bg-amber-50 rounded border border-amber-200 text-amber-800 text-xs">
                      <span className="font-semibold">Status:</span> {m.scientific_status || 'PRETRAINED_NOT_YET_VALIDATED'}
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-4 text-xs text-slate-400 col-span-2">
                  No model tracking entries available.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: STREAMING OPERATIONS */}
      {activeTab === 'streaming' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Radio className="w-5 h-5 text-sky-600" />
              Real-Time Streaming Telemetry &amp; Protection Controls
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs font-semibold text-slate-500 uppercase">Active / Max Sessions</div>
                <div className="text-2xl font-bold text-slate-800 mt-1">
                  {streaming ? `${streamActive} / ${streamMax || 20}` : '0 / 20'}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Created: {safeNumber(metrics?.streaming?.total_sessions_created, 0)}
                </div>
              </div>

              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs font-semibold text-slate-500 uppercase">Avg Window Latency</div>
                <div className="text-2xl font-bold text-slate-800 mt-1">
                  {metrics?.streaming?.average_window_latency_ms !== undefined
                    ? `${formatNumber(metrics.streaming.average_window_latency_ms, 1)}ms`
                    : '0.0ms'}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Max Lag: {formatNumber(metrics?.streaming?.max_processing_lag_ms, 1)}ms
                </div>
              </div>

              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-xs font-semibold text-slate-500 uppercase">Rate Limit Events</div>
                <div className="text-2xl font-bold text-slate-800 mt-1">
                  {safeNumber(streaming?.rate_limit_count, 0)}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Cap: 20 chunks/sec
                </div>
              </div>
            </div>

            <div className="border border-slate-200 rounded-lg p-4 bg-slate-50">
              <h4 className="font-semibold text-sm text-slate-800 mb-2">Streaming Production Safeguards</h4>
              <ul className="space-y-1.5 text-xs text-slate-600 list-disc list-inside">
                <li>Strict rolling audio buffer cap at {formatNumber(streaming?.buffer_memory_cap_sec, 1, '15.0')}s per session</li>
                <li>Idle session TTL automatic reap timeout: 120s</li>
                <li>Maximum message chunk size restricted to 64KB (preventing memory exhaustion DOS)</li>
                <li>WebSocket upgrade reverse proxy buffering disabled for minimal end-to-end latency</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: ERROR TAXONOMY */}
      {activeTab === 'errors' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-rose-600" />
              Unified Operational Error Taxonomy &amp; Failure Log
            </h2>

            {/* Error Distribution Badges */}
            <div className="mb-6">
              <div className="text-xs font-semibold uppercase text-slate-500 mb-2">Error Distribution by Taxonomy Code</div>
              <div className="flex flex-wrap gap-2">
                {errors?.error_counts_by_code && Object.keys(errors.error_counts_by_code).length > 0 ? (
                  Object.entries(errors.error_counts_by_code).map(([code, count]) => (
                    <span key={code} className="inline-flex items-center gap-1.5 px-3 py-1 bg-slate-100 border border-slate-200 rounded-full text-xs font-medium text-slate-700">
                      <span className="font-mono text-rose-700">{code}:</span>
                      <span className="font-bold">{safeNumber(count, 0)}</span>
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-400">No operational errors logged in this lifecycle.</span>
                )}
              </div>
            </div>

            {/* Error Summary Card */}
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1">
              <div className="text-slate-700 font-semibold">Total Errors Recorded: <span className="font-mono text-rose-600">{safeNumber(errors?.total_errors_recorded, 0)}</span></div>
              <div className="text-slate-500">Last Telemetry Snapshot: <span className="font-mono">{errors?.timestamp || 'N/A'}</span></div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: SECURITY & DEPLOY SPECS */}
      {activeTab === 'security' && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Lock className="w-5 h-5 text-emerald-600" />
              Security Hardening &amp; Deployment Specifications
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
                <h3 className="font-bold text-sm text-slate-900 mb-2">Container Hardening (Dockerfile)</h3>
                <ul className="space-y-1 text-xs text-slate-600 list-disc list-inside">
                  <li>Multi-stage build separating builder and minimal runtime</li>
                  <li>Non-root execution under unprivileged user <code className="bg-slate-200 px-1 rounded font-mono">voiceshield (uid: 10001)</code></li>
                  <li>Docker health check probing non-blocking <code className="bg-slate-200 px-1 rounded font-mono">/api/health/live</code></li>
                  <li>Minimal package surface excluding build tooling from final image</li>
                </ul>
              </div>

              <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
                <h3 className="font-bold text-sm text-slate-900 mb-2">Observability &amp; Data Protection</h3>
                <ul className="space-y-1 text-xs text-slate-600 list-disc list-inside">
                  <li>Strict sensitive data redactor scrubbing raw audio and 512-d embeddings</li>
                  <li>Structured JSON logging with request correlation IDs (<code className="bg-slate-200 px-1 rounded font-mono">X-Request-ID</code>)</li>
                  <li>Internal Python tracebacks suppressed from production client responses</li>
                  <li>CORS origin restriction via environment variable <code className="bg-slate-200 px-1 rounded font-mono">CORS_ORIGINS</code></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
export default OperationsDashboard;

