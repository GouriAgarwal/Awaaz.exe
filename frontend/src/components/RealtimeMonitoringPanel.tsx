import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Radio,
  Mic,
  Square,
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  Flame,
  Activity,
  Layers,
  Clock,
  Cpu,
  Info,
  CheckCircle2,
  XCircle,
  Play,
  RotateCcw,
  Zap,
  Volume2,
  Fingerprint,
} from 'lucide-react';
import {
  StreamingConnectionState,
  StreamEventInfo,
  RollingRiskMetricsInfo,
  SpeakerProfileSummaryInfo,
} from '../types';
import { getStreamingWsUrl, listSpeakerProfiles } from '../services/api';

export function RealtimeMonitoringPanel() {
  const [connectionState, setConnectionState] = useState<StreamingConnectionState>('DISCONNECTED');
  const [enrolledProfiles, setEnrolledProfiles] = useState<string[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');
  
  // Streaming state
  const [elapsedDurationSec, setElapsedDurationSec] = useState<number>(0);
  const [analyzedWindowsCount, setAnalyzedWindowsCount] = useState<number>(0);
  const [latestLatencyMs, setLatestLatencyMs] = useState<number | null>(null);
  const [processingLagMs, setProcessingLagMs] = useState<number | null>(null);
  const [latestEvent, setLatestEvent] = useState<StreamEventInfo | null>(null);
  const [eventsLog, setEventsLog] = useState<StreamEventInfo[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Audio stream refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const chunkSeqRef = useRef<number>(0);
  const timerRef = useRef<any>(null);

  // Load enrolled profiles
  useEffect(() => {
    listSpeakerProfiles()
      .then((profiles) => {
        if (profiles && profiles.length > 0) {
          const ids = profiles.map((p) => p.profile_id);
          setEnrolledProfiles(ids);
          setSelectedProfileId(ids[0]);
        }
      })
      .catch(console.error);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  const startStreaming = async () => {
    setErrorMessage(null);
    setConnectionState('CONNECTING');
    setEventsLog([]);
    setLatestEvent(null);
    setElapsedDurationSec(0);
    setAnalyzedWindowsCount(0);
    setLatestLatencyMs(null);
    setProcessingLagMs(null);
    chunkSeqRef.current = 0;

    try {
      // 1. Initialize Microphone Audio Stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      // Buffer size 4096 samples at 16kHz = ~0.256s chunk
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorNodeRef.current = processor;

      // 2. Initialize WebSocket
      const wsUrl = getStreamingWsUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionState('CONNECTED');
        // Send start control message
        const startMsg = {
          action: 'start',
          sample_rate: 16000,
          channels: 1,
          encoding: 'pcm_f32le',
          enrolled_profile_id: selectedProfileId.trim() || undefined,
          analysis_window_sec: 3.0,
          analysis_hop_sec: 1.5,
        };
        ws.send(JSON.stringify(startMsg));

        // Start elapsed timer
        timerRef.current = setInterval(() => {
          setElapsedDurationSec((prev) => prev + 0.5);
        }, 500);
      };

      ws.onmessage = (event) => {
        try {
          const ev: StreamEventInfo = JSON.parse(event.data);
          setLatestEvent(ev);
          setEventsLog((prev) => [ev, ...prev.slice(0, 49)]);

          if (ev.processing_latency_ms !== undefined && ev.processing_latency_ms !== null) {
            setLatestLatencyMs(ev.processing_latency_ms);
          }
          if (ev.processing_lag_ms !== undefined && ev.processing_lag_ms !== null) {
            setProcessingLagMs(ev.processing_lag_ms);
          }
          if (ev.timing) {
            setAnalyzedWindowsCount(ev.timing.window_index + 1);
          }
          if (ev.event_type === 'STREAM_ERROR') {
            setErrorMessage(ev.message);
            setConnectionState('ERROR');
          }
        } catch (err: any) {
          console.error('Error parsing streaming event:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('Streaming WebSocket error:', err);
        setErrorMessage('WebSocket connection error occurred.');
        setConnectionState('ERROR');
      };

      ws.onclose = () => {
        setConnectionState('DISCONNECTED');
        if (timerRef.current) clearInterval(timerRef.current);
      };

      // 3. Audio Chunk Ingestion Handler
      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          // Create a copy of float32 samples
          const samplesCopy = new Float32Array(inputData);
          
          // Convert to Base64
          const uint8 = new Uint8Array(samplesCopy.buffer);
          let binary = '';
          for (let i = 0; i < uint8.byteLength; i++) {
            binary += String.fromCharCode(uint8[i]);
          }
          const base64Data = window.btoa(binary);

          const chunkMsg = {
            action: 'chunk',
            sequence_number: chunkSeqRef.current++,
            data: base64Data,
          };
          ws.send(JSON.stringify(chunkMsg));
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err: any) {
      console.error('Failed to start streaming:', err);
      setErrorMessage(err.message || 'Microphone capture failed.');
      setConnectionState('ERROR');
      stopStreaming();
    }
  };

  const stopStreaming = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionState('DISCONNECTED');
  };

  // Helper for risk styles
  const getRiskColor = (level?: string | null) => {
    switch (level) {
      case 'LOW':
        return { text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40', bg: 'from-emerald-500 to-teal-600' };
      case 'MEDIUM':
        return { text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40', bg: 'from-amber-500 to-yellow-600' };
      case 'HIGH':
        return { text: 'text-orange-400', badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40', bg: 'from-orange-500 to-amber-600' };
      case 'CRITICAL':
        return { text: 'text-rose-400', badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40 font-black animate-pulse', bg: 'from-rose-500 to-red-600' };
      default:
        return { text: 'text-slate-400', badge: 'bg-slate-800 text-slate-300 border-slate-700', bg: 'from-slate-600 to-slate-700' };
    }
  };

  const riskScore = latestEvent?.risk_score ?? 0;
  const riskLevel = latestEvent?.risk_level ?? 'LOW';
  const riskStyle = getRiskColor(riskLevel);
  const rolling = latestEvent?.rolling_metrics;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Top Banner & Stream Controls */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-rose-500/20 text-rose-400">
                <Radio className="w-5 h-5 animate-pulse" />
              </span>
              <h3 className="text-lg font-bold text-white tracking-tight">
                Real-Time Streaming Impersonation Defense
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Live microphone audio is chunked into rolling 3.0-second analysis windows with a 1.5-second hop, evaluating AASIST synthetic voice artifacts, ECAPA speaker verification, and Step 9 fusion.
            </p>
          </div>

          {/* Connection Status Badge */}
          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1.5 border ${
                connectionState === 'CONNECTED'
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : connectionState === 'CONNECTING'
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : connectionState === 'ERROR'
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  connectionState === 'CONNECTED'
                    ? 'bg-emerald-400 animate-ping'
                    : connectionState === 'CONNECTING'
                    ? 'bg-amber-400 animate-pulse'
                    : connectionState === 'ERROR'
                    ? 'bg-rose-400'
                    : 'bg-slate-500'
                }`}
              />
              <span>{connectionState}</span>
            </span>
          </div>
        </div>

        {/* Target Profile Selector & Actions */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-3 border-t border-slate-800/80">
          <div className="flex items-center gap-2.5 w-full sm:w-auto">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 shrink-0">
              <Fingerprint className="w-4 h-4 text-blue-400" />
              <span>Target Enrolled Identity:</span>
            </label>
            <input
              type="text"
              value={selectedProfileId}
              onChange={(e) => setSelectedProfileId(e.target.value)}
              disabled={connectionState === 'CONNECTED'}
              placeholder="e.g. executive_spk_01 (optional)"
              className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-white focus:outline-none focus:border-indigo-500 disabled:opacity-50 w-full sm:w-64"
            />
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            {connectionState !== 'CONNECTED' ? (
              <button
                onClick={startStreaming}
                disabled={connectionState === 'CONNECTING'}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white text-xs font-bold uppercase tracking-wider shadow-lg shadow-rose-600/30 transition disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Mic className="w-4 h-4" />
                <span>Start Live Voice Stream</span>
              </button>
            ) : (
              <button
                onClick={stopStreaming}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold uppercase tracking-wider shadow-lg shadow-rose-600/30 transition flex items-center justify-center gap-2"
              >
                <Square className="w-4 h-4 fill-current" />
                <span>Stop Stream</span>
              </button>
            )}
          </div>
        </div>

        {errorMessage && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}
      </div>

      {/* Live Stream Telemetry Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase">
            <Clock className="w-3.5 h-3.5 text-indigo-400" />
            <span>Elapsed Audio</span>
          </div>
          <p className="text-2xl font-black text-white font-mono">
            {elapsedDurationSec.toFixed(1)} <span className="text-xs text-slate-500 font-normal">sec</span>
          </p>
          <p className="text-[10px] font-mono text-slate-500">Observation window: 3.0s</p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Analyzed Windows</span>
          </div>
          <p className="text-2xl font-black text-white font-mono">
            {analyzedWindowsCount} <span className="text-xs text-slate-500 font-normal">evaluated</span>
          </p>
          <p className="text-[10px] font-mono text-slate-500">Hop rate: 1.5s</p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>Inference Latency</span>
          </div>
          <p className="text-2xl font-black text-white font-mono">
            {latestLatencyMs !== null ? `${Math.round(latestLatencyMs)}` : '—'}{' '}
            <span className="text-xs text-slate-500 font-normal">ms</span>
          </p>
          <p className="text-[10px] font-mono text-slate-500">AASIST + ECAPA + Fusion</p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center gap-1.5 text-slate-400 text-[11px] font-semibold uppercase">
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            <span>Processing Lag</span>
          </div>
          <p className="text-2xl font-black text-white font-mono">
            {processingLagMs !== null ? `${Math.round(processingLagMs)}` : '—'}{' '}
            <span className="text-xs text-slate-500 font-normal">ms</span>
          </p>
          <p className="text-[10px] font-mono text-slate-500">Stream buffer delta</p>
        </div>
      </div>

      {/* Main Rolling Risk Card */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex items-center gap-5">
            {/* Risk Gauge Circle */}
            <div className="relative w-24 h-24 rounded-full flex flex-col items-center justify-center bg-slate-950/80 border-4 border-slate-800 shadow-inner">
              <span className="text-3xl font-black text-white tracking-tight">
                {Math.round(riskScore)}
              </span>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                / 100
              </span>
            </div>

            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-black uppercase tracking-wider border ${riskStyle.badge}`}>
                  {riskLevel} RISK
                </span>
                {rolling && (
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-mono bg-slate-950 border border-slate-800 text-slate-400">
                    Rolling Max: {rolling.rolling_max_risk.toFixed(1)}
                  </span>
                )}
              </div>
              <h3 className="text-lg font-bold text-white tracking-tight">
                {latestEvent ? latestEvent.recommended_action || 'EVALUATING' : 'Awaiting Audio Stream (3.0s window)'}
              </h3>
              <p className="text-xs text-slate-300 max-w-xl">
                {latestEvent?.action_rationale || 'Real-time multi-modal streaming analysis is active. Results update every 1.5 seconds.'}
              </p>
            </div>
          </div>

          {/* Action Policy Box */}
          {latestEvent && (
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1 text-right min-w-[200px]">
              <p className="text-[10px] uppercase font-bold text-slate-400">Operational Policy</p>
              <p className="text-sm font-black text-rose-400 tracking-tight">
                {latestEvent.recommended_action}
              </p>
              {rolling && (
                <p className="text-[10px] font-mono text-slate-500">
                  Critical streak: {rolling.consecutive_critical_windows} win
                </p>
              )}
            </div>
          )}
        </div>

        {/* Progress Bar */}
        <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-slate-400">Current Impersonation Risk Index</span>
            <span className={riskStyle.text}>{riskScore.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden p-0.5 border border-slate-800">
            <div
              className={`h-full rounded-full bg-gradient-to-r ${riskStyle.bg} transition-all duration-300`}
              style={{ width: `${Math.min(Math.max(riskScore, 2), 100)}%` }}
            />
          </div>
        </div>

        {/* Dual Signals Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* AASIST Synthetic Signal */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-indigo-400">
                <Radio className="w-4 h-4" />
                <h4 className="text-xs font-bold text-slate-200 uppercase">AASIST Synthetic Detector</h4>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                {latestEvent?.synthetic_signal?.classification || 'WAITING'}
              </span>
            </div>

            <div className="flex justify-between text-xs font-mono pt-1">
              <span className="text-slate-400">Softmax Spoof Score:</span>
              <span className="font-bold text-white">
                {latestEvent?.synthetic_signal ? `${(latestEvent.synthetic_signal.score * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <p className="text-[10px] font-mono text-slate-500">
              Confidence: {latestEvent?.synthetic_signal?.confidence_band || '—'}
            </p>
          </div>

          {/* ECAPA Speaker Verification Signal */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-blue-400">
                <Fingerprint className="w-4 h-4" />
                <h4 className="text-xs font-bold text-slate-200 uppercase">ECAPA Speaker Verification</h4>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/10 text-blue-300 border border-blue-500/30">
                {latestEvent?.speaker_signal?.status || (selectedProfileId ? 'WAITING' : 'NOT_AVAILABLE')}
              </span>
            </div>

            <div className="flex justify-between text-xs font-mono pt-1">
              <span className="text-slate-400">Cosine Similarity:</span>
              <span className="font-bold text-white">
                {latestEvent?.speaker_signal?.similarity_score !== undefined && latestEvent?.speaker_signal?.similarity_score !== null
                  ? latestEvent.speaker_signal.similarity_score.toFixed(3)
                  : 'N/A'}
              </span>
            </div>
            <p className="text-[10px] font-mono text-slate-500">
              Profile: {selectedProfileId || 'None (Synthetic Detection Only)'}
            </p>
          </div>
        </div>
      </div>

      {/* Chronological Live Event Log */}
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Activity className="w-4 h-4 text-rose-400" />
            <span>Chronological Streaming Event Log ({eventsLog.length} events)</span>
          </h4>
          <span className="text-[10px] font-mono text-slate-500">Streaming evaluation: NOT_VALIDATED</span>
        </div>

        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {eventsLog.length === 0 ? (
            <div className="p-4 text-center text-xs text-slate-500 font-mono">
              No streaming events yet. Click &quot;Start Live Voice Stream&quot; to begin.
            </div>
          ) : (
            eventsLog.map((ev, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800/80 text-xs flex items-start justify-between gap-3"
              >
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono font-bold text-slate-400">
                      {new Date(ev.timestamp * 1000).toLocaleTimeString()}
                    </span>
                    <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-slate-800 text-indigo-300 border border-slate-700">
                      {ev.event_type}
                    </span>
                  </div>
                  <p className="text-slate-300 text-[11px]">{ev.message}</p>
                </div>

                {ev.risk_score !== undefined && ev.risk_score !== null && (
                  <div className="text-right shrink-0">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${getRiskColor(ev.risk_level).badge}`}>
                      {ev.risk_level} ({ev.risk_score.toFixed(1)})
                    </span>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Scientific Disclosure Notice */}
      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-3">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold text-slate-300">Observation Latency Notice: </span>
          <span>
            Real-time monitoring does not mean instantaneous detection. AASIST requires sufficient audio for its 3-second analysis window, so the system has an inherent 3-second observation delay in addition to computational inference latency (~700ms).
          </span>
          <div className="flex gap-4 font-mono text-[10px] text-slate-500 pt-1">
            <span>Evaluation Status: NOT_VALIDATED</span>
            <span>Calibration: NOT_CALIBRATED</span>
            <span>Policy: Raw Audio RAM-Only</span>
          </div>
        </div>
      </div>
    </div>
  );
}
