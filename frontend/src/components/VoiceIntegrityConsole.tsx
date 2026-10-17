import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  UserCheck,
  Activity,
  DollarSign,
  AlertTriangle,
  Play,
  Mic,
  Square,
  Upload,
  RefreshCw,
  Info,
  CheckCircle2,
  XCircle,
  PhoneCall,
  Lock,
  Zap,
  Sliders,
  Sparkles,
} from 'lucide-react';
import {
  MultiLayerRiskAnalysisResultInfo,
  AttackScenarioInfo,
  ExtendedActionType,
  SpeakerProfileSummaryInfo,
} from '../types';
import {
  getAttackScenarios,
  simulateContextualRisk,
  analyzeContextualRisk,
  getEnrolledProfiles,
} from '../services/api';

export function VoiceIntegrityConsole() {
  // Scenarios and Profiles
  const [scenarios, setScenarios] = useState<AttackScenarioInfo[]>([]);
  const [profiles, setProfiles] = useState<SpeakerProfileSummaryInfo[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('');

  // Active Mode: 'scenario' | 'live' | 'custom'
  const [activeMode, setActiveMode] = useState<'scenario' | 'live' | 'custom'>('scenario');

  // Context State
  const [callType, setCallType] = useState<string>('FINANCIAL_TRANSACTION');
  const [callerTrust, setCallerTrust] = useState<string>('VIP_OR_EXECUTIVE');
  const [requestedAction, setRequestedAction] = useState<string>('FUND_TRANSFER');
  const [transactionAmount, setTransactionAmount] = useState<number | ''>(250000);
  const [historicalRisk, setHistoricalRisk] = useState<string>('MEDIUM');
  const [profileId, setProfileId] = useState<string>('vip_executive_ceo');

  // Custom simulation sliders
  const [syntheticScore, setSyntheticScore] = useState<number>(0.92);
  const [speakerSimilarity, setSpeakerSimilarity] = useState<number>(0.86);
  const [prosodyClass, setProsodyClass] = useState<string>('LOW_VARIATION');

  // Audio Upload & Recording
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<any>(null);

  // Analysis Result & Loading State
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<MultiLayerRiskAnalysisResultInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load scenarios and enrolled profiles on mount
  useEffect(() => {
    async function initData() {
      try {
        const [loadedScenarios, loadedProfiles] = await Promise.all([
          getAttackScenarios().catch(() => []),
          getEnrolledProfiles().catch(() => []),
        ]);
        setScenarios(loadedScenarios);
        setProfiles(loadedProfiles);

        if (loadedScenarios.length > 0) {
          selectScenario(loadedScenarios[0]);
        }
      } catch (err: any) {
        console.warn('Failed initializing console metadata:', err);
      }
    }
    initData();
  }, []);

  const selectScenario = (sc: AttackScenarioInfo) => {
    setSelectedScenarioId(sc.id);
    setSyntheticScore(sc.synthetic_score);
    setSpeakerSimilarity(sc.speaker_similarity);
    setProsodyClass(sc.prosody_classification);
    setCallType(sc.call_type);
    setCallerTrust(sc.caller_trust);
    setRequestedAction(sc.requested_action);
    setTransactionAmount(sc.transaction_amount ?? '');
    setHistoricalRisk(sc.historical_risk);

    // Auto-run simulation for the chosen scenario
    executeSimulation({
      synthetic_score: sc.synthetic_score,
      speaker_similarity: sc.speaker_similarity,
      prosody_classification: sc.prosody_classification,
      call_type: sc.call_type,
      caller_trust: sc.caller_trust,
      requested_action: sc.requested_action,
      transaction_amount: sc.transaction_amount ?? null,
      historical_risk: sc.historical_risk,
      profile_id: 'target_enrolled_profile',
    });
  };

  const executeSimulation = async (payloadOverride?: any) => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = payloadOverride || {
        synthetic_score: syntheticScore,
        speaker_similarity: speakerSimilarity,
        prosody_classification: prosodyClass,
        call_type: callType,
        caller_trust: callerTrust,
        requested_action: requestedAction,
        transaction_amount: transactionAmount === '' ? null : Number(transactionAmount),
        historical_risk: historicalRisk,
        profile_id: profileId.trim() || 'simulated_profile',
      };
      const data = await simulateContextualRisk(payload);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Simulation failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioFile(file);
    executeLiveAnalysis(file);
  };

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([blob], 'live_recording.wav', { type: 'audio/wav' });
        setAudioFile(recordedFile);
        executeLiveAnalysis(recordedFile);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start(200);
      setIsRecording(true);
      setRecordingSeconds(0);

      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 15) {
            stopRecording();
            return 15;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err: any) {
      setError(`Microphone access error: ${err.message || 'Permission denied'}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const executeLiveAnalysis = async (fileToAnalyze: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', fileToAnalyze);
      if (profileId.trim()) formData.append('profile_id', profileId.trim());
      formData.append('call_type', callType);
      formData.append('caller_trust', callerTrust);
      formData.append('requested_action', requestedAction);
      if (transactionAmount !== '') {
        formData.append('transaction_amount', String(transactionAmount));
      }
      formData.append('historical_risk', historicalRisk);

      const data = await analyzeContextualRisk(formData);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Live audio analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // Helper colors for risk levels
  const getRiskColor = (level?: string) => {
    switch (level) {
      case 'CRITICAL':
        return {
          badge: 'bg-rose-500/20 text-rose-400 border-rose-500/40',
          gauge: 'text-rose-500 border-rose-500',
          bg: 'from-rose-950/40 to-slate-900/60 border-rose-500/30',
        };
      case 'HIGH':
        return {
          badge: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
          gauge: 'text-amber-500 border-amber-500',
          bg: 'from-amber-950/40 to-slate-900/60 border-amber-500/30',
        };
      case 'MEDIUM':
        return {
          badge: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
          gauge: 'text-yellow-500 border-yellow-500',
          bg: 'from-yellow-950/40 to-slate-900/60 border-yellow-500/30',
        };
      default:
        return {
          badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
          gauge: 'text-emerald-500 border-emerald-500',
          bg: 'from-emerald-950/40 to-slate-900/60 border-emerald-500/30',
        };
    }
  };

  const getActionBanner = (action: ExtendedActionType) => {
    switch (action) {
      case 'BLOCK_TRANSACTION_AND_ESCALATE':
        return {
          title: 'IMMEDIATE TRANSACTION HOLD & ESCALATE',
          color: 'bg-rose-600/30 border-rose-500 text-rose-200',
          icon: <XCircle className="w-5 h-5 text-rose-400" />,
          actionMsg: 'Block outbound funds immediately and notify Security Operations Center (SOC).',
        };
      case 'BLOCK_OR_ESCALATE':
        return {
          title: 'BLOCK SESSION & ESCALATE FRAUD INCIDENT',
          color: 'bg-rose-600/20 border-rose-500/60 text-rose-200',
          icon: <ShieldAlert className="w-5 h-5 text-rose-400" />,
          actionMsg: 'Terminate interaction. Flag voice channel for targeted voice cloning investigation.',
        };
      case 'CALL_BACK_REQUIRED':
        return {
          title: 'OUT-OF-BAND TELEPHONE CALL-BACK REQUIRED',
          color: 'bg-amber-600/20 border-amber-500/60 text-amber-200',
          icon: <PhoneCall className="w-5 h-5 text-amber-400" />,
          actionMsg: 'Do not authorize funds until primary registered phone is verified via verified callback.',
        };
      case 'MFA_REQUIRED':
        return {
          title: 'STEP-UP MULTI-FACTOR AUTHENTICATION REQUIRED',
          color: 'bg-amber-600/20 border-amber-500/60 text-amber-200',
          icon: <Lock className="w-5 h-5 text-amber-400" />,
          actionMsg: 'Enforce hardware security key or push authorization before modifying credentials.',
        };
      case 'STEP_UP_VERIFICATION':
        return {
          title: 'STEP-UP SECURITY CHALLENGE REQUIRED',
          color: 'bg-yellow-600/20 border-yellow-500/60 text-yellow-200',
          icon: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
          actionMsg: 'Secondary knowledge-based or biometric confirmation recommended.',
        };
      case 'MONITOR':
        return {
          title: 'MONITOR INTERACTION (CAUTION)',
          color: 'bg-slate-800/80 border-slate-700 text-slate-200',
          icon: <Activity className="w-5 h-5 text-blue-400" />,
          actionMsg: 'Log conversation telemetry; monitor for further anomalous requests.',
        };
      default:
        return {
          title: 'PERMIT INTERACTION (LOW RISK)',
          color: 'bg-emerald-600/20 border-emerald-500/60 text-emerald-200',
          icon: <CheckCircle2 className="w-5 h-5 text-emerald-400" />,
          actionMsg: 'Acoustic voice characteristics and business context are within authorized thresholds.',
        };
    }
  };

  const riskColors = getRiskColor(result?.risk_level);

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Flagship Header */}
      <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-r from-indigo-950/40 via-slate-900/60 to-purple-950/40 p-6 backdrop-blur-md shadow-xl">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 text-xs font-semibold mb-2">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>Step 13 • Multi-Layer Voice Cloning &amp; Impersonation Defense</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
              Voice Integrity Console
            </h1>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl">
              Real-time multi-layer defense uniting AI synthetic voice detection (AASIST), biometric speaker matching (ECAPA-TDNN),
              prosodic behavioral dynamics, and business transaction intelligence to neutralize voice cloning fraud.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800 self-stretch lg:self-auto">
            <button
              onClick={() => setActiveMode('scenario')}
              className={`flex-1 lg:flex-none px-4 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                activeMode === 'scenario'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>Attack Scenarios</span>
            </button>
            <button
              onClick={() => setActiveMode('live')}
              className={`flex-1 lg:flex-none px-4 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                activeMode === 'live'
                  ? 'bg-rose-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Live Audio / File</span>
            </button>
            <button
              onClick={() => setActiveMode('custom')}
              className={`flex-1 lg:flex-none px-4 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                activeMode === 'custom'
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Custom Tuning</span>
            </button>
          </div>
        </div>

        {/* Quick Attack Scenario Selector */}
        {activeMode === 'scenario' && scenarios.length > 0 && (
          <div className="mt-6 pt-5 border-t border-slate-800/80">
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
              Select Pre-Configured Enterprise Attack Scenario:
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {scenarios.map((sc) => (
                <button
                  key={sc.id}
                  onClick={() => selectScenario(sc)}
                  className={`text-left p-3.5 rounded-xl border transition-all ${
                    selectedScenarioId === sc.id
                      ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-lg ring-1 ring-indigo-500/50'
                      : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-850'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold truncate">{sc.name}</span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-bold ${
                        sc.expected_outcome.risk_level === 'CRITICAL'
                          ? 'bg-rose-500/20 text-rose-300'
                          : sc.expected_outcome.risk_level === 'HIGH'
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-emerald-500/20 text-emerald-300'
                      }`}
                    >
                      {sc.expected_outcome.risk_level}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                    {sc.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input Controls Container */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Audio / Signal Inputs (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-5">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span>1. Voice Acoustic &amp; Biometric Signal</span>
          </h2>

          {activeMode === 'live' ? (
            <div className="space-y-4">
              {/* Mic Recording */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center text-center space-y-3">
                <div className="text-xs text-slate-400 font-medium">Record live speech from microphone</div>
                <div className="flex items-center gap-3">
                  {!isRecording ? (
                    <button
                      onClick={startRecording}
                      disabled={isLoading}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold shadow-lg shadow-rose-600/30 transition disabled:opacity-50"
                    >
                      <Mic className="w-4 h-4" />
                      <span>Start Recording</span>
                    </button>
                  ) : (
                    <button
                      onClick={stopRecording}
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-rose-400 text-xs font-bold border border-rose-500/50 animate-pulse transition"
                    >
                      <Square className="w-4 h-4 text-rose-500 fill-current" />
                      <span>Stop ({recordingSeconds}s / 15s)</span>
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-slate-500">Audio is processed in RAM and discarded immediately.</p>
              </div>

              {/* File Upload */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center text-center space-y-2">
                <Upload className="w-6 h-6 text-slate-400" />
                <div className="text-xs text-slate-300 font-semibold">Or upload pre-recorded audio file</div>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioUpload}
                  disabled={isLoading || isRecording}
                  className="text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
                />
                {audioFile && (
                  <span className="text-[11px] text-indigo-300 font-mono">Loaded: {audioFile.name}</span>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Sliders for Simulation */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-semibold">Synthetic Voice Detector Score (AASIST)</span>
                  <span className="font-mono text-indigo-300 font-bold">{(syntheticScore * 100).toFixed(1)}%</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={syntheticScore}
                  onChange={(e) => setSyntheticScore(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                  <span>0% (Natural Speech)</span>
                  <span>50% (Uncertain)</span>
                  <span>100% (Synthetic TTS / Clone)</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-semibold">Speaker Cosine Similarity (ECAPA-TDNN)</span>
                  <span className="font-mono text-blue-300 font-bold">{speakerSimilarity.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-0.5"
                  max="1.0"
                  step="0.01"
                  value={speakerSimilarity}
                  onChange={(e) => setSpeakerSimilarity(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                  <span>&lt; 0.50 (Non-Match)</span>
                  <span>0.65 (Decision Threshold)</span>
                  <span>&gt; 0.80 (Strong Match)</span>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Prosodic &amp; Behavioral Dynamics
                </label>
                <select
                  value={prosodyClass}
                  onChange={(e) => setProsodyClass(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="NATURAL_VARIATION">NATURAL_VARIATION (Dynamic conversational modulation)</option>
                  <option value="LOW_VARIATION">LOW_VARIATION (Monotone / robotic flat pitch &amp; energy)</option>
                  <option value="UNUSUAL_PROSODY">UNUSUAL_PROSODY (Atypical jitter / pitch discontinuities)</option>
                  <option value="INSUFFICIENT_AUDIO">INSUFFICIENT_AUDIO (Brief audio / low voiced frames)</option>
                </select>
              </div>
            </div>
          )}

          {/* Profile Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Target Enrolled Speaker Profile
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={profileId}
                onChange={(e) => setProfileId(e.target.value)}
                placeholder="e.g. vip_executive_ceo"
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              {profiles.length > 0 && (
                <select
                  onChange={(e) => setProfileId(e.target.value)}
                  value={profileId}
                  className="bg-slate-950 border border-slate-800 rounded-xl px-2 py-2 text-xs text-slate-300"
                >
                  <option value="">Select Enrolled...</option>
                  {profiles.map((p) => (
                    <option key={p.profile_id} value={p.profile_id}>
                      {p.profile_id}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>
        </div>

        {/* Right Col: Business & Transactional Context (7 cols) */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-emerald-400" />
              <span>2. Business &amp; Transactional Context</span>
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Call Type</label>
                <select
                  value={callType}
                  onChange={(e) => setCallType(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="NORMAL_CALL">NORMAL_CALL (Inquiry / Support)</option>
                  <option value="FINANCIAL_TRANSACTION">FINANCIAL_TRANSACTION (Banking / Wire)</option>
                  <option value="PRIVILEGED_ACCESS">PRIVILEGED_ACCESS (IT / Credentials)</option>
                  <option value="GOVERNMENT_INSTRUCTION">GOVERNMENT_INSTRUCTION (Authority)</option>
                  <option value="ENTERPRISE_APPROVAL">ENTERPRISE_APPROVAL (Corporate sign-off)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Caller Trust Level</label>
                <select
                  value={callerTrust}
                  onChange={(e) => setCallerTrust(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="VERIFIED_CONTACT">VERIFIED_CONTACT (Known enrolled channel)</option>
                  <option value="KNOWN_CONTACT">KNOWN_CONTACT (Existing customer contact)</option>
                  <option value="UNKNOWN_CALLER">UNKNOWN_CALLER (Unrecognized phone number)</option>
                  <option value="VIP_OR_EXECUTIVE">VIP_OR_EXECUTIVE (CEO / High-Value Target)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Requested Action</label>
                <select
                  value={requestedAction}
                  onChange={(e) => setRequestedAction(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="INFORMATION_ONLY">INFORMATION_ONLY (Status Check)</option>
                  <option value="PAYMENT_APPROVAL">PAYMENT_APPROVAL (Invoice Approval)</option>
                  <option value="FUND_TRANSFER">FUND_TRANSFER (Wire / ACH Transfer)</option>
                  <option value="CREDENTIAL_RESET">CREDENTIAL_RESET (Password / MFA Reset)</option>
                  <option value="SENSITIVE_DATA_DISCLOSURE">SENSITIVE_DATA_DISCLOSURE (PII)</option>
                  <option value="PRIVILEGED_ACCESS_CHANGE">PRIVILEGED_ACCESS_CHANGE (Admin Role)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Transaction Amount (USD)
                </label>
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-xs text-slate-500">$</span>
                  <input
                    type="number"
                    value={transactionAmount}
                    onChange={(e) =>
                      setTransactionAmount(e.target.value === '' ? '' : Math.max(0, parseFloat(e.target.value)))
                    }
                    placeholder="e.g. 250000"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-7 pr-3 py-2 text-xs font-mono text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Historical Account Risk</label>
                <select
                  value={historicalRisk}
                  onChange={(e) => setHistoricalRisk(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="NONE">NONE (Clean security record)</option>
                  <option value="LOW">LOW (Occasional travel / device changes)</option>
                  <option value="MEDIUM">MEDIUM (Prior suspicious login attempts)</option>
                  <option value="HIGH">HIGH (Previous confirmed compromise)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-slate-800 flex items-center justify-between gap-3">
            <span className="text-xs text-slate-400">
              Evaluates non-linear risk across all 4 defense pillars simultaneously.
            </span>
            <button
              onClick={() => executeSimulation()}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-bold shadow-lg shadow-indigo-600/30 transition disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Evaluating Defense Layers...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Evaluate Voice Integrity</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-rose-300 text-xs flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Analysis Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Actionable Prevention Recommendation Banner */}
          {(() => {
            const banner = getActionBanner(result.recommended_action);
            return (
              <div
                className={`rounded-2xl border p-5 backdrop-blur-md shadow-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${banner.color}`}
              >
                <div className="flex items-start sm:items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-slate-950/40 border border-white/10 shrink-0">
                    {banner.icon}
                  </div>
                  <div>
                    <div className="text-[11px] font-mono tracking-wider uppercase opacity-80">
                      Recommended Prevention Policy Action:
                    </div>
                    <h3 className="text-base sm:text-lg font-black tracking-tight">{banner.title}</h3>
                    <p className="text-xs opacity-90 mt-0.5">{banner.actionMsg}</p>
                    <p className="text-[10px] opacity-75 mt-1 font-mono italic">
                      {result.enforcement_disclaimer}
                    </p>
                  </div>
                </div>

                <div className="shrink-0 text-right">
                  <span className={`px-3 py-1 rounded-full text-xs font-bold border ${riskColors.badge}`}>
                    RISK LEVEL: {result.risk_level}
                  </span>
                  <div className="text-[11px] text-slate-300 font-mono mt-1">
                    Latency: {result.latency_ms.toFixed(1)}ms
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Primary Metric & Four Pillars Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Overall Score Gauge Card (4 cols) */}
            <div
              className={`lg:col-span-4 rounded-2xl border bg-gradient-to-b ${riskColors.bg} p-6 flex flex-col justify-between items-center text-center shadow-xl`}
            >
              <div className="w-full flex justify-between items-center text-xs text-slate-400">
                <span className="font-bold uppercase tracking-wider">Overall Impersonation Risk</span>
                <span className="font-mono text-[11px]">0 – 100</span>
              </div>

              <div className="my-6 relative flex items-center justify-center">
                <div
                  className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center bg-slate-950/80 shadow-2xl ${riskColors.gauge}`}
                >
                  <span className="text-4xl font-black text-white tracking-tight">
                    {result.overall_risk_score}
                  </span>
                  <span className="text-[11px] font-bold uppercase tracking-wider mt-0.5">
                    {result.risk_level} RISK
                  </span>
                </div>
              </div>

              <div className="w-full space-y-2 text-left bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80">
                <div className="text-[11px] text-slate-400 flex justify-between">
                  <span>Score Type:</span>
                  <span className="font-mono text-slate-300">heuristic_contextual</span>
                </div>
                <div className="text-[11px] text-slate-400 flex justify-between">
                  <span>Privacy Mode:</span>
                  <span className="font-mono text-emerald-400">RAM-Only (Ephemeral)</span>
                </div>
              </div>
            </div>

            {/* Four Pillars Breakdown (8 cols) */}
            <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Pillar 1: Synthetic Voice Detection */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase">
                    <ShieldAlert className="w-4 h-4 text-indigo-400" />
                    <span>Synthetic Detection</span>
                  </div>
                  <span
                    className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                      result.synthetic_signal.classification === 'SYNTHETIC'
                        ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                        : result.synthetic_signal.classification === 'NATURAL'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                    }`}
                  >
                    {result.synthetic_signal.classification}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>AASIST Score:</span>
                    <span className="font-mono text-white font-bold">
                      {(result.synthetic_signal.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Confidence:</span>
                    <span className="font-mono text-slate-300">{result.synthetic_signal.confidence_band}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Detector:</span>
                    <span className="font-mono text-slate-400 text-[11px]">
                      {result.synthetic_signal.detector_model}
                    </span>
                  </div>
                </div>
              </div>

              {/* Pillar 2: Speaker Identity Match */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase">
                    <UserCheck className="w-4 h-4 text-blue-400" />
                    <span>Speaker Verification</span>
                  </div>
                  <span
                    className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                      result.speaker_signal.decision === 'MATCH'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}
                  >
                    {result.speaker_signal.decision}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Cosine Similarity:</span>
                    <span className="font-mono text-white font-bold">
                      {result.speaker_signal.similarity_score.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Claimed Profile:</span>
                    <span className="font-mono text-indigo-300 truncate max-w-[120px]">
                      {result.speaker_signal.profile_id || 'unclaimed'}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Model:</span>
                    <span className="font-mono text-slate-400 text-[11px]">SpeechBrain ECAPA</span>
                  </div>
                </div>
              </div>

              {/* Pillar 3: Prosody & Behavioral Dynamics */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <span>Prosody Dynamics</span>
                  </div>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded font-mono ${
                      result.prosody_signal.classification === 'NATURAL_VARIATION'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : result.prosody_signal.classification === 'LOW_VARIATION'
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                        : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}
                  >
                    {result.prosody_signal.classification}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Pitch Variation (CV):</span>
                    <span className="font-mono text-white font-bold">
                      {result.prosody_signal.features.pitch_variation_coef !== null &&
                      result.prosody_signal.features.pitch_variation_coef !== undefined
                        ? (result.prosody_signal.features.pitch_variation_coef * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Pitch Range:</span>
                    <span className="font-mono text-slate-300">
                      {result.prosody_signal.features.pitch_range_hz !== null &&
                      result.prosody_signal.features.pitch_range_hz !== undefined
                        ? result.prosody_signal.features.pitch_range_hz.toFixed(1) + ' Hz'
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Dynamic Range:</span>
                    <span className="font-mono text-slate-300">
                      {result.prosody_signal.features.energy_dynamic_range_db.toFixed(1)} dB
                    </span>
                  </div>
                </div>
              </div>

              {/* Pillar 4: Business Context Risk */}
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase">
                    <DollarSign className="w-4 h-4 text-emerald-400" />
                    <span>Context Stakes</span>
                  </div>
                  <span className="text-[11px] font-bold px-2 py-0.5 rounded font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {result.context_signal.policy_sensitivity}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Context Score:</span>
                    <span className="font-mono text-white font-bold">
                      {result.context_signal.context_risk_score} / 100
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Sensitivity Multiplier:</span>
                    <span className="font-mono text-slate-300">
                      {result.context_signal.sensitivity_multiplier.toFixed(2)}x
                    </span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Target Action:</span>
                    <span className="font-mono text-emerald-400 truncate max-w-[130px]">
                      {result.context_signal.requested_action}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Why This Was Flagged (Structured Evidence & Rationale) */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Info className="w-4 h-4 text-indigo-400" />
              <span>3. Why This Was Flagged (Security Rationale &amp; Evidence)</span>
            </h2>

            {/* Primary plain English rationale */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-200 leading-relaxed">
              <span className="font-bold text-indigo-400 mr-1.5">Executive Summary:</span>
              {result.primary_rationale}
            </div>

            {/* Evidence items */}
            <div className="space-y-2">
              {result.evidence.map((ev, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-xl bg-slate-950/60 border border-slate-850 flex items-start justify-between gap-3 text-xs"
                >
                  <div className="flex items-start gap-2.5">
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded uppercase mt-0.5 ${
                        ev.severity === 'CRITICAL'
                          ? 'bg-rose-500/20 text-rose-400'
                          : ev.severity === 'HIGH'
                          ? 'bg-amber-500/20 text-amber-400'
                          : ev.severity === 'MEDIUM'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {ev.severity}
                    </span>
                    <div>
                      <div className="text-[11px] font-mono text-indigo-300">{ev.code}</div>
                      <div className="text-slate-300 mt-0.5">{ev.message}</div>
                    </div>
                  </div>

                  <span className="text-[10px] font-mono text-slate-500 uppercase shrink-0">
                    [{ev.layer}]
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default VoiceIntegrityConsole;
