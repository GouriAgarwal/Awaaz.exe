import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  PhoneCall,
  PhoneOff,
  Lock,
  UserCheck,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  PauseCircle,
  PlayCircle,
  Clock,
  DollarSign,
  Activity,
  Zap,
  Building,
  Radio,
  FileCheck,
  Send,
  RefreshCw,
  Info,
  ChevronRight,
  Globe,
  Sliders,
  Volume2,
  Server,
  Layers,
  ArrowRight,
  KeyRound,
  UserX,
  Sparkles,
  ExternalLink,
  Cpu,
  Fingerprint,
  Mic,
  Shield,
  HelpCircle
} from 'lucide-react';
import {
  PreventionEvaluationResponseInfo,
  PreventionScenarioInfo,
  PolicyProfileType,
  SensitiveActionTypeValue,
  PreventionStatusType,
  PreventionActionType,
  TimelineEventInfo,
} from '../types';
import {
  evaluatePrevention,
  verifyPreventionWorkflow,
  resolvePreventionWorkflow,
  getPreventionScenarios,
} from '../services/api';

// Safe numeric helpers
const safeNumber = (val: unknown, fallback = 0): number => {
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
};

const formatNumber = (val: unknown, decimals = 2, fallback = '0.00'): string => {
  const n = Number(val);
  return Number.isFinite(n) ? n.toFixed(decimals) : fallback;
};

const formatCurrency = (val: unknown, currency = '₹'): string => {
  const n = Number(val);
  if (!Number.isFinite(n) || n === 0) return 'N/A';
  return `${currency}${n.toLocaleString('en-IN')}`;
};

// Multilingual scam data
interface MultilingualPhrase {
  language: string;
  nativeName: string;
  phrase: string;
  translation: string;
  riskTrigger: string;
  context: string;
}

const MULTILINGUAL_PHRASES: Record<string, MultilingualPhrase> = {
  hi: {
    language: 'Hindi',
    nativeName: 'हिंदी',
    phrase: 'तत्काल 10 लाख रुपये ट्रांसफर करें, मैं बेहद जरूरी बोर्ड मीटिंग में हूं। OTP किसी को मत बताना।',
    translation: 'Immediately transfer ₹10,00,000, I am in an urgent board meeting. Do not share OTP with anyone.',
    riskTrigger: 'Extreme Urgency + Wire Instruction + OTP Secrecy',
    context: 'Corporate CEO Impersonation',
  },
  mr: {
    language: 'Marathi',
    nativeName: 'मराठी',
    phrase: 'मी विक्रम बोलतोय. कंपनीच्या खात्यातून तत्काळ ₹१०,००,००० पाठवा. वेळ वाया घालवू नका.',
    translation: 'This is Vikram speaking. Immediately transfer ₹10,00,000 from company account. Do not waste time.',
    riskTrigger: 'Authority Pressure + Direct Financial Demand',
    context: 'Treasury Wire Fraud',
  },
  ta: {
    language: 'Tamil',
    nativeName: 'தமிழ்',
    phrase: 'நான் தலைமை நிர்வாக அதிகாரி பேசுகிறேன். உடனடியாக ₹10,00,000 பணப்பரிமாற்றம் செய்யவும்.',
    translation: 'I am the CEO speaking. Immediately execute the ₹10,00,000 funds transfer.',
    riskTrigger: 'Executive Authority Demand + Immediate Execution',
    context: 'Executive Phone Spoofing',
  },
  te: {
    language: 'Telugu',
    nativeName: 'తెలుగు',
    phrase: 'నేను సిఇఒ విక్రమ్ మాట్లాడుతున్నాను. వెంటనే ₹10,00,000 వైర్ బదిలీ చేయండి, ఆలస్యం చేయవద్దు.',
    translation: 'I am CEO Vikram speaking. Wire ₹10,00,000 immediately without any delay.',
    riskTrigger: 'High-Value Urgent Request + Time Constraint',
    context: 'Financial Impersonation',
  },
  bn: {
    language: 'Bengali',
    nativeName: 'বাংলা',
    phrase: 'আমি বিক্রম বলছি। জরুরি ভিত্তিতে অবিলম্বে ১০,০০,০০০ টাকা স্থানান্তর করুন। কোনো প্রশ্ন করবেন না।',
    translation: 'This is Vikram. Transfer ₹10,00,000 immediately on emergency basis. Ask no questions.',
    riskTrigger: 'Urgency + Inquiry Suppression',
    context: 'Banking Voice Clone Attack',
  },
  en: {
    language: 'English',
    nativeName: 'English',
    phrase: 'This is Vikram. Wire $150,000 / ₹10,00,000 to the offshore account immediately for the acquisition. I am in a meeting.',
    translation: 'Standard executive spear-phishing wire transfer attack.',
    riskTrigger: 'Acquisition Pretext + Wire Transfer Urgency',
    context: 'CEO Business Email & Voice Compromise',
  },
};

// 10-Step Story Mode Steps
interface StoryStep {
  step: number;
  title: string;
  subtitle: string;
  activeComponent: 'call' | 'synth' | 'speaker' | 'prosody' | 'context' | 'risk' | 'action' | 'prevention' | 'timeline';
  riskScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  status: PreventionStatusType;
  action: PreventionActionType;
  highlightText: string;
}

const STORY_STEPS: StoryStep[] = [
  {
    step: 1,
    title: 'Incoming Call Ingestion & Carrier Intercept',
    subtitle: 'High-priority call received from +91 98201 XXXXX claiming to be CEO Vikram Malhotra.',
    activeComponent: 'call',
    riskScore: 15,
    riskLevel: 'LOW',
    status: 'MONITORING',
    action: 'MONITOR',
    highlightText: 'SIP trunk connected via secure carrier gateway. Live 16kHz audio stream ingested for real-time inspection.',
  },
  {
    step: 2,
    title: 'Acoustic Feature Extraction Pipeline',
    subtitle: 'Stream buffered into 500ms sliding windows. FFT, spectral centroid, and pitch extracted in 14ms.',
    activeComponent: 'call',
    riskScore: 22,
    riskLevel: 'LOW',
    status: 'MONITORING',
    action: 'MONITOR',
    highlightText: 'Standard acoustic preprocessing complete. Extracted spectral flatness and fundamental pitch (F0).',
  },
  {
    step: 3,
    title: 'AASIST Synthetic Voice Detection Initiated',
    subtitle: 'Pretrained Graph Neural Network (AASIST) inspects spectral sub-bands for neural synthesis artifacts.',
    activeComponent: 'synth',
    riskScore: 48,
    riskLevel: 'MEDIUM',
    status: 'WARNING',
    action: 'SHOW_WARNING',
    highlightText: 'Neural vocoder phase inconsistencies detected. Model flags potential AI synthetic generation.',
  },
  {
    step: 4,
    title: 'Synthetic Voice Detector Flags Clone (0.96)',
    subtitle: 'Confidence score crosses critical synthetic boundary: 0.96 synthetic probability.',
    activeComponent: 'synth',
    riskScore: 78,
    riskLevel: 'HIGH',
    status: 'WARNING',
    action: 'SHOW_WARNING',
    highlightText: 'AASIST identifies high-frequency vocoder synthesis artifacts characteristic of diffusion/autoregressive models.',
  },
  {
    step: 5,
    title: 'ECAPA-TDNN Speaker Biometric Verification',
    subtitle: 'Voice embedding compared against enrolled voice profile VP-9021 (CEO Vikram Malhotra).',
    activeComponent: 'speaker',
    riskScore: 82,
    riskLevel: 'HIGH',
    status: 'WARNING',
    action: 'SHOW_WARNING',
    highlightText: 'Cosine similarity is 0.89 (High acoustic similarity to CEO, confirming targeted voice cloning attack).',
  },
  {
    step: 6,
    title: 'Prosody & Behavioral Dynamics Analysis',
    subtitle: 'Pitch variance analysis indicates abnormally flat pitch micro-tremor (14.2 Hz) and robotic cadence.',
    activeComponent: 'prosody',
    riskScore: 86,
    riskLevel: 'HIGH',
    status: 'WARNING',
    action: 'SHOW_WARNING',
    highlightText: 'Human micro-prosodic tremors are absent. Synthetic speech dynamics confirmed.',
  },
  {
    step: 7,
    title: 'Contextual Risk & Intent Analysis',
    subtitle: 'NLP transcript engine detects urgent wire transfer request of ₹10,00,000 ($150,000) to unverified account.',
    activeComponent: 'context',
    riskScore: 91,
    riskLevel: 'CRITICAL',
    status: 'PAUSED',
    action: 'PAUSE_SENSITIVE_ACTION',
    highlightText: 'Extreme urgency cues + executive authority pressure + high-value financial movement detected.',
  },
  {
    step: 8,
    title: 'Multi-Layer Fusion Engine Computes Final Threat',
    subtitle: 'Multi-layer fusion engine combines 4 orthogonal vectors into composite threat score: 94/100.',
    activeComponent: 'risk',
    riskScore: 94,
    riskLevel: 'CRITICAL',
    status: 'PAUSED',
    action: 'BLOCK_TRANSACTION',
    highlightText: 'Composite formula: 0.40(Synth: 96) + 0.35(Mismatch/Clone) + 0.15(Prosody) + 0.10(Context) = 94/100.',
  },
  {
    step: 9,
    title: 'Automated Prevention: Transaction HARD-BLOCKED',
    subtitle: 'Banking policy engine intercepts core payment switch before transaction settlement.',
    activeComponent: 'action',
    riskScore: 94,
    riskLevel: 'CRITICAL',
    status: 'BLOCKED',
    action: 'BLOCK_TRANSACTION',
    highlightText: 'POLICY ENFORCEMENT: Sensitive wire transfer ₹10,00,000 has been BLOCKED. Account flagged for security review.',
  },
  {
    step: 10,
    title: 'Out-of-Band Challenge & SOC Alert Dispatched',
    subtitle: 'Real-time out-of-band verification challenge sent to CEO registered device; SIEM incident logged.',
    activeComponent: 'prevention',
    riskScore: 94,
    riskLevel: 'CRITICAL',
    status: 'BLOCKED',
    action: 'BLOCK_TRANSACTION',
    highlightText: 'Attack successfully prevented in 118ms latency. Audit timeline updated with zero raw audio leakage.',
  },
];

export function VoiceShieldCommandCenter() {
  const [scenarios, setScenarios] = useState<PreventionScenarioInfo[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('scenario_1_ceo_wire_transfer_attack');

  // Interactive Live Call State
  const [callActive, setCallActive] = useState<boolean>(true);
  const [callPaused, setCallPaused] = useState<boolean>(false);
  const [callDuration, setCallDuration] = useState<number>(42);

  // Active Evaluation Values
  const [policyProfile, setPolicyProfile] = useState<PolicyProfileType>('BANKING');
  const [sensitiveAction, setSensitiveAction] = useState<SensitiveActionTypeValue>('FUND_TRANSFER');
  const [transactionAmount, setTransactionAmount] = useState<number | ''>(150000);
  const [riskScore, setRiskScore] = useState<number>(94);
  const [riskLevel, setRiskLevel] = useState<string>('CRITICAL');
  const [syntheticScore, setSyntheticScore] = useState<number>(0.96);
  const [speakerSimilarity, setSpeakerSimilarity] = useState<number>(0.89);
  const [contextRiskScore, setContextRiskScore] = useState<number>(92);
  const [callerTrust, setCallerTrust] = useState<string>('VIP_OR_EXECUTIVE');

  // Backend Workflow Response
  const [workflow, setWorkflow] = useState<PreventionEvaluationResponseInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Story Mode State
  const [storyModeActive, setStoryModeActive] = useState<boolean>(false);
  const [currentStoryStep, setCurrentStoryStep] = useState<number>(0);
  const [storyPlaying, setStoryPlaying] = useState<boolean>(false);
  const storyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Indian Multilingual Selector
  const [selectedLang, setSelectedLang] = useState<string>('hi');

  // Live Timer Effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (callActive && !callPaused) {
      interval = setInterval(() => {
        setCallDuration((prev) => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [callActive, callPaused]);

  // Load scenarios on mount
  useEffect(() => {
    async function loadScenarios() {
      try {
        const list = await getPreventionScenarios();
        setScenarios(list);
        if (list.length > 0) {
          const ceoScenario = list.find((s) => s.id === 'scenario_1_ceo_wire_transfer_attack') || list[0];
          applyScenario(ceoScenario);
        }
      } catch (err: any) {
        console.warn('Failed loading demo scenarios:', err);
      }
    }
    loadScenarios();
  }, []);

  const applyScenario = (sc: PreventionScenarioInfo) => {
    setSelectedScenarioId(sc.id);
    setPolicyProfile(sc.policy_profile);
    setSensitiveAction(sc.sensitive_action);
    setTransactionAmount(sc.transaction_amount ?? '');
    setRiskScore(sc.risk_score);
    setRiskLevel(sc.risk_level);
    setSyntheticScore(sc.synthetic_score);
    setSpeakerSimilarity(sc.speaker_similarity);
    setContextRiskScore(sc.context_risk_score);
    setCallerTrust(sc.caller_trust);
    setCallActive(true);
    setCallPaused(false);
    setActionError(null);

    // Trigger backend evaluation
    triggerEvaluation({
      risk_score: sc.risk_score,
      risk_level: sc.risk_level,
      synthetic_score: sc.synthetic_score,
      speaker_similarity: sc.speaker_similarity,
      context_risk_score: sc.context_risk_score,
      sensitive_action_type: sc.sensitive_action,
      transaction_amount: sc.transaction_amount ?? null,
      policy_profile: sc.policy_profile,
      caller_trust: sc.caller_trust,
    });
  };

  const triggerEvaluation = async (payloadOverride?: any) => {
    setIsLoading(true);
    setActionError(null);
    try {
      const payload = payloadOverride || {
        risk_score: riskScore,
        risk_level: riskLevel,
        synthetic_score: syntheticScore,
        speaker_similarity: speakerSimilarity,
        context_risk_score: contextRiskScore,
        sensitive_action_type: sensitiveAction,
        transaction_amount: transactionAmount === '' ? null : Number(transactionAmount),
        policy_profile: policyProfile,
        caller_trust: callerTrust,
      };
      const result = await evaluatePrevention(payload);
      setWorkflow(result);
    } catch (err: any) {
      setActionError(err.message || 'Prevention evaluation failed.');
    } finally {
      setIsLoading(false);
    }
  };

  // Verification simulations (MFA, Callback, Supervisor)
  const handleSimulateVerification = async (verificationType: string) => {
    if (!workflow) return;
    setActionLoading(verificationType);
    setActionError(null);
    try {
      const updated = await verifyPreventionWorkflow(workflow.workflow_id, {
        verification_type: verificationType,
        actor: 'security_operator_01',
        notes: `Simulated hackathon response: ${verificationType}`,
      });
      setWorkflow(updated);
    } catch (err: any) {
      setActionError(err.message || 'Verification challenge failed.');
    } finally {
      setActionLoading(null);
    }
  };

  const handleResolveIncident = async (action: 'ALLOW' | 'BLOCK_TRANSACTION') => {
    if (!workflow) return;
    setActionLoading('resolve');
    setActionError(null);
    try {
      const updated = await resolvePreventionWorkflow(workflow.workflow_id, {
        resolution_reason: `Security Operator manual confirmation: ${action}`,
        resolved_by: 'soc_commander_01',
        final_action: action,
      });
      setWorkflow(updated);
    } catch (err: any) {
      setActionError(err.message || 'Failed to resolve workflow.');
    } finally {
      setActionLoading(null);
    }
  };

  // 10-Step Story Mode Controller
  const startStoryMode = () => {
    setStoryModeActive(true);
    setCurrentStoryStep(0);
    setStoryPlaying(true);
    applyStoryStep(0);
  };

  const applyStoryStep = (stepIdx: number) => {
    const step = STORY_STEPS[stepIdx];
    if (!step) return;

    setRiskScore(step.riskScore);
    setRiskLevel(step.riskLevel);
    if (step.step >= 4) {
      setSyntheticScore(0.96);
    } else if (step.step === 3) {
      setSyntheticScore(0.65);
    } else {
      setSyntheticScore(0.08);
    }

    if (step.step >= 5) {
      setSpeakerSimilarity(0.89);
    } else {
      setSpeakerSimilarity(0.40);
    }

    if (step.step >= 7) {
      setContextRiskScore(92);
      setSensitiveAction('FUND_TRANSFER');
      setTransactionAmount(150000);
    }

    triggerEvaluation({
      risk_score: step.riskScore,
      risk_level: step.riskLevel,
      synthetic_score: step.step >= 4 ? 0.96 : 0.1,
      speaker_similarity: step.step >= 5 ? 0.89 : 0.4,
      context_risk_score: step.step >= 7 ? 92 : 20,
      sensitive_action_type: 'FUND_TRANSFER',
      transaction_amount: 150000,
      policy_profile: 'BANKING',
      caller_trust: 'VIP_OR_EXECUTIVE',
    });
  };

  const nextStoryStep = () => {
    if (currentStoryStep < STORY_STEPS.length - 1) {
      const nextIdx = currentStoryStep + 1;
      setCurrentStoryStep(nextIdx);
      applyStoryStep(nextIdx);
    } else {
      setStoryPlaying(false);
    }
  };

  const prevStoryStep = () => {
    if (currentStoryStep > 0) {
      const prevIdx = currentStoryStep - 1;
      setCurrentStoryStep(prevIdx);
      applyStoryStep(prevIdx);
    }
  };

  const toggleStoryPlay = () => {
    setStoryPlaying((prev) => !prev);
  };

  useEffect(() => {
    if (storyModeActive && storyPlaying) {
      storyTimerRef.current = setTimeout(() => {
        if (currentStoryStep < STORY_STEPS.length - 1) {
          nextStoryStep();
        } else {
          setStoryPlaying(false);
        }
      }, 2400);
    }
    return () => {
      if (storyTimerRef.current) clearTimeout(storyTimerRef.current);
    };
  }, [storyModeActive, storyPlaying, currentStoryStep]);

  const closeStoryMode = () => {
    setStoryModeActive(false);
    setStoryPlaying(false);
    if (storyTimerRef.current) clearTimeout(storyTimerRef.current);
  };

  // Helper formatting for timer
  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Component score contributions
  const synthContrib = Math.round(safeNumber(syntheticScore) * 100 * 0.4);
  const speakerContrib = Math.round((1.0 - Math.min(1.0, Math.max(0.0, safeNumber(speakerSimilarity)))) * 100 * 0.35);
  const prosodyContrib = Math.round(75 * 0.15); // normalized anomaly
  const contextContrib = Math.round(safeNumber(contextRiskScore) * 0.1);

  // Status visual mapping
  const effectiveStatus: PreventionStatusType = workflow?.prevention_status || (riskScore >= 75 ? 'BLOCKED' : riskScore >= 50 ? 'PAUSED' : 'ALLOWED');

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* ------------------------------------------------------------- */}
      {/* SECTION 1: HACKATHON HERO & QUICK DEMO TRIGGER                */}
      {/* ------------------------------------------------------------- */}
      <div className="relative overflow-hidden rounded-3xl border border-rose-500/30 bg-gradient-to-br from-slate-900 via-slate-950 to-rose-950/40 p-6 sm:p-8 shadow-2xl">
        <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-rose-500/10 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -left-24 w-96 h-96 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          <div className="space-y-3 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-bold uppercase tracking-wider shadow-sm">
                <ShieldAlert className="w-3.5 h-3.5 text-rose-400 animate-pulse" />
                Hackathon Flagship Showcase
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-800/80 border border-slate-700 text-slate-300 text-xs font-mono">
                Real-Time Latency: ~118ms
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-semibold">
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                Zero Raw Audio Retained
              </span>
            </div>

            <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
              Real-Time Voice Cloning Impersonation Defense
            </h1>

            <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
              Detects AI-cloned voices in live audio streams and halts unauthorized financial transfers,
              credential resets, and executive spear-phishing attacks before transactions commit.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full lg:w-auto">
            <button
              onClick={startStoryMode}
              className="inline-flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-2xl bg-gradient-to-r from-rose-600 via-rose-500 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white font-bold text-sm shadow-xl shadow-rose-600/30 ring-2 ring-rose-400/40 transition transform active:scale-95"
            >
              <Sparkles className="w-4 h-4 text-amber-200 animate-spin" />
              <span>START FULL DEMO (10-STEP STORY)</span>
            </button>
          </div>
        </div>

        {/* Story Mode Interactive Banner */}
        {storyModeActive && (
          <div className="mt-6 pt-6 border-t border-slate-800/80 rounded-2xl bg-slate-900/90 p-5 backdrop-blur-md ring-1 ring-rose-500/40 animate-in slide-in-from-top-4 duration-300">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-3">
              <div className="flex items-center gap-3">
                <span className="px-2.5 py-1 rounded-lg bg-rose-500 text-white font-mono font-bold text-xs">
                  STEP {currentStoryStep + 1} OF {STORY_STEPS.length}
                </span>
                <h3 className="text-base font-bold text-white">
                  {STORY_STEPS[currentStoryStep]?.title}
                </h3>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={prevStoryStep}
                  disabled={currentStoryStep === 0}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-semibold hover:bg-slate-700 disabled:opacity-40 transition"
                >
                  Previous
                </button>
                <button
                  onClick={toggleStoryPlay}
                  className="px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs font-semibold hover:bg-rose-500 transition"
                >
                  {storyPlaying ? 'Pause Story' : 'Auto Play'}
                </button>
                <button
                  onClick={nextStoryStep}
                  disabled={currentStoryStep === STORY_STEPS.length - 1}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-semibold hover:bg-slate-700 disabled:opacity-40 transition"
                >
                  Next
                </button>
                <button
                  onClick={closeStoryMode}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 text-xs font-semibold transition"
                >
                  Exit
                </button>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-3">
              <div
                className="bg-gradient-to-r from-rose-500 to-amber-400 h-full transition-all duration-500"
                style={{ width: `${((currentStoryStep + 1) / STORY_STEPS.length) * 100}%` }}
              />
            </div>

            <p className="text-xs text-slate-300 font-sans">
              <span className="text-rose-400 font-semibold">Narrative Insight:</span>{' '}
              {STORY_STEPS[currentStoryStep]?.highlightText}
            </p>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 2: 6 HACKATHON DEMO SCENARIOS SELECTOR                */}
      {/* ------------------------------------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-rose-400" />
            <h2 className="text-base font-bold text-white tracking-wide">
              Hackathon Attack &amp; Defense Scenarios
            </h2>
          </div>
          <span className="text-xs text-slate-400">Select any scenario to evaluate instantly</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {scenarios.map((sc, idx) => {
            const isSelected = selectedScenarioId === sc.id;
            const isCritical = sc.risk_level === 'CRITICAL';
            const isHigh = sc.risk_level === 'HIGH';
            const isMedium = sc.risk_level === 'MEDIUM';

            return (
              <button
                key={sc.id}
                onClick={() => applyScenario(sc)}
                className={`text-left p-4 rounded-2xl border transition-all duration-200 flex flex-col justify-between relative overflow-hidden ${
                  isSelected
                    ? 'border-rose-500/80 bg-rose-950/20 shadow-lg shadow-rose-950/40 ring-1 ring-rose-500/50'
                    : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900'
                }`}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-mono font-bold text-slate-400">
                      SCENARIO 0{idx + 1}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        isCritical
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          : isHigh
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                          : isMedium
                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}
                    >
                      {sc.expected_status}
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-white line-clamp-1">{sc.name.replace(/^Scenario \d+:\s*/, '')}</h4>

                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                    {sc.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px]">
                  <span className="text-slate-400 font-mono">
                    Risk: <strong className="text-white">{sc.risk_score}/100</strong>
                  </span>
                  <span className="font-semibold text-rose-300 flex items-center gap-1">
                    {sc.expected_action.replace(/_/g, ' ')}
                    <ChevronRight className="w-3 h-3" />
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 3: TWO-COLUMN MAIN COMMAND CENTER COCKPIT             */}
      {/* ------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: LIVE CALL + 5-LAYER SIGNALS (7 COLS) */}
        <div className="lg:col-span-7 space-y-6">
          {/* 3.1 LIVE / SIMULATED INCOMING CALL PANEL */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl backdrop-blur-md space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="relative">
                  <div className={`w-3 h-3 rounded-full ${callActive && !callPaused ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
                  <div className={`w-3 h-3 rounded-full absolute top-0 left-0 ${callActive && !callPaused ? 'bg-emerald-500' : 'bg-slate-500'}`} />
                </div>
                <h3 className="text-sm font-bold text-white tracking-wide">Live Intercepted Voice Call</h3>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                  {callActive ? (callPaused ? 'PAUSED' : 'STREAMING') : 'DISCONNECTED'}
                </span>
              </div>

              {/* Call Controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCallPaused((p) => !p)}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition inline-flex items-center gap-1"
                >
                  {callPaused ? <PlayCircle className="w-3 h-3 text-emerald-400" /> : <PauseCircle className="w-3 h-3 text-amber-400" />}
                  <span>{callPaused ? 'Resume' : 'Pause'}</span>
                </button>
                <button
                  onClick={() => {
                    setCallActive((a) => !a);
                    if (!callActive) setCallDuration(0);
                  }}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition inline-flex items-center gap-1 ${
                    callActive
                      ? 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30'
                      : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'
                  }`}
                >
                  {callActive ? <PhoneOff className="w-3 h-3 text-rose-400" /> : <PhoneCall className="w-3 h-3 text-emerald-400" />}
                  <span>{callActive ? 'Terminate' : 'Dial In'}</span>
                </button>
              </div>
            </div>

            {/* Caller Identity Card */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
              <div className="space-y-1">
                <div className="text-[11px] text-slate-400">Caller Identity Claimed</div>
                <div className="text-sm font-bold text-white flex items-center gap-1.5">
                  <UserCheck className="w-4 h-4 text-indigo-400" />
                  <span>Vikram Malhotra (CEO)</span>
                </div>
                <div className="text-xs font-mono text-slate-400">+91 98201 44821 (Mumbai, India)</div>
              </div>

              <div className="space-y-1">
                <div className="text-[11px] text-slate-400">Enrolled Voice Profile</div>
                <div className="text-sm font-bold text-emerald-400 flex items-center gap-1.5">
                  <Fingerprint className="w-4 h-4 text-emerald-400" />
                  <span>Profile VP-9021 (ECAPA Verified)</span>
                </div>
                <div className="text-xs font-mono text-slate-400">SIP / Opus 48kHz • TLS/SRTP Intercept</div>
              </div>

              <div className="space-y-1 pt-2 border-t border-slate-800/60 sm:col-span-2 flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  Call Duration: <strong className="text-white font-mono">{formatTimer(callDuration)}</strong>
                </span>
                <span className="text-slate-400 flex items-center gap-1">
                  <Activity className="w-3.5 h-3.5 text-rose-400" />
                  Inspection Mode: <strong className="text-rose-300">Continuous 500ms Sliding Window</strong>
                </span>
              </div>
            </div>
          </div>

          {/* 3.2 5-LAYER VOICE SECURITY SIGNALS */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span>5-Layer Voice Security Signals</span>
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">Orthogonal Inspection Layers</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Layer 1: Synthetic Voice Detection */}
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-rose-400" />
                    Layer 1: AASIST Synthetic
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      safeNumber(syntheticScore) >= 0.7
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {safeNumber(syntheticScore) >= 0.7 ? 'CLONED VOICE' : 'NATURAL SPEECH'}
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-extrabold text-white font-mono">
                    {formatNumber(safeNumber(syntheticScore) * 100, 1)}%
                  </span>
                  <span className="text-[11px] text-slate-400">Vocoder Phase Artifacts</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${safeNumber(syntheticScore) >= 0.7 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.min(100, safeNumber(syntheticScore) * 100)}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400">
                  Pretrained AASIST Graph Neural Network • High-frequency subband inspection
                </div>
              </div>

              {/* Layer 2: Speaker Biometric Verification */}
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <Fingerprint className="w-3.5 h-3.5 text-indigo-400" />
                    Layer 2: ECAPA-TDNN Biometric
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      safeNumber(speakerSimilarity) >= 0.65
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    }`}
                  >
                    {safeNumber(speakerSimilarity) >= 0.65 ? 'TARGET MATCH' : 'MISMATCH'}
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-extrabold text-white font-mono">
                    {formatNumber(speakerSimilarity, 2)}
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">Threshold: 0.65</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500"
                    style={{ width: `${Math.min(100, Math.max(0, safeNumber(speakerSimilarity) * 100))}%` }}
                  />
                </div>
                <div className="text-[10px] text-slate-400">
                  SpeechBrain ECAPA-TDNN 192-dim biometric cosine similarity vs profile
                </div>
              </div>

              {/* Layer 3: Acoustic & Spectral Integrity */}
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-teal-400" />
                    Layer 3: Acoustic Integrity
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/40">
                    28.5 dB SNR
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                  <div>
                    <span className="text-[10px] text-slate-400">Spectral Centroid:</span>
                    <div className="font-mono font-bold text-white">2,840 Hz</div>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400">Pitch Stability:</span>
                    <div className="font-mono font-bold text-white">98.4%</div>
                  </div>
                </div>
                <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/60">
                  Channel artifacts consistent with VoIP transmission; no microphone distortion
                </div>
              </div>

              {/* Layer 4: Prosody & Behavioral Dynamics */}
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <Mic className="w-3.5 h-3.5 text-purple-400" />
                    Layer 4: Prosody Dynamics
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      safeNumber(syntheticScore) >= 0.7
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {safeNumber(syntheticScore) >= 0.7 ? 'UNNATURAL CADENCE' : 'NATURAL TREMOR'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                  <div>
                    <span className="text-[10px] text-slate-400">Pitch Variance:</span>
                    <div className="font-mono font-bold text-white">14.2 Hz (Flat)</div>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400">Speaking Rate:</span>
                    <div className="font-mono font-bold text-white">4.8 syl/sec</div>
                  </div>
                </div>
                <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/60">
                  Absence of micro-prosodic vocal fold jitter indicates synthetic generation
                </div>
              </div>

              {/* Layer 5: Contextual Risk Intelligence (Full Width) */}
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/80 space-y-2 sm:col-span-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                    Layer 5: Contextual Intent &amp; Social Engineering Intelligence
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      safeNumber(contextRiskScore) >= 70
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {safeNumber(contextRiskScore) >= 70 ? 'CRITICAL INTENT RISK' : 'NORMAL INQUIRY'}
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-1">
                  <div>
                    <span className="text-[10px] text-slate-400">Urgency Score:</span>
                    <div className="font-mono font-bold text-rose-300">
                      {safeNumber(contextRiskScore) >= 70 ? '92/100 (EXTREME)' : '18/100 (LOW)'}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400">Emotional Pressure:</span>
                    <div className="font-mono font-bold text-rose-300">
                      {safeNumber(contextRiskScore) >= 70 ? 'HIGH (Authority)' : 'NONE'}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400">Financial Movement:</span>
                    <div className="font-mono font-bold text-amber-300">
                      {transactionAmount ? formatCurrency(transactionAmount) : 'None'}
                    </div>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400">Credential Inquiry:</span>
                    <div className="font-mono font-bold text-slate-300">
                      {sensitiveAction === 'PRIVILEGED_ACCESS' || sensitiveAction === 'ACCOUNT_CHANGE' ? 'DETECTED' : 'NONE'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: RISK ENGINE + ACTION PROTECTION + RESPONSE CONTROLS (5 COLS) */}
        <div className="lg:col-span-5 space-y-6">
          {/* 3.3 DYNAMIC IMPERSONATION RISK ENGINE GAUGE */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl backdrop-blur-md space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Shield className="w-4 h-4 text-rose-400" />
                <span>Impersonation Risk Engine</span>
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40">
                {riskLevel} THREAT
              </span>
            </div>

            {/* Gauge and Threat Classification */}
            <div className="flex items-center gap-5 p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
              <div className="relative flex items-center justify-center">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle
                    cx="48"
                    cy="48"
                    r="38"
                    stroke="currentColor"
                    strokeWidth="8"
                    className="text-slate-800"
                    fill="transparent"
                  />
                  <circle
                    cx="48"
                    cy="48"
                    r="38"
                    stroke="currentColor"
                    strokeWidth="8"
                    strokeDasharray={2 * Math.PI * 38}
                    strokeDashoffset={2 * Math.PI * 38 * (1 - Math.min(100, Math.max(0, riskScore)) / 100)}
                    strokeLinecap="round"
                    className={`${
                      riskScore >= 75
                        ? 'text-rose-500'
                        : riskScore >= 50
                        ? 'text-amber-500'
                        : riskScore >= 25
                        ? 'text-blue-500'
                        : 'text-emerald-500'
                    } transition-all duration-700`}
                    fill="transparent"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-2xl font-black text-white font-mono">{riskScore}</span>
                  <span className="text-[9px] text-slate-400 uppercase tracking-widest font-bold">/ 100</span>
                </div>
              </div>

              <div className="space-y-1.5 flex-1">
                <div className="text-[11px] text-slate-400">Real-Time Threat Classification</div>
                <div
                  className={`text-sm font-extrabold tracking-wide ${
                    riskScore >= 75
                      ? 'text-rose-400'
                      : riskScore >= 50
                      ? 'text-amber-400'
                      : riskScore >= 25
                      ? 'text-blue-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {riskScore >= 75
                    ? 'CONFIRMED IMPERSONATION ATTACK'
                    : riskScore >= 50
                    ? 'PROBABLE VOICE CLONE'
                    : riskScore >= 25
                    ? 'POSSIBLE SPOOF / UNVERIFIED'
                    : 'LEGITIMATE CALLER'}
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  {riskScore >= 75
                    ? 'Synthetic acoustics matched against executive profile during sensitive instruction.'
                    : 'Acoustic metrics and caller parameters within normal authorized tolerances.'}
                </p>
              </div>
            </div>

            {/* Component Breakdown */}
            <div className="space-y-2 pt-2 text-xs">
              <div className="flex justify-between text-slate-400 text-[11px]">
                <span>Component Score Contribution Breakdown</span>
                <span className="font-mono text-slate-300 font-bold">Weights</span>
              </div>
              <div className="space-y-1.5 font-mono text-[11px]">
                <div className="flex justify-between items-center">
                  <span className="text-slate-300">1. Synthetic Detector (40%)</span>
                  <span className="text-rose-400 font-bold">+{synthContrib} pts</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-300">2. Speaker Mismatch (35%)</span>
                  <span className="text-indigo-400 font-bold">+{speakerContrib} pts</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-300">3. Prosody Anomaly (15%)</span>
                  <span className="text-purple-400 font-bold">+{prosodyContrib} pts</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-300">4. Contextual Risk (10%)</span>
                  <span className="text-amber-400 font-bold">+{contextContrib} pts</span>
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[10px] text-slate-400 leading-tight">
                <span className="font-semibold text-slate-300">Heuristic Fusion:</span>{' '}
                Risk = 0.40(Synthetic) + 0.35(Speaker Mismatch) + 0.15(Prosody) + 0.10(Context).
                Empirically calibrated to prevent threshold bypass.
              </div>
            </div>
          </div>

          {/* 3.4 SENSITIVE ACTION PROTECTION PANEL */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl backdrop-blur-md space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Lock className="w-4 h-4 text-amber-400" />
                <span>Sensitive Action Protection</span>
              </h3>
              <span
                className={`text-xs font-black px-2.5 py-1 rounded-lg tracking-wider ${
                  effectiveStatus === 'BLOCKED'
                    ? 'bg-rose-600 text-white shadow-md shadow-rose-600/40 animate-pulse'
                    : effectiveStatus === 'PAUSED'
                    ? 'bg-amber-500 text-slate-950 font-bold'
                    : effectiveStatus === 'VERIFICATION_REQUIRED'
                    ? 'bg-blue-600 text-white'
                    : 'bg-emerald-600 text-white'
                }`}
              >
                {effectiveStatus === 'BLOCKED'
                  ? 'TRANSACTION BLOCKED'
                  : effectiveStatus === 'PAUSED'
                  ? 'TRANSACTION PAUSED'
                  : effectiveStatus === 'VERIFICATION_REQUIRED'
                  ? 'VERIFICATION REQUIRED'
                  : 'PERMITTED'}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Target Sensitive Action:</span>
                <strong className="text-white font-mono">{sensitiveAction.replace(/_/g, ' ')}</strong>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Transaction Amount:</span>
                <strong className="text-amber-300 font-mono text-sm">
                  {transactionAmount ? formatCurrency(transactionAmount) : 'N/A'}
                </strong>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Beneficiary:</span>
                <span className="font-mono text-slate-300">Offshore Acct ****8921 (Unverified)</span>
              </div>
              <div className="pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                <strong className="text-rose-400">Policy Rule Triggered:</strong>{' '}
                Banking Profile Rule: Critical Risk (&gt;70) + Wire Transfer = Immediate Hard Block &amp; Payment Switch Intercept.
              </div>
            </div>
          </div>

          {/* 3.5 AUTOMATED PREVENTION & RESPONSE CONTROLS (INTERACTIVE) */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-xl backdrop-blur-md space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Zap className="w-4 h-4 text-emerald-400" />
                <span>Interactive Incident Response</span>
              </h3>
              <span className="text-[10px] font-mono text-slate-400">Live API Endpoints</span>
            </div>

            {actionError && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{actionError}</span>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {/* Trigger MFA */}
              <button
                onClick={() => handleSimulateVerification('mfa_passed')}
                disabled={actionLoading !== null || !workflow || effectiveStatus === 'BLOCKED'}
                className="p-3 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700/80 hover:border-slate-600 text-xs font-bold transition flex flex-col items-start gap-1 disabled:opacity-40"
              >
                <div className="flex items-center gap-1.5 text-indigo-300">
                  <KeyRound className="w-3.5 h-3.5" />
                  <span>Simulate Step-Up MFA</span>
                </div>
                <span className="text-[10px] text-slate-400 font-normal">Push WebAuthn challenge</span>
              </button>

              {/* Trigger Callback */}
              <button
                onClick={() => handleSimulateVerification('callback_passed')}
                disabled={actionLoading !== null || !workflow || effectiveStatus === 'BLOCKED'}
                className="p-3 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 border border-slate-700/80 hover:border-slate-600 text-xs font-bold transition flex flex-col items-start gap-1 disabled:opacity-40"
              >
                <div className="flex items-center gap-1.5 text-emerald-300">
                  <PhoneCall className="w-3.5 h-3.5" />
                  <span>Out-of-Band Callback</span>
                </div>
                <span className="text-[10px] text-slate-400 font-normal">Call registered phone</span>
              </button>

              {/* Block & Terminate */}
              <button
                onClick={() => handleResolveIncident('BLOCK_TRANSACTION')}
                disabled={actionLoading !== null || !workflow || effectiveStatus === 'BLOCKED'}
                className="p-3 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/40 text-xs font-bold transition flex flex-col items-start gap-1 disabled:opacity-40"
              >
                <div className="flex items-center gap-1.5 text-rose-300">
                  <UserX className="w-3.5 h-3.5" />
                  <span>Hard Block &amp; Lock</span>
                </div>
                <span className="text-[10px] text-rose-400/80 font-normal">Halt session permanently</span>
              </button>

              {/* Supervisor Override */}
              <button
                onClick={() => handleSimulateVerification('supervisor_approved')}
                disabled={actionLoading !== null || !workflow}
                className="p-3 rounded-xl bg-amber-600/20 hover:bg-amber-600/30 text-amber-300 border border-amber-500/40 text-xs font-bold transition flex flex-col items-start gap-1 disabled:opacity-40"
              >
                <div className="flex items-center gap-1.5 text-amber-300">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>Supervisor Override</span>
                </div>
                <span className="text-[10px] text-amber-400/80 font-normal">Authorized exception protocol</span>
              </button>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span>State Machine Guard:</span>
              <strong className="text-slate-300 font-mono">
                BLOCKED state cannot be cleared without supervisor sign-off
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 4: MULTILINGUAL & INDIAN CONTEXT SECTION              */}
      {/* ------------------------------------------------------------- */}
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl backdrop-blur-md space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Globe className="w-4 h-4 text-indigo-400" />
              <h3 className="text-base font-bold text-white tracking-wide">
                Multilingual &amp; Indian Regional Context (UPI &amp; Banking Impersonation)
              </h3>
            </div>
            <p className="text-xs text-slate-400">
              Voice cloning attacks in India frequently leverage regional languages and urgent pretexts (KYC expiry, emergency transfers, OTP secrecy).
            </p>
          </div>

          {/* Language Selector Buttons */}
          <div className="flex flex-wrap items-center gap-1.5">
            {Object.entries(MULTILINGUAL_PHRASES).map(([code, item]) => (
              <button
                key={code}
                onClick={() => setSelectedLang(code)}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                  selectedLang === code
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
                }`}
              >
                {item.nativeName} ({item.language})
              </button>
            ))}
          </div>
        </div>

        {/* Active Language Card */}
        {MULTILINGUAL_PHRASES[selectedLang] && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 p-4 rounded-2xl bg-slate-950/70 border border-slate-800">
            <div className="md:col-span-8 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold text-indigo-400">
                  {MULTILINGUAL_PHRASES[selectedLang].context}
                </span>
                <span className="text-xs text-slate-500">•</span>
                <span className="text-xs font-semibold text-rose-400">
                  Trigger: {MULTILINGUAL_PHRASES[selectedLang].riskTrigger}
                </span>
              </div>
              <p className="text-base font-medium text-white italic">
                "{MULTILINGUAL_PHRASES[selectedLang].phrase}"
              </p>
              <p className="text-xs text-slate-400">
                <strong className="text-slate-300">English Translation:</strong>{' '}
                {MULTILINGUAL_PHRASES[selectedLang].translation}
              </p>
            </div>

            <div className="md:col-span-4 p-3 rounded-xl bg-slate-900/90 border border-slate-800/80 flex flex-col justify-between text-xs space-y-2">
              <div className="text-[11px] font-bold text-slate-300">Acoustic Signal Compatibility</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                AASIST and ECAPA-TDNN feature extractors operate on language-agnostic physical acoustic properties
                (phase distortion, vocoder harmonics, vocal tract resonance).
              </p>
              <div className="text-[10px] text-amber-300/80 font-mono">
                Scientific Disclosure: Performance validation requires representative Indian dialect and accent datasets.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 5: REAL-TIME INCIDENT TIMELINE & AUDIT TRAIL          */}
      {/* ------------------------------------------------------------- */}
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl backdrop-blur-md space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white tracking-wide flex items-center gap-2">
              <Clock className="w-4 h-4 text-rose-400" />
              <span>Real-Time Incident Timeline &amp; Privacy-Preserving Audit Trail</span>
            </h3>
            <p className="text-xs text-slate-400">
              Chronological decision log with zero raw audio or biometric embeddings.
            </p>
          </div>

          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>DPDP Act &amp; GDPR Privacy Compliant</span>
          </div>
        </div>

        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {workflow && workflow.timeline && workflow.timeline.length > 0 ? (
            workflow.timeline.map((event, idx) => {
              const isBlocked = event.prevention_status === 'BLOCKED';
              const isPaused = event.prevention_status === 'PAUSED';
              const isResolved = event.prevention_status === 'RESOLVED';
              const isAllowed = event.prevention_status === 'ALLOWED';

              return (
                <div
                  key={event.event_id || idx}
                  className={`p-3.5 rounded-xl border transition flex items-start gap-3 text-xs ${
                    isBlocked
                      ? 'border-rose-500/50 bg-rose-950/20 text-rose-200'
                      : isPaused
                      ? 'border-amber-500/50 bg-amber-950/20 text-amber-200'
                      : isResolved
                      ? 'border-emerald-500/50 bg-emerald-950/20 text-emerald-200'
                      : 'border-slate-800 bg-slate-950/60 text-slate-300'
                  }`}
                >
                  <div className="mt-0.5 shrink-0">
                    {isBlocked ? (
                      <XCircle className="w-4 h-4 text-rose-400" />
                    ) : isPaused ? (
                      <PauseCircle className="w-4 h-4 text-amber-400" />
                    ) : isResolved || isAllowed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Activity className="w-4 h-4 text-indigo-400" />
                    )}
                  </div>

                  <div className="flex-1 space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono font-bold tracking-wide text-white">
                        {event.event_type}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 leading-relaxed">{event.details}</p>

                    <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono pt-1">
                      <span>Status: <strong className="text-white">{event.prevention_status}</strong></span>
                      <span>Action: <strong className="text-white">{event.prevention_action}</strong></span>
                      {event.risk_score !== null && (
                        <span>Risk: <strong className="text-rose-400">{event.risk_score}/100</strong></span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-center py-8 text-slate-500 text-xs">
              Timeline will populate as events occur during the call session.
            </div>
          )}
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 6: INTEGRATION ARCHITECTURE SHOWCASE                  */}
      {/* ------------------------------------------------------------- */}
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl backdrop-blur-md space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-white tracking-wide flex items-center gap-2">
              <Server className="w-4 h-4 text-indigo-400" />
              <span>Enterprise &amp; Banking Deployment Touchpoints</span>
            </h3>
            <p className="text-xs text-slate-400">
              VoiceShield deploys as a transparent streaming proxy or API sidecar across 4 major enterprise infrastructures.
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300">
            Architectural Blueprint
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Touchpoint 1: Core Banking */}
          <div className="p-4 rounded-2xl border border-slate-800 bg-slate-950/60 space-y-2">
            <div className="flex items-center gap-2 text-rose-400 font-bold text-xs">
              <Building className="w-4 h-4" />
              <span>Core Banking &amp; UPI</span>
            </div>
            <h4 className="text-xs font-bold text-white">Payment Switch Intercept</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Integrates with ISO 8583 / UPI switch before debit settlement. Rejection code 91 (Fraud Risk) emitted on critical clone detection.
            </p>
            <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-800">
              Hook: Pre-Debit Auth Intercept
            </div>
          </div>

          {/* Touchpoint 2: Telecom SIP */}
          <div className="p-4 rounded-2xl border border-slate-800 bg-slate-950/60 space-y-2">
            <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs">
              <PhoneCall className="w-4 h-4" />
              <span>Telecom &amp; Call Center</span>
            </div>
            <h4 className="text-xs font-bold text-white">SIP Trunk Gateway Sidecar</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Transparent media proxy (Asterisk / FreeSWITCH / Genesys) forks RTP Opus packets via WebSocket without adding call delay.
            </p>
            <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-800">
              Protocol: RFC 3550 RTP Forking
            </div>
          </div>

          {/* Touchpoint 3: Enterprise Teams */}
          <div className="p-4 rounded-2xl border border-slate-800 bg-slate-950/60 space-y-2">
            <div className="flex items-center gap-2 text-purple-400 font-bold text-xs">
              <Radio className="w-4 h-4" />
              <span>Enterprise Collaboration</span>
            </div>
            <h4 className="text-xs font-bold text-white">Teams &amp; Zoom Bot Intercept</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Microsoft Graph / Zoom Real-time Audio API bot listens to executive calls and alerts security channels on impersonation attempts.
            </p>
            <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-800">
              Target: Executive Video/Voice
            </div>
          </div>

          {/* Touchpoint 4: SOC SIEM */}
          <div className="p-4 rounded-2xl border border-slate-800 bg-slate-950/60 space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs">
              <Shield className="w-4 h-4" />
              <span>Security Operations SOC</span>
            </div>
            <h4 className="text-xs font-bold text-white">SIEM &amp; SOAR Dispatch</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              CEF / Syslog streaming to Splunk, Microsoft Sentinel, and Elastic. Automated playbooks lock compromised executive accounts.
            </p>
            <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-800">
              Format: Common Event Format (CEF)
            </div>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-400 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <span>
            <strong className="text-slate-200">Integration Notice:</strong> Touchpoints are presented as enterprise architectural blueprints and API integration interfaces.
          </span>
          <span className="font-mono text-[11px] text-indigo-400 shrink-0">
            REST API v1 • WebSocket RFC 6455
          </span>
        </div>
      </div>
    </div>
  );
}

export default VoiceShieldCommandCenter;
