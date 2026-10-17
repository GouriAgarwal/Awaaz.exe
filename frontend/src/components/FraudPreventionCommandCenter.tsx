import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  PhoneCall,
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
} from 'lucide-react';
import {
  PreventionEvaluationResponseInfo,
  PreventionScenarioInfo,
  PolicyProfileType,
  SensitiveActionTypeValue,
  PreventionStatusType,
  PreventionActionType,
} from '../types';
import {
  evaluatePrevention,
  verifyPreventionWorkflow,
  resolvePreventionWorkflow,
  getPreventionScenarios,
} from '../services/api';

export function FraudPreventionCommandCenter() {
  const [scenarios, setScenarios] = useState<PreventionScenarioInfo[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('scenario_2_voice_clone_fund_transfer');

  // Interactive Form Parameters
  const [policyProfile, setPolicyProfile] = useState<PolicyProfileType>('BANKING');
  const [sensitiveAction, setSensitiveAction] = useState<SensitiveActionTypeValue>('FUND_TRANSFER');
  const [transactionAmount, setTransactionAmount] = useState<number | ''>(150000);
  const [riskScore, setRiskScore] = useState<number>(92);
  const [riskLevel, setRiskLevel] = useState<string>('CRITICAL');
  const [syntheticScore, setSyntheticScore] = useState<number>(0.96);
  const [speakerSimilarity, setSpeakerSimilarity] = useState<number>(0.89);
  const [contextRiskScore, setContextRiskScore] = useState<number>(85);
  const [callerTrust, setCallerTrust] = useState<string>('VIP_OR_EXECUTIVE');

  // Workflow State & Loading
  const [workflow, setWorkflow] = useState<PreventionEvaluationResponseInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load predefined demo scenarios on mount
  useEffect(() => {
    async function loadScenarios() {
      try {
        const list = await getPreventionScenarios();
        setScenarios(list);
        if (list.length > 0) {
          // Select Scenario 2 (Voice Clone Fund Transfer Attack) by default as hackathon centerpiece
          const defaultScenario = list.find((s) => s.id === 'scenario_2_voice_clone_fund_transfer') || list[0];
          applyScenario(defaultScenario);
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

    // Auto-evaluate
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

  const triggerEvaluation = async (overridePayload?: any) => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = overridePayload || {
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
      setError(err.message || 'Prevention evaluation failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSimulateVerification = async (verificationType: string) => {
    if (!workflow) return;
    setActionLoading(verificationType);
    setError(null);
    try {
      const updated = await verifyPreventionWorkflow(workflow.workflow_id, {
        verification_type: verificationType,
        actor: 'security_operator',
        notes: `Simulated challenge: ${verificationType}`,
      });
      setWorkflow(updated);
    } catch (err: any) {
      setError(err.message || 'Verification challenge simulation failed.');
    } finally {
      setActionLoading(null);
    }
  };

  const handleResolve = async (finalAction: 'ALLOW' | 'BLOCK_TRANSACTION') => {
    if (!workflow) return;
    setActionLoading('resolve');
    setError(null);
    try {
      const resolved = await resolvePreventionWorkflow(workflow.workflow_id, {
        resolution_reason: `Manual supervisor intervention: marked as ${finalAction}.`,
        resolved_by: 'supervisor_badge_402',
        final_action: finalAction,
      });
      setWorkflow(resolved);
    } catch (err: any) {
      setError(err.message || 'Resolution failed.');
    } finally {
      setActionLoading(null);
    }
  };

  const getStatusBadge = (status?: PreventionStatusType) => {
    switch (status) {
      case 'BLOCKED':
        return {
          title: '⛔ TRANSACTION BLOCKED',
          color: 'bg-rose-500/20 text-rose-300 border-rose-500/50',
          desc: 'High-confidence impersonation attack. Action aborted to prevent capital loss.',
        };
      case 'PAUSED':
        return {
          title: '⏸ TRANSACTION PAUSED',
          color: 'bg-amber-500/20 text-amber-300 border-amber-500/50',
          desc: 'Action held in pending queue awaiting mandatory secondary verification.',
        };
      case 'VERIFICATION_REQUIRED':
        return {
          title: '⚠️ VERIFICATION REQUIRED',
          color: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50',
          desc: 'Secondary challenge required before action release.',
        };
      case 'ESCALATED':
        return {
          title: '🚨 ESCALATED TO SOC',
          color: 'bg-purple-500/20 text-purple-300 border-purple-500/50',
          desc: 'Incident escalated to Security Operations Center analysts.',
        };
      case 'RESOLVED':
        return {
          title: '✅ INCIDENT RESOLVED',
          color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50',
          desc: 'Verification challenge completed or supervisor exception granted.',
        };
      default:
        return {
          title: '✓ TRANSACTION PERMITTED',
          color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50',
          desc: 'Legitimate voice acoustic and identity indicators within normal parameters.',
        };
    }
  };

  const currentStatusInfo = getStatusBadge(workflow?.prevention_status);

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Flagship Header Banner */}
      <div className="rounded-2xl border border-rose-500/30 bg-gradient-to-r from-rose-950/40 via-slate-900/70 to-indigo-950/40 p-6 backdrop-blur-md shadow-xl">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold mb-2">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>Step 14 • Automated Prevention &amp; Incident Response</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Fraud Prevention Command Center
            </h1>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl">
              Real-time intervention workflow translating synthetic voice and speaker impersonation signals into
              automated containment decisions: pausing wire transfers, requiring out-of-band callbacks, and blocking attacks.
            </p>
          </div>

          <div className="flex items-center gap-2 self-stretch lg:self-auto">
            <button
              onClick={() => triggerEvaluation()}
              disabled={isLoading}
              className="w-full lg:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-indigo-600 hover:from-rose-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg shadow-rose-600/30 transition disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Evaluating Policy Engine...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Re-Evaluate Policy</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Demo Scenario Selector (Section 9.G) */}
        {scenarios.length > 0 && (
          <div className="mt-6 pt-5 border-t border-slate-800/80">
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Select Hackathon Attack Demonstration Scenario:</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
              {scenarios.map((sc) => (
                <button
                  key={sc.id}
                  onClick={() => applyScenario(sc)}
                  className={`text-left p-3 rounded-xl border transition-all ${
                    selectedScenarioId === sc.id
                      ? 'bg-rose-600/20 border-rose-500 text-white shadow-lg ring-1 ring-rose-500/50'
                      : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-850'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] font-bold truncate">{sc.name.split(':')[1] || sc.name}</span>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold ${
                        sc.expected_status === 'BLOCKED'
                          ? 'bg-rose-500/20 text-rose-300'
                          : sc.expected_status === 'PAUSED' || sc.expected_status === 'VERIFICATION_REQUIRED'
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-emerald-500/20 text-emerald-300'
                      }`}
                    >
                      {sc.expected_status}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 line-clamp-2 leading-tight">
                    {sc.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-rose-300 text-xs flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Grid: Call Security + Signals + Risk Display */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Live Call Security Panel (Section 9.A) (4 cols) */}
        <div className="lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Radio className="w-4 h-4 text-rose-400 animate-pulse" />
            <span>A. Live Call &amp; Security Policy</span>
          </h2>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-400 mb-1 font-medium">Policy Profile</label>
              <select
                value={policyProfile}
                onChange={(e) => setPolicyProfile(e.target.value as PolicyProfileType)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white font-semibold"
              >
                <option value="BANKING">BANKING (Strict Fund Transfer Defense)</option>
                <option value="ENTERPRISE">ENTERPRISE (Privileged Access / IT Takeover)</option>
                <option value="GOVERNMENT">GOVERNMENT (Zero-Trust Confidential Data)</option>
                <option value="TELECOM">TELECOM (SIM-Swap &amp; Porting Defense)</option>
                <option value="DEFAULT">DEFAULT (Balanced Corporate Baseline)</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 font-medium">Requested Sensitive Action</label>
              <select
                value={sensitiveAction}
                onChange={(e) => setSensitiveAction(e.target.value as SensitiveActionTypeValue)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white font-semibold"
              >
                <option value="FUND_TRANSFER">FUND_TRANSFER (Wire Transfer / ACH)</option>
                <option value="PAYMENT_APPROVAL">PAYMENT_APPROVAL (Vendor Payment)</option>
                <option value="ACCOUNT_CHANGE">ACCOUNT_CHANGE (Phone / Address / SIM)</option>
                <option value="PRIVILEGED_ACCESS">PRIVILEGED_ACCESS (Admin Console / IAM)</option>
                <option value="CONFIDENTIAL_DISCLOSURE">CONFIDENTIAL_DISCLOSURE (PII / Vault)</option>
                <option value="EXECUTIVE_INSTRUCTION">EXECUTIVE_INSTRUCTION (Urgent Order)</option>
                <option value="GENERAL_CALL">GENERAL_CALL (Standard Inquiry)</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 font-medium">Transaction Amount ($ USD)</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-500 font-mono">$</span>
                <input
                  type="number"
                  value={transactionAmount}
                  onChange={(e) =>
                    setTransactionAmount(e.target.value === '' ? '' : Math.max(0, parseFloat(e.target.value)))
                  }
                  placeholder="e.g. 150000"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-7 pr-3 py-2 font-mono text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 font-medium">Caller Trust Profile</label>
              <select
                value={callerTrust}
                onChange={(e) => setCallerTrust(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white"
              >
                <option value="VIP_OR_EXECUTIVE">VIP_OR_EXECUTIVE (CEO / High Target)</option>
                <option value="UNKNOWN_CALLER">UNKNOWN_CALLER (Unrecognized ID)</option>
                <option value="KNOWN_CONTACT">KNOWN_CONTACT (Existing Record)</option>
                <option value="VERIFIED_CONTACT">VERIFIED_CONTACT (Prior Channel Auth)</option>
              </select>
            </div>

            {workflow && (
              <div className="pt-2 border-t border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1">
                <div>Workflow ID: <span className="text-indigo-300">{workflow.workflow_id}</span></div>
                <div>Updated: {new Date(workflow.updated_at).toLocaleTimeString()}</div>
              </div>
            )}
          </div>
        </div>

        {/* Center Col: Voice Signals (Section 9.B) (4 cols) */}
        <div className="lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span>B. Live Voice Integrity Signals</span>
          </h2>

          <div className="space-y-3 text-xs">
            {/* Synthetic Score Slider */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <div className="flex justify-between items-center mb-1">
                <span className="text-slate-300 font-semibold">AASIST Synthetic Score</span>
                <span className="font-mono text-rose-300 font-bold">{(syntheticScore * 100).toFixed(1)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={syntheticScore}
                onChange={(e) => setSyntheticScore(parseFloat(e.target.value))}
                className="w-full accent-rose-500"
              />
              <div className="text-[10px] text-slate-500 flex justify-between">
                <span>0% Natural</span>
                <span className="font-mono text-slate-400">uncalibrated_model_score</span>
                <span>100% Synthetic</span>
              </div>
            </div>

            {/* Speaker Similarity Slider */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <div className="flex justify-between items-center mb-1">
                <span className="text-slate-300 font-semibold">ECAPA Cosine Similarity</span>
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
              <div className="text-[10px] text-slate-500 flex justify-between">
                <span>&lt; 0.50 Non-Match</span>
                <span className="font-mono text-slate-400">cosine_similarity</span>
                <span>&gt; 0.65 Match</span>
              </div>
            </div>

            {/* Context Risk Score */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-850">
              <div className="flex justify-between items-center mb-1">
                <span className="text-slate-300 font-semibold">Contextual Fraud Risk</span>
                <span className="font-mono text-purple-300 font-bold">{contextRiskScore} / 100</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={contextRiskScore}
                onChange={(e) => setContextRiskScore(parseInt(e.target.value, 10))}
                className="w-full accent-purple-500"
              />
              <div className="text-[10px] text-slate-500 flex justify-between">
                <span>Low Exposure</span>
                <span className="font-mono text-slate-400">heuristic_contextual_risk</span>
                <span>High Exposure</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Primary Risk Display (Section 9.C) (4 cols) */}
        <div className="lg:col-span-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 flex flex-col justify-between items-center text-center">
          <div className="w-full flex justify-between items-center text-xs text-slate-400 font-bold uppercase tracking-wider">
            <span>C. Impersonation Risk</span>
            <span className="font-mono text-[11px]">0 – 100</span>
          </div>

          <div className="my-4 relative flex items-center justify-center">
            <div
              className={`w-36 h-36 rounded-full border-4 flex flex-col items-center justify-center bg-slate-950/90 shadow-2xl ${
                riskScore >= 75
                  ? 'text-rose-500 border-rose-500'
                  : riskScore >= 50
                  ? 'text-amber-500 border-amber-500'
                  : riskScore >= 25
                  ? 'text-yellow-500 border-yellow-500'
                  : 'text-emerald-500 border-emerald-500'
              }`}
            >
              <span className="text-4xl font-black text-white tracking-tight">{riskScore}</span>
              <span className="text-[10px] font-bold uppercase tracking-wider mt-0.5">{riskLevel}</span>
            </div>
          </div>

          <div className="w-full bg-slate-950 p-3 rounded-xl border border-slate-850 text-left space-y-1 text-[11px]">
            <div className="flex justify-between">
              <span className="text-slate-400">Threat Status:</span>
              <span className={`font-bold ${riskScore >= 75 ? 'text-rose-400' : 'text-slate-300'}`}>
                {riskScore >= 75
                  ? 'POSSIBLE VOICE CLONING ATTACK'
                  : riskScore >= 50
                  ? 'ELEVATED IMPERSONATION RISK'
                  : 'NORMAL VOICE PATTERNS'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Score Provenance:</span>
              <span className="font-mono text-indigo-300">heuristic_fusion_score</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sensitive Action Protection Panel (Section 9.D) */}
      <div className={`rounded-2xl border p-6 backdrop-blur-md shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${currentStatusInfo.color}`}>
        <div className="flex items-start md:items-center gap-4">
          <div className="p-3 rounded-xl bg-slate-950/50 border border-white/10 shrink-0">
            {workflow?.prevention_status === 'BLOCKED' ? (
              <XCircle className="w-7 h-7 text-rose-400" />
            ) : workflow?.prevention_status === 'PAUSED' ? (
              <PauseCircle className="w-7 h-7 text-amber-400" />
            ) : workflow?.prevention_status === 'VERIFICATION_REQUIRED' ? (
              <AlertTriangle className="w-7 h-7 text-yellow-400" />
            ) : (
              <CheckCircle2 className="w-7 h-7 text-emerald-400" />
            )}
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider opacity-80">
              D. Sensitive Action Containment Status
            </div>
            <h3 className="text-xl sm:text-2xl font-black tracking-tight">{currentStatusInfo.title}</h3>
            <p className="text-xs opacity-90 mt-1 max-w-2xl">{workflow?.explanation || currentStatusInfo.desc}</p>
          </div>
        </div>

        <div className="shrink-0 text-right bg-slate-950/40 px-4 py-2.5 rounded-xl border border-white/10 self-stretch md:self-auto">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Protected Transaction Value</div>
          <div className="text-lg font-black font-mono text-white">
            {transactionAmount !== '' ? `$${Number(transactionAmount).toLocaleString()}` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Grid: Recommended Response Panel (E) + Live Incident Timeline (F) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Recommended Response Panel with LIVE Action Buttons (Section 9.E) (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4 flex flex-col justify-between">
          <div>
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2 mb-3">
              <Lock className="w-4 h-4 text-indigo-400" />
              <span>E. Prevention Response Controls</span>
            </h2>

            <p className="text-xs text-slate-300 leading-relaxed mb-4">
              Directly interact with the prevention workflow state machine. Each button executes real backend transitions and logs into the incident timeline:
            </p>

            <div className="space-y-2.5">
              {/* Callback Challenge Button */}
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-850 space-y-2">
                <div className="text-xs font-semibold text-white flex items-center gap-2">
                  <PhoneCall className="w-3.5 h-3.5 text-amber-400" />
                  <span>Out-of-Band Callback Challenge</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleSimulateVerification('callback_passed')}
                    disabled={actionLoading !== null || workflow?.prevention_status === 'BLOCKED'}
                    className="px-3 py-2 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Simulate: Verified
                  </button>
                  <button
                    onClick={() => handleSimulateVerification('callback_failed')}
                    disabled={actionLoading !== null}
                    className="px-3 py-2 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Simulate: Failed / Fraud
                  </button>
                </div>
              </div>

              {/* MFA Challenge Button */}
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-850 space-y-2">
                <div className="text-xs font-semibold text-white flex items-center gap-2">
                  <Lock className="w-3.5 h-3.5 text-blue-400" />
                  <span>Step-Up Multi-Factor Auth (MFA)</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleSimulateVerification('mfa_passed')}
                    disabled={actionLoading !== null || workflow?.prevention_status === 'BLOCKED'}
                    className="px-3 py-2 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Simulate: MFA Passed
                  </button>
                  <button
                    onClick={() => handleSimulateVerification('mfa_failed')}
                    disabled={actionLoading !== null}
                    className="px-3 py-2 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Simulate: MFA Denied
                  </button>
                </div>
              </div>

              {/* Supervisor Approval */}
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-850 space-y-2">
                <div className="text-xs font-semibold text-white flex items-center gap-2">
                  <UserCheck className="w-3.5 h-3.5 text-purple-400" />
                  <span>Supervisor Escalation &amp; Override</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleSimulateVerification('supervisor_approved')}
                    disabled={actionLoading !== null}
                    className="px-3 py-2 rounded-lg bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 border border-purple-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Authorize Override
                  </button>
                  <button
                    onClick={() => handleSimulateVerification('supervisor_rejected')}
                    disabled={actionLoading !== null}
                    className="px-3 py-2 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/40 text-[11px] font-bold transition disabled:opacity-40"
                  >
                    Reject &amp; Block
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500 italic">
            * All actions enforce strict state machine guards; blocked workflows require explicit supervisor authorization.
          </div>
        </div>

        {/* Right: Live Chronological Incident Timeline (Section 9.F) (7 cols) */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-indigo-400" />
              <span>F. Live Incident &amp; Decision Timeline</span>
            </h2>
            <span className="text-[11px] font-mono text-slate-400">
              {workflow?.timeline.length || 0} events logged
            </span>
          </div>

          {workflow && workflow.timeline.length > 0 ? (
            <div className="space-y-3 max-h-[420px] overflow-y-auto pr-2 custom-scrollbar">
              {workflow.timeline.map((evt, idx) => (
                <div
                  key={evt.event_id || idx}
                  className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-850 flex items-start gap-3 text-xs"
                >
                  <div className="mt-0.5">
                    {evt.event_type.includes('BLOCKED') || evt.event_type.includes('FAILED') ? (
                      <XCircle className="w-4 h-4 text-rose-400" />
                    ) : evt.event_type.includes('PAUSED') ? (
                      <PauseCircle className="w-4 h-4 text-amber-400" />
                    ) : evt.event_type.includes('VERIFIED') || evt.event_type.includes('AUTHENTICATED') || evt.event_type.includes('APPROVED') || evt.event_type.includes('RESOLVED') ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Activity className="w-4 h-4 text-blue-400" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono font-bold text-indigo-300 text-[11px]">
                        {evt.event_type}
                      </span>
                      <span className="text-[10px] font-mono text-slate-500">
                        {new Date(evt.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-slate-300 mt-1 leading-relaxed text-[11px]">
                      {evt.details}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-xs text-slate-500">
              No timeline events recorded yet. Evaluate a scenario to initialize the timeline.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FraudPreventionCommandCenter;
