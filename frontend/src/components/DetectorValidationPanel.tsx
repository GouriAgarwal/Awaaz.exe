import React, { useEffect, useState } from 'react';
import {
  EvaluationReportInfo,
  ModelProvenanceInfo,
} from '../types';
import { getEvaluationStatus, getModelProvenance } from '../services/api';

export const DetectorValidationPanel: React.FC = () => {
  const [report, setReport] = useState<EvaluationReportInfo | null>(null);
  const [provenanceMap, setProvenanceMap] = useState<Record<string, ModelProvenanceInfo>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [evalReport, provMap] = await Promise.all([
        getEvaluationStatus(),
        getModelProvenance(),
      ]);
      setReport(evalReport);
      setProvenanceMap(provMap);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load scientific validation data';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'NOT_VALIDATED':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'BLOCKED':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'NOT_RUN':
        return 'bg-slate-700/50 text-slate-300 border-slate-600';
      default:
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
    }
  };

  const getPretrainedStatusBadge = (status: string) => {
    switch (status) {
      case 'VALIDATED_PRETRAINED':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'PRETRAINED_NOT_YET_VALIDATED':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      case 'UNTRAINED_NEURAL_BASELINE':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'HEURISTIC_FALLBACK':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40';
      default:
        return 'bg-slate-700 text-slate-300 border-slate-600';
    }
  };

  if (loading) {
    return (
      <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-8 text-center text-slate-400">
        <div className="flex items-center justify-center gap-3">
          <svg className="animate-spin h-5 w-5 text-indigo-400" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <span>Loading scientific validation records...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                  Scientific Validation &amp; Benchmark Audit
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-medium border border-indigo-500/30">
                    Step 7
                  </span>
                </h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  Reproducible benchmark evaluation, machine-readable provenance, threshold analysis, and calibration audits.
                </p>
              </div>
            </div>
          </div>

          {/* Model & Evaluation Status Badges */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-900/80 border border-slate-700 text-xs font-mono text-slate-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>AASIST Pretrained (SHA: 51d2d9cf...)</span>
            </div>
            {report && (
              <span className={`px-3 py-1 rounded-lg text-xs font-bold border ${getStatusBadge(report.evaluation_status)}`}>
                STATUS: {report.evaluation_status}
              </span>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {/* Main Status / Benchmark Section */}
      {report && (report.evaluation_status === 'NOT_RUN' || report.evaluation_status === 'BLOCKED') ? (
        <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-8 space-y-6">
          <div className="text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-400 mx-auto">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-200">Independent Scientific Validation Pending</h3>
              <p className="text-xs text-slate-400 max-w-2xl mx-auto mt-1 leading-relaxed">
                The genuine pretrained AASIST model checkpoint (<code className="text-indigo-300 font-mono">AASIST.pth</code>, 297,866 parameters) is loaded and operational.
                However, no local real-world benchmark dataset (e.g. ASVspoof 2019/2021 LA or In-the-Wild) has been executed on this installation.
                In accordance with VoiceShield scientific integrity policies, placeholder metrics and fabricated accuracy numbers are strictly withheld.
              </p>
            </div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 text-xs font-mono">
              <span>Scientific Status: PRETRAINED_NOT_YET_VALIDATED • NOT_CALIBRATED</span>
            </div>
          </div>

          {/* Dataset Manifest Documentation Guide */}
          <div className="bg-slate-900/60 rounded-xl p-5 border border-slate-800 space-y-4 text-xs">
            <h4 className="font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              How to Run Benchmark Evaluation on Real Datasets
            </h4>
            <p className="text-slate-400 leading-relaxed">
              To independently validate the detector, provide a CSV or JSON manifest pointing to real human (<code className="text-emerald-300">REAL</code>) and synthetic (<code className="text-rose-300">SYNTHETIC</code>) audio recordings.
            </p>
            
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300 overflow-x-auto">
              <div className="text-slate-500 mb-1"># Example Manifest Format (manifest.csv):</div>
              <div>file,label,split,speaker_id,generator,codec,noise_condition</div>
              <div>data/asvspoof/sample_001.flac,REAL,eval,speaker_101,none,flac,clean</div>
              <div>data/asvspoof/sample_002.flac,SYNTHETIC,eval,speaker_101,elevenlabs_v2,flac,clean</div>
              <div>data/asvspoof/sample_003.flac,SYNTHETIC,eval,speaker_102,tts_vits,flac,office_noise</div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-400">
              <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800">
                <span className="font-semibold text-slate-200 block mb-1">CLI Evaluation Command:</span>
                <code className="text-indigo-300 font-mono text-[11px] block break-all">
                  python -m app.detection.evaluation.runner --manifest path/to/manifest.csv --threshold 0.50
                </code>
              </div>
              <div className="bg-slate-950/40 p-3 rounded-lg border border-slate-800">
                <span className="font-semibold text-slate-200 block mb-1">REST API Endpoint:</span>
                <code className="text-indigo-300 font-mono text-[11px] block">
                  POST /api/detection/evaluation/run
                </code>
              </div>
            </div>
          </div>
        </div>
      ) : report && report.metrics ? (
        /* Populated Metrics Display */
        <div className="space-y-6">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">Accuracy</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{(report.metrics.accuracy * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">Precision</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{(report.metrics.precision * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">Recall</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{(report.metrics.recall * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">Specificity</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{(report.metrics.specificity * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">F1-Score</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{report.metrics.f1_score.toFixed(3)}</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">FPR</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{(report.metrics.fpr * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">ROC-AUC</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{report.metrics.roc_auc !== null ? report.metrics.roc_auc.toFixed(3) : 'N/A'}</p>
            </div>
            <div className="bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-center">
              <span className="text-[10px] text-slate-400 font-semibold uppercase">EER</span>
              <p className="text-base font-extrabold text-slate-100 mt-1">{report.metrics.eer !== null ? `${report.metrics.eer.toFixed(1)}%` : 'N/A'}</p>
            </div>
          </div>

          {/* Confusion Matrix & Latency Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confusion Matrix */}
            <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">
                Confusion Matrix (Samples: {report.sample_count} • REAL: {report.real_count}, SYNTHETIC: {report.synthetic_count})
              </h3>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="bg-emerald-500/10 border border-emerald-500/30 p-4 rounded-lg">
                  <span className="text-[11px] text-emerald-300 font-semibold block">True Negatives (TN)</span>
                  <span className="text-2xl font-black text-emerald-200 mt-1 block">{report.metrics.confusion_matrix.tn}</span>
                  <span className="text-[10px] text-slate-400">REAL classified as REAL</span>
                </div>
                <div className="bg-rose-500/10 border border-rose-500/30 p-4 rounded-lg">
                  <span className="text-[11px] text-rose-300 font-semibold block">False Positives (FP)</span>
                  <span className="text-2xl font-black text-rose-200 mt-1 block">{report.metrics.confusion_matrix.fp}</span>
                  <span className="text-[10px] text-slate-400">REAL flagged as SYNTHETIC</span>
                </div>
                <div className="bg-amber-500/10 border border-amber-500/30 p-4 rounded-lg">
                  <span className="text-[11px] text-amber-300 font-semibold block">False Negatives (FN)</span>
                  <span className="text-2xl font-black text-amber-200 mt-1 block">{report.metrics.confusion_matrix.fn}</span>
                  <span className="text-[10px] text-slate-400">SYNTHETIC missed as REAL</span>
                </div>
                <div className="bg-emerald-500/10 border border-emerald-500/30 p-4 rounded-lg">
                  <span className="text-[11px] text-emerald-300 font-semibold block">True Positives (TP)</span>
                  <span className="text-2xl font-black text-emerald-200 mt-1 block">{report.metrics.confusion_matrix.tp}</span>
                  <span className="text-[10px] text-slate-400">SYNTHETIC classified as SYNTHETIC</span>
                </div>
              </div>
            </div>

            {/* Latency & Processing Speed */}
            {report.latency && (
              <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
                <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">
                  Inference Latency &amp; Real-Time Factor
                </h3>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 font-medium">Cold Start Latency</span>
                    <p className="text-slate-100 font-semibold text-sm mt-0.5">{report.latency.cold_start_ms.toFixed(1)} ms</p>
                  </div>
                  <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 font-medium">Warm Average</span>
                    <p className="text-slate-100 font-semibold text-sm mt-0.5">{report.latency.warm_avg_ms.toFixed(1)} ms</p>
                  </div>
                  <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 font-medium">p50 / p95 Latency</span>
                    <p className="text-slate-100 font-semibold text-sm mt-0.5">
                      {report.latency.p50_ms.toFixed(1)} / {report.latency.p95_ms.toFixed(1)} ms
                    </p>
                  </div>
                  <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 font-medium">Real-Time Factor (RTF)</span>
                    <p className="text-slate-100 font-semibold text-sm mt-0.5">{report.latency.real_time_factor.toFixed(4)}x</p>
                    <span className="text-[10px] text-emerald-400 font-medium">Faster than real-time (&lt;1.0)</span>
                  </div>
                </div>
                <div className="mt-4 text-[11px] text-slate-400">
                  Total Evaluated Audio: <span className="text-slate-200 font-medium">{report.latency.total_audio_duration_sec.toFixed(1)}s</span>
                </div>
              </div>
            )}
          </div>

          {/* Notice on benchmark results */}
          <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-200/90 leading-relaxed">
            <span className="font-semibold text-amber-300">Scientific Evaluation Scope Notice: </span>
            These metrics apply to this specific evaluation dataset distribution and do not guarantee production impersonation prevention performance across unknown acoustic conditions.
          </div>

          {/* Threshold Sweep Table */}
          {report.metrics.threshold_sweep && report.metrics.threshold_sweep.length > 0 && (
            <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">
                Operating Threshold Sweep Analysis
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[11px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-700">
                    <tr>
                      <th className="py-2 px-3">Threshold</th>
                      <th className="py-2 px-3">TP</th>
                      <th className="py-2 px-3">FP</th>
                      <th className="py-2 px-3">TN</th>
                      <th className="py-2 px-3">FN</th>
                      <th className="py-2 px-3">Precision</th>
                      <th className="py-2 px-3">Recall (TPR)</th>
                      <th className="py-2 px-3">FPR</th>
                      <th className="py-2 px-3">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {report.metrics.threshold_sweep.map((pt) => (
                      <tr key={pt.threshold} className="hover:bg-slate-700/20 font-mono">
                        <td className="py-1.5 px-3 font-semibold text-slate-200">{pt.threshold.toFixed(2)}</td>
                        <td className="py-1.5 px-3 text-slate-300">{pt.tp}</td>
                        <td className="py-1.5 px-3 text-slate-300">{pt.fp}</td>
                        <td className="py-1.5 px-3 text-slate-300">{pt.tn}</td>
                        <td className="py-1.5 px-3 text-slate-300">{pt.fn}</td>
                        <td className="py-1.5 px-3 text-slate-300">{(pt.precision * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-3 text-slate-300">{(pt.recall * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-3 text-slate-300">{(pt.fpr * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-3 font-semibold text-slate-200">{pt.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : null}

      {/* Model Provenance & Audit Registry Table */}
      <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
        <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
          </svg>
          Machine-Readable Model Provenance Records
        </h3>

        <div className="space-y-4">
          {Object.values(provenanceMap).map((prov) => (
            <div key={prov.detector_name} className="bg-slate-900/60 rounded-xl p-4 border border-slate-800">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800">
                <div>
                  <h4 className="text-sm font-bold text-slate-100">{prov.detector_name}</h4>
                  <p className="text-xs text-slate-400">{prov.architecture}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`px-2.5 py-0.5 rounded text-[11px] font-bold border ${getPretrainedStatusBadge(prov.pretrained_status)}`}>
                    {prov.pretrained_status}
                  </span>
                  <span className="px-2.5 py-0.5 rounded text-[11px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                    {prov.calibration_status}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs mt-3">
                <div>
                  <span className="text-slate-500 font-medium">Checkpoint:</span>
                  <p className="text-slate-300 font-mono text-[11px] mt-0.5">{prov.checkpoint_identifier}</p>
                </div>
                <div>
                  <span className="text-slate-500 font-medium">License / Repo:</span>
                  <p className="text-slate-300 mt-0.5">{prov.license} • {prov.source_repository}</p>
                </div>
                <div>
                  <span className="text-slate-500 font-medium">Training Dataset:</span>
                  <p className="text-slate-300 mt-0.5">{prov.training_dataset}</p>
                </div>
              </div>

              <div className="mt-3 text-xs text-slate-400 bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/80">
                <span className="text-slate-500 font-semibold">Audit Notes: </span>
                {prov.notes}
              </div>
            </div>
          ))}
        </div>

        {/* Scientific Disclaimer */}
        <div className="mt-6 p-3.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-xs text-blue-200/90 leading-relaxed flex items-start gap-2.5">
          <svg className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold text-blue-300">Scientific Integrity Commitment: </span>
            A pretrained model is not equivalent to a VoiceShield-validated model. Model scores represent uncalibrated neural softmax outputs.
            Formal certification requires verified benchmark evaluation on independent, disjoint speaker partitions.
          </div>
        </div>
      </div>
    </div>
  );
};
