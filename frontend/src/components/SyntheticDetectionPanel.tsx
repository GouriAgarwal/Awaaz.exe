import React, { useState, useRef, useEffect } from 'react';
import {
  DetectionResultInfo,
  DetectorMetadataInfo,
  ClassificationLabel,
} from '../types';
import { analyzeSyntheticVoice, getDetectorModels } from '../services/api';

export const SyntheticDetectionPanel: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const [selectedDetector, setSelectedDetector] = useState<string>('aasist');
  const [models, setModels] = useState<DetectorMetadataInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DetectionResultInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load available models on mount
  useEffect(() => {
    getDetectorModels()
      .then((data) => setModels(data))
      .catch((err) => console.warn('Could not load detector models:', err));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setRecordedBlob(null);
      setError(null);
      setResult(null);
    }
  };

  const startRecording = async () => {
    setError(null);
    setResult(null);
    setFile(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        setRecordedBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(200);
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = window.setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Microphone access denied';
      setError(`Microphone error: ${msg}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleAnalyze = async () => {
    const payload = file || recordedBlob;
    if (!payload) {
      setError('Please select an audio file or record from microphone.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const filename = file ? file.name : 'microphone_recording.webm';
      const response = await analyzeSyntheticVoice(payload, filename, selectedDetector);
      setResult(response.data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Detection analysis failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const getBadgeStyle = (label: ClassificationLabel) => {
    switch (label) {
      case 'NATURAL':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'SYNTHETIC':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'UNCERTAIN':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      default:
        return 'bg-slate-700/50 text-slate-300 border-slate-600';
    }
  };

  const getScoreColor = (score: number) => {
    if (score > 0.65) return 'from-rose-500 to-red-600';
    if (score < 0.35) return 'from-emerald-500 to-teal-500';
    return 'from-amber-500 to-yellow-500';
  };

  return (
    <div className="space-y-6">
      {/* Header card */}
      <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                  AI Synthetic Voice Detection
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-medium border border-indigo-500/30">
                    Step 4
                  </span>
                </h2>
                <p className="text-sm text-slate-400 mt-0.5">
                  Deep learning acoustic and raw-waveform classifier detecting synthetic, neural vocoder, and cloned speech.
                </p>
              </div>
            </div>
          </div>

          {/* Model Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400 font-medium whitespace-nowrap">Active Detector:</label>
            <select
              value={selectedDetector}
              onChange={(e) => setSelectedDetector(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              <option value="aasist">VoiceShield-AASIST-v1 (Raw Waveform Graph)</option>
              <option value="spec_cnn">VoiceShield-SpecCNN-v1 (Spectral ResNet)</option>
              <option value="fallback">Fallback Heuristic (Acoustic Regularity)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Upload and Recording Input Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* File Upload Box */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center min-h-[160px] ${
            file
              ? 'border-indigo-500/60 bg-indigo-500/5'
              : 'border-slate-700 hover:border-slate-500 bg-slate-800/40 hover:bg-slate-800/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.webm,.ogg,.flac,.m4a,.aac"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="w-10 h-10 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-300 mb-3">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          {file ? (
            <div>
              <p className="text-sm font-medium text-indigo-300">{file.name}</p>
              <p className="text-xs text-slate-400 mt-1">{(file.size / 1024).toFixed(1)} KB • Click to change</p>
            </div>
          ) : (
            <div>
              <p className="text-sm font-medium text-slate-300">Upload audio file for AI detection</p>
              <p className="text-xs text-slate-500 mt-1">WAV, MP3, WebM, M4A, OGG, FLAC (up to 25MB)</p>
            </div>
          )}
        </div>

        {/* Live Microphone Box */}
        <div className="border border-slate-700/80 bg-slate-800/40 rounded-xl p-6 flex flex-col items-center justify-center min-h-[160px]">
          <div className="flex flex-col items-center">
            {isRecording ? (
              <div className="flex flex-col items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-rose-500 animate-ping" />
                  <span className="text-sm font-medium text-rose-400">Recording live audio... {recordingTime}s</span>
                </div>
                <button
                  onClick={stopRecording}
                  className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-lg shadow-rose-600/30 transition"
                >
                  Stop Recording
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-300">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
                {recordedBlob ? (
                  <div className="text-center">
                    <p className="text-xs font-medium text-emerald-400">Recorded sample ready ({recordingTime}s)</p>
                    <button
                      onClick={startRecording}
                      className="mt-2 text-xs text-slate-400 hover:text-slate-200 underline"
                    >
                      Record again
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={startRecording}
                    className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-medium transition"
                  >
                    Start Microphone Test
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center">
        <button
          onClick={handleAnalyze}
          disabled={loading || (!file && !recordedBlob)}
          className={`px-8 py-3 rounded-xl font-semibold text-sm shadow-xl flex items-center gap-2 transition duration-200 ${
            loading || (!file && !recordedBlob)
              ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30 hover:scale-[1.02]'
          }`}
        >
          {loading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Running AI Neural Inference...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Analyze Synthetic Speech Evidence
            </>
          )}
        </button>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-3">
          <svg className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <p className="font-semibold">Analysis Failed</p>
            <p className="mt-0.5 text-slate-300">{error}</p>
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Main Verdict Card */}
          <div className="bg-slate-800/90 border border-slate-700/80 rounded-xl p-6 shadow-xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-slate-700/60">
              <div>
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Classification Verdict</span>
                <div className="flex items-center gap-3 mt-2">
                  <span className={`px-4 py-1.5 rounded-lg text-lg font-black tracking-wider border ${getBadgeStyle(result.classification)}`}>
                    {result.classification}
                  </span>
                  <div className="text-xs text-slate-400">
                    Confidence Band: <span className="text-slate-200 font-semibold">{result.confidence_band}</span>
                  </div>
                </div>
              </div>

              {/* Latency & Processing Speed */}
              <div className="flex items-center gap-4 bg-slate-900/60 px-4 py-3 rounded-lg border border-slate-800">
                <div>
                  <p className="text-[10px] uppercase text-slate-500 font-bold">Inference Latency</p>
                  <p className="text-sm font-semibold text-slate-200">{result.inference_latency_ms.toFixed(1)} ms</p>
                </div>
                <div className="h-6 w-px bg-slate-700" />
                <div>
                  <p className="text-[10px] uppercase text-slate-500 font-bold">Duration</p>
                  <p className="text-sm font-semibold text-slate-200">{result.audio_duration_sec.toFixed(2)}s</p>
                </div>
                <div className="h-6 w-px bg-slate-700" />
                <div>
                  <p className="text-[10px] uppercase text-slate-500 font-bold">Windows</p>
                  <p className="text-sm font-semibold text-slate-200">{result.total_windows}</p>
                </div>
              </div>
            </div>

            {/* Continuous Likelihood Gauge */}
            <div className="mt-6">
              <div className="flex justify-between items-center text-xs mb-2">
                <span className="text-slate-400 font-medium">Synthetic Likelihood Score:</span>
                <div className="flex items-center gap-2">
                  <span className="text-base font-bold text-slate-100">{result.score.toFixed(4)}</span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">
                    {result.score_type}
                  </span>
                </div>
              </div>

              {/* Progress bar with threshold markers */}
              <div className="relative w-full h-4 bg-slate-900 rounded-full overflow-hidden border border-slate-700/80">
                <div
                  className={`h-full bg-gradient-to-r ${getScoreColor(result.score)} transition-all duration-500`}
                  style={{ width: `${Math.min(100, Math.max(0, result.score * 100))}%` }}
                />
              </div>

              {/* Threshold Labels */}
              <div className="flex justify-between text-[11px] text-slate-500 mt-2 font-mono">
                <span>0.0 (Natural &lt; {result.thresholds_applied.natural_threshold})</span>
                <span className="text-amber-400/80">Uncertain Zone [0.35 - 0.65]</span>
                <span>(&gt; {result.thresholds_applied.synthetic_threshold} Synthetic) 1.0</span>
              </div>
            </div>

            {/* Warnings list if any */}
            {result.warnings && result.warnings.length > 0 && (
              <div className="mt-6 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
                <p className="font-semibold mb-1">Operational Warnings:</p>
                <ul className="list-disc list-inside space-y-0.5 text-slate-300">
                  {result.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Temporal Windows Breakdown Table */}
          {result.window_scores && result.window_scores.length > 0 && (
            <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Temporal Window Analysis ({result.window_scores.length} Segments • {result.aggregation_method} aggregated)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[11px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-700">
                    <tr>
                      <th className="py-2.5 px-3">Window #</th>
                      <th className="py-2.5 px-3">Time Range</th>
                      <th className="py-2.5 px-3">Raw Score</th>
                      <th className="py-2.5 px-3">Synthetic Score</th>
                      <th className="py-2.5 px-3">Window Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {result.window_scores.map((win) => (
                      <tr key={win.window_index} className="hover:bg-slate-700/20">
                        <td className="py-2 px-3 font-mono text-slate-400">#{win.window_index + 1}</td>
                        <td className="py-2 px-3 font-mono text-slate-300">
                          {win.start_sec.toFixed(2)}s – {win.end_sec.toFixed(2)}s
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-300">{win.raw_score.toFixed(4)}</td>
                        <td className="py-2 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-slate-900 rounded-full overflow-hidden">
                              <div
                                className={`h-full bg-gradient-to-r ${getScoreColor(win.synthetic_score)}`}
                                style={{ width: `${win.synthetic_score * 100}%` }}
                              />
                            </div>
                            <span className="font-mono text-slate-200">{win.synthetic_score.toFixed(4)}</span>
                          </div>
                        </td>
                        <td className="py-2 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getBadgeStyle(win.label)}`}>
                            {win.label}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Model Provenance & Architecture Card */}
          <div className="bg-slate-800/80 border border-slate-700/70 rounded-xl p-6 shadow-xl">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
              <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Model Provenance & Configuration
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">Model Name</span>
                <p className="text-slate-200 font-semibold mt-0.5">{result.detector_metadata.model_name}</p>
              </div>

              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">Architecture</span>
                <p className="text-slate-200 font-semibold mt-0.5">{result.detector_metadata.architecture}</p>
              </div>

              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">Model Category</span>
                <p className="text-slate-200 font-semibold mt-0.5">{result.detector_metadata.model_type}</p>
              </div>

              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">Compute Device</span>
                <p className="text-slate-200 font-semibold mt-0.5 uppercase">{result.detector_metadata.device}</p>
              </div>

              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">Expected Sample Rate</span>
                <p className="text-slate-200 font-semibold mt-0.5">{result.detector_metadata.expected_sample_rate} Hz (Mono)</p>
              </div>

              <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-500 font-medium">License & Checkpoint</span>
                <p className="text-slate-200 font-semibold mt-0.5">{result.detector_metadata.license}</p>
              </div>
            </div>

            {/* Score Semantics */}
            <div className="mt-4 p-3 rounded-lg bg-slate-900/40 border border-slate-800 text-xs text-slate-300">
              <span className="text-slate-400 font-semibold">Score Semantics: </span>
              {result.detector_metadata.score_interpretation}
            </div>

            {/* Scientific Disclaimer */}
            <div className="mt-4 p-3.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-xs text-blue-200/90 leading-relaxed flex items-start gap-2.5">
              <svg className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <span className="font-semibold text-blue-300">Scientific Evaluation Notice: </span>
                {result.detector_metadata.scientific_disclaimer}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
