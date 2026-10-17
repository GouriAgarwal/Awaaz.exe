import React, { useState, useRef, useEffect } from 'react';
import {
  Activity,
  AudioWaveform,
  Sliders,
  Gauge,
  Volume2,
  Mic,
  Square,
  UploadCloud,
  FileAudio,
  Radio,
  Sparkles,
  Info,
  Clock,
  HelpCircle,
} from 'lucide-react';
import { extractFeatures } from '../services/api';
import { FeatureExtractionResponse } from '../types';

export const AcousticFeaturesPanel: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isExtracting, setIsExtracting] = useState(false);
  const [featureData, setFeatureData] = useState<FeatureExtractionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const getSupportedMimeType = (): string => {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      'audio/wav',
    ];
    for (const type of types) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return '';
  };

  const handleExtract = async (audioBlobOrFile: Blob | File, filename?: string) => {
    setIsExtracting(true);
    setError(null);
    try {
      const res = await extractFeatures(audioBlobOrFile, filename);
      setFeatureData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to extract acoustic features.');
      setFeatureData(null);
    } finally {
      setIsExtracting(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      handleExtract(file, file.name);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      handleExtract(file, file.name);
    }
  };

  const startRecording = async () => {
    setError(null);
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blobType = mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: blobType });
        const ext = blobType.includes('ogg') ? 'ogg' : blobType.includes('wav') ? 'wav' : 'webm';
        handleExtract(audioBlob, `features_recording.${ext}`);
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setRecordingSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setError(`Microphone error: ${err.message || 'Permission denied or device not found.'}`);
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

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const feat = featureData?.features;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-white">Acoustic &amp; Spectral Feature Extraction</h2>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-bold">
                Step 3 Engine
              </span>
            </div>
            <p className="text-xs text-slate-400">
              STFT, 20-MFCC, Autocorrelation F0 Pitch, Jitter, Shimmer, HNR &amp; Prosodic Dynamics
            </p>
          </div>
        </div>

        <button
          onClick={() => setShowExplainability(!showExplainability)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium transition"
        >
          <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
          <span>{showExplainability ? 'Hide Explanations' : 'Feature Definitions'}</span>
        </button>
      </div>

      {/* Input Hub */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition text-center ${
            isDragging
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.webm,.ogg,.flac,audio/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="p-3 rounded-2xl bg-slate-800 text-purple-400 mb-2 shadow-inner">
            <UploadCloud className="w-5 h-5" />
          </div>
          <p className="text-xs font-semibold text-slate-200">
            Upload audio to extract full acoustic vector
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">WAV • MP3 • M4A • WebM • OGG • FLAC</p>
          {selectedFile && (
            <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 rounded-md bg-slate-800 text-xs text-slate-300">
              <FileAudio className="w-3.5 h-3.5 text-purple-400" />
              <span className="truncate max-w-[180px]">{selectedFile.name}</span>
            </div>
          )}
        </div>

        <div className="bg-slate-950/60 rounded-xl p-6 border border-slate-800 flex flex-col justify-between items-center text-center">
          <div>
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-1">
              Extract from Live Microphone
            </h3>
            <p className="text-[11px] text-slate-500">
              Streams directly to feature pipeline via WebM/Opus buffer
            </p>
          </div>

          <div className="my-2">
            {isRecording ? (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono font-bold animate-pulse">
                <Radio className="w-3.5 h-3.5 animate-spin" />
                <span>RECORDING: {recordingSeconds}s</span>
              </div>
            ) : (
              <span className="text-xs text-slate-500">Ready to capture voice sample</span>
            )}
          </div>

          {isRecording ? (
            <button
              onClick={stopRecording}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-lg shadow-rose-600/30 transition active:scale-95"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Stop &amp; Extract Features</span>
            </button>
          ) : (
            <button
              onClick={startRecording}
              disabled={isExtracting}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-xs font-bold shadow-lg shadow-purple-600/30 transition active:scale-95 disabled:opacity-50"
            >
              <Mic className="w-4 h-4" />
              <span>Record &amp; Extract</span>
            </button>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300">
          <p className="font-bold">Feature Extraction Error:</p>
          <p className="mt-0.5">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {isExtracting && (
        <div className="p-8 rounded-xl bg-slate-950/40 border border-slate-800 flex flex-col items-center justify-center gap-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400"></div>
          <p className="text-xs text-slate-400 font-medium">
            Computing STFT, 20-MFCCs, Autocorrelation F0, Jitter, Shimmer &amp; Prosodic Dynamics...
          </p>
        </div>
      )}

      {/* Explainability Section */}
      {showExplainability && (
        <div className="bg-slate-950/90 rounded-xl p-4 border border-indigo-500/30 space-y-3">
          <div className="flex items-center gap-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
            <Info className="w-4 h-4" />
            <span>Acoustic Parameter Definitions (Explainability Foundation)</span>
          </div>
          <p className="text-xs text-slate-400">
            These features represent deterministic physical and mathematical acoustic measurements.
            They are calculated as quantitative baseline vectors for future detection layers.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-[11px]">
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="font-bold text-slate-200">Spectral Centroid:</span>
              <p className="text-slate-400 mt-0.5">Frequency center of gravity (Hz), correlating with perceived sound brightness.</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="font-bold text-slate-200">Spectral Flatness:</span>
              <p className="text-slate-400 mt-0.5">Ratio of geometric to arithmetic mean; 0 indicates tonal pitch, 1 indicates white noise.</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="font-bold text-slate-200">Harmonics-to-Noise Ratio (HNR):</span>
              <p className="text-slate-400 mt-0.5">Logarithmic ratio of periodic harmonic energy versus aperiodic noise floor in dB.</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="font-bold text-slate-200">Jitter &amp; Shimmer:</span>
              <p className="text-slate-400 mt-0.5">Cycle-to-cycle perturbation in pitch period (Jitter) and amplitude (Shimmer).</p>
            </div>
          </div>
        </div>
      )}

      {/* Feature Results Cards */}
      {feat && !isExtracting && (
        <div className="space-y-6 pt-2">
          {/* Status and Latency Badge */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-xl bg-purple-500/10 border border-purple-500/20">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-bold text-purple-300">Feature Extraction Complete</span>
              <span className="text-[11px] text-slate-400 font-mono">
                • {feat.audio_duration_seconds.toFixed(2)}s Audio Processed
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-purple-300 font-mono">
              <Clock className="w-3.5 h-3.5 text-purple-400" />
              <span>{featureData.processing_time_ms} ms Latency</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Spectral Metrics */}
            <div className="bg-slate-950/70 rounded-xl p-4 border border-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold uppercase tracking-wider">
                <Sliders className="w-4 h-4" />
                <span>STFT Spectral Features</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Centroid</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.centroid_hz.toFixed(1)} Hz</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Bandwidth</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.bandwidth_hz.toFixed(1)} Hz</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Rolloff (85%)</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.rolloff_hz.toFixed(1)} Hz</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Flatness</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.flatness.toFixed(4)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Spectral Entropy</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.entropy.toFixed(4)}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Spectral Flux</span>
                  <span className="font-bold font-mono text-white">{feat.spectral.flux.toFixed(4)}</span>
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 text-[11px] space-y-1">
                <span className="text-slate-400 block text-[10px]">Sub-Band Energy Distribution:</span>
                <div className="flex justify-between font-mono text-slate-300">
                  <span>Low (&lt;1kHz): {(feat.spectral.low_energy_ratio * 100).toFixed(1)}%</span>
                  <span>Mid (1-4kHz): {(feat.spectral.mid_energy_ratio * 100).toFixed(1)}%</span>
                  <span>High (&gt;4kHz): {(feat.spectral.high_energy_ratio * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Pitch & Voice Quality */}
            <div className="bg-slate-950/70 rounded-xl p-4 border border-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
                <Gauge className="w-4 h-4" />
                <span>Pitch, Voice Quality &amp; Harmonics</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Mean F0 (Pitch)</span>
                  <span className="font-bold font-mono text-white">
                    {feat.pitch.f0_mean_hz !== null ? `${feat.pitch.f0_mean_hz.toFixed(1)} Hz` : 'Unvoiced'}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">HNR (Harmonics-to-Noise)</span>
                  <span className="font-bold font-mono text-white">
                    {feat.voice_quality.hnr_db !== null ? `${feat.voice_quality.hnr_db.toFixed(1)} dB` : 'N/A'}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Jitter (Local Perturbation)</span>
                  <span className="font-bold font-mono text-white">
                    {feat.voice_quality.jitter !== null ? `${(feat.voice_quality.jitter * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Shimmer (Amplitude Var)</span>
                  <span className="font-bold font-mono text-white">
                    {feat.voice_quality.shimmer !== null ? `${(feat.voice_quality.shimmer * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Voiced Speech Ratio</span>
                  <span className="font-bold font-mono text-white">
                    {(feat.pitch.voiced_ratio * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 block">Speaking Activity</span>
                  <span className="font-bold font-mono text-white">
                    {(feat.prosody.speaking_ratio * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 text-[11px] space-y-1 font-mono text-slate-300">
                <div className="flex justify-between">
                  <span>Voiced Segments: {feat.prosody.voiced_segment_count}</span>
                  <span>Avg Segment Dur: {feat.prosody.avg_voiced_duration_s.toFixed(2)}s</span>
                  <span>Pause Ratio: {(feat.prosody.pause_ratio * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* 20-Coefficient MFCC Spectrum */}
          <div className="bg-slate-950/70 rounded-xl p-4 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
                <Volume2 className="w-4 h-4" />
                <span>20-Band Mel-Frequency Cepstral Coefficients (MFCC Vector)</span>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">C00 – C19</span>
            </div>

            <div className="grid grid-cols-5 sm:grid-cols-10 gap-2">
              {feat.mfcc.means.map((val, idx) => (
                <div
                  key={idx}
                  className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-center"
                >
                  <span className="text-[9px] text-slate-500 font-mono block">C{idx < 10 ? `0${idx}` : idx}</span>
                  <span className="text-xs font-bold font-mono text-slate-200">{val.toFixed(1)}</span>
                  <span className="text-[8px] text-slate-500 block font-mono">±{feat.mfcc.stds[idx].toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
