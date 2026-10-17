import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  UploadCloud,
  Mic,
  Square,
  Volume2,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FileAudio,
  Radio,
  Gauge,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { inspectAudio } from '../services/api';
import { AudioInspectResponse } from '../types';

export const AudioInspector: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AudioInspectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Determine best supported MIME type for recording
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

  const handleProcessAudio = async (audioBlobOrFile: Blob | File, filename?: string) => {
    setIsProcessing(true);
    setError(null);
    try {
      const inspectRes = await inspectAudio(audioBlobOrFile, filename);
      setResult(inspectRes);
    } catch (err: any) {
      setError(err.message || 'Failed to inspect audio file.');
      setResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      handleProcessAudio(file, file.name);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      handleProcessAudio(file, file.name);
    }
  };

  const startRecording = async () => {
    setError(null);
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, options);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blobType = mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: blobType });
        const ext = blobType.includes('ogg') ? 'ogg' : blobType.includes('wav') ? 'wav' : 'webm';
        handleProcessAudio(audioBlob, `mic_recording.${ext}`);
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setRecordingSeconds(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setError(`Microphone access error: ${err.message || 'Permission denied or no microphone found.'}`);
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

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Volume2 className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-white">Audio Ingestion &amp; Inspector</h2>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                Step 2 Active
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Multi-format decoding, 16kHz mono float32 standardization &amp; acoustic signal validation
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-slate-400 bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
            Supported: WAV • MP3 • M4A • WebM/Opus • OGG • FLAC
          </span>
        </div>
      </div>

      {/* Upload and Recording Hub */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dropzone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition text-center ${
            isDragging
              ? 'border-indigo-500 bg-indigo-500/10'
              : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.webm,.ogg,.oga,.flac,audio/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="p-3.5 rounded-2xl bg-slate-800 text-indigo-400 mb-3 shadow-inner">
            <UploadCloud className="w-6 h-6" />
          </div>
          <p className="text-xs font-semibold text-slate-200">
            Drag &amp; drop audio file here or <span className="text-indigo-400 underline">browse</span>
          </p>
          <p className="text-[11px] text-slate-500 mt-1">
            Max 25 MB • 0.5s – 300s duration
          </p>
          {selectedFile && (
            <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-md bg-slate-800/80 border border-slate-700 text-xs text-slate-300">
              <FileAudio className="w-3.5 h-3.5 text-indigo-400" />
              <span className="truncate max-w-[200px]">{selectedFile.name}</span>
            </div>
          )}
        </div>

        {/* Microphone Recording Panel */}
        <div className="bg-slate-950/60 rounded-xl p-6 border border-slate-800 flex flex-col justify-between items-center text-center">
          <div>
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-1">
              Live Browser Microphone
            </h3>
            <p className="text-[11px] text-slate-500">
              Records directly via MediaRecorder (WebM/Opus) and transmits to /api/audio/inspect
            </p>
          </div>

          <div className="my-4">
            {isRecording ? (
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono font-bold animate-pulse">
                  <Radio className="w-3.5 h-3.5 animate-spin" />
                  <span>RECORDING: {recordingSeconds}s</span>
                </div>
                <p className="text-[10px] text-slate-400">Speak into your microphone</p>
              </div>
            ) : (
              <div className="text-xs text-slate-500">
                Click below to start live microphone capture
              </div>
            )}
          </div>

          <div>
            {isRecording ? (
              <button
                onClick={stopRecording}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-lg shadow-rose-600/30 transition active:scale-95"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Stop &amp; Inspect Audio</span>
              </button>
            ) : (
              <button
                onClick={startRecording}
                disabled={isProcessing}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold shadow-lg shadow-indigo-600/30 transition active:scale-95 disabled:opacity-50"
              >
                <Mic className="w-4 h-4" />
                <span>Start Microphone Record</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Error Message Display */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3">
          <XCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-xs font-bold text-rose-300">Audio Ingestion Error</h4>
            <p className="text-xs text-rose-400/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Loading Spinner */}
      {isProcessing && (
        <div className="p-8 rounded-xl bg-slate-950/40 border border-slate-800 flex flex-col items-center justify-center gap-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400"></div>
          <p className="text-xs text-slate-400 font-medium">
            Decoding, resampling to 16kHz mono &amp; calculating acoustic metrics...
          </p>
        </div>
      )}

      {/* Results Display */}
      {result && !isProcessing && (
        <div className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span>Standardized Acoustic Inspection Results</span>
            </h3>
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${
                result.quality.quality === 'good'
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : result.quality.quality === 'warning'
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
              }`}
            >
              Signal Quality: {result.quality.quality}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block mb-1">Detected Format</span>
              <span className="text-base font-bold font-mono text-cyan-400 uppercase">
                {result.audio.original_format}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Container / Codec</span>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block mb-1">Duration</span>
              <span className="text-base font-bold font-mono text-white">
                {result.audio.duration_seconds.toFixed(2)} s
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Calculated audio length</span>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block mb-1">Standardized Output</span>
              <span className="text-base font-bold font-mono text-emerald-400">16 kHz Mono</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Float32 PCM Format</span>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800">
              <span className="text-[11px] text-slate-400 block mb-1">Original Input</span>
              <span className="text-base font-bold font-mono text-slate-300">
                {result.audio.original_sample_rate / 1000} kHz • {result.audio.original_channels} Ch
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Pre-resampled stream</span>
            </div>
          </div>

          {/* Quality Breakdown Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1">
            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
              <Sliders className="w-4 h-4 text-indigo-400 shrink-0" />
              <div>
                <span className="text-[10px] text-slate-400 block">RMS Energy</span>
                <span className="text-xs font-bold font-mono text-slate-200">
                  {result.quality.rms_db.toFixed(1)} dBFS
                </span>
              </div>
            </div>

            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
              <Gauge className="w-4 h-4 text-cyan-400 shrink-0" />
              <div>
                <span className="text-[10px] text-slate-400 block">Peak Amplitude</span>
                <span className="text-xs font-bold font-mono text-slate-200">
                  {result.quality.peak_db.toFixed(1)} dBFS
                </span>
              </div>
            </div>

            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <div>
                <span className="text-[10px] text-slate-400 block">Clipping Ratio</span>
                <span className="text-xs font-bold font-mono text-slate-200">
                  {(result.quality.clipping_ratio * 100).toFixed(2)}%
                </span>
              </div>
            </div>

            <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 flex items-center gap-3">
              <Volume2 className="w-4 h-4 text-purple-400 shrink-0" />
              <div>
                <span className="text-[10px] text-slate-400 block">Silence Ratio</span>
                <span className="text-xs font-bold font-mono text-slate-200">
                  {(result.quality.silence_ratio * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* Diagnostic Observations */}
          {result.quality.notes && result.quality.notes.length > 0 && (
            <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800 text-left">
              <span className="text-[11px] font-semibold text-slate-400 block mb-1">
                Diagnostic Observations:
              </span>
              <ul className="space-y-1">
                {result.quality.notes.map((note, idx) => (
                  <li key={idx} className="text-xs text-slate-300 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
