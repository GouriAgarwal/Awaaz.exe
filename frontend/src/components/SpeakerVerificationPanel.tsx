import React, { useState, useEffect, useRef } from 'react';
import {
  UserCheck,
  UserPlus,
  Shield,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Upload,
  Mic,
  Square,
  CheckCircle,
  XCircle,
  Clock,
  Cpu,
  Trash2,
  RefreshCw,
  Lock,
  Layers,
  HelpCircle
} from 'lucide-react';
import {
  SpeakerProfileSummaryInfo,
  SpeakerEnrollmentResponse,
  SpeakerVerificationResponse,
  SpeakerModelProvenanceInfo
} from '../types';

export const SpeakerVerificationPanel: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'enroll' | 'verify'>('enroll');

  // Provenance & Model info
  const [provenance, setProvenance] = useState<SpeakerModelProvenanceInfo | null>(null);
  const [profiles, setProfiles] = useState<SpeakerProfileSummaryInfo[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');

  // Enrollment state
  const [enrollProfileId, setEnrollProfileId] = useState<string>('');
  const [enrollFiles, setEnrollFiles] = useState<File[]>([]);
  const [isEnrollRecording, setIsEnrollRecording] = useState<boolean>(false);
  const [enrollRecordDuration, setEnrollRecordDuration] = useState<number>(0);
  const [enrollLoading, setEnrollLoading] = useState<boolean>(false);
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const [enrollResult, setEnrollResult] = useState<SpeakerEnrollmentResponse | null>(null);

  // Verification state
  const [verifyFile, setVerifyFile] = useState<File | null>(null);
  const [isVerifyRecording, setIsVerifyRecording] = useState<boolean>(false);
  const [verifyRecordDuration, setVerifyRecordDuration] = useState<number>(0);
  const [verifyLoading, setVerifyLoading] = useState<boolean>(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [verifyResult, setVerifyResult] = useState<SpeakerVerificationResponse | null>(null);

  // MediaRecorder refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [provRes, profRes] = await Promise.all([
        fetch('/api/speaker/provenance'),
        fetch('/api/speaker/profiles')
      ]);

      if (provRes.ok) {
        const provData = await provRes.json();
        setProvenance(provData);
      }

      if (profRes.ok) {
        const profData = await profRes.json();
        setProfiles(profData);
        if (profData.length > 0 && !selectedProfileId) {
          setSelectedProfileId(profData[0].profile_id);
        }
      }
    } catch (err) {
      console.error('Failed to load speaker verification metadata:', err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Handle Enrollment Recording
  const startEnrollRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([audioBlob], `enroll_mic_${Date.now()}.wav`, { type: 'audio/wav' });
        setEnrollFiles((prev) => [...prev, recordedFile]);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsEnrollRecording(true);
      setEnrollRecordDuration(0);

      timerIntervalRef.current = window.setInterval(() => {
        setEnrollRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setEnrollError('Microphone access denied or unsupported browser.');
    }
  };

  const stopEnrollRecording = () => {
    if (mediaRecorderRef.current && isEnrollRecording) {
      mediaRecorderRef.current.stop();
      setIsEnrollRecording(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }
  };

  // Handle Verification Recording
  const startVerifyRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([audioBlob], `verify_mic_${Date.now()}.wav`, { type: 'audio/wav' });
        setVerifyFile(recordedFile);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsVerifyRecording(true);
      setVerifyRecordDuration(0);

      timerIntervalRef.current = window.setInterval(() => {
        setVerifyRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setVerifyError('Microphone access denied or unsupported browser.');
    }
  };

  const stopVerifyRecording = () => {
    if (mediaRecorderRef.current && isVerifyRecording) {
      mediaRecorderRef.current.stop();
      setIsVerifyRecording(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }
  };

  // Execute Enrollment
  const handleEnrollSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enrollProfileId.trim()) {
      setEnrollError('Please provide a valid Profile ID.');
      return;
    }
    if (enrollFiles.length === 0) {
      setEnrollError('Please upload or record at least one audio sample.');
      return;
    }

    setEnrollLoading(true);
    setEnrollError(null);
    setEnrollResult(null);

    const formData = new FormData();
    formData.append('profile_id', enrollProfileId.trim());
    enrollFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('/api/speaker/enroll', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Enrollment failed.');
      }

      setEnrollResult(data);
      setEnrollFiles([]);
      fetchData();
    } catch (err: any) {
      setEnrollError(err.message || 'An unexpected error occurred during enrollment.');
    } finally {
      setEnrollLoading(false);
    }
  };

  // Execute Verification
  const handleVerifySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedProfileId) {
      setVerifyError('Please select an enrolled profile to verify against.');
      return;
    }
    if (!verifyFile) {
      setVerifyError('Please upload or record verification audio.');
      return;
    }

    setVerifyLoading(true);
    setVerifyError(null);
    setVerifyResult(null);

    const formData = new FormData();
    formData.append('profile_id', selectedProfileId);
    formData.append('file', verifyFile);

    try {
      const response = await fetch('/api/speaker/verify', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Verification failed.');
      }

      setVerifyResult(data);
    } catch (err: any) {
      setVerifyError(err.message || 'Verification error.');
    } finally {
      setVerifyLoading(false);
    }
  };

  // Delete Profile
  const handleDeleteProfile = async (profileId: string) => {
    if (!window.confirm(`Are you sure you want to delete profile '${profileId}'?`)) return;
    try {
      const res = await fetch(`/api/speaker/profiles/${profileId}`, { method: 'DELETE' });
      if (res.ok) {
        fetchData();
        if (selectedProfileId === profileId) {
          setSelectedProfileId('');
        }
      }
    } catch (err) {
      console.error('Failed to delete profile:', err);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner: Provenance & Model Info */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-white">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start space-x-3">
            <div className="p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400 mt-1">
              <UserCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-xl font-bold tracking-tight">Speaker Identity Verification Subsystem</h2>
                <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full">
                  Step 8 Active
                </span>
              </div>
              <p className="text-slate-400 text-sm mt-1">
                Answers <span className="text-blue-300 font-medium font-mono">“WHO is speaking?”</span> using official pretrained{' '}
                <span className="text-slate-200 font-semibold">SpeechBrain ECAPA-TDNN</span> (192-dim biometric embeddings).
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="bg-slate-800/80 px-3.5 py-2 rounded-lg border border-slate-700/60 text-right">
              <div className="text-xs text-slate-400">Enrolled Profiles</div>
              <div className="text-lg font-bold text-blue-400">{profiles.length} Active</div>
            </div>
            <button
              onClick={fetchData}
              className="p-2 text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition"
              title="Refresh profiles"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Model Provenance Chips */}
        {provenance && (
          <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-slate-300">
            <div>
              <span className="text-slate-500">Model:</span> <span className="font-semibold text-slate-200">ECAPA-TDNN</span>
            </div>
            <div>
              <span className="text-slate-500">Params:</span> <span className="font-semibold text-slate-200">20,767,552</span>
            </div>
            <div>
              <span className="text-slate-500">Embedding:</span> <span className="font-semibold text-slate-200">192-dim (L2 Norm)</span>
            </div>
            <div>
              <span className="text-slate-500">License:</span> <span className="font-semibold text-slate-200">Apache-2.0</span>
            </div>
          </div>
        )}
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex border-b border-slate-800 space-x-4">
        <button
          onClick={() => setActiveSubTab('enroll')}
          className={`flex items-center space-x-2 pb-3 px-2 border-b-2 font-medium text-sm transition ${
            activeSubTab === 'enroll'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <UserPlus className="w-4 h-4" />
          <span>Speaker Enrollment</span>
        </button>
        <button
          onClick={() => setActiveSubTab('verify')}
          className={`flex items-center space-x-2 pb-3 px-2 border-b-2 font-medium text-sm transition ${
            activeSubTab === 'verify'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldCheck className="w-4 h-4" />
          <span>Identity Verification</span>
        </button>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* TAB 1: SPEAKER ENROLLMENT */}
      {/* ------------------------------------------------------------------- */}
      {activeSubTab === 'enroll' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Form: Enrollment Controls */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-2 flex items-center space-x-2">
              <UserPlus className="w-5 h-5 text-blue-400" />
              <span>Enroll Speaker Profile</span>
            </h3>
            <p className="text-slate-400 text-sm mb-5">
              Provide approximately 10–20 seconds of natural speech. Multiple samples are automatically combined into an
              L2-normalized centroid embedding.
            </p>

            {enrollError && (
              <div className="mb-4 p-3.5 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300 text-sm flex items-start space-x-2">
                <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                <span>{enrollError}</span>
              </div>
            )}

            <form onSubmit={handleEnrollSubmit} className="space-y-5">
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                  Speaker / Profile ID <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  value={enrollProfileId}
                  onChange={(e) => setEnrollProfileId(e.target.value)}
                  placeholder="e.g. executive_alice, agent_104"
                  className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3.5 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 text-sm"
                  required
                />
              </div>

              {/* Audio Upload / Record Options */}
              <div className="space-y-3">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Enrollment Audio Material
                </label>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {/* File Upload Button */}
                  <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-4 cursor-pointer bg-slate-800/40 hover:bg-slate-800/70 transition">
                    <Upload className="w-6 h-6 text-slate-400 mb-1.5" />
                    <span className="text-xs font-medium text-slate-300">Upload WAV/MP3 Audio</span>
                    <span className="text-[10px] text-slate-500 mt-0.5">Supports multi-file select</span>
                    <input
                      type="file"
                      multiple
                      accept="audio/*"
                      onChange={(e) => {
                        if (e.target.files) {
                          setEnrollFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
                        }
                      }}
                      className="hidden"
                    />
                  </label>

                  {/* Microphone Record Button */}
                  <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-lg p-4 bg-slate-800/40">
                    {!isEnrollRecording ? (
                      <button
                        type="button"
                        onClick={startEnrollRecording}
                        className="flex flex-col items-center justify-center text-blue-400 hover:text-blue-300 transition"
                      >
                        <Mic className="w-6 h-6 mb-1.5" />
                        <span className="text-xs font-medium">Record from Microphone</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={stopEnrollRecording}
                        className="flex flex-col items-center justify-center text-rose-400 hover:text-rose-300 animate-pulse transition"
                      >
                        <Square className="w-6 h-6 mb-1.5" />
                        <span className="text-xs font-medium">Stop Recording ({enrollRecordDuration}s)</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Staged files list */}
                {enrollFiles.length > 0 && (
                  <div className="space-y-1.5 mt-2">
                    <div className="text-xs text-slate-400 font-medium">Staged Samples ({enrollFiles.length}):</div>
                    <div className="max-h-32 overflow-y-auto space-y-1 pr-1">
                      {enrollFiles.map((f, i) => (
                        <div key={i} className="flex items-center justify-between bg-slate-800/70 px-3 py-1.5 rounded text-xs text-slate-300">
                          <span className="truncate max-w-[240px]">{f.name}</span>
                          <button
                            type="button"
                            onClick={() => setEnrollFiles(enrollFiles.filter((_, idx) => idx !== i))}
                            className="text-slate-500 hover:text-rose-400"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Privacy Guarantee Note */}
              <div className="p-3 bg-slate-800/60 rounded-lg border border-slate-700/50 flex items-start space-x-2 text-xs text-slate-400">
                <Lock className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-slate-300">Biometric Privacy Guarantee:</strong> Raw audio is processed in RAM and
                  immediately destroyed. Only the mathematical centroid embedding is retained.
                </span>
              </div>

              <button
                type="submit"
                disabled={enrollLoading || (!enrollProfileId.trim() && enrollFiles.length === 0)}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold rounded-lg text-sm transition flex items-center justify-center space-x-2"
              >
                {enrollLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Extracting Embeddings & Enrolling...</span>
                  </>
                ) : (
                  <>
                    <UserPlus className="w-4 h-4" />
                    <span>Complete Speaker Enrollment</span>
                  </>
                )}
              </button>
            </form>

            {/* Success Result Box */}
            {enrollResult && (
              <div className="mt-5 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-white space-y-2">
                <div className="flex items-center space-x-2 text-emerald-400 font-semibold">
                  <CheckCircle className="w-5 h-5" />
                  <span>{enrollResult.message}</span>
                </div>
                <div className="text-xs text-slate-300 grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2 border-t border-emerald-500/20">
                  <div>
                    <span className="text-slate-400">Samples Enrolled:</span> {enrollResult.sample_count}
                  </div>
                  <div>
                    <span className="text-slate-400">Total Duration:</span> {enrollResult.total_audio_duration_seconds}s
                  </div>
                  <div>
                    <span className="text-slate-400">Avg RMS:</span> {enrollResult.audio_quality.rms_db} dBFS
                  </div>
                </div>
                {enrollResult.warning && (
                  <div className="mt-2 text-xs text-amber-300 bg-amber-500/10 p-2 rounded border border-amber-500/30 flex items-start space-x-1.5">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    <span>{enrollResult.warning}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Column: Enrolled Profiles List */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
            <h3 className="text-md font-bold text-white flex items-center space-x-2">
              <Layers className="w-4 h-4 text-blue-400" />
              <span>Enrolled Profiles ({profiles.length})</span>
            </h3>

            {profiles.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-xs">
                No speaker profiles enrolled yet. Enroll an identity to enable verification.
              </div>
            ) : (
              <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
                {profiles.map((p) => (
                  <div
                    key={p.profile_id}
                    className="p-3 bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 rounded-lg flex items-center justify-between text-xs"
                  >
                    <div>
                      <div className="font-semibold text-slate-200">{p.profile_id}</div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {p.sample_count} samples • {p.total_audio_duration_seconds}s speech
                      </div>
                    </div>
                    <button
                      onClick={() => handleDeleteProfile(p.profile_id)}
                      className="p-1.5 text-slate-500 hover:text-rose-400 transition"
                      title="Delete profile"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* TAB 2: SPEAKER VERIFICATION */}
      {/* ------------------------------------------------------------------- */}
      {activeSubTab === 'verify' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Form: Verification Controls */}
          <div className="lg:col-span-1 bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
            <h3 className="text-lg font-bold text-white flex items-center space-x-2">
              <ShieldCheck className="w-5 h-5 text-blue-400" />
              <span>Identity Verification</span>
            </h3>
            <p className="text-slate-400 text-xs">
              Upload or record a speech utterance to test whether the voice matches an enrolled biometric identity.
            </p>

            {verifyError && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300 text-xs flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{verifyError}</span>
              </div>
            )}

            <form onSubmit={handleVerifySubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                  Target Enrolled Profile <span className="text-rose-400">*</span>
                </label>
                <select
                  value={selectedProfileId}
                  onChange={(e) => setSelectedProfileId(e.target.value)}
                  className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-blue-500"
                  required
                >
                  <option value="">-- Select Profile --</option>
                  {profiles.map((p) => (
                    <option key={p.profile_id} value={p.profile_id}>
                      {p.profile_id} ({p.sample_count} samples, {p.total_audio_duration_seconds}s)
                    </option>
                  ))}
                </select>
              </div>

              {/* Audio Upload / Record */}
              <div className="space-y-2">
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Verification Utterance
                </label>

                <div className="grid grid-cols-2 gap-2">
                  <label className="flex flex-col items-center justify-center border border-dashed border-slate-700 hover:border-slate-500 rounded-lg p-3 cursor-pointer bg-slate-800/40 hover:bg-slate-800/70 transition text-center">
                    <Upload className="w-5 h-5 text-slate-400 mb-1" />
                    <span className="text-[11px] font-medium text-slate-300 truncate w-full">
                      {verifyFile ? verifyFile.name : 'Upload File'}
                    </span>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          setVerifyFile(e.target.files[0]);
                        }
                      }}
                      className="hidden"
                    />
                  </label>

                  <div className="flex flex-col items-center justify-center border border-dashed border-slate-700 rounded-lg p-3 bg-slate-800/40">
                    {!isVerifyRecording ? (
                      <button
                        type="button"
                        onClick={startVerifyRecording}
                        className="flex flex-col items-center justify-center text-blue-400 hover:text-blue-300 transition"
                      >
                        <Mic className="w-5 h-5 mb-1" />
                        <span className="text-[11px] font-medium">Record Mic</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={stopVerifyRecording}
                        className="flex flex-col items-center justify-center text-rose-400 animate-pulse transition"
                      >
                        <Square className="w-5 h-5 mb-1" />
                        <span className="text-[11px] font-medium">Stop ({verifyRecordDuration}s)</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={verifyLoading || !selectedProfileId || !verifyFile}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold rounded-lg text-sm transition flex items-center justify-center space-x-2"
              >
                {verifyLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Verifying Identity...</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-4 h-4" />
                    <span>Run Speaker Verification</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Right Column: Verification Results */}
          <div className="lg:col-span-2 space-y-4">
            {!verifyResult ? (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-500 flex flex-col items-center justify-center">
                <Shield className="w-12 h-12 text-slate-700 mb-3" />
                <h4 className="text-slate-400 font-semibold mb-1">Awaiting Verification Request</h4>
                <p className="text-xs max-w-sm">
                  Select an enrolled profile and provide an audio sample to inspect cosine similarity and verification decision.
                </p>
              </div>
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-white space-y-5">
                {/* Result Header Badge */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
                  <div>
                    <span className="text-xs text-slate-400">Target Profile:</span>
                    <h4 className="text-lg font-bold text-slate-200">{verifyResult.profile_id}</h4>
                  </div>

                  <div className="flex items-center space-x-2">
                    {verifyResult.decision === 'MATCH' && (
                      <span className="px-3.5 py-1.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center space-x-1.5">
                        <CheckCircle className="w-4 h-4" />
                        <span>MATCH</span>
                      </span>
                    )}
                    {verifyResult.decision === 'NON_MATCH' && (
                      <span className="px-3.5 py-1.5 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 flex items-center space-x-1.5">
                        <XCircle className="w-4 h-4" />
                        <span>NON_MATCH</span>
                      </span>
                    )}
                    {verifyResult.decision === 'UNCERTAIN' && (
                      <span className="px-3.5 py-1.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/40 flex items-center space-x-1.5">
                        <HelpCircle className="w-4 h-4" />
                        <span>UNCERTAIN</span>
                      </span>
                    )}
                  </div>
                </div>

                {/* Main Metrics Display */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-slate-800/80 p-4 rounded-xl border border-slate-700/60">
                    <div className="text-xs text-slate-400">Cosine Similarity</div>
                    <div className="text-2xl font-black text-blue-400 mt-1">
                      {verifyResult.similarity_score.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">Threshold: {verifyResult.threshold}</div>
                  </div>

                  <div className="bg-slate-800/80 p-4 rounded-xl border border-slate-700/60">
                    <div className="text-xs text-slate-400">Confidence Band</div>
                    <div className="text-xl font-bold text-slate-200 mt-1">
                      {verifyResult.confidence_band}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">Status: {verifyResult.calibration_status}</div>
                  </div>

                  <div className="bg-slate-800/80 p-4 rounded-xl border border-slate-700/60">
                    <div className="text-xs text-slate-400">Inference Latency</div>
                    <div className="text-xl font-bold text-slate-200 mt-1 flex items-center space-x-1">
                      <Clock className="w-4 h-4 text-slate-400" />
                      <span>{verifyResult.latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">{verifyResult.duration_seconds}s audio duration</div>
                  </div>
                </div>

                {/* Scientific Notice Banner */}
                {verifyResult.provisional_warning && (
                  <div className="p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-300 text-xs flex items-start space-x-2.5">
                    <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                    <div>
                      <strong className="block font-semibold mb-0.5">Scientific Disclaimer:</strong>
                      {verifyResult.provisional_warning}
                    </div>
                  </div>
                )}

                {/* Orthogonal Security Notice */}
                <div className="p-3.5 bg-blue-500/10 border border-blue-500/30 rounded-xl text-slate-300 text-xs flex items-start space-x-2.5">
                  <ShieldAlert className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="block font-semibold text-blue-300 mb-0.5">
                      Orthogonal Biometric Dimension Notice:
                    </strong>
                    High speaker similarity indicates acoustic proximity to the enrolled speaker, but does{' '}
                    <strong>NOT</strong> guarantee the voice is natural. A cloned voice impersonating this speaker may also
                    exhibit high similarity. Cross-modal anti-spoofing fusion with AASIST is required for holistic impersonation defense.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
