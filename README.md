# VoiceShield — AI-Powered Real-Time Voice Cloning Impersonation Defense

**VoiceShield** is an advanced cybersecurity and AI defense system designed to detect and prevent real-time voice cloning and acoustic impersonation attacks.

> **Current Phase**: **Step 15: Final Hackathon Product Integration, End-to-End Demo & Presentation Readiness**  
> *Official Problem Statement*: **"AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks"**  
> *Notice: VoiceShield unifies pretrained AASIST synthetic voice detection, SpeechBrain ECAPA-TDNN biometric verification, acoustic prosody dynamics, and automated zero-trust fraud prevention workflows. It stops fraudulent wire transfers, credential resets, and executive spear-phishing attacks in real time before transactions commit.*

---

## Architecture Overview

```
voiceshield/
├── backend/
│   ├── .venv/                      # Python virtual environment
│   ├── requirements.txt            # Dependencies (FastAPI, PyAV, SoundFile, NumPy, PyTorch, etc.)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application factory & startup checksum gate
│   │   ├── api/
│   │   │   ├── router.py           # Central API router
│   │   │   └── v1/
│   │   │       ├── health.py       # GET /api/health/live, GET /api/health/ready
│   │   │       ├── system.py       # GET /api/system/status
│   │   │       ├── audio.py        # POST /api/audio/inspect
│   │   │       ├── features.py     # POST /api/features/extract
│   │   │       ├── detection.py    # POST /api/detection/analyze
│   │   │       ├── speaker.py      # POST /api/speaker/enroll, POST /api/speaker/verify
│   │   │       ├── risk.py         # POST /api/risk/analyze, POST /api/risk/simulate
│   │   │       ├── streaming.py    # WS /api/stream/ws, POST /api/stream/sessions
│   │   │       ├── evaluation.py   # GET /api/evaluation/status, POST /api/evaluation/audit
│   │   │       ├── observability.py# GET /api/observability/metrics, GET /api/observability/models
│   │   │       ├── contextual_risk.py # POST /api/contextual-risk/analyze, POST /api/contextual-risk/simulate
│   │   │       └── prevention.py   # POST /api/prevention/evaluate, POST /api/prevention/workflows/{id}/verify
│   │   ├── audio/                  # Step 2 Ingestion & Preprocessing (16 kHz mono float32)
│   │   ├── features/               # Step 3 Feature Extraction (STFT, F0, MFCCs, Jitter, Shimmer, HNR)
│   │   ├── detection/              # Step 4/6 Pretrained AASIST Synthetic Voice Detection Engine
│   │   ├── speaker_verification/   # Step 8 Pretrained ECAPA-TDNN Speaker Verification Subsystem
│   │   ├── risk/                   # Step 9 Controlled Multi-Modal Impersonation Risk Engine
│   │   ├── streaming/              # Step 10 Real-Time Streaming Orchestration Pipeline
│   │   ├── evaluation/             # Step 11 Scientific Validation, Calibration & Security Subsystem
│   │   ├── observability/          # Step 12 Health Probes, Tracing, Metrics & Model Monitoring
│   │   ├── prosody/                # Step 13 Acoustic Prosody & Behavioral Dynamics Subsystem
│   │   ├── context/                # Step 13 Contextual Fraud Risk Intelligence Engine
│   │   └── prevention/             # Step 14 Automated Prevention & Incident Response Workflow
│   │       ├── schemas.py          # Actions, statuses, verification methods, timeline events
│   │       ├── policies.py         # Banking, Enterprise, Government, Telecom threshold profiles
│   │       ├── engine.py           # Prevention decision engine
│   │       ├── workflows.py        # State machine with strict transition guards
│   │       ├── timeline.py         # Privacy-preserving in-memory audit ledger
│   │       └── service.py          # PreventionService coordinator & 6 demo scenarios
│   └── tests/                      # 244 automated unit & integration tests across 15 steps
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Full-stack console with flagship Hackathon Command Center
│   │   ├── components/
│   │   │   ├── VoiceShieldCommandCenter.tsx # Step 15 Flagship Hackathon Presentation & Live Cockpit
│   │   │   ├── FraudPreventionCommandCenter.tsx # Step 14 Automated Prevention Command Center
│   │   │   ├── VoiceIntegrityConsole.tsx # Step 13 Multi-Layer Voice Cloning Defense Console
│   │   │   ├── OperationsDashboard.tsx # Step 12 Production Observability & Health Dashboard
│   │   │   ├── ScientificValidationPanel.tsx # Step 11 Scientific Validation & Security Panel
│   │   │   ├── RealtimeMonitoringPanel.tsx # Step 10 Real-Time Streaming Panel
│   │   │   ├── ImpersonationRiskPanel.tsx # Step 9 Impersonation Risk Fusion Dashboard
│   │   │   ├── SpeakerVerificationPanel.tsx # Step 8 Speaker Identity Verification Panel
│   │   │   └── SyntheticDetectionPanel.tsx # Step 4/6 AI Voice Clone Detection Panel
│   │   └── types/                  # TypeScript definitions
│   └── docs/
│       ├── hackathon_solution.md   # Step 15 Hackathon Solution & Presentation Guide
│       ├── prevention_workflows.md # Step 14 Prevention Documentation
│       ├── speaker_verification.md # Step 8 Scientific Documentation
│       └── realtime_streaming.md   # Step 10 Scientific Documentation
└── README.md
```

---

## Mathematical & Algorithmic Feature Formulations

### 1. STFT & Spectral Analysis
- **STFT Configuration**:
  - Sample Rate: `16,000 Hz`
  - FFT Window Size: `512` (32.0 ms)
  - Window Length: `400` (25.0 ms with periodic Hann window)
  - Hop Length: `160` (10.0 ms frame shift)
  - Frequency Bins: `257` discrete bins from `0 Hz` to `8,000 Hz` (resolution: `31.25 Hz/bin`)
- **Spectral Centroid**: Frequency center of mass $\text{Centroid} = \frac{\sum f_k |X(f_k)|}{\sum |X(f_k)|}$ in Hz.
- **Spectral Bandwidth**: Standard deviation around centroid $\sqrt{\frac{\sum (f_k - \text{Centroid})^2 |X(f_k)|}{\sum |X(f_k)|}}$ in Hz.
- **Spectral Rolloff**: Frequency bin below which 85% of cumulative spectral power is concentrated.
- **Spectral Flatness**: Ratio of geometric mean to arithmetic mean of spectral power (0 = pure tone, 1 = white noise).
- **Spectral Entropy**: Normalized Shannon entropy over power distribution $-\frac{1}{\log_2(K)}\sum p_k \log_2(p_k + 10^{-12})$.
- **Sub-Band Energy Ratios**: Low ($0 - 1\text{ kHz}$), Mid ($1 - 4\text{ kHz}$), and High ($4 - 8\text{ kHz}$).

### 2. 20 Mel-Frequency Cepstral Coefficients (MFCCs)
- **Mel Filterbank**: 20 triangular overlapping filters spaced linearly on the HTK Mel scale: $m = 2595 \log_{10}(1 + f/700)$.
- **Log Compression**: $\log(\text{MelEnergies} + 10^{-6})$.
- **Discrete Cosine Transform (DCT-II)**: Computes 20 cepstral coefficients per frame ($C_0 - C_{19}$) with mean and standard deviation across all frames.

### 3. Fundamental Frequency (F0) & Pitch Tracking
- **Algorithm**: Normalized Cross-Correlation Function (NCCF) / Autocorrelation search in speech range ($50\text{ Hz} - 500\text{ Hz}$).
- **Parabolic Interpolation**: Quadratic 3-point peak interpolation for sub-sample lag precision.
- **Voiced Decision**: Frame is voiced if normalized autocorrelation peak $\ge 0.35$ and frame $\text{RMS} \ge 0.003$.
- **Voiced F0 Statistics**: Mean, median, standard deviation, min, max, and percentiles (p10, p25, p75, p90) computed strictly over voiced frames.

### 4. Voice Quality & Perturbation Metrics
- **Local Jitter**: Relative cycle-to-cycle F0 period perturbation: $\text{Jitter} = \frac{\frac{1}{M-1}\sum |T_{i+1} - T_i|}{\frac{1}{M}\sum T_i}$ (returns `null` if $<3$ voiced cycles).
- **Local Shimmer**: Relative cycle-to-cycle peak amplitude perturbation: $\text{Shimmer} = \frac{\frac{1}{M-1}\sum |A_{i+1} - A_i|}{\frac{1}{M}\sum A_i}$ (returns `null` if $<3$ voiced cycles).
- **Harmonics-to-Noise Ratio (HNR)**: Logarithmic ratio between periodic autocorrelation peak $r_{\text{max}}$ and residual aperiodic noise floor: $\text{HNR}_{\text{dB}} = 10 \log_{10}\left(\frac{r_{\text{max}}}{r_0 - r_{\text{max}} + 10^{-9}}\right)$.

### 5. Prosodic Speech Dynamics
- **Voiced Ratio**: Proportion of speech frames that exhibit voiced periodic excitation.
- **Pause / Silence Ratio**: Fraction of frames falling below silence energy threshold ($-50\text{ dBFS}$).
- **Speaking Activity Ratio**: Fraction of recording containing active speech ($1.0 - \text{Pause Ratio}$).
- **Voiced Segments**: Count of contiguous voiced frame blocks and their average duration in seconds.
- **Variability Coefficients**: Coefficients of variation for frame energy ($\text{std}/\text{mean}$) and pitch.

---

## REST API Endpoints

### Feature Extraction (`POST /api/features/extract`)
Accepts `multipart/form-data` audio upload, decodes and standardizes in memory, and returns comprehensive acoustic metrics.

#### Request Example (cURL)
```bash
curl -X POST "http://localhost:8000/api/features/extract" \
     -H "Accept: application/json" \
     -F "file=@voice_sample.wav"
```

#### Response Example (JSON)
```json
{
  "success": true,
  "processing_time_ms": 38.4,
  "features": {
    "audio_duration_seconds": 4.82,
    "time_domain": {
      "rms_db": -18.4,
      "peak_db": -1.2,
      "zero_crossing_rate": 0.082,
      "energy_mean": 0.0152,
      "energy_std": 0.0118,
      "energy_p10": 0.0012,
      "energy_p25": 0.0041,
      "energy_p50": 0.0114,
      "energy_p75": 0.0241,
      "energy_p90": 0.0382
    },
    "spectral": {
      "centroid_hz": 1650.4,
      "bandwidth_hz": 1220.8,
      "rolloff_hz": 3400.0,
      "flatness": 0.0421,
      "entropy": 0.6854,
      "flux": 0.1284,
      "low_energy_ratio": 0.582,
      "mid_energy_ratio": 0.338,
      "high_energy_ratio": 0.080
    },
    "mfcc": {
      "coefficients": 20,
      "means": [-12.4, 4.2, -1.8, 0.9, -0.4, 0.8, -0.2, 0.5, -0.1, 0.3, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.0, 0.0],
      "stds": [3.2, 2.1, 1.8, 1.4, 1.2, 1.1, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3]
    },
    "pitch": {
      "f0_mean_hz": 142.5,
      "f0_median_hz": 139.8,
      "f0_std_hz": 24.2,
      "f0_min_hz": 95.0,
      "f0_max_hz": 210.0,
      "f0_p10_hz": 110.0,
      "f0_p25_hz": 125.0,
      "f0_p75_hz": 160.0,
      "f0_p90_hz": 182.0,
      "voiced_ratio": 0.682
    },
    "voice_quality": {
      "jitter": 0.0124,
      "shimmer": 0.0382,
      "hnr_db": 18.5,
      "harmonic_energy_ratio": 0.824,
      "noise_energy_estimate": 0.176
    },
    "prosody": {
      "voiced_ratio": 0.682,
      "pause_ratio": 0.184,
      "speaking_ratio": 0.816,
      "voiced_segment_count": 7,
      "avg_voiced_duration_s": 0.384,
      "energy_variability": 0.621,
      "f0_variability": 0.170
    }
  },
  "explainability": {
    "spectral_centroid": "Center of mass of the audio spectrum in Hz, reflecting perceived brightness or sharpness of sound.",
    "jitter": "Relative cycle-to-cycle perturbation in fundamental period, reflecting vocal fold stability.",
    "shimmer": "Relative cycle-to-cycle perturbation in peak amplitude, reflecting vocal intensity stability.",
    "hnr_db": "Harmonics-to-Noise Ratio in dB; quantifies relative energy of periodic harmonics versus aperiodic noise."
  }
}
```

---

## Quick Start Guide

### Launch All Services (Full Stack)
```powershell
.\start_all.ps1
```

### Run Full Test Suite (38 Tests)
```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```

### Build Frontend Application
```powershell
cd frontend
npm run build
```
