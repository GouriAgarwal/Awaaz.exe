# VoiceShield — Hackathon Solution Guide

## "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks"

**VoiceShield** is an autonomous, multi-layer voice integrity, biometric verification, and zero-trust fraud prevention platform. Designed for enterprise core banking, contact center telecommunications, and corporate security teams, VoiceShield neutralizes generative AI voice cloning attacks *in real time* before high-value financial movements or privileged security actions can execute.

---

## 1. The Threat Landscape

Generative voice synthesis (diffusion vocoders, neural codec models, and zero-shot voice cloning) allows adversaries to replicate a target's vocal timbre, prosody, and accent using as little as 3 seconds of reference audio. 

Common high-impact attack vectors include:
1. **CEO Business Email & Voice Compromise (BVC)**: Cloned voices of C-suite executives instructing treasury departments to authorize urgent multi-million offshore wire transfers.
2. **Family Emergency UPI & Wire Extortion**: Cloned voices of children or family members calling distress lines demanding instantaneous peer-to-peer transfers.
3. **Contact Center Social Engineering**: Cloned synthetic voices calling customer support agents to execute unauthorized SIM swaps, credential resets, and two-factor bypasses.
4. **IT Helpdesk Privileged Takeover**: Synthetic administrator voices requesting root/administrative credentials to critical database infrastructure.

Traditional voice biometrics (acoustic voiceprints alone) fail because generative clones match the spectral profile of the legitimate speaker. Conversely, standalone synthetic voice detectors fail because compression and channel noise create high false-alarm rates. 

**VoiceShield solves this by combining 5 orthogonal inspection layers into an explainable, deterministic risk engine backed by an automated prevention state machine.**

---

## 2. 5-Layer Multi-Vector Defense Architecture

VoiceShield processes live audio streams in 500ms sliding windows with sub-second decision latency (~118ms), evaluating five distinct signal domains:

```
               Live VoIP / SIP / Audio Stream (16kHz PCM)
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ▼                           ▼                           ▼
[Layer 1: AASIST]       [Layer 2: ECAPA-TDNN]       [Layer 3: Acoustics]
Synthetic Detection      Speaker Verification        Spectral Centroid & SNR
(Vocoder Artifacts)      (192-dim Cosine Sim)        (VoIP Distortion)
     │                           │                           │
     └─────────────┬─────────────┴─────────────┬─────────────┘
                   ▼                           ▼
          [Layer 4: Prosody]          [Layer 5: Context]
          Pitch Jitter / Shimmer      Urgency & Financial Intent
          (Micro-tremor Analysis)     (NLP Pretext Analysis)
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
              [Dynamic Impersonation Risk Engine]
              Composite Threat Score (0–100) & Band
                                 │
                                 ▼
             [Automated Prevention Decision Engine]
             Policy Profiles: Banking, Telecom, Enterprise
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
 [BLOCKED]                   [PAUSED]                    [ALLOWED]
 Immediate Asset Hold    Step-Up MFA / Callback      Audit Stream Logged
 SOC SIEM Dispatch       Out-of-Band Verification    Call Continues
```

### Layer 1: Pretrained AASIST Synthetic Voice Detection
- **Model**: Graph Neural Network with Heterogeneous Spectral Graph Attention (AASIST).
- **Function**: Inspects high-frequency sub-bands and phase relationships for vocoder synthesis artifacts characteristic of diffusion and autoregressive audio models.
- **Output**: Calibrated synthetic suspicion probability ($S_{\text{synth}} \in [0.0, 1.0]$).

### Layer 2: SpeechBrain ECAPA-TDNN Biometric Verification
- **Model**: 192-dimensional Emphasized Channel Attention TDNN (ECAPA-TDNN).
- **Function**: Extracts speaker identity embeddings from audio frames and computes cosine similarity against enrolled biometric profiles (e.g. CEO voice profile VP-9021).
- **Output**: Biometric similarity score ($S_{\text{sim}} \in [-1.0, 1.0]$) with explicit threshold disclosure (0.65 match threshold).

### Layer 3: Acoustic & Spectral Integrity Analysis
- **Function**: Computes time-domain RMS, signal-to-noise ratio (SNR), zero-crossing rate (ZCR), spectral centroid, and rolloff.
- **Purpose**: Distinguishes true neural synthesis anomalies from telecommunication codec compression (e.g., G.711, Opus, AMR-WB) and channel noise.

### Layer 4: Prosody & Behavioral Dynamics Engine
- **Function**: Measures fundamental pitch frequency ($F_0$), pitch variance, vocal jitter, shimmer, and syllabic speech rate.
- **Signature Detection**: Natural human vocal cords produce involuntary micro-tremors and natural expressive pitch drift. Synthesized audio exhibits abnormally flat pitch contours ($< 18 \text{ Hz}$ variance) and unnaturally uniform syllabic timing.

### Layer 5: Contextual Fraud & Intent Intelligence
- **Function**: Natural language intent extraction analyzing speech transcripts for high-risk social engineering markers.
- **Cues Tracked**: Authority pressure, extreme urgency ("transfer immediately", "in a meeting"), high-value financial requests (₹/$, wire instructions), and secrecy demands ("do not share OTP").

---

## 3. Dynamic Impersonation Risk Engine

VoiceShield avoids black-box averaging by applying a deterministic, explainable heuristic fusion formula that penalizes simultaneous high synthetic probability and high biometric match:

$$\text{Impersonation Risk} = 0.40 \cdot S_{\text{synth}} + 0.35 \cdot (1.0 - S_{\text{sim\_clamped}}) + 0.15 \cdot S_{\text{prosody}} + 0.10 \cdot S_{\text{context}}$$

### Threat Band Classification
| Score Band | Threat Level | Primary Operational Response |
| :--- | :--- | :--- |
| **0 – 24** | `LOW` | **ALLOWED**: Acoustic integrity verified. Continuous passive monitoring. |
| **25 – 49** | `MEDIUM` | **VERIFICATION REQUIRED**: Step-up security challenge before sensitive access. |
| **50 – 74** | `HIGH` | **PAUSED**: Action held. Out-of-band callback or push MFA required. |
| **75 – 100** | `CRITICAL` | **BLOCKED**: Immediate transaction halt. Payment switch intercept & SOC alert. |

---

## 4. Automated Prevention & Response Workflow

VoiceShield does not stop at generating a risk score—it demonstrates **what happens after detection** through an automated policy engine and state machine.

### State Lifecycle
```
[INITIAL] ──> [MONITORING] ──> [WARNING] ──> [VERIFICATION_REQUIRED] ──> [PAUSED] ──> [BLOCKED]
                                                                             │             │
                                                                             ▼             ▼
                                                                        [RESOLVED]   [SUPERVISOR
                                                                        (MFA/Call)     OVERRIDE]
```

### Critical Security Invariant
> **Unblocking Protection**: A transaction in `BLOCKED` status cannot be cleared by automated callback or standard SMS MFA (which an attacker could intercept via SIM swap). Clearing a hard-blocked attack requires explicit human-in-the-loop **Supervisor Exception Authorization** or formal SOC incident resolution.

---

## 5. 6 Hackathon Demonstration Scenarios

VoiceShield includes 6 pre-configured end-to-end attack and defense scenarios accessible directly from the command center:

### Scenario 1: CEO Urgent Wire Transfer Attack
- **Pretext**: AI clone of CEO Vikram Malhotra calling treasury demanding urgent ₹10,00,000 ($150,000) wire transfer for an unannounced acquisition.
- **Detection**: AASIST flags synthetic speech (0.96); ECAPA matches CEO profile (0.89); Context flags extreme urgency.
- **Prevention Outcome**: **BLOCKED (94/100 CRITICAL)**. Transaction intercepted at payment switch; SOC alert dispatched.

### Scenario 2: Bank Customer Legitimate Transfer
- **Pretext**: Enrolled retail customer calling customer support to execute routine ₹25,000 balance transfer.
- **Detection**: Natural human speech dynamics; verified biometric match; no urgency or anomaly flags.
- **Prevention Outcome**: **ALLOWED (12/100 LOW)**. Transaction proceeds seamlessly.

### Scenario 3: Replay Attack on Enrolled Account
- **Pretext**: Attacker plays back a recorded sample of a legitimate customer to approve a ₹50,000 transaction.
- **Detection**: Strong biometric match (0.82) but acoustic channel distortion, flat prosody, and missing ambient room resonance.
- **Prevention Outcome**: **VERIFICATION REQUIRED (38/100 MEDIUM)**. Out-of-band challenge required before funds release.

### Scenario 4: Customer Care Social Engineering
- **Pretext**: Synthetic voice agent calling telecom support requesting an urgent PIN and password reset on a high-value account.
- **Detection**: Synthetic voice indicators (0.86) + credential modification intent.
- **Prevention Outcome**: **PAUSED (72/100 HIGH)**. Account change halted; mandatory push MFA dispatched to customer device.

### Scenario 5: Family Emergency Impersonation (UPI Scam)
- **Pretext**: Cloned voice of customer's child claiming urgent medical distress and demanding instant ₹1,50,000 UPI transfer.
- **Detection**: Synthetic voice synthesis (0.82) + extreme emotional duress cues.
- **Prevention Outcome**: **PAUSED (68/100 HIGH)**. Immediate hold applied; automated out-of-band telephone callback triggered.

### Scenario 6: IT Support Privileged Access Request
- **Pretext**: Synthetic clone posing as senior IT systems engineer requesting root database credentials during off-hours.
- **Detection**: Synthetic vocoder signature (0.92) targeting administrative security credentials.
- **Prevention Outcome**: **BLOCKED (89/100 CRITICAL)**. Credential distribution blocked; active session terminated.

---

## 6. Multilingual & Indian Regional Context

Financial voice cloning attacks in India exploit regional languages, colloquial phrasing, and authority pressures. VoiceShield demonstrates compatibility across 6 languages:

1. **Hindi (हिंदी)**: *"तत्काल 10 लाख रुपये ट्रांसफर करें, मैं बेहद जरूरी बोर्ड मीटिंग में हूं।"*
2. **Marathi (मराठी)**: *"मी विक्रम बोलतोय. कंपनीच्या खात्यातून तत्काळ ₹१०,००,००० पाठवा."*
3. **Tamil (தமிழ்)**: *"நான் தலைமை நிர்வாக அதிகாரி பேசுகிறேன். உடனடியாக ₹10,00,000 பணப்பரிமாற்றம் செய்யவும்."*
4. **Telugu (తెలుగు)**: *"నేను సిఇఒ విక్రమ్ మాట్లాడుతున్నాను. వెంటనే ₹10,00,000 వైర్ బదిలీ చేయండి."*
5. **Bengali (বাংলা)**: *"আমি বিক্রম বলছি। জরুরি ভিত্তিতে অবিলম্বে ১০,০০,০০০ টাকা স্থানান্তর করুন।"*
6. **English**: *"Wire $150,000 immediately for the acquisition. I am in a board meeting."*

> **Scientific Disclosure**: VoiceShield's AASIST and ECAPA-TDNN acoustic feature extractors operate on language-agnostic physical characteristics (vocal tract resonance, vocoder phase distortion, micro-prosody). Full production deployment requires regional accent validation across representative dialect datasets.

---

## 7. Enterprise Integration Touchpoints

| Touchpoint | Target System | Integration Protocol | Mechanism |
| :--- | :--- | :--- | :--- |
| **Core Banking** | Finacle, Temenos, UPI Switches | ISO 8583 / REST Webhooks | Intercepts transaction auth before debit settlement. Emits Rejection Code 91 (Fraud Suspicion). |
| **Telecommunications** | Asterisk, FreeSWITCH, Genesys | RFC 3550 RTP Forking / WebSockets | Media proxy forks audio packets to VoiceShield without adding latency to the conversation. |
| **Enterprise UC** | Microsoft Teams, Zoom, Webex | Graph API Real-time Audio Bot | Passive bot joins executive conference calls and alerts security channels on clone detection. |
| **Enterprise SOC** | Splunk, Microsoft Sentinel, Elastic | Common Event Format (CEF) Syslog | Dispatches real-time incident alerts and triggers automated SOAR playbooks. |

---

## 8. Privacy, Security & Scientific Integrity Safeguards

1. **Zero Raw Audio Retention**: The incident timeline and audit trail record only privacy-preserving metadata (timestamps, risk scores, event classifications, and decision outcomes). Raw PCM audio buffers, base64 payloads, and biometric embeddings are never logged or stored.
2. **Regulatory Compliance**: Designed in accordance with India's Digital Personal Data Protection (DPDP) Act 2023, GDPR Article 9 (biometric data safeguards), and ISO/IEC 27001 security controls.
3. **Model Provenance**: Built on verified open research models (AASIST, SpeechBrain ECAPA-TDNN) with explicit disclaimers separating genuine benchmark results from live simulated demonstrations.
