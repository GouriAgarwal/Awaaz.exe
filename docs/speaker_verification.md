# VoiceShield — Speaker Identity Verification Subsystem

---

## 1. What Speaker Verification Means
Speaker Verification is a **1:1 biometric identity matching task** designed to answer the fundamental question:
> **“Does this speech utterance belong to the claimed or enrolled speaker identity?”**

Unlike speaker identification (a 1:N classification search across an entire speaker database), speaker verification compares a single input sample against a specific enrolled reference profile (an acoustic-biometric centroid vector) to determine whether both audio signals originated from the same vocal tract and articulatory mechanism.

---

## 2. Why Speaker Verification is Different from Synthetic-Voice Detection
Speaker Verification and Synthetic Voice Detection operate across **orthogonal security dimensions**:

| Attribute | Speaker Verification (Step 8) | Synthetic Voice Detection (AASIST / Step 6-7) |
| :--- | :--- | :--- |
| **Core Question** | *“WHO is speaking?”* | *“HOW was the speech generated?”* |
| **Analyzed Dimension** | Vocal tract anatomy, pitch habitus, formant ratios | Waveform continuity, phase consistency, spectral artifacts |
| **Target Phenomenon** | Acoustic identity / Speaker similarity | Synthetic generation / Vocoder artifacts / Splicing |
| **Output Metric** | Cosine similarity $\in [-1.0, 1.0]$ | Spoof posterior probability $\in [0.0, 1.0]$ |
| **Vulnerability** | Susceptible to voice cloning (clones sound like target) | Blind to speaker identity (cannot verify authorization) |

> [!IMPORTANT]
> A high-quality AI voice clone will often produce **high speaker similarity** because it accurately mimics the acoustic formants and pitch of the victim. Therefore, **speaker verification alone cannot detect voice cloning**. Holistic defense requires combining speaker verification with independent synthetic-voice detection.

---

## 3. Pretrained Model Provenance

VoiceShield integrates the official pretrained **SpeechBrain ECAPA-TDNN** model trained on the massive VoxCeleb 1 & VoxCeleb 2 benchmarks.

| Property | Value |
| :--- | :--- |
| **Model Identifier** | `speechbrain_ecapa_tdnn_voxceleb` |
| **Model Name** | `SpeechBrain ECAPA-TDNN (VoxCeleb)` |
| **Official Architecture** | ECAPA-TDNN (Conv1D + SE-Res2Net + MFA + Attentive Statistics Pooling) |
| **Source Repository** | [https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| **Checkpoint URL** | `https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt` |
| **Checkpoint SHA-256** | `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2` |
| **Parameter Count** | `20,767,552` learnable parameters |
| **Embedding Dimension** | `192` dimensions (float32) |
| **Training Benchmark** | VoxCeleb 1 + VoxCeleb 2 (7,205 unique speakers) |
| **License** | Apache-2.0 (SpeechBrain Community) |
| **Scientific Status** | `SPEAKER_MODEL_INTEGRATED_NOT_YET_VALIDATED` |

---

## 4. Embedding Generation & Preprocessing

The speaker embedding pipeline converts raw speech into a compact 192-dimensional vector:
```
Raw Audio (WAV/MP3/WebM/FLAC)
  ↓
Step 2 Audio Ingestion (16 kHz mono float32 PCM, normalized)
  ↓
80-Channel Log Mel-Filterbanks (25ms Hann window, 10ms hop stride)
  ↓
Sentence-Level Mean Normalization (Fbank - mean(Fbank))
  ↓
ECAPA-TDNN Neural Network Forward Pass
  ↓
L2 Normalization: e_norm = e / max(||e||_2, 1e-12)
  ↓
192-Dimensional Unit Biometric Vector
```

1. **Deterministic Execution**: The encoder runs deterministically in evaluation mode (`eval()`), ensuring identical outputs for identical acoustic inputs.
2. **Finite Value Guarantees**: All embeddings are validated against non-finite values (NaN / Inf) with numerical epsilon clamping.
3. **Consistent Normalization**: Embeddings are strictly L2-normalized upon extraction, ensuring valid inner product geometric semantics.

---

## 5. Enrollment Subsystem

Speaker enrollment captures the acoustic fingerprint of an authorized identity:
- **Audio Duration Guidance**: Recommended 10–20 seconds of continuous or multi-sample speech.
- **Multi-Sample Ingestion**: Users can upload multiple shorter utterances rather than one continuous file.
- **Centroid Aggregation**:
  $$\mathbf{c} = \frac{1}{M} \sum_{i=1}^M \mathbf{e}_i \quad \implies \quad \mathbf{c}_{\text{enrolled}} = \frac{\mathbf{c}}{\|\mathbf{c}\|_2}$$
  Averaging multi-sample unit embeddings dampens phonetic and background variance, producing a robust identity centroid.
- **Quality Gates**: Samples $< 1.0\text{s}$, silent signals ($\text{RMS} < -60\text{ dBFS}$), or heavily clipped signals are rejected.

---

## 6. Verification Subsystem

Verification compares an incoming speech sample against an enrolled reference profile:
```
Incoming Verification Audio (≥ 0.5s)
  ↓
Audio Decoding & Preprocessing (16 kHz float32)
  ↓
ECAPA-TDNN Embedding: v_emb (192-dim, L2-normalized)
  ↓
Cosine Similarity with Enrolled Centroid: dot(c_enrolled, v_emb)
  ↓
Decision Engine (MATCH / NON_MATCH / UNCERTAIN)
```

---

## 7. Cosine Similarity Semantics

For L2-normalized vectors $\mathbf{c}$ and $\mathbf{v}$:
$$\text{Cosine Similarity} = \mathbf{c} \cdot \mathbf{v} = \sum_{d=1}^{192} c_d v_d \quad \in [-1.0, 1.0]$$

> [!CAUTION]
> **Cosine similarity is NOT an identity probability.**  
> A score of `0.85` means geometric proximity in high-dimensional feature space, NOT an "85% probability that the caller is genuine." VoiceShield strictly reports similarity scores as raw mathematical metrics unless calibrated on target verification trials.

---

## 8. Thresholding & Operating Points

The decision engine applies configurable thresholds:
- **Provisional Threshold ($\theta = 0.65$)**: Used for interface and prototype testing with uncertainty margin $\pm 0.04$.
  - $\text{Similarity} \ge 0.69 \implies \text{MATCH}$ (High Confidence)
  - $0.65 \le \text{Similarity} < 0.69 \implies \text{MATCH}$ (Medium Confidence)
  - $0.61 \le \text{Similarity} < 0.65 \implies \text{UNCERTAIN}$ (Low Confidence)
  - $\text{Similarity} < 0.61 \implies \text{NON\_MATCH}$
- **Operating Points for Production Deployment**:
  1. **Equal Error Rate (EER)**: Minimizes $|\text{FAR} - \text{FRR}|$.
  2. **High Security / Low-FAR ($\text{FAR} \le 1.0\%$ or $\le 0.1\%$)**: Prioritizes rejecting impostors in high-risk banking / authorization contexts.
  3. **High Usability / Low-FRR ($\text{FRR} \le 1.0\%$)**: Prioritizes minimizing false rejections for frictionless user experience.

---

## 9. Calibration Framework

Speaker verification similarity distributions require empirical score calibration before probability claims can be rendered:
- **Status**: `NOT_CALIBRATED` (Default)
- **Supported Future Calibration Methods**:
  - **Platt Scaling**: $P(\text{Target} \mid s) = \frac{1}{1 + \exp(A \cdot s + B)}$
  - **Isotonic Regression**: Non-parametric piecewise constant monotonic mapping.
  - **Temperature Scaling**: Logit scaling against empirical ground truth.

---

## 10. Evaluation Methodology & Metrics

When a labeled benchmark trial dataset is supplied, VoiceShield computes:
1. **Target Trials**: Utterances where claim speaker == actual speaker.
2. **Non-Target Trials**: Utterances where claim speaker $\ne$ actual speaker (impostor trials).
3. **False Acceptance Rate (FAR)**: $\text{FAR} = \frac{\text{False Accepts}}{\text{Non-Target Trials}}$
4. **False Rejection Rate (FRR)**: $\text{FRR} = \frac{\text{False Rejects}}{\text{Target Trials}}$
5. **True Acceptance Rate (TAR)**: $\text{TAR} = 1.0 - \text{FRR}$
6. **Equal Error Rate (EER)**: The threshold where $\text{FAR} = \text{FRR}$.
7. **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
8. **Subgroup & Robustness Stratification**: Metrics across gender, language, accent, audio codec, and acoustic noise conditions.

---

## 11. Biometric Privacy Protection

Biometric voice data is sensitive personal information. VoiceShield enforces:
1. **Zero Raw Audio Persistence**: Voice recordings are processed in volatile memory and immediately discarded. No audio buffers or raw waveforms are written to disk, database, or logs.
2. **Zero Embedding Vector Logging**: 192-dimensional numerical vectors are omitted from logs and client responses.
3. **Sanitized Profile Metadata**: The `GET /api/speaker/profiles` endpoint returns only administrative metadata (`profile_id`, `sample_count`, `duration_seconds`).

---

## 12. Known Scientific Limitations

1. **Acoustic Environment & Noise**: Reverb, background speech, and poor microphone transducers degrade embedding fidelity.
2. **Utterance Duration**: Utterances $< 1.5\text{s}$ yield higher variance embeddings than longer, phonetically rich speech.
3. **Channel & Codec Mismatch**: Audio recorded via cellular telephony (AMR-WB / G.711) compared against studio microphones can experience embedding distribution shift.
4. **Vocal Pathologies & Emotional Stress**: Illness, laryngitis, or acute stress alter pitch and vocal tract resonant frequencies.
5. **Vulnerability to Synthetic Clones**: An authentic neural clone trained on victim voice samples will produce high cosine similarity.

---

## 13. Domain Shift Considerations

ECAPA-TDNN was trained on YouTube interview audio from the VoxCeleb dataset. In real-world call center or mobile microphone deployments, domain shift may shift similarity baselines. System administrators should benchmark on representative target audio before locking production thresholds.

---

## 14. Dataset Manifest Requirements

To execute reproducible offline evaluation, supply a trial manifest (`.csv` or `.jsonl`):
```csv
trial_id,enrollment_audio_path,verification_audio_path,enrollment_speaker_id,verification_speaker_id,trial_type,generator,language,accent,gender,codec,noise_condition
trial_001,/data/spk1_enr.wav,/data/spk1_ver.wav,spk_01,spk_01,target,,en,us,female,pcm,clean
trial_002,/data/spk1_enr.wav,/data/spk2_ver.wav,spk_01,spk_02,non_target,,en,uk,male,mp3_128k,clean
```

---

## 15. Cross-Modal Fusion Architecture (Future Roadmap)

In future steps, VoiceShield will combine the **Step 8 Speaker Identity** result and the **Step 6-7 AASIST Synthetic Detector** result:

```
                      ┌────────────────────────────────────────┐
                      │             Incoming Audio             │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                      ┌────────────────────────┐
     │  Speaker Verification  │                      │   Synthetic Detector   │
     │      (ECAPA-TDNN)      │                      │        (AASIST)        │
     └────────────┬───────────┘                      └────────────┬───────────┘
                  │                                               │
      Speaker Match Score (0.89)                     Synthetic Score (0.92)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │ Impersonation Engine    │
                             │ (Target + Clone Attack) │
                             └─────────────────────────┘
```

| Speaker Verification | Synthetic Voice Detection | Security Interpretation |
| :--- | :--- | :--- |
| **MATCH** | **NATURAL** | **Authorized Legitimate Speaker** |
| **MATCH** | **SYNTHETIC** | **Targeted Voice Cloning Impersonation Attack** |
| **NON_MATCH** | **NATURAL** | **Unknown Third-Party Speaker** |
| **NON_MATCH** | **SYNTHETIC** | **Untargeted Synthetic Impostor** |

---

*End of Speaker Verification Scientific Specification.*
