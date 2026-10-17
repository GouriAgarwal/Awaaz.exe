/**
 * VoiceShield API and Frontend Type Definitions
 */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface SystemStatusResponse {
  backend: string;
  version: string;
  environment: string;
}

export type ConnectionStatus = 'connecting' | 'online' | 'offline';

export interface DashboardState {
  status: ConnectionStatus;
  health: HealthResponse | null;
  system: SystemStatusResponse | null;
  lastChecked: Date | null;
  latencyMs: number | null;
  error: string | null;
}

export interface AudioMetadataInfo {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  original_format: string;
  original_sample_rate: number;
  original_channels: number;
}

export interface AudioQualityMetricsInfo {
  rms_db: number;
  peak_db: number;
  clipping_ratio: number;
  silence_ratio: number;
  quality: 'good' | 'warning' | 'invalid';
  notes: string[];
}

export interface AudioInspectResponse {
  success: boolean;
  audio: AudioMetadataInfo;
  quality: AudioQualityMetricsInfo;
}

// ---------------------------------------------------------------------------
// Step 3: Acoustic & Spectral Feature Extraction Types
// ---------------------------------------------------------------------------

export interface TimeDomainFeaturesInfo {
  rms_db: number;
  peak_db: number;
  zero_crossing_rate: number;
  energy_mean: number;
  energy_std: number;
  energy_p10: number;
  energy_p25: number;
  energy_p50: number;
  energy_p75: number;
  energy_p90: number;
}

export interface SpectralFeaturesInfo {
  centroid_hz: number;
  bandwidth_hz: number;
  rolloff_hz: number;
  flatness: number;
  entropy: number;
  flux: number;
  low_energy_ratio: number;
  mid_energy_ratio: number;
  high_energy_ratio: number;
}

export interface MFCCFeaturesInfo {
  coefficients: number;
  means: number[];
  stds: number[];
}

export interface PitchFeaturesInfo {
  f0_mean_hz: number | null;
  f0_median_hz: number | null;
  f0_std_hz: number | null;
  f0_min_hz: number | null;
  f0_max_hz: number | null;
  f0_p10_hz: number | null;
  f0_p25_hz: number | null;
  f0_p75_hz: number | null;
  f0_p90_hz: number | null;
  voiced_ratio: number;
}

export interface VoiceQualityFeaturesInfo {
  jitter: number | null;
  shimmer: number | null;
  hnr_db: number | null;
  harmonic_energy_ratio: number | null;
  noise_energy_estimate: number | null;
}

export interface ProsodyFeaturesInfo {
  voiced_ratio: number;
  pause_ratio: number;
  speaking_ratio: number;
  voiced_segment_count: number;
  avg_voiced_duration_s: number;
  energy_variability: number;
  f0_variability: number | null;
}

export interface AcousticFeaturesInfo {
  audio_duration_seconds: number;
  time_domain: TimeDomainFeaturesInfo;
  spectral: SpectralFeaturesInfo;
  mfcc: MFCCFeaturesInfo;
  pitch: PitchFeaturesInfo;
  voice_quality: VoiceQualityFeaturesInfo;
  prosody: ProsodyFeaturesInfo;
}

export interface FeatureExtractionResponse {
  success: boolean;
  processing_time_ms: number;
  features: AcousticFeaturesInfo;
  explainability: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Step 4: AI-Based Synthetic / Cloned Voice Detection Types
// ---------------------------------------------------------------------------

export type ScoreType = 'uncalibrated_model_score' | 'calibrated_probability' | 'heuristic_fallback_score';

export type ClassificationLabel = 'NATURAL' | 'SYNTHETIC' | 'UNCERTAIN' | 'MODEL_UNAVAILABLE';

export interface DetectorMetadataInfo {
  model_name: string;
  model_type: string;
  architecture: string;
  checkpoint_or_source: string;
  license: string;
  expected_sample_rate: number;
  window_size_sec: number;
  window_hop_sec: number;
  score_type: ScoreType;
  score_interpretation: string;
  scientific_disclaimer: string;
  device: string;
  is_fallback: boolean;
}

export interface WindowScoreInfo {
  window_index: number;
  start_sec: number;
  end_sec: number;
  raw_score: number;
  synthetic_score: number;
  label: ClassificationLabel;
}

export interface DetectionResultInfo {
  detector_metadata: DetectorMetadataInfo;
  classification: ClassificationLabel;
  score: number;
  score_type: ScoreType;
  confidence_band: string;
  thresholds_applied: Record<string, number>;
  window_scores: WindowScoreInfo[];
  aggregation_method: string;
  inference_latency_ms: number;
  audio_duration_sec: number;
  total_windows: number;
  warnings: string[];
}

export interface DetectionResponse {
  success: boolean;
  data: DetectionResultInfo;
  message: string;
}

// ---------------------------------------------------------------------------
// Step 5: Scientific Validation and Model Provenance Types
// ---------------------------------------------------------------------------

export type PretrainedStatus =
  | 'UNTRAINED_NEURAL_BASELINE'
  | 'PRETRAINED_NOT_YET_VALIDATED'
  | 'VALIDATED_PRETRAINED'
  | 'HEURISTIC_FALLBACK';

export type CalibrationStatus =
  | 'NOT_CALIBRATED'
  | 'CALIBRATED'
  | 'CALIBRATION_FAILED'
  | 'CALIBRATED_ISOTONIC'
  | 'CALIBRATED_PLATT';


export interface ModelProvenanceInfo {
  detector_name: string;
  architecture: string;
  checkpoint_identifier: string;
  source_repository: string;
  source_url: string;
  revision: string;
  license: string;
  training_dataset: string;
  intended_task: string;
  sample_rate: number;
  input_format: string;
  output_classes: string[];
  score_semantics: string;
  pretrained: boolean;
  validated: boolean;
  pretrained_status: PretrainedStatus;
  calibration_status: CalibrationStatus;
  notes: string;
}

export interface ConfusionMatrixInfo {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface ThresholdPointInfo {
  threshold: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  specificity: number;
  fpr: number;
  fnr: number;
  f1: number;
}

export interface EvaluationMetricsSummaryInfo {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1_score: number;
  fpr: number;
  fnr: number;
  roc_auc: number | null;
  eer: number | null;
  eer_threshold: number | null;
  confusion_matrix: ConfusionMatrixInfo;
  threshold_sweep: ThresholdPointInfo[];
}

export interface LatencySummaryInfo {
  cold_start_ms: number;
  warm_avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  total_audio_duration_sec: number;
  real_time_factor: number;
}

export interface EvaluationReportInfo {
  evaluation_status: 'NOT_RUN' | 'COMPLETED' | 'NOT_VALIDATED' | 'BLOCKED' | 'ERROR';
  detector_name: string;
  model_id?: string;
  checkpoint_sha256?: string;
  provenance: ModelProvenanceInfo;
  dataset_name: string;
  sample_count: number;
  real_count: number;
  synthetic_count: number;
  dataset?: Record<string, any>;
  windowing?: Record<string, any>;
  metrics: EvaluationMetricsSummaryInfo | null;
  threshold_analysis?: Record<string, any>;
  calibration?: Record<string, any>;
  subgroups?: Record<string, any>;
  robustness?: Record<string, any>;
  speaker_leakage?: Record<string, any>;
  latency: LatencySummaryInfo | null;
  calibration_status: string;
  speaker_leakage_detected: boolean;
  environment_info: Record<string, string>;
  limitations?: string[];
  scientific_disclaimer: string;
  notes_or_reason: string;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Step 8: Speaker Identity Verification Types
// ---------------------------------------------------------------------------

export type SpeakerDecisionType = 'MATCH' | 'NON_MATCH' | 'UNCERTAIN';
export type ConfidenceBandType = 'LOW' | 'MEDIUM' | 'HIGH';

export interface PrivacyMetadataInfo {
  raw_audio_persisted: boolean;
  embeddings_logged: boolean;
  in_memory_only: boolean;
  policy: string;
}

export interface SpeakerProfileSummaryInfo {
  profile_id: string;
  model_id: string;
  embedding_dimension: number;
  sample_count: number;
  total_audio_duration_seconds: number;
  created_at: string;
  updated_at: string;
  in_memory_only: boolean;
}

export interface SpeakerEnrollmentResponse {
  success: boolean;
  profile_id: string;
  sample_count: number;
  total_audio_duration_seconds: number;
  audio_quality: {
    duration_seconds: number;
    sample_rate: number;
    rms_db: number;
    peak_db: number;
    snr_estimate_db: number;
    clipping_detected: boolean;
    silence_ratio: number;
  };
  privacy: PrivacyMetadataInfo;
  message: string;
  warning?: string | null;
}

export interface SpeakerVerificationResponse {
  profile_id: string;
  model_id: string;
  similarity_score: number;
  score_type: string;
  decision: SpeakerDecisionType;
  confidence_band: ConfidenceBandType;
  threshold: number;
  threshold_version: string;
  calibration_status: string;
  provisional_warning?: string | null;
  duration_seconds: number;
  latency_ms: number;
  audio_quality: {
    duration_seconds: number;
    sample_rate: number;
    rms_db: number;
    peak_db: number;
    snr_estimate_db: number;
    clipping_detected: boolean;
    silence_ratio: number;
  };
  privacy: PrivacyMetadataInfo;
}

export interface SpeakerModelProvenanceInfo {
  model_id: string;
  model_name: string;
  version: string;
  architecture: string;
  source_repository: string;
  checkpoint_url: string;
  checkpoint_sha256: string;
  license: string;
  embedding_dimension: number;
  expected_sample_rate: number;
  parameter_count: number;
  training_dataset: string;
  is_l2_normalized: boolean;
  calibration_status: string;
  scientific_status: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Step 9: Controlled Fusion & Impersonation Risk Types
// ---------------------------------------------------------------------------

export type RiskLevelType = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type RecommendedActionType = 'ALLOW' | 'MONITOR' | 'STEP_UP_VERIFICATION' | 'BLOCK_OR_ESCALATE';
export type EvidencePolarityType = 'SUPPORTS_LEGITIMATE' | 'SUPPORTS_IMPERSONATION' | 'NEUTRAL' | 'DEGRADED_QUALITY';
export type EvidenceImpactType = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface EvidenceItemInfo {
  description: string;
  polarity: EvidencePolarityType;
  impact: EvidenceImpactType;
  source: string;
}

export interface SyntheticSignalSummaryInfo {
  score: number;
  band: string;
  classification: string;
  detector_id: string;
  confidence: string;
}

export interface SpeakerSignalSummaryInfo {
  similarity_score: number;
  band: string;
  decision: string;
  profile_id: string;
  model_id: string;
  confidence: string;
}

export interface ContextualSignalsInfo {
  call_metadata?: Record<string, any>;
  transaction_amount?: number;
  urgency_cues_detected?: boolean;
  channel_flags?: string[];
}

export interface RiskAnalysisResultInfo {
  risk_score: number;
  risk_level: RiskLevelType;
  recommended_action: RecommendedActionType;
  action_rationale: string;
  confidence: ConfidenceBandType;
  confidence_rationale: string;
  primary_scenario: string;
  signals: {
    synthetic: SyntheticSignalSummaryInfo;
    speaker: SpeakerSignalSummaryInfo;
    contextual: ContextualSignalsInfo;
    audio_quality: {
      duration_seconds: number;
      quality_status: string;
    };
  };
  evidence: EvidenceItemInfo[];
  risk_score_type: string;
  calibration_status: string;
  scientific_disclaimer: string;
  evaluation_status: string;
  latency_ms: number;
  timestamp: string;
}

export interface RiskSimulationRequestInfo {
  synthetic_score: number;
  speaker_similarity: number;
  duration_seconds?: number;
  audio_quality_status?: string;
  contextual_signals?: ContextualSignalsInfo;
}

export interface RiskProvenanceResponseInfo {
  fusion_engine: string;
  version: string;
  fusion_type: string;
  calibration_status: string;
  evaluation_status: string;
  synthetic_detector: Record<string, any>;
  speaker_verifier: SpeakerModelProvenanceInfo;
  scientific_disclaimer: string;
}

// ---------------------------------------------------------------------------
// Step 10: Real-Time Streaming Pipeline Types
// ---------------------------------------------------------------------------

export type StreamingConnectionState = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'PROCESSING' | 'ERROR';

export type StreamingEventType =
  | 'STREAM_STARTED'
  | 'STREAM_STOPPED'
  | 'STREAM_ERROR'
  | 'ANALYSIS_WINDOW_COMPLETED'
  | 'RISK_LEVEL_CHANGED'
  | 'HIGH_RISK_DETECTED'
  | 'CRITICAL_RISK_DETECTED'
  | 'SYNTHETIC_SCORE_UPDATED'
  | 'SPEAKER_MATCH_UPDATED'
  | 'AUDIO_QUALITY_DEGRADED'
  | 'HEARTBEAT';

export interface WindowTimingInfo {
  window_index: number;
  window_start_sec: number;
  window_end_sec: number;
  duration_sec: number;
}

export interface StreamingSyntheticSignalInfo {
  score: number;
  score_type: string;
  classification: string;
  confidence_band: string;
  detector_id: string;
  latency_ms: number;
}

export interface StreamingSpeakerSignalInfo {
  status: string;
  similarity_score?: number | null;
  confidence_band?: string | null;
  profile_id?: string | null;
  model_id?: string | null;
  latency_ms?: number | null;
}

export interface RollingRiskMetricsInfo {
  latest_risk_score: number;
  latest_risk_level: string;
  latest_action: string;
  previous_risk_score?: number | null;
  rolling_max_risk: number;
  rolling_mean_risk: number;
  analyzed_windows_count: number;
  consecutive_high_risk_windows: number;
  consecutive_critical_windows: number;
}

export interface StreamEventInfo {
  event_type: StreamingEventType;
  session_id: string;
  timestamp: number;
  message: string;
  timing?: WindowTimingInfo | null;
  synthetic_signal?: StreamingSyntheticSignalInfo | null;
  speaker_signal?: StreamingSpeakerSignalInfo | null;
  risk_score?: number | null;
  risk_level?: string | null;
  recommended_action?: string | null;
  action_rationale?: string | null;
  evidence?: Array<Record<string, any>> | null;
  rolling_metrics?: RollingRiskMetricsInfo | null;
  processing_latency_ms?: number | null;
  processing_lag_ms?: number | null;
  audio_duration_received_sec?: number | null;
}

export interface StreamingSessionSummaryInfo {
  session_id: string;
  state: string;
  created_at: number;
  last_event_at: number;
  sample_rate: number;
  channels: number;
  enrolled_profile_id?: string | null;
  total_chunks_received: number;
  total_samples_received: number;
  audio_duration_received_sec: number;
  analyzed_windows_count: number;
  rolling_metrics: RollingRiskMetricsInfo;
  evaluation_status: string;
  privacy_policy: string;
}

// ---------------------------------------------------------------------------
// Step 11: Production Scientific Validation, Calibration & Security Types
// ---------------------------------------------------------------------------

export type DatasetAvailabilityStatus =
  | 'AVAILABLE_VALIDATED'
  | 'AVAILABLE_WITH_WARNINGS'
  | 'NOT_AVAILABLE_LOCALLY'
  | 'CORRUPTED_OR_INVALID';

export type CalibrationMethod =

  | 'NONE'
  | 'TEMPERATURE_SCALING'
  | 'PLATT_SCALING'
  | 'ISOTONIC_REGRESSION';

export type ScientificEvaluationStatus =
  | 'DATASET_NOT_AVAILABLE'
  | 'NOT_RUN'
  | 'BLOCKED'
  | 'PRETRAINED_NOT_YET_VALIDATED'
  | 'VALIDATED_ON_DEFINED_BENCHMARK'
  | 'NOT_CALIBRATED'
  | 'CALIBRATED_ON_DEFINED_VALIDATION_DISTRIBUTION'
  | 'FUSION_IMPLEMENTED_NOT_YET_VALIDATED'
  | 'FUSION_VALIDATED_ON_DEFINED_DATASET'
  | 'STREAMING_NOT_VALIDATED'
  | 'STREAMING_VALIDATED_ON_DEFINED_TEST_PROTOCOL'
  | 'INSUFFICIENT_DATA';

export interface ConfidenceIntervalInfo {
  metric: string;
  point_estimate: number;
  ci_lower: number;
  ci_upper: number;
  confidence_level: number;
  method: string;
  bootstrap_iterations: number;
}

export interface ThresholdSweepPointInfo {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  fpr: number;
  fnr: number;
  f1: number;
}

export interface ThresholdSweepReportInfo {
  points: ThresholdSweepPointInfo[];
  default_threshold: number;
  best_f1_threshold: number;
  best_f1_value: number;
  eer_threshold: number;
  eer_value: number;
  low_fpr_threshold: number;
  low_fpr_value: number;
}

export interface SubgroupMetricInfo {
  subgroup_dimension: string;
  subgroup_value: string;
  sample_count: number;
  status: 'VALIDATED' | 'INSUFFICIENT_DATA';
  accuracy?: number | null;
  eer?: number | null;
  f1?: number | null;
  note?: string | null;
}

export interface DatasetAuditReportInfo {
  dataset_name: string;
  status: DatasetAvailabilityStatus;
  manifest_path?: string | null;
  total_samples: number;
  real_count: number;
  synthetic_count: number;
  split_counts: Record<string, number>;
  speaker_count: number;
  generator_counts: Record<string, number>;
  has_speaker_leakage: boolean;
  speaker_leakage_notes: string[];
  has_generator_leakage: boolean;
  generator_leakage_notes: string[];
  duplicate_count: number;
  duplicate_files: string[];
  missing_files: string[];
  integrity_notes: string[];
}

export interface CalibrationReportInfo {
  model_name: string;
  calibration_status: CalibrationStatus;
  calibration_method: CalibrationMethod;
  calibration_dataset?: string | null;
  calibration_split?: string | null;
  calibration_timestamp?: string | null;
  model_checksum: string;
  parameters: Record<string, any>;
  expected_calibration_error_before?: number | null;
  expected_calibration_error_after?: number | null;
  scientific_notice: string;
}

export interface AASISTBenchmarkReportInfo {
  evaluation_status: ScientificEvaluationStatus;
  dataset_name?: string | null;
  score_type: string;
  total_evaluated: number;
  confusion_matrix?: { tp: number; tn: number; fp: number; fn: number } | null;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  specificity?: number | null;
  fpr?: number | null;
  fnr?: number | null;
  f1_score?: number | null;
  eer?: number | null;
  threshold_sweep?: ThresholdSweepReportInfo | null;
  confidence_intervals: ConfidenceIntervalInfo[];
  subgroups: SubgroupMetricInfo[];
  model_checksum: string;
  disclaimer: string;
}

export interface StreamingStressTestConditionInfo {
  test_id: string;
  condition_name: string;
  description: string;
  passed: boolean;
  details: Record<string, any>;
  error_message?: string | null;
}

export interface StreamingValidationReportInfo {
  streaming_evaluation_status: ScientificEvaluationStatus;
  total_conditions_tested: number;
  passed_conditions_count: number;
  all_passed: boolean;
  conditions: StreamingStressTestConditionInfo[];
  avg_processing_lag_ms: number;
  max_memory_allocated_mb: number;
  ephemeral_privacy_verified: boolean;
  session_stability_score: number;
  timestamp: string;
}

export interface ModelIntegrityAuditInfo {
  model_name: string;
  weights_path: string;
  expected_sha256: string;
  actual_sha256: string;
  integrity_verified: boolean;
  parameter_count: number;
  device: string;
  status: string;
}

export interface SecurityAuditReportInfo {
  timestamp: string;
  model_integrity: ModelIntegrityAuditInfo[];
  websocket_security: Record<string, any>;
  api_security: Record<string, any>;
  privacy_guarantees: Record<string, any>;
  overall_hardened: boolean;
}

export interface SystemScientificStatusSummaryInfo {
  timestamp: string;
  datasets_status: DatasetAvailabilityStatus;
  aasist_evaluation_status: ScientificEvaluationStatus;
  aasist_calibration_status: CalibrationStatus;
  speaker_evaluation_status: ScientificEvaluationStatus;
  speaker_threshold_status: string;
  fusion_evaluation_status: ScientificEvaluationStatus;
  fusion_score_type: string;
  streaming_evaluation_status: ScientificEvaluationStatus;
  security_hardening_status: string;
  model_integrity_verified: boolean;
  scientific_disclaimer: string;
}

// ---------------------------------------------------------------------------
// Step 12: Production Deployment, Observability & Operational Readiness Types
// ---------------------------------------------------------------------------

export type ReadinessState = 'READY' | 'DEGRADED' | 'NOT_READY';

export interface LivenessResponseInfo {
  status: string;
  timestamp: string;
}

export interface ComponentHealthInfo {
  name: string;
  status: string;
  healthy: boolean;
  details?: Record<string, any>;
  message?: string | null;
}

export interface ReadinessResponseInfo {
  status: ReadinessState;
  components: Record<string, ComponentHealthInfo>;
  timestamp: string;
  strict_integrity_mode: boolean;
  message: string;
}

export interface OperationalStatusResponseInfo {
  application_status: string;
  environment: string;
  uptime_seconds: number;
  active_detector: string;
  speaker_model_available: boolean;
  streaming_available: boolean;
  active_sessions: number;
  max_active_sessions: number;
  model_integrity_status: Record<string, boolean>;
  evaluation_status_summary: Record<string, string>;
  timestamp: string;
}

export interface HttpMetricsInfo {
  total_requests: number;
  status_2xx: number;
  status_4xx: number;
  status_5xx: number;
  average_latency_ms: number;
}

export interface AasistMetricsInfo {
  inference_count: number;
  inference_failure_count: number;
  average_latency_ms: number;
  model_available: boolean;
  integrity_verified: boolean;
  scientific_status: string;
}

export interface EcapaMetricsInfo {
  embedding_extraction_count: number;
  verification_count: number;
  failure_count: number;
  average_latency_ms: number;
  model_available: boolean;
  integrity_verified: boolean;
  operating_threshold_status: string;
}

export interface FusionMetricsInfo {
  fusion_count: number;
  risk_distribution: Record<string, number>;
  action_distribution: Record<string, number>;
  scientific_status: string;
}

export interface StreamingMetricsInfo {
  active_sessions: number;
  total_sessions_created: number;
  total_sessions_closed: number;
  total_chunks_processed: number;
  total_windows_analyzed: number;
  average_window_latency_ms: number;
  max_processing_lag_ms: number;
  rate_limit_events: number;
  message_size_rejections: number;
  timeout_events: number;
  scientific_status: string;
}

export interface ApplicationMetricsResponseInfo {
  timestamp: string;
  uptime_seconds: number;
  http: HttpMetricsInfo;
  aasist: AasistMetricsInfo;
  ecapa: EcapaMetricsInfo;
  fusion: FusionMetricsInfo;
  streaming: StreamingMetricsInfo;
  session_capacity: {
    current_active: number;
    max_capacity: number;
    utilization_percent: number;
  };
}

export interface ModelMonitoringInfo {
  model_id: string;
  display_name: string;
  architecture: string;
  weights_path: string;
  expected_sha256: string;
  actual_sha256: string;
  integrity_status: string;
  checkpoint_verified: boolean;
  availability: string;
  load_count: number;
  inference_count: number;
  inference_failure_count: number;
  average_latency_ms: number;
  scientific_status: string;
}

export interface StreamingOperationalInfo {
  active_sessions: number;
  max_active_sessions: number;
  utilization_percentage: number;
  total_chunks_ingested: number;
  total_windows_analyzed: number;
  rate_limit_count: number;
  buffer_memory_cap_sec: number;
  ephemeral_privacy_policy: string;
  status: string;
}

export interface OperationalErrorInfo {
  error_code: string;
  message: string;
  request_id?: string | null;
  timestamp: string;
}

export interface OperationalErrorSummaryInfo {
  error_counts_by_code: Record<string, number>;
  total_errors_recorded: number;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Step 13: Multi-Layer Voice Integrity & Contextual Risk Intelligence Types
// ---------------------------------------------------------------------------

export type ProsodyClassificationType =
  | 'NATURAL_VARIATION'
  | 'LOW_VARIATION'
  | 'UNUSUAL_PROSODY'
  | 'INSUFFICIENT_AUDIO';

export type ExtendedActionType =
  | 'ALLOW'
  | 'MONITOR'
  | 'STEP_UP_VERIFICATION'
  | 'CALL_BACK_REQUIRED'
  | 'MFA_REQUIRED'
  | 'BLOCK_OR_ESCALATE'
  | 'BLOCK_TRANSACTION_AND_ESCALATE';

export interface MultiLayerEvidenceItemInfo {
  layer: string;
  code: string;
  severity: string;
  message: string;
}

export interface MultiLayerRiskAnalysisResultInfo {
  overall_risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_score_type: string;
  recommended_action: ExtendedActionType;
  action_reason: string;
  enforcement_disclaimer: string;
  synthetic_signal: {
    score: number;
    classification: string;
    confidence_band: string;
    detector_model: string;
    score_type: string;
  };
  speaker_signal: {
    similarity_score: number;
    decision: string;
    confidence_band: string;
    model: string;
    profile_id?: string | null;
    score_type: string;
  };
  prosody_signal: {
    classification: string;
    quality_score: number;
    confidence: string;
    features: {
      pitch_mean_hz?: number | null;
      pitch_std_hz?: number | null;
      pitch_min_hz?: number | null;
      pitch_max_hz?: number | null;
      pitch_range_hz?: number | null;
      pitch_variation_coef?: number | null;
      energy_rms_mean: number;
      energy_rms_std: number;
      energy_dynamic_range_db: number;
      voiced_unvoiced_ratio: number;
      speech_activity_ratio: number;
      pause_ratio: number;
      pause_count: number;
      pause_mean_duration_ms: number;
      pause_max_duration_ms: number;
      speech_rhythm_proxy: number;
      microvariation_jitter_proxy?: number | null;
      energy_delta_mean: number;
    };
  };
  context_signal: {
    context_risk_score: number;
    context_risk_level: string;
    policy_sensitivity: string;
    sensitivity_multiplier: number;
    call_type: string;
    caller_trust: string;
    requested_action: string;
    transaction_amount?: number | null;
    historical_risk: string;
  };
  evidence: MultiLayerEvidenceItemInfo[];
  primary_rationale: string;
  scientific_disclosure: string;
  latency_ms: number;
  privacy_policy: string;
}

export interface AttackScenarioInfo {
  id: string;
  name: string;
  description: string;
  synthetic_score: number;
  speaker_similarity: number;
  prosody_classification: string;
  call_type: string;
  caller_trust: string;
  requested_action: string;
  transaction_amount?: number | null;
  historical_risk: string;
  expected_outcome: {
    risk_level: string;
    recommended_action: string;
  };
}

// ---------------------------------------------------------------------------
// Step 14: Automated Prevention & Incident Response Types
// ---------------------------------------------------------------------------

export type PreventionActionType =
  | 'ALLOW'
  | 'MONITOR'
  | 'SHOW_WARNING'
  | 'PAUSE_SENSITIVE_ACTION'
  | 'REQUIRE_CALLBACK'
  | 'REQUIRE_MFA'
  | 'REQUIRE_SECONDARY_VERIFICATION'
  | 'ESCALATE_TO_SUPERVISOR'
  | 'ESCALATE_TO_SOC'
  | 'BLOCK_TRANSACTION';

export type PreventionStatusType =
  | 'ALLOWED'
  | 'MONITORING'
  | 'WARNING'
  | 'VERIFICATION_REQUIRED'
  | 'PAUSED'
  | 'ESCALATED'
  | 'BLOCKED'
  | 'RESOLVED';

export type SensitiveActionTypeValue =
  | 'FUND_TRANSFER'
  | 'PAYMENT_APPROVAL'
  | 'ACCOUNT_CHANGE'
  | 'PRIVILEGED_ACCESS'
  | 'CONFIDENTIAL_DISCLOSURE'
  | 'EXECUTIVE_INSTRUCTION'
  | 'GENERAL_CALL';

export type VerificationMethodType =
  | 'CALLBACK_TO_REGISTERED_NUMBER'
  | 'MULTI_FACTOR_AUTHENTICATION'
  | 'SUPERVISOR_APPROVAL'
  | 'OUT_OF_BAND_VERIFICATION'
  | 'SECURITY_QUESTION'
  | 'BIOMETRIC_REAUTHENTICATION';

export type PolicyProfileType =
  | 'BANKING'
  | 'ENTERPRISE'
  | 'GOVERNMENT'
  | 'TELECOM'
  | 'DEFAULT';

export interface TimelineEventInfo {
  event_id: string;
  timestamp: string;
  event_type: string;
  risk_level?: string | null;
  risk_score?: number | null;
  prevention_status: PreventionStatusType;
  prevention_action: PreventionActionType;
  details: string;
  correlation_id?: string | null;
}

export interface PreventionEvaluationResponseInfo {
  workflow_id: string;
  prevention_status: PreventionStatusType;
  primary_action: PreventionActionType;
  required_actions: PreventionActionType[];
  is_blocked: boolean;
  is_paused: boolean;
  requires_verification: boolean;
  recommended_verification_methods: VerificationMethodType[];
  policy_profile: PolicyProfileType;
  explanation: string;
  evidence: Array<Record<string, any>>;
  timeline: TimelineEventInfo[];
  created_at: string;
  updated_at: string;
  enforcement_disclaimer: string;
}

export interface PreventionScenarioInfo {
  id: string;
  name: string;
  description: string;
  sensitive_action: SensitiveActionTypeValue;
  transaction_amount?: number | null;
  policy_profile: PolicyProfileType;
  risk_score: number;
  risk_level: string;
  synthetic_score: number;
  speaker_similarity: number;
  context_risk_score: number;
  caller_trust: string;
  expected_status: PreventionStatusType;
  expected_action: PreventionActionType;
  key_evidence: string;
}

