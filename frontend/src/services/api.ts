/**
 * VoiceShield API Client Service
 */

import {
  HealthResponse,
  SystemStatusResponse,
  AudioInspectResponse,
  FeatureExtractionResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Perform health check against VoiceShield API.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Health check returned HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Retrieve system operational status from VoiceShield API.
 */
export async function getSystemStatus(): Promise<SystemStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/api/system/status`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`System status check returned HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Perform a full diagnostic ping measuring latency and retrieving all statuses.
 */
export async function testConnection(): Promise<{
  health: HealthResponse;
  system: SystemStatusResponse;
  latencyMs: number;
}> {
  const startTime = performance.now();

  const [health, system] = await Promise.all([
    checkHealth(),
    getSystemStatus(),
  ]);

  const latencyMs = Math.round(performance.now() - startTime);

  return {
    health,
    system,
    latencyMs,
  };
}

/**
 * Upload an audio File or recorded Blob to /api/audio/inspect for validation and acoustic analysis.
 */
export async function inspectAudio(file: File | Blob, filename?: string): Promise<AudioInspectResponse> {
  const formData = new FormData();
  const effectiveFilename = filename || (file instanceof File ? file.name : 'microphone_recording.webm');
  formData.append('file', file, effectiveFilename);

  const response = await fetch(`${API_BASE_URL}/api/audio/inspect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Audio inspection failed (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      } else if (errorJson.error?.message) {
        errorMessage = errorJson.error.message;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Upload audio and extract 16kHz standardized acoustic, spectral, MFCC, pitch, voice quality, and prosody features.
 */
export async function extractFeatures(file: File | Blob, filename?: string): Promise<FeatureExtractionResponse> {
  const formData = new FormData();
  const effectiveFilename = filename || (file instanceof File ? file.name : 'audio_input.webm');
  formData.append('file', file, effectiveFilename);

  const response = await fetch(`${API_BASE_URL}/api/features/extract`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Feature extraction failed (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      } else if (errorJson.error?.message) {
        errorMessage = errorJson.error.message;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Upload audio and run AI-based synthetic / cloned voice detection (Step 4).
 */
export async function analyzeSyntheticVoice(
  file: File | Blob,
  filename?: string,
  detector?: string,
): Promise<import('../types').DetectionResponse> {
  const formData = new FormData();
  const effectiveFilename = filename || (file instanceof File ? file.name : 'audio_input.webm');
  formData.append('file', file, effectiveFilename);
  if (detector) {
    formData.append('detector', detector);
  }

  const response = await fetch(`${API_BASE_URL}/api/detection/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Synthetic voice analysis failed (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      } else if (errorJson.error?.message) {
        errorMessage = errorJson.error.message;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Retrieve metadata for all available detector models.
 */
export async function getDetectorModels(): Promise<import('../types').DetectorMetadataInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/detection/models`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve detector models (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Retrieve the latest scientific evaluation report (Step 5).
 */
export async function getEvaluationStatus(): Promise<import('../types').EvaluationReportInfo> {
  const response = await fetch(`${API_BASE_URL}/api/detection/evaluation/status`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve evaluation status (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Retrieve machine-readable model provenance records for all detectors (Step 5).
 */
export async function getModelProvenance(): Promise<Record<string, import('../types').ModelProvenanceInfo>> {
  const response = await fetch(`${API_BASE_URL}/api/detection/provenance`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve model provenance (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Perform multi-modal impersonation risk analysis on audio payload against an enrolled profile ID.
 */
export async function analyzeRisk(
  file: File | Blob,
  profileId: string,
  contextualSignals?: import('../types').ContextualSignalsInfo,
  filename?: string
): Promise<import('../types').RiskAnalysisResultInfo> {
  const formData = new FormData();
  const effectiveFilename = filename || (file instanceof File ? file.name : 'risk_evaluation.wav');
  formData.append('file', file, effectiveFilename);
  formData.append('profile_id', profileId);

  if (contextualSignals) {
    formData.append('contextual_signals', JSON.stringify(contextualSignals));
  }

  const response = await fetch(`${API_BASE_URL}/api/risk/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Risk analysis failed (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Simulate controlled fusion evaluation given synthetic and speaker signals.
 */
export async function simulateRisk(
  payload: import('../types').RiskSimulationRequestInfo
): Promise<import('../types').RiskAnalysisResultInfo> {
  const response = await fetch(`${API_BASE_URL}/api/risk/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Risk simulation failed (HTTP ${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = errorJson.detail;
      }
    } catch {
      errorMessage = response.statusText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Retrieve Risk Fusion provenance and model architecture metadata.
 */
export async function getRiskProvenance(): Promise<import('../types').RiskProvenanceResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/risk/provenance`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve risk provenance (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Retrieve list of all enrolled speaker profile metadata summaries.
 */
export async function listSpeakerProfiles(): Promise<import('../types').SpeakerProfileSummaryInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/speaker/profiles`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list speaker profiles (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Get WebSocket URL for real-time speech streaming.
 */
export function getStreamingWsUrl(): string {
  const base = API_BASE_URL.replace(/^http/, 'ws');
  return `${base}/api/stream/ws`;
}

/**
 * List active streaming sessions.
 */
export async function listStreamingSessions(): Promise<import('../types').StreamingSessionSummaryInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/stream/sessions`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to list streaming sessions (HTTP ${response.status})`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Step 11: Production Scientific Validation, Calibration & Security APIs
// ---------------------------------------------------------------------------

/**
 * Retrieve system-wide scientific validation and calibration statuses.
 */
export async function getSystemScientificStatus(): Promise<import('../types').SystemScientificStatusSummaryInfo> {
  const response = await fetch(`${API_BASE_URL}/api/evaluation/status`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve scientific evaluation status (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Search workspace for benchmark datasets and manifests.
 */
export async function discoverDatasets(): Promise<{
  found_count: number;
  manifest_paths: string[];
  status: string;
  message: string;
}> {
  const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to discover datasets (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Run structural and cryptographic integrity audit on a dataset manifest.
 */
export async function auditDatasetManifest(
  manifestPath: string,
  baseDir?: string
): Promise<import('../types').DatasetAuditReportInfo> {
  const response = await fetch(`${API_BASE_URL}/api/evaluation/audit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      manifest_path: manifestPath,
      base_dir: baseDir,
      verify_audio_decoding: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to audit manifest (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Retrieve active AASIST calibration status.
 */
export async function getDetectionCalibrationStatus(): Promise<import('../types').CalibrationReportInfo> {
  const response = await fetch(`${API_BASE_URL}/api/detection/calibration/status`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to retrieve calibration status (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Execute automated real-time streaming stress test suite across 15 conditions.
 */
export async function runStreamingStressTests(): Promise<import('../types').StreamingValidationReportInfo> {
  const response = await fetch(`${API_BASE_URL}/api/stream/evaluation/stress-test`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to execute streaming stress tests (HTTP ${response.status})`);
  }

  return response.json();
}

/**
 * Run cryptographic model checksums and security audit.
 */
export async function runSecurityAudit(): Promise<import('../types').SecurityAuditReportInfo> {
  const response = await fetch(`${API_BASE_URL}/api/security/audit`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to execute security audit (HTTP ${response.status})`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Step 12: Production Deployment, Observability & Operational Readiness APIs
// ---------------------------------------------------------------------------

/**
 * Check service liveness (fast non-blocking).
 */
export async function getLiveness(): Promise<import('../types').LivenessResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/health/live`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Liveness check failed (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Check comprehensive system readiness.
 */
export async function getReadiness(): Promise<import('../types').ReadinessResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/health/ready`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  return response.json();
}

/**
 * Get detailed operational status across all subsystems.
 */
export async function getOperationalStatus(): Promise<import('../types').OperationalStatusResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/health/status`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch operational status (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Get real-time application metrics (latency, HTTP codes, model throughput).
 */
export async function getApplicationMetrics(): Promise<import('../types').ApplicationMetricsResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/observability/metrics`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch application metrics (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Get monitored models operational lifecycle and telemetry.
 */
export async function getMonitoredModels(): Promise<import('../types').ModelMonitoringInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/observability/models`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch model monitoring info (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Get real-time streaming operations telemetry.
 */
export async function getStreamingTelemetry(): Promise<import('../types').StreamingOperationalInfo> {
  const response = await fetch(`${API_BASE_URL}/api/observability/streaming`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch streaming telemetry (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Get structured operational error summary and recent failures.
 */
export async function getOperationalErrors(limit = 20): Promise<import('../types').OperationalErrorSummaryInfo> {
  const response = await fetch(`${API_BASE_URL}/api/observability/errors?limit=${limit}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch error summary (HTTP ${response.status})`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Step 13: Multi-Layer Voice Integrity & Contextual Risk API
// ---------------------------------------------------------------------------

/**
 * Get pre-configured banking and enterprise attack scenarios.
 */
export async function getAttackScenarios(): Promise<import('../types').AttackScenarioInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/contextual-risk/scenarios`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch attack scenarios (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * List all enrolled speaker profiles (metadata only).
 */
export async function getEnrolledProfiles(): Promise<import('../types').SpeakerProfileSummaryInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/speaker/profiles`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    return [];
  }
  return response.json();
}

/**
 * Simulate multi-layer voice cloning & contextual risk analysis.
 */
export async function simulateContextualRisk(
  payload: any
): Promise<import('../types').MultiLayerRiskAnalysisResultInfo> {
  const response = await fetch(`${API_BASE_URL}/api/contextual-risk/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Simulation failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) errorMessage = err.detail;
    } catch {}
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Execute full multi-layer voice integrity analysis on audio payload.
 */
export async function analyzeContextualRisk(
  formData: FormData
): Promise<import('../types').MultiLayerRiskAnalysisResultInfo> {
  const response = await fetch(`${API_BASE_URL}/api/contextual-risk/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Analysis failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) errorMessage = err.detail;
    } catch {}
    throw new Error(errorMessage);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Step 14: Automated Prevention & Incident Response API
// ---------------------------------------------------------------------------

/**
 * Evaluate prevention policies against voice risk and transaction context.
 */
export async function evaluatePrevention(
  payload: any
): Promise<import('../types').PreventionEvaluationResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Prevention evaluation failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) errorMessage = err.detail;
    } catch {}
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Retrieve active prevention workflow state and incident timeline.
 */
export async function getPreventionWorkflow(
  workflowId: string
): Promise<import('../types').PreventionEvaluationResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/workflows/${workflowId}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch workflow (HTTP ${response.status})`);
  }
  return response.json();
}

/**
 * Simulate verification challenge outcome (callback, MFA, supervisor approval).
 */
export async function verifyPreventionWorkflow(
  workflowId: string,
  payload: { verification_type: string; actor?: string; notes?: string }
): Promise<import('../types').PreventionEvaluationResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/workflows/${workflowId}/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Verification simulation failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) errorMessage = err.detail;
    } catch {}
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Explicitly resolve an incident workflow following operator review.
 */
export async function resolvePreventionWorkflow(
  workflowId: string,
  payload: { resolution_reason: string; resolved_by: string; final_action?: string }
): Promise<import('../types').PreventionEvaluationResponseInfo> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/workflows/${workflowId}/resolve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorMessage = `Workflow resolution failed (HTTP ${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) errorMessage = err.detail;
    } catch {}
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Get available prevention policy profiles.
 */
export async function getPreventionPolicies(): Promise<any[]> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/policies`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    return [];
  }
  return response.json();
}

/**
 * Get 5 pre-configured demonstration scenarios.
 */
export async function getPreventionScenarios(): Promise<import('../types').PreventionScenarioInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/prevention/demo/scenarios`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });
  if (!response.ok) {
    return [];
  }
  return response.json();
}
