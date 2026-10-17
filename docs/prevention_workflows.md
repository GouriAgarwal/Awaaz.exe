# VoiceShield — Step 14: Automated Prevention & Incident Response Workflow

## Overview
VoiceShield Step 14 addresses the core hackathon problem:
> **"AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks"**

While Steps 1–13 detect synthetic speech artifacts, verify biometric speaker identity, and analyze contextual risk, **Step 14 operationalizes what happens AFTER detection**. It converts dynamic impersonation risk into concrete, policy-driven fraud prevention actions before high-stakes sensitive transactions (wire transfers, privileged credential resets, confidential data disclosures) are executed.

---

## 1. Core Prevention Workflow Architecture

```
LIVE VOICE / CALL
        |
        v
REAL-TIME VOICE ANALYSIS
        |
        +--> Synthetic Voice Detection (AASIST)
        |
        +--> Speaker Identity Verification (SpeechBrain ECAPA-TDNN)
        |
        +--> Prosody & Behavioral Dynamics
        |
        +--> Contextual Fraud Intelligence
        |
        v
DYNAMIC IMPERSONATION RISK SCORE (0–100)
        |
        v
PREVENTION POLICY ENGINE
        |
        +--------------------------------+
        |                                |
        v                                v
   LOW / MEDIUM                    HIGH / CRITICAL
  (Risk: 0–49)                      (Risk: 50–100)
        |                                |
        v                                v
 MONITOR / VERIFY                PREVENT / ESCALATE
        |                                |
        |                                +--> Pause Sensitive Action
        |                                +--> Require Secondary Verification
        |                                +--> Require Callback
        |                                +--> Require Step-Up MFA
        |                                +--> Escalate to Supervisor
        |                                +--> Escalate to SOC
        |                                +--> Block Transaction
        |                                |
        +--------------------------------+
                        |
                        v
          SAFE INCIDENT & DECISION TIMELINE
```

---

## 2. Prevention Policy Matrix

The prevention decision engine implements bounded, deterministic rules mapping impersonation risk and transaction context to operational actions:

| Risk Tier | Risk Score Band | Action Type | Sensitive Action Context | Prescribed Prevention Action | Workflow Status | Recommended Verification Pathway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LOW** | 0 – 24 | Natural Speech + Enrolled Match | Any | `ALLOW` / `MONITOR` | `ALLOWED` | None required |
| **MEDIUM** | 25 – 49 | Identity Uncertainty | General Inquiry | `MONITOR` | `MONITORING` | None required |
| **MEDIUM** | 25 – 49 | Identity Uncertainty | Sensitive Action / Account Change | `REQUIRE_SECONDARY_VERIFICATION` | `VERIFICATION_REQUIRED` | Security Question / Out-of-Band Auth |
| **HIGH** | 50 – 74 | Synthetic Cues / Unverified Identity | General Call | `SHOW_WARNING` | `WARNING` | Operator Advisory |
| **HIGH** | 50 – 74 | Synthetic Cues / Unverified Identity | Fund Transfer / Wire | `REQUIRE_CALLBACK` + `PAUSE_SENSITIVE_ACTION` | `PAUSED` | Callback to Registered Primary Number |
| **HIGH** | 50 – 74 | Synthetic Cues / Unverified Identity | High-Value Transfer ($\ge \$50,000$) | `PAUSE_SENSITIVE_ACTION` + `REQUIRE_CALLBACK` + `ESCALATE_TO_SUPERVISOR` | `PAUSED` | Callback + Supervisor Override |
| **HIGH** | 50 – 74 | Synthetic Cues / Unverified Identity | Privileged IAM Access / Admin Reset | `REQUIRE_MFA` + `PAUSE_SENSITIVE_ACTION` | `PAUSED` | Hardware Token / Step-Up Push MFA |
| **HIGH** | 50 – 74 | Synthetic Cues / Unverified Identity | Confidential Data Disclosure | `PAUSE_SENSITIVE_ACTION` | `PAUSED` | Out-of-Band Auth Challenge |
| **CRITICAL** | 75 – 100 | Targeted Clone + High Stakes | Financial Transfer / Privileged Access | `BLOCK_TRANSACTION` + `ESCALATE_TO_SOC` + `REQUIRE_CALLBACK` | `BLOCKED` | SOC Security Investigation + Supervisor |
| **CRITICAL** | 75 – 100 | Target Clone Artifacts | Non-Financial Call | `ESCALATE_TO_SOC` + `PAUSE_SENSITIVE_ACTION` | `ESCALATED` | SOC Review |

---

## 3. Organizational Policy Profiles

VoiceShield supports 5 configurable policy profiles tailoring sensitivity thresholds:

1. **`BANKING`** (Financial Fraud Shield):
   - High sensitivity on `FUND_TRANSFER` and `PAYMENT_APPROVAL`.
   - Low high-risk threshold (45) and critical threshold (70).
   - Mandatory out-of-band telephone callback for all paused wire transfers.
   - High-value transactions ($\ge \$50,000$) require dual-custody supervisor authorization.
2. **`ENTERPRISE`** (IT & Privileged Access Policy):
   - Focused on preventing executive spear-phishing and IT helpdesk social engineering.
   - Mandatory step-up hardware token MFA for admin console changes.
3. **`GOVERNMENT`** (Critical Infrastructure & Classified Data):
   - Zero-trust policy on confidential data disclosure.
   - Low thresholds (high: 40, critical: 65) with immediate session halting.
4. **`TELECOM`** (SIM-Swap & Account Defense):
   - Optimized to thwart fraudulent SIM-swaps, number porting, and SMS redirect requests.
5. **`DEFAULT`** (Balanced Corporate Baseline):
   - Standard balanced thresholds suitable for enterprise contact centers.

---

## 4. Prevention Workflow State Machine

The workflow state machine governs the lifecycle of every evaluated interaction and enforces strict audit integrity:

```
                  INITIAL
                     |
                     v
                 MONITORING <----+
                 /   |    \      |
                /    |     \     | (Verification Passed)
               v     v      v    |
          WARNING  PAUSED  VERIFICATION_REQUIRED
               \     |      /
                v    v     v
                 ESCALATED
                     |
                     v (Critical Threat)
                  BLOCKED
                     |
                     | (Explicit Supervisor Override Only)
                     v
                  RESOLVED
```

### Strict Transition Guard
- **A `BLOCKED` workflow CANNOT automatically transition to `ALLOWED` or `MONITORING`.**
- If an attack is blocked, automated callback verification or standard MFA is rejected by policy.
- Unblocking or clearing a blocked incident requires explicit human supervisor review via `/api/prevention/workflows/{id}/resolve` or `supervisor_approved` override.

---

## 5. Incident & Decision Timeline

The incident timeline provides an immutable, in-memory audit ledger recording chronological defense events:
- Safe metadata only: timestamps (ISO UTC), event codes, risk levels, prescribed actions, and correlation IDs.
- **Strict Privacy Guarantee**: Never stores or logs raw audio waveforms, base64 data, audio arrays, or 512-dimensional speaker embeddings.

---

## 6. Predefined Hackathon Demonstration Scenarios

1. **Scenario 1: Genuine Authorized Call**:
   - Customer inquiring about balance from verified telephone number.
   - Outcome: `ALLOW` / `ALLOWED`.
2. **Scenario 2: Voice Clone Fund Transfer Attack**:
   - Cloned voice of CEO requesting urgent foreign wire transfer ($150,000).
   - Outcome: `BLOCK_TRANSACTION` / `BLOCKED` + SOC escalation + callback challenge.
3. **Scenario 3: Unknown Natural Caller**:
   - Unrecognized caller attempting account phone/address change.
   - Outcome: `REQUIRE_SECONDARY_VERIFICATION` / `VERIFICATION_REQUIRED`.
4. **Scenario 4: Synthetic Unknown Caller (Robocall Probe)**:
   - Automated text-to-speech probe dialer.
   - Outcome: `SHOW_WARNING` / `WARNING` + continuous monitoring.
5. **Scenario 5: High-Risk Executive Impersonation**:
   - Cloned voice of Senior VP demanding emergency IAM admin privileges.
   - Outcome: `BLOCK_TRANSACTION` / `BLOCKED` + MFA challenge + supervisor sign-off.

---

## 7. REST API Endpoints

- `POST /api/prevention/evaluate`: Evaluates risk against policy profile and returns workflow state and actions.
- `POST /api/prevention/workflows`: Initializes a tracked prevention workflow.
- `GET /api/prevention/workflows/{workflow_id}`: Retrieves workflow state and chronological incident timeline.
- `POST /api/prevention/workflows/{workflow_id}/verify`: Simulates verification challenges (`callback_passed`, `callback_failed`, `mfa_passed`, `mfa_failed`, `supervisor_approved`, `supervisor_rejected`).
- `POST /api/prevention/workflows/{workflow_id}/resolve`: Resolves or overrides an incident workflow with supervisor audit notes.
- `GET /api/prevention/policies`: Lists configurable policy profiles and threshold definitions.
- `GET /api/prevention/demo/scenarios`: Retrieves 5 hackathon demonstration scenarios.

---

## 8. Scientific & Operational Disclosures

1. **Advisory Decision Support**: VoiceShield generates automated prevention recommendations and state workflows; physical disconnection of telephone calls or core banking hold placement requires downstream API integration.
2. **Simulation Boundaries**: Demo verification controls (callback, MFA, supervisor approvals) represent simulated workflow testing endpoints and do not imply external commercial telco/banking gateway connections unless configured.
3. **Heuristic Risk Transparency**: Overall impersonation scores remain heuristic decision-support metrics and must not be interpreted as empirical Bayesian probabilities of fraud.
