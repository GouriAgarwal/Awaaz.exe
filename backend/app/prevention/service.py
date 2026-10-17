"""
Prevention Service Layer.
Coordinates policy evaluation, workflow state management, verification simulations,
and chronological incident timeline recording.
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.prevention.schemas import (
    PreventionAction,
    PreventionStatus,
    SensitiveActionType,
    VerificationMethod,
    PolicyProfile,
    PreventionEvaluationRequest,
    PreventionEvaluationResponse,
    TimelineEvent,
    DemoScenario,
)
from app.prevention.policies import POLICY_PROFILES, get_policy_config
from app.prevention.engine import PreventionDecisionEngine
from app.prevention.timeline import IncidentTimeline
from app.prevention.workflows import PreventionWorkflowStateMachine, InvalidStateTransitionError

logger = logging.getLogger(__name__)


class PreventionService:
    """
    Singleton service managing fraud prevention workflows and incident responses.
    """
    _instance: Optional["PreventionService"] = None

    def __init__(self):
        self._workflows: Dict[str, PreventionEvaluationResponse] = {}
        self._timelines: Dict[str, IncidentTimeline] = {}

    @classmethod
    def get_instance(cls) -> "PreventionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate_and_create_workflow(self, req: PreventionEvaluationRequest) -> PreventionEvaluationResponse:
        """
        Executes policy evaluation and initializes a new prevention workflow with initial timeline events.
        """
        workflow_id = f"prev_{uuid.uuid4().hex[:12]}"
        timeline = IncidentTimeline(workflow_id=workflow_id)
        self._timelines[workflow_id] = timeline

        # Timeline event 1: Call / Session initialized
        now_iso = datetime.now(timezone.utc).isoformat()
        timeline.add_event(
            event_type="CALL_CONNECTED",
            prevention_status=PreventionStatus.MONITORING,
            prevention_action=PreventionAction.MONITOR,
            details=f"Call channel initiated. Action requested: {req.sensitive_action_type.value}.",
            risk_level=req.risk_level,
            risk_score=req.risk_score,
            correlation_id=req.session_id,
        )

        # Timeline event 2: Voice analysis evaluated
        synth_desc = f"synthetic_score={req.synthetic_score:.3f}" if req.synthetic_score is not None else "synthetic=N/A"
        spk_desc = f"spk_sim={req.speaker_similarity:.3f}" if req.speaker_similarity is not None else "spk_sim=N/A"
        timeline.add_event(
            event_type="VOICE_ANALYSIS_COMPLETED",
            prevention_status=PreventionStatus.MONITORING,
            prevention_action=PreventionAction.MONITOR,
            details=f"Voice signals received: {synth_desc}, {spk_desc}. Impersonation risk: {req.risk_score}/100 ({req.risk_level}).",
            risk_level=req.risk_level,
            risk_score=req.risk_score,
            correlation_id=req.session_id,
        )

        # Execute prevention engine
        (
            status,
            primary_action,
            required_actions,
            is_blocked,
            is_paused,
            requires_verification,
            verification_methods,
            explanation,
            evidence,
        ) = PreventionDecisionEngine.evaluate(req)

        # Timeline event 3: Prevention decision rendered
        timeline.add_event(
            event_type=f"POLICY_{status.value}",
            prevention_status=status,
            prevention_action=primary_action,
            details=explanation,
            risk_level=req.risk_level,
            risk_score=req.risk_score,
            correlation_id=req.session_id,
        )

        # Timeline event 4: Specific action flags
        if is_blocked:
            timeline.add_event(
                event_type="TRANSACTION_BLOCKED",
                prevention_status=status,
                prevention_action=PreventionAction.BLOCK_TRANSACTION,
                details=f"Sensitive action {req.sensitive_action_type.value} has been blocked by policy.",
                risk_level=req.risk_level,
                risk_score=req.risk_score,
            )
        elif is_paused:
            timeline.add_event(
                event_type="ACTION_PAUSED",
                prevention_status=status,
                prevention_action=PreventionAction.PAUSE_SENSITIVE_ACTION,
                details=f"Sensitive action {req.sensitive_action_type.value} temporarily paused awaiting verification.",
                risk_level=req.risk_level,
                risk_score=req.risk_score,
            )

        if PreventionAction.ESCALATE_TO_SOC in required_actions:
            timeline.add_event(
                event_type="SOC_ESCALATION_CREATED",
                prevention_status=status,
                prevention_action=PreventionAction.ESCALATE_TO_SOC,
                details="Security Operations Center incident alert dispatched for voice cloning investigation.",
                risk_level=req.risk_level,
                risk_score=req.risk_score,
            )

        response = PreventionEvaluationResponse(
            workflow_id=workflow_id,
            prevention_status=status,
            primary_action=primary_action,
            required_actions=required_actions,
            is_blocked=is_blocked,
            is_paused=is_paused,
            requires_verification=requires_verification,
            recommended_verification_methods=verification_methods,
            policy_profile=req.policy_profile,
            explanation=explanation,
            evidence=evidence,
            timeline=timeline.get_events(),
            created_at=now_iso,
            updated_at=now_iso,
        )

        self._workflows[workflow_id] = response
        return response

    def get_workflow(self, workflow_id: str) -> Optional[PreventionEvaluationResponse]:
        """Retrieves active workflow state and latest timeline events."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        timeline = self._timelines.get(workflow_id)
        if timeline:
            workflow.timeline = timeline.get_events()
        return workflow

    def process_verification(
        self, workflow_id: str, verification_type: str, actor: str = "security_operator", notes: Optional[str] = None
    ) -> PreventionEvaluationResponse:
        """
        Simulates out-of-band verification challenge outcomes (e.g. callback, MFA, supervisor).
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow '{workflow_id}' not found.")

        timeline = self._timelines[workflow_id]
        v_type = verification_type.lower().strip()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Guard: BLOCKED actions cannot be unblocked by automated callback or standard MFA
        if workflow.prevention_status == PreventionStatus.BLOCKED and v_type in ("callback_passed", "mfa_passed"):
            raise InvalidStateTransitionError(
                "A BLOCKED critical transaction cannot be automatically cleared via standard callback or MFA. "
                "Explicit supervisor approval ('supervisor_approved') or incident resolution is required."
            )

        if v_type == "callback_passed":
            target_status = PreventionStatus.RESOLVED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.ALLOW
            workflow.is_paused = False
            workflow.is_blocked = False
            workflow.requires_verification = False
            workflow.explanation = f"Out-of-band callback verification passed by {actor}. Transaction unpaused."
            timeline.add_event(
                event_type="CALLBACK_VERIFIED",
                prevention_status=target_status,
                prevention_action=PreventionAction.ALLOW,
                details=f"Primary registered telephone number verified successfully by {actor}. {notes or ''}".strip(),
            )

        elif v_type == "callback_failed":
            target_status = PreventionStatus.BLOCKED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.BLOCK_TRANSACTION
            workflow.is_paused = False
            workflow.is_blocked = True
            workflow.explanation = f"Out-of-band callback verification failed or rejected by customer. Action blocked."
            timeline.add_event(
                event_type="CALLBACK_FAILED",
                prevention_status=target_status,
                prevention_action=PreventionAction.BLOCK_TRANSACTION,
                details=f"Callback to registered phone failed. Fraud confirmation alert created. {notes or ''}".strip(),
            )

        elif v_type == "mfa_passed":
            target_status = PreventionStatus.RESOLVED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.ALLOW
            workflow.is_paused = False
            workflow.is_blocked = False
            workflow.requires_verification = False
            workflow.explanation = f"Multi-factor hardware/push authentication succeeded. Action permitted."
            timeline.add_event(
                event_type="MFA_AUTHENTICATED",
                prevention_status=target_status,
                prevention_action=PreventionAction.ALLOW,
                details=f"Step-up MFA challenge satisfied. {notes or ''}".strip(),
            )

        elif v_type == "mfa_failed":
            target_status = PreventionStatus.BLOCKED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.BLOCK_TRANSACTION
            workflow.is_paused = False
            workflow.is_blocked = True
            workflow.explanation = "Step-up MFA authentication failed. Transaction blocked."
            timeline.add_event(
                event_type="MFA_FAILED",
                prevention_status=target_status,
                prevention_action=PreventionAction.BLOCK_TRANSACTION,
                details=f"MFA challenge failed or timed out. {notes or ''}".strip(),
            )

        elif v_type == "supervisor_approved":
            target_status = PreventionStatus.RESOLVED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.ALLOW
            workflow.is_paused = False
            workflow.is_blocked = False
            workflow.requires_verification = False
            workflow.explanation = f"Supervisor override authorized by {actor}. Transaction permitted under exception protocol."
            timeline.add_event(
                event_type="SUPERVISOR_OVERRIDE_APPROVED",
                prevention_status=target_status,
                prevention_action=PreventionAction.ALLOW,
                details=f"Supervisor {actor} approved exception override. Audit note: {notes or 'Standard exception'}",
            )

        elif v_type == "supervisor_rejected":
            target_status = PreventionStatus.BLOCKED
            PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)
            workflow.prevention_status = target_status
            workflow.primary_action = PreventionAction.BLOCK_TRANSACTION
            workflow.is_paused = False
            workflow.is_blocked = True
            workflow.explanation = f"Supervisor rejected transaction authorization."
            timeline.add_event(
                event_type="SUPERVISOR_REJECTED",
                prevention_status=target_status,
                prevention_action=PreventionAction.BLOCK_TRANSACTION,
                details=f"Supervisor {actor} rejected request. Confirmed suspicious interaction. {notes or ''}".strip(),
            )

        else:
            raise ValueError(f"Unknown verification simulation type: '{verification_type}'")

        workflow.updated_at = now_iso
        workflow.timeline = timeline.get_events()
        return workflow

    def resolve_workflow(
        self, workflow_id: str, reason: str, resolved_by: str, final_action: PreventionAction = PreventionAction.ALLOW
    ) -> PreventionEvaluationResponse:
        """
        Explicitly resolves an open or blocked incident workflow following security operator review.
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise KeyError(f"Workflow '{workflow_id}' not found.")

        timeline = self._timelines[workflow_id]
        target_status = PreventionStatus.RESOLVED
        PreventionWorkflowStateMachine.validate_transition(workflow.prevention_status, target_status)

        workflow.prevention_status = target_status
        workflow.primary_action = final_action
        workflow.is_paused = False
        workflow.is_blocked = (final_action == PreventionAction.BLOCK_TRANSACTION)
        workflow.requires_verification = False
        workflow.explanation = f"Incident resolved by {resolved_by}. Decision: {final_action.value}. Rationale: {reason}"
        workflow.updated_at = datetime.now(timezone.utc).isoformat()

        timeline.add_event(
            event_type="INCIDENT_RESOLVED",
            prevention_status=target_status,
            prevention_action=final_action,
            details=f"Resolved by {resolved_by}. Final action: {final_action.value}. Notes: {reason}",
        )
        workflow.timeline = timeline.get_events()
        return workflow

    def get_policy_profiles(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available policy profiles."""
        return [p.to_dict() for p in POLICY_PROFILES.values()]

    def get_demo_scenarios(self) -> List[DemoScenario]:
        """Returns 6 pre-configured demonstration scenarios directly addressing the hackathon problem statement."""
        return [
            DemoScenario(
                id="scenario_1_ceo_wire_transfer_attack",
                name="Scenario 1: CEO Urgent Wire Transfer Attack",
                description="High-fidelity AI voice clone of CEO calling treasury demanding urgent ₹10,00,000 ($150,000) wire transfer to an unverified beneficiary.",
                sensitive_action=SensitiveActionType.FUND_TRANSFER,
                transaction_amount=150000.0,
                policy_profile=PolicyProfile.BANKING,
                risk_score=94,
                risk_level="CRITICAL",
                synthetic_score=0.96,
                speaker_similarity=0.89,
                context_risk_score=92,
                caller_trust="VIP_OR_EXECUTIVE",
                expected_status=PreventionStatus.BLOCKED,
                expected_action=PreventionAction.BLOCK_TRANSACTION,
                key_evidence="High-confidence synthetic speech artifacts + targeted CEO voice match during high-stakes financial transfer.",
            ),
            DemoScenario(
                id="scenario_2_bank_customer_legitimate",
                name="Scenario 2: Bank Customer Legitimate Transfer",
                description="Legitimate retail customer calling from enrolled phone number requesting normal ₹25,000 transfer. Natural human voice, verified biometric match.",
                sensitive_action=SensitiveActionType.FUND_TRANSFER,
                transaction_amount=25000.0,
                policy_profile=PolicyProfile.BANKING,
                risk_score=12,
                risk_level="LOW",
                synthetic_score=0.04,
                speaker_similarity=0.88,
                context_risk_score=10,
                caller_trust="VERIFIED_CONTACT",
                expected_status=PreventionStatus.ALLOWED,
                expected_action=PreventionAction.ALLOW,
                key_evidence="Natural acoustic modulation + enrolled biometric match within normal limits. Transaction permitted.",
            ),
            DemoScenario(
                id="scenario_3_replay_attack_enrolled",
                name="Scenario 3: Replay Attack on Enrolled Account",
                description="Real voice sample replayed through phone channel attempting ₹50,000 payment. Acoustic channel distortion and flat prosody detected.",
                sensitive_action=SensitiveActionType.PAYMENT_APPROVAL,
                transaction_amount=50000.0,
                policy_profile=PolicyProfile.BANKING,
                risk_score=38,
                risk_level="MEDIUM",
                synthetic_score=0.35,
                speaker_similarity=0.82,
                context_risk_score=45,
                caller_trust="UNKNOWN_CALLER",
                expected_status=PreventionStatus.VERIFICATION_REQUIRED,
                expected_action=PreventionAction.REQUIRE_SECONDARY_VERIFICATION,
                key_evidence="High biometric match but significant replay transmission artifacts and prosodic anomalies require step-up challenge.",
            ),
            DemoScenario(
                id="scenario_4_customer_care_social_eng",
                name="Scenario 4: Customer Care Social Engineering",
                description="Synthetic voice agent attempting social engineering on customer support to execute unauthorized credential/password reset.",
                sensitive_action=SensitiveActionType.ACCOUNT_CHANGE,
                transaction_amount=None,
                policy_profile=PolicyProfile.TELECOM,
                risk_score=72,
                risk_level="HIGH",
                synthetic_score=0.86,
                speaker_similarity=0.20,
                context_risk_score=75,
                caller_trust="UNKNOWN_CALLER",
                expected_status=PreventionStatus.PAUSED,
                expected_action=PreventionAction.REQUIRE_MFA,
                key_evidence="Synthetic voice indicators combined with credential reset intent. Action paused pending out-of-band challenge.",
            ),
            DemoScenario(
                id="scenario_5_family_emergency_upi",
                name="Scenario 5: Family Emergency Impersonation",
                description="Cloned voice of victim's child claiming urgent emergency distress and requesting immediate ₹1,50,000 UPI transfer to unfamiliar contact.",
                sensitive_action=SensitiveActionType.FUND_TRANSFER,
                transaction_amount=150000.0,
                policy_profile=PolicyProfile.BANKING,
                risk_score=68,
                risk_level="HIGH",
                synthetic_score=0.82,
                speaker_similarity=0.45,
                context_risk_score=88,
                caller_trust="UNKNOWN_CALLER",
                expected_status=PreventionStatus.PAUSED,
                expected_action=PreventionAction.REQUIRE_CALLBACK,
                key_evidence="Synthetic vocal synthesis + extreme urgency and emotional pressure flags. Automatic hold applied.",
            ),
            DemoScenario(
                id="scenario_6_it_privileged_access",
                name="Scenario 6: IT Support Privileged Access Request",
                description="Synthetic voice clone posing as IT administrator demanding emergency privileged root access to database cluster.",
                sensitive_action=SensitiveActionType.PRIVILEGED_ACCESS,
                transaction_amount=None,
                policy_profile=PolicyProfile.ENTERPRISE,
                risk_score=89,
                risk_level="CRITICAL",
                synthetic_score=0.92,
                speaker_similarity=0.35,
                context_risk_score=90,
                caller_trust="UNKNOWN_CALLER",
                expected_status=PreventionStatus.BLOCKED,
                expected_action=PreventionAction.BLOCK_TRANSACTION,
                key_evidence="Synthetic speech signature targeting administrative access. Immediate block and mandatory SOC escalation.",
            ),
        ]
