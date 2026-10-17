"""
Test Suite for Step 14: Automated Prevention & Incident Response Workflow.
Verifies policy-based prevention decision engine, state machine transitions,
incident timeline guarantees, verification simulations, and REST APIs.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import create_application
from app.prevention.schemas import (
    PreventionAction,
    PreventionStatus,
    SensitiveActionType,
    VerificationMethod,
    PolicyProfile,
    PreventionEvaluationRequest,
    VerificationSimulationRequest,
    PreventionResolutionRequest,
)
from app.prevention.engine import PreventionDecisionEngine
from app.prevention.workflows import PreventionWorkflowStateMachine, InvalidStateTransitionError
from app.prevention.service import PreventionService


@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)


@pytest.fixture
def prevention_service():
    return PreventionService.get_instance()


# 1. Low risk general call allowed
def test_low_risk_general_call_allowed():
    req = PreventionEvaluationRequest(
        risk_score=15,
        risk_level="LOW",
        sensitive_action_type=SensitiveActionType.GENERAL_CALL,
        policy_profile=PolicyProfile.DEFAULT,
    )
    status, primary_action, required_actions, is_blocked, is_paused, requires_verification, methods, explanation, evidence = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.ALLOWED
    assert primary_action == PreventionAction.ALLOW
    assert not is_blocked
    assert not is_paused
    assert not requires_verification


# 2. Medium risk sensitive action requires verification
def test_medium_risk_sensitive_action_requires_verification():
    req = PreventionEvaluationRequest(
        risk_score=38,
        risk_level="MEDIUM",
        sensitive_action_type=SensitiveActionType.ACCOUNT_CHANGE,
        policy_profile=PolicyProfile.DEFAULT,
    )
    status, primary_action, required_actions, is_blocked, is_paused, requires_verification, methods, explanation, evidence = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.VERIFICATION_REQUIRED
    assert primary_action == PreventionAction.REQUIRE_SECONDARY_VERIFICATION
    assert requires_verification
    assert not is_blocked
    assert VerificationMethod.SECURITY_QUESTION in methods or VerificationMethod.OUT_OF_BAND_VERIFICATION in methods


# 3. High risk fund transfer pauses action
def test_high_risk_fund_transfer_pauses_action():
    req = PreventionEvaluationRequest(
        risk_score=65,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=25000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    status, primary_action, required_actions, is_blocked, is_paused, requires_verification, methods, explanation, evidence = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.PAUSED
    assert primary_action == PreventionAction.REQUIRE_CALLBACK
    assert is_paused
    assert requires_verification
    assert PreventionAction.PAUSE_SENSITIVE_ACTION in required_actions
    assert VerificationMethod.CALLBACK_TO_REGISTERED_NUMBER in methods


# 4. Critical clone attack blocks transaction
def test_critical_clone_attack_blocks_transaction():
    req = PreventionEvaluationRequest(
        risk_score=92,
        risk_level="CRITICAL",
        synthetic_score=0.96,
        speaker_similarity=0.88,
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=150000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    status, primary_action, required_actions, is_blocked, is_paused, requires_verification, methods, explanation, evidence = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.BLOCKED
    assert primary_action == PreventionAction.BLOCK_TRANSACTION
    assert is_blocked
    assert PreventionAction.ESCALATE_TO_SOC in required_actions
    assert PreventionAction.REQUIRE_CALLBACK in required_actions


# 5. Context changes prevention decision
def test_context_changes_prevention_decision():
    # Same risk score (60), different context
    req_general = PreventionEvaluationRequest(
        risk_score=60,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.GENERAL_CALL,
        policy_profile=PolicyProfile.DEFAULT,
    )
    status_gen, action_gen, _, is_blocked_gen, is_paused_gen, _, _, _, _ = (
        PreventionDecisionEngine.evaluate(req_general)
    )

    req_wire = PreventionEvaluationRequest(
        risk_score=60,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=60000.0,
        policy_profile=PolicyProfile.DEFAULT,
    )
    status_wire, action_wire, _, is_blocked_wire, is_paused_wire, _, _, _, _ = (
        PreventionDecisionEngine.evaluate(req_wire)
    )

    assert status_gen == PreventionStatus.WARNING
    assert action_gen == PreventionAction.SHOW_WARNING
    assert not is_paused_gen

    assert status_wire == PreventionStatus.PAUSED
    assert action_wire == PreventionAction.REQUIRE_CALLBACK
    assert is_paused_wire


# 6. Banking policy behavior
def test_banking_policy_behavior():
    # Banking profile has a lower high risk threshold (45)
    req = PreventionEvaluationRequest(
        risk_score=48,
        risk_level="MEDIUM",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=12000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    status, primary_action, required_actions, is_blocked, is_paused, _, _, _, _ = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.PAUSED
    assert primary_action == PreventionAction.REQUIRE_CALLBACK
    assert is_paused


# 7. Enterprise policy behavior
def test_enterprise_policy_behavior():
    req = PreventionEvaluationRequest(
        risk_score=70,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.PRIVILEGED_ACCESS,
        policy_profile=PolicyProfile.ENTERPRISE,
    )
    status, primary_action, required_actions, is_blocked, is_paused, _, methods, _, _ = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.PAUSED
    assert primary_action == PreventionAction.REQUIRE_MFA
    assert VerificationMethod.MULTI_FACTOR_AUTHENTICATION in methods
    assert PreventionAction.ESCALATE_TO_SUPERVISOR in required_actions


# 8. Government policy behavior
def test_government_policy_behavior():
    req = PreventionEvaluationRequest(
        risk_score=55,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.CONFIDENTIAL_DISCLOSURE,
        policy_profile=PolicyProfile.GOVERNMENT,
    )
    status, primary_action, required_actions, is_blocked, is_paused, _, methods, _, _ = (
        PreventionDecisionEngine.evaluate(req)
    )

    assert status == PreventionStatus.PAUSED
    assert primary_action == PreventionAction.PAUSE_SENSITIVE_ACTION
    assert VerificationMethod.OUT_OF_BAND_VERIFICATION in methods


# 9. Invalid state transition rejected
def test_invalid_state_transition_rejected():
    # BLOCKED cannot automatically become ALLOWED
    with pytest.raises(InvalidStateTransitionError):
        PreventionWorkflowStateMachine.validate_transition(
            current_status=PreventionStatus.BLOCKED,
            target_status=PreventionStatus.ALLOWED,
        )


# 10. Callback verification passed
def test_callback_verification_passed(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=68,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=20000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)
    assert workflow.prevention_status == PreventionStatus.PAUSED

    updated = prevention_service.process_verification(
        workflow_id=workflow.workflow_id,
        verification_type="callback_passed",
        actor="senior_operator",
        notes="Customer confirmed transaction via primary phone.",
    )

    assert updated.prevention_status == PreventionStatus.RESOLVED
    assert updated.primary_action == PreventionAction.ALLOW
    assert not updated.is_paused
    assert not updated.is_blocked


# 11. Callback verification failed
def test_callback_verification_failed(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=68,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=20000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    updated = prevention_service.process_verification(
        workflow_id=workflow.workflow_id,
        verification_type="callback_failed",
        actor="system",
        notes="Customer reported unauthorized attempt.",
    )

    assert updated.prevention_status == PreventionStatus.BLOCKED
    assert updated.primary_action == PreventionAction.BLOCK_TRANSACTION
    assert updated.is_blocked


# 12. MFA verification passed
def test_mfa_verification_passed(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=65,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.PRIVILEGED_ACCESS,
        policy_profile=PolicyProfile.ENTERPRISE,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    updated = prevention_service.process_verification(
        workflow_id=workflow.workflow_id,
        verification_type="mfa_passed",
        actor="iam_service",
    )

    assert updated.prevention_status == PreventionStatus.RESOLVED
    assert updated.primary_action == PreventionAction.ALLOW
    assert not updated.is_paused


# 13. Supervisor approval
def test_supervisor_approval(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=72,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.PAYMENT_APPROVAL,
        transaction_amount=80000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    updated = prevention_service.process_verification(
        workflow_id=workflow.workflow_id,
        verification_type="supervisor_approved",
        actor="branch_manager",
        notes="Customer in branch presenting ID.",
    )

    assert updated.prevention_status == PreventionStatus.RESOLVED
    assert updated.primary_action == PreventionAction.ALLOW


# 14. Supervisor rejection
def test_supervisor_rejection(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=72,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.PAYMENT_APPROVAL,
        transaction_amount=80000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    updated = prevention_service.process_verification(
        workflow_id=workflow.workflow_id,
        verification_type="supervisor_rejected",
        actor="compliance_officer",
        notes="Confirmed suspicious caller.",
    )

    assert updated.prevention_status == PreventionStatus.BLOCKED
    assert updated.primary_action == PreventionAction.BLOCK_TRANSACTION
    assert updated.is_blocked


# 15. Blocked workflow cannot automatically allow itself
def test_blocked_workflow_cannot_automatically_allow(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=95,
        risk_level="CRITICAL",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=200000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)
    assert workflow.prevention_status == PreventionStatus.BLOCKED

    # Attempting callback_passed on a BLOCKED workflow must raise InvalidStateTransitionError
    # because a blocked critical attack requires explicit supervisor resolution.
    with pytest.raises(InvalidStateTransitionError):
        prevention_service.process_verification(
            workflow_id=workflow.workflow_id,
            verification_type="callback_passed",
        )


# 16. Timeline contains prevention events
def test_timeline_contains_prevention_events(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=85,
        risk_level="CRITICAL",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=50000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    assert len(workflow.timeline) >= 4
    event_types = [e.event_type for e in workflow.timeline]
    assert "CALL_CONNECTED" in event_types
    assert "VOICE_ANALYSIS_COMPLETED" in event_types
    assert "POLICY_BLOCKED" in event_types
    assert "TRANSACTION_BLOCKED" in event_types


# 17. Timeline contains no raw audio
def test_timeline_contains_no_raw_audio(prevention_service):
    req = PreventionEvaluationRequest(
        risk_score=80,
        risk_level="CRITICAL",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        policy_profile=PolicyProfile.DEFAULT,
    )
    workflow = prevention_service.evaluate_and_create_workflow(req)

    for evt in workflow.timeline:
        assert "audio" not in evt.details.lower() or "voice" in evt.details.lower()
        assert len(evt.details) < 2000
        # No base64 strings or float arrays
        assert "data:audio" not in evt.details
        assert "[" not in evt.details or "TRUNCATED" in evt.details


# 18. API endpoints function
def test_api_endpoints_function(client):
    # 1. /api/prevention/policies
    res = client.get("/api/prevention/policies")
    assert res.status_code == 200
    policies = res.json()
    assert len(policies) >= 5

    # 2. /api/prevention/demo/scenarios
    res = client.get("/api/prevention/demo/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) >= 5

    # 3. /api/prevention/evaluate
    eval_payload = {
        "risk_score": 88,
        "risk_level": "CRITICAL",
        "sensitive_action_type": "FUND_TRANSFER",
        "transaction_amount": 100000.0,
        "policy_profile": "BANKING",
    }
    res = client.post("/api/prevention/evaluate", json=eval_payload)
    assert res.status_code == 200
    wf_data = res.json()
    wf_id = wf_data["workflow_id"]
    assert wf_data["prevention_status"] == "BLOCKED"
    assert wf_data["is_blocked"] is True

    # 4. /api/prevention/workflows/{id}
    res = client.get(f"/api/prevention/workflows/{wf_id}")
    assert res.status_code == 200
    assert res.json()["workflow_id"] == wf_id

    # 5. /api/prevention/workflows/{id}/resolve
    resolve_payload = {
        "resolution_reason": "Senior fraud investigator confirmed legitimate emergency exception.",
        "resolved_by": "fraud_director_01",
        "final_action": "ALLOW",
    }
    res = client.post(f"/api/prevention/workflows/{wf_id}/resolve", json=resolve_payload)
    assert res.status_code == 200
    assert res.json()["prevention_status"] == "RESOLVED"


# 19. Demo scenarios produce expected prevention decisions
def test_demo_scenarios_produce_expected_decisions(prevention_service):
    scenarios = prevention_service.get_demo_scenarios()
    assert len(scenarios) >= 5

    for sc in scenarios:
        req = PreventionEvaluationRequest(
            risk_score=sc.risk_score,
            risk_level=sc.risk_level,
            synthetic_score=sc.synthetic_score,
            speaker_similarity=sc.speaker_similarity,
            context_risk_score=sc.context_risk_score,
            sensitive_action_type=sc.sensitive_action,
            transaction_amount=sc.transaction_amount,
            policy_profile=sc.policy_profile,
            caller_trust=sc.caller_trust,
        )
        resp = prevention_service.evaluate_and_create_workflow(req)
        assert resp.prevention_status == sc.expected_status, (
            f"Scenario {sc.name} expected status {sc.expected_status.value}, got {resp.prevention_status.value}"
        )
        assert resp.primary_action == sc.expected_action, (
            f"Scenario {sc.name} expected action {sc.expected_action.value}, got {resp.primary_action.value}"
        )
