"""
REST API Endpoints for VoiceShield Automated Prevention & Incident Response Workflow.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.prevention.schemas import (
    PreventionEvaluationRequest,
    PreventionEvaluationResponse,
    VerificationSimulationRequest,
    PreventionResolutionRequest,
    DemoScenario,
)
from app.prevention.service import PreventionService
from app.prevention.workflows import InvalidStateTransitionError

router = APIRouter(prefix="/prevention", tags=["Automated Prevention & Response"])


@router.post(
    "/evaluate",
    response_model=PreventionEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate voice cloning risk against organizational policy and initiate prevention workflow."
)
async def evaluate_prevention_policy(req: PreventionEvaluationRequest):
    """
    Core Step 14 Endpoint:
    Translates dynamic impersonation risk and transaction context into concrete fraud prevention decisions
    (ALLOW, PAUSE, REQUIRE_CALLBACK, REQUIRE_MFA, BLOCK_TRANSACTION, ESCALATE_TO_SOC).
    """
    service = PreventionService.get_instance()
    try:
        response = service.evaluate_and_create_workflow(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prevention evaluation failed: {str(e)}")


@router.post(
    "/workflows",
    response_model=PreventionEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize a new fraud prevention workflow."
)
async def create_prevention_workflow(req: PreventionEvaluationRequest):
    """
    Initializes a tracked fraud prevention workflow for a call or transaction session.
    """
    service = PreventionService.get_instance()
    try:
        response = service.evaluate_and_create_workflow(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed creating prevention workflow: {str(e)}")


@router.get(
    "/workflows/{workflow_id}",
    response_model=PreventionEvaluationResponse,
    summary="Get prevention workflow state, required actions, and chronological incident timeline."
)
async def get_workflow_status(workflow_id: str):
    """
    Returns active prevention workflow state, blocking indicators, and privacy-preserving audit timeline.
    """
    service = PreventionService.get_instance()
    workflow = service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Prevention workflow '{workflow_id}' not found.")
    return workflow


@router.post(
    "/workflows/{workflow_id}/verify",
    response_model=PreventionEvaluationResponse,
    summary="Simulate an out-of-band verification challenge outcome (callback, MFA, supervisor approval)."
)
async def process_workflow_verification(workflow_id: str, req: VerificationSimulationRequest):
    """
    Simulates secondary verification challenge results:
    - callback_passed / callback_failed
    - mfa_passed / mfa_failed
    - supervisor_approved / supervisor_rejected
    """
    service = PreventionService.get_instance()
    try:
        updated = service.process_verification(
            workflow_id=workflow_id,
            verification_type=req.verification_type,
            actor=req.actor or "operator",
            notes=req.notes,
        )
        return updated
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification simulation failed: {str(e)}")


@router.post(
    "/workflows/{workflow_id}/resolve",
    response_model=PreventionEvaluationResponse,
    summary="Explicitly resolve or close an incident workflow following human review."
)
async def resolve_workflow_incident(workflow_id: str, req: PreventionResolutionRequest):
    """
    Operator or supervisor resolution of an incident workflow.
    """
    service = PreventionService.get_instance()
    try:
        resolved = service.resolve_workflow(
            workflow_id=workflow_id,
            reason=req.resolution_reason,
            resolved_by=req.resolved_by,
            final_action=req.final_action,
        )
        return resolved
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resolution error: {str(e)}")


@router.get(
    "/policies",
    response_model=List[Dict[str, Any]],
    summary="List available organizational policy profiles and threshold matrices."
)
async def list_prevention_policies():
    """
    Returns available policy profiles (BANKING, ENTERPRISE, GOVERNMENT, TELECOM, DEFAULT) and threshold rules.
    """
    service = PreventionService.get_instance()
    return service.get_policy_profiles()


@router.get(
    "/demo/scenarios",
    response_model=List[DemoScenario],
    summary="Get 5 pre-configured hackathon demonstration scenarios."
)
async def list_demo_scenarios():
    """
    Returns pre-configured scenarios directly addressing the hackathon problem statement.
    """
    service = PreventionService.get_instance()
    return service.get_demo_scenarios()
