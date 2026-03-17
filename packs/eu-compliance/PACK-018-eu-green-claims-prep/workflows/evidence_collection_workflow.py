# -*- coding: utf-8 -*-
"""
Evidence Collection Workflow - PACK-018 EU Green Claims Prep
==============================================================

5-phase workflow that systematically collects, validates, and archives
the evidence chain required to substantiate environmental claims under
the EU Green Claims Directive. Covers requirement identification,
evidence gathering, validation against Directive criteria, chain-of-
custody building, and long-term archival with full provenance tracking.

Phases:
    1. RequirementsIdentification -- Determine what evidence is needed
    2. EvidenceGathering          -- Collect evidence artefacts
    3. Validation                 -- Validate evidence quality/relevance
    4. ChainBuilding              -- Build chain-of-custody for audit trail
    5. Archival                   -- Archive evidence for regulatory retention

Reference:
    EU Green Claims Directive (COM/2023/166)
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceType(str, Enum):
    """Recognised evidence categories under the Directive."""
    LCA_REPORT = "lca_report"
    LIFECYCLE_ASSESSMENT = "lifecycle_assessment"
    THIRD_PARTY_VERIFICATION = "third_party_verification"
    TEST_REPORT = "test_report"
    LAB_TEST = "lab_test"
    CERTIFICATION = "certification"
    SUPPLIER_DECLARATION = "supplier_declaration"
    AUDIT_REPORT = "audit_report"
    MONITORING_DATA = "monitoring_data"
    OTHER = "other"


class ValidationOutcome(str, Enum):
    """Outcome of evidence validation."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


# =============================================================================
# DATA MODELS
# =============================================================================


class WorkflowInput(BaseModel):
    """Input model for EvidenceCollectionWorkflow."""
    claim_id: str = Field(default="", description="ID of the claim requiring evidence")
    required_evidence_types: List[str] = Field(
        default_factory=list,
        description="List of evidence type strings required for the claim",
    )
    existing_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence artefacts already available",
    )
    claim_text: str = Field(default="", description="Text of the claim being substantiated")
    entity_name: str = Field(default="", description="Reporting entity name")
    config: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class WorkflowResult(BaseModel):
    """Complete result from EvidenceCollectionWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="evidence_collection")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EvidenceCollectionWorkflow:
    """
    5-phase evidence collection workflow for EU Green Claims substantiation.

    Determines evidence requirements for a claim, gathers and validates
    evidence artefacts, builds a chain-of-custody record, and archives
    everything for regulatory retention periods.

    Zero-hallucination: all scoring and gap analysis uses deterministic
    set operations and arithmetic. No LLM calls in calculation paths.

    Example:
        >>> wf = EvidenceCollectionWorkflow()
        >>> result = wf.execute(
        ...     claim_id="CLM-001",
        ...     required_evidence_types=["lca_report", "certification"],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "evidence_collection"

    # Minimum evidence requirements per claim category
    DEFAULT_REQUIREMENTS: Dict[str, List[str]] = {
        "environmental": ["lca_report", "third_party_verification"],
        "carbon": ["lca_report", "third_party_verification", "monitoring_data"],
        "circular": ["test_report", "certification"],
        "biodiversity": ["audit_report", "monitoring_data"],
        "generic": ["lca_report"],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EvidenceCollectionWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 5-phase evidence collection pipeline.

        Keyword Args:
            claim_id: Identifier of the target claim.
            required_evidence_types: List of evidence type strings needed.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            claim_id=kwargs.get("claim_id", ""),
            required_evidence_types=kwargs.get("required_evidence_types", []),
            existing_evidence=kwargs.get("existing_evidence", []),
            claim_text=kwargs.get("claim_text", ""),
            entity_name=kwargs.get("entity_name", ""),
            config=kwargs.get("config", {}),
        )

        started_at = _utcnow()
        self.logger.info("Starting %s workflow %s for claim %s",
                         self.WORKFLOW_NAME, self.workflow_id, input_data.claim_id)
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Requirements Identification
            phase_results.append(self._phase_requirements_identification(input_data))

            # Phase 2 -- Evidence Gathering
            requirements = phase_results[0].result_data
            phase_results.append(self._phase_evidence_gathering(input_data, requirements))

            # Phase 3 -- Validation
            gathered = phase_results[1].result_data
            phase_results.append(self._phase_validation(input_data, gathered))

            # Phase 4 -- Chain Building
            validated = phase_results[2].result_data
            phase_results.append(self._phase_chain_building(input_data, validated))

            # Phase 5 -- Archival
            chain_data = phase_results[3].result_data
            phase_results.append(self._phase_archival(input_data, chain_data))

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=_utcnow(),
                completed_at=_utcnow(),
                error_message=str(exc),
            ))

        completed_at = _utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "claim_id": input_data.claim_id,
            "phases_completed": len(completed_phases),
            "phases_total": 5,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Workflow %s %s -- claim %s",
                         self.workflow_id, overall_status.value, input_data.claim_id)
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Phase 1: Requirements Identification
    # ------------------------------------------------------------------

    def _phase_requirements_identification(self, input_data: WorkflowInput) -> PhaseResult:
        """Determine the evidence types required for the claim."""
        started = _utcnow()
        self.logger.info("Phase 1/5 RequirementsIdentification")

        # If caller supplied explicit types, use them; otherwise derive defaults
        if input_data.required_evidence_types:
            required = list(input_data.required_evidence_types)
        else:
            required = list(self.DEFAULT_REQUIREMENTS.get("environmental", []))

        existing_types = {e.get("type", "unknown") for e in input_data.existing_evidence}
        missing = [r for r in required if r not in existing_types]
        covered = [r for r in required if r in existing_types]

        result_data: Dict[str, Any] = {
            "required_types": required,
            "existing_types": sorted(existing_types),
            "missing_types": missing,
            "covered_types": covered,
            "coverage_pct": round(
                (len(covered) / len(required) * 100) if required else 100.0, 1
            ),
        }

        return PhaseResult(
            phase_name="RequirementsIdentification",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Evidence Gathering
    # ------------------------------------------------------------------

    def _phase_evidence_gathering(
        self, input_data: WorkflowInput, requirements: Dict[str, Any],
    ) -> PhaseResult:
        """Catalogue available evidence items and tag gaps."""
        started = _utcnow()
        self.logger.info("Phase 2/5 EvidenceGathering")

        inventory: List[Dict[str, Any]] = []
        for idx, ev in enumerate(input_data.existing_evidence):
            inventory.append({
                "evidence_id": ev.get("id", f"ev-{idx}"),
                "type": ev.get("type", "unknown"),
                "source": ev.get("source", "unknown"),
                "date": ev.get("date", ""),
                "size_bytes": ev.get("size_bytes", 0),
                "matched_requirement": ev.get("type", "unknown") in requirements.get("required_types", []),
            })

        gap_items: List[Dict[str, str]] = []
        for missing_type in requirements.get("missing_types", []):
            gap_items.append({
                "type": missing_type,
                "action": f"Collect {missing_type} evidence for claim {input_data.claim_id}",
                "priority": "high" if missing_type in ("lca_report", "third_party_verification") else "medium",
            })

        result_data: Dict[str, Any] = {
            "inventory": inventory,
            "inventory_count": len(inventory),
            "gap_items": gap_items,
            "gap_count": len(gap_items),
        }

        return PhaseResult(
            phase_name="EvidenceGathering",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Validation
    # ------------------------------------------------------------------

    def _phase_validation(
        self, input_data: WorkflowInput, gathered: Dict[str, Any],
    ) -> PhaseResult:
        """Validate each evidence item for quality, recency, and relevance."""
        started = _utcnow()
        self.logger.info("Phase 3/5 Validation")

        validated_items: List[Dict[str, Any]] = []
        accepted_count = 0
        rejected_count = 0

        for item in gathered.get("inventory", []):
            outcome = self._validate_evidence_item(item)
            validated_items.append({
                **item,
                "validation_outcome": outcome.value,
                "validation_hash": _compute_hash(item),
            })
            if outcome == ValidationOutcome.ACCEPTED:
                accepted_count += 1
            elif outcome == ValidationOutcome.REJECTED:
                rejected_count += 1

        total = len(validated_items)
        result_data: Dict[str, Any] = {
            "validated_items": validated_items,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "needs_review_count": total - accepted_count - rejected_count,
            "acceptance_rate_pct": round(
                (accepted_count / total * 100) if total else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name="Validation",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Chain Building
    # ------------------------------------------------------------------

    def _phase_chain_building(
        self, input_data: WorkflowInput, validated: Dict[str, Any],
    ) -> PhaseResult:
        """Build chain-of-custody linking claim to validated evidence."""
        started = _utcnow()
        self.logger.info("Phase 4/5 ChainBuilding")

        chain_entries: List[Dict[str, Any]] = []
        accepted_items = [
            v for v in validated.get("validated_items", [])
            if v.get("validation_outcome") == ValidationOutcome.ACCEPTED.value
        ]

        for seq, item in enumerate(accepted_items, start=1):
            chain_entries.append({
                "sequence": seq,
                "evidence_id": item.get("evidence_id", ""),
                "evidence_type": item.get("type", ""),
                "claim_id": input_data.claim_id,
                "validation_hash": item.get("validation_hash", ""),
                "chain_timestamp": _utcnow().isoformat(),
            })

        chain_hash = _compute_hash(chain_entries)

        result_data: Dict[str, Any] = {
            "chain_entries": chain_entries,
            "chain_length": len(chain_entries),
            "chain_hash": chain_hash,
            "claim_id": input_data.claim_id,
            "is_complete": len(chain_entries) >= len(
                validated.get("validated_items", [])
            ) and len(chain_entries) > 0,
        }

        return PhaseResult(
            phase_name="ChainBuilding",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 5: Archival
    # ------------------------------------------------------------------

    def _phase_archival(
        self, input_data: WorkflowInput, chain_data: Dict[str, Any],
    ) -> PhaseResult:
        """Archive the evidence chain for regulatory retention compliance."""
        started = _utcnow()
        self.logger.info("Phase 5/5 Archival")

        # Directive requires evidence retention for the lifetime of the claim
        # plus a regulatory buffer (typically 5 years)
        retention_years = self.config.get("retention_years", 10)

        archive_record: Dict[str, Any] = {
            "archive_id": _new_uuid(),
            "claim_id": input_data.claim_id,
            "entity_name": input_data.entity_name,
            "chain_hash": chain_data.get("chain_hash", ""),
            "chain_length": chain_data.get("chain_length", 0),
            "archived_at": _utcnow().isoformat(),
            "retention_years": retention_years,
            "retention_expiry": str(
                _utcnow().year + retention_years
            ),
            "storage_format": "json",
            "is_complete": chain_data.get("is_complete", False),
        }

        result_data: Dict[str, Any] = {
            "archive_record": archive_record,
            "archival_status": "archived",
            "retention_years": retention_years,
            "evidence_chain_complete": chain_data.get("is_complete", False),
        }

        return PhaseResult(
            phase_name="Archival",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_evidence_item(self, item: Dict[str, Any]) -> ValidationOutcome:
        """Validate a single evidence item deterministically."""
        # Reject if type is unknown
        if item.get("type", "unknown") == "unknown":
            return ValidationOutcome.REJECTED

        # Reject if no source attribution
        if not item.get("source"):
            return ValidationOutcome.REJECTED

        # Flag for review if no date (cannot confirm recency)
        if not item.get("date"):
            return ValidationOutcome.NEEDS_REVIEW

        # Accept if matched to a requirement
        if item.get("matched_requirement", False):
            return ValidationOutcome.ACCEPTED

        return ValidationOutcome.NEEDS_REVIEW
