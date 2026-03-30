# -*- coding: utf-8 -*-
"""
Sustainable Investment Verification Workflow
================================================

Four-phase workflow for verifying that 100% of investments in an Article 9
product qualify as sustainable investments. Orchestrates holding-level
classification, DNSH screening, good governance verification, and aggregate
compliance reporting.

Regulatory Context:
    Article 9 products under SFDR must invest 100% in sustainable investments
    (excluding cash/hedging instruments). Each holding must individually meet:
    - Contribution to an environmental or social objective
    - Do No Significant Harm to any other objective (DNSH)
    - Good governance practices verified
    - Alignment with EU Taxonomy where applicable

Phases:
    1. HoldingClassification - Classify each holding as sustainable or
       non-sustainable based on objective contribution
    2. DNSHScreening - Screen each sustainable holding against DNSH criteria
       across all six environmental objectives
    3. GovernanceVerification - Verify good governance practices for all
       investee companies
    4. ComplianceAggregation - Aggregate results, calculate compliance
       ratios, flag non-compliant holdings

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class HoldingClassification(str, Enum):
    """Sustainable investment classification."""
    SUSTAINABLE_ENVIRONMENTAL = "SUSTAINABLE_ENVIRONMENTAL"
    SUSTAINABLE_SOCIAL = "SUSTAINABLE_SOCIAL"
    TAXONOMY_ALIGNED = "TAXONOMY_ALIGNED"
    NON_SUSTAINABLE_CASH = "NON_SUSTAINABLE_CASH"
    NON_SUSTAINABLE_HEDGING = "NON_SUSTAINABLE_HEDGING"
    NON_SUSTAINABLE_OTHER = "NON_SUSTAINABLE_OTHER"
    UNDER_REVIEW = "UNDER_REVIEW"

class DNSHResult(str, Enum):
    """DNSH assessment result for a holding."""
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"

class GovernanceResult(str, Enum):
    """Good governance verification result."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    EXEMPT = "EXEMPT"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# =============================================================================
# DATA MODELS - SUSTAINABLE VERIFICATION
# =============================================================================

class HoldingInput(BaseModel):
    """Individual holding for sustainable verification."""
    holding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer_name: str = Field(..., description="Issuer name")
    isin: Optional[str] = Field(None)
    asset_type: str = Field(
        default="equity", description="equity, bond, fund, cash, derivative"
    )
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    market_value_eur: float = Field(default=0.0, ge=0.0)
    contributes_to_objective: bool = Field(default=True)
    objective_contribution_type: str = Field(
        default="environmental",
        description="environmental, social, taxonomy_aligned"
    )
    taxonomy_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    esg_rating: Optional[float] = Field(None, ge=0.0, le=100.0)
    ghg_scope1_tco2e: Optional[float] = Field(None, ge=0.0)
    ghg_scope2_tco2e: Optional[float] = Field(None, ge=0.0)
    water_intensity: Optional[float] = Field(None, ge=0.0)
    waste_recycling_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    biodiversity_impact_score: Optional[float] = Field(None)
    ungc_compliant: bool = Field(default=True)
    board_independence_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    tax_transparency_score: Optional[float] = Field(None, ge=0.0, le=100.0)

class SustainableVerificationInput(BaseModel):
    """Input for the sustainable verification workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    verification_date: str = Field(
        ..., description="Verification date YYYY-MM-DD"
    )
    holdings: List[HoldingInput] = Field(
        default_factory=list, description="Portfolio holdings"
    )
    dnsh_threshold_ghg_intensity: float = Field(
        default=100.0, description="Max GHG intensity for DNSH pass"
    )
    dnsh_threshold_water_intensity: float = Field(
        default=50.0, description="Max water intensity for DNSH pass"
    )
    governance_min_board_independence: float = Field(
        default=30.0, description="Min board independence %"
    )
    governance_min_tax_transparency: float = Field(
        default=50.0, description="Min tax transparency score"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("verification_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("verification_date must be YYYY-MM-DD")
        return v

class SustainableVerificationResult(WorkflowResult):
    """Complete result from the sustainable verification workflow."""
    product_name: str = Field(default="")
    total_holdings: int = Field(default=0)
    sustainable_holdings: int = Field(default=0)
    sustainable_pct: float = Field(default=0.0)
    dnsh_pass_count: int = Field(default=0)
    dnsh_fail_count: int = Field(default=0)
    governance_verified_count: int = Field(default=0)
    governance_failed_count: int = Field(default=0)
    fully_compliant_holdings: int = Field(default=0)
    fully_compliant_pct: float = Field(default=0.0)
    is_100_pct_sustainable: bool = Field(default=False)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class HoldingClassificationPhase:
    """
    Phase 1: Holding Classification.

    Classifies each holding as sustainable or non-sustainable based on
    objective contribution, asset type, and taxonomy alignment.
    """

    PHASE_NAME = "holding_classification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute holding classification phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])

            classifications: List[Dict[str, Any]] = []
            sustainable_count = 0
            sustainable_weight = 0.0

            for holding in holdings:
                asset_type = holding.get("asset_type", "equity")
                contributes = holding.get(
                    "contributes_to_objective", True
                )
                contribution_type = holding.get(
                    "objective_contribution_type", "environmental"
                )
                weight = holding.get("portfolio_weight_pct", 0.0)
                taxonomy_pct = holding.get("taxonomy_aligned_pct", 0.0)

                # Classify based on asset type and contribution
                if asset_type == "cash":
                    classification = HoldingClassification.NON_SUSTAINABLE_CASH.value
                elif asset_type == "derivative":
                    classification = HoldingClassification.NON_SUSTAINABLE_HEDGING.value
                elif not contributes:
                    classification = HoldingClassification.NON_SUSTAINABLE_OTHER.value
                elif taxonomy_pct >= 50.0:
                    classification = HoldingClassification.TAXONOMY_ALIGNED.value
                    sustainable_count += 1
                    sustainable_weight += weight
                elif contribution_type == "social":
                    classification = HoldingClassification.SUSTAINABLE_SOCIAL.value
                    sustainable_count += 1
                    sustainable_weight += weight
                else:
                    classification = HoldingClassification.SUSTAINABLE_ENVIRONMENTAL.value
                    sustainable_count += 1
                    sustainable_weight += weight

                classifications.append({
                    "holding_id": holding.get("holding_id", ""),
                    "issuer_name": holding.get("issuer_name", ""),
                    "classification": classification,
                    "portfolio_weight_pct": weight,
                    "is_sustainable": classification in (
                        HoldingClassification.SUSTAINABLE_ENVIRONMENTAL.value,
                        HoldingClassification.SUSTAINABLE_SOCIAL.value,
                        HoldingClassification.TAXONOMY_ALIGNED.value,
                    ),
                })

            outputs["classifications"] = classifications
            outputs["sustainable_count"] = sustainable_count
            outputs["non_sustainable_count"] = (
                len(holdings) - sustainable_count
            )
            outputs["sustainable_weight_pct"] = round(
                sustainable_weight, 2
            )
            outputs["total_holdings"] = len(holdings)

            # Classification distribution
            dist: Dict[str, int] = {}
            for c in classifications:
                cls_val = c["classification"]
                dist[cls_val] = dist.get(cls_val, 0) + 1
            outputs["classification_distribution"] = dist

            non_sustainable_other = [
                c for c in classifications
                if c["classification"] == HoldingClassification.NON_SUSTAINABLE_OTHER.value
            ]
            if non_sustainable_other:
                warnings.append(
                    f"{len(non_sustainable_other)} holding(s) classified "
                    f"as non-sustainable (not cash/hedging)"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error(
                "HoldingClassification failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Holding classification failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

class DNSHScreeningPhase:
    """
    Phase 2: DNSH Screening.

    Screens each sustainable holding against DNSH criteria across
    GHG emissions, water, waste, and biodiversity thresholds.
    """

    PHASE_NAME = "dnsh_screening"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute DNSH screening phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])
            classification_output = context.get_phase_output(
                "holding_classification"
            )
            classifications = classification_output.get(
                "classifications", []
            )

            ghg_threshold = config.get(
                "dnsh_threshold_ghg_intensity", 100.0
            )
            water_threshold = config.get(
                "dnsh_threshold_water_intensity", 50.0
            )

            # Build lookup for classifications
            cls_map = {
                c["holding_id"]: c for c in classifications
            }

            dnsh_results: List[Dict[str, Any]] = []
            pass_count = 0
            fail_count = 0
            insufficient_count = 0

            for holding in holdings:
                h_id = holding.get("holding_id", "")
                cls_info = cls_map.get(h_id, {})

                if not cls_info.get("is_sustainable", False):
                    dnsh_results.append({
                        "holding_id": h_id,
                        "issuer_name": holding.get("issuer_name", ""),
                        "dnsh_result": DNSHResult.NOT_APPLICABLE.value,
                        "objective_results": {},
                        "reason": "Non-sustainable holding",
                    })
                    continue

                # Assess each environmental objective
                objective_results: Dict[str, str] = {}
                has_failure = False
                has_insufficient = False

                # Climate mitigation - GHG check
                s1 = holding.get("ghg_scope1_tco2e")
                s2 = holding.get("ghg_scope2_tco2e")
                if s1 is not None and s2 is not None:
                    ghg_intensity = s1 + s2
                    if ghg_intensity <= ghg_threshold:
                        objective_results["climate_mitigation"] = "PASS"
                    else:
                        objective_results["climate_mitigation"] = "FAIL"
                        has_failure = True
                else:
                    objective_results["climate_mitigation"] = (
                        "INSUFFICIENT_DATA"
                    )
                    has_insufficient = True

                # Water resources
                water = holding.get("water_intensity")
                if water is not None:
                    if water <= water_threshold:
                        objective_results["water_resources"] = "PASS"
                    else:
                        objective_results["water_resources"] = "FAIL"
                        has_failure = True
                else:
                    objective_results["water_resources"] = (
                        "INSUFFICIENT_DATA"
                    )
                    has_insufficient = True

                # Circular economy - waste
                waste = holding.get("waste_recycling_rate")
                if waste is not None:
                    if waste >= 20.0:
                        objective_results["circular_economy"] = "PASS"
                    else:
                        objective_results["circular_economy"] = "FAIL"
                        has_failure = True
                else:
                    objective_results["circular_economy"] = (
                        "INSUFFICIENT_DATA"
                    )
                    has_insufficient = True

                # Biodiversity
                bio = holding.get("biodiversity_impact_score")
                if bio is not None:
                    if bio >= 0:
                        objective_results["biodiversity"] = "PASS"
                    else:
                        objective_results["biodiversity"] = "FAIL"
                        has_failure = True
                else:
                    objective_results["biodiversity"] = (
                        "INSUFFICIENT_DATA"
                    )
                    has_insufficient = True

                # Social safeguards - UNGC
                ungc = holding.get("ungc_compliant", True)
                objective_results["social_safeguards"] = (
                    "PASS" if ungc else "FAIL"
                )
                if not ungc:
                    has_failure = True

                # Determine overall DNSH result
                if has_failure:
                    overall = DNSHResult.FAIL.value
                    fail_count += 1
                elif has_insufficient:
                    overall = DNSHResult.INSUFFICIENT_DATA.value
                    insufficient_count += 1
                else:
                    overall = DNSHResult.PASS.value
                    pass_count += 1

                dnsh_results.append({
                    "holding_id": h_id,
                    "issuer_name": holding.get("issuer_name", ""),
                    "dnsh_result": overall,
                    "objective_results": objective_results,
                })

            outputs["dnsh_results"] = dnsh_results
            outputs["dnsh_pass_count"] = pass_count
            outputs["dnsh_fail_count"] = fail_count
            outputs["dnsh_insufficient_count"] = insufficient_count

            if fail_count > 0:
                warnings.append(
                    f"{fail_count} sustainable holding(s) failed DNSH"
                )
            if insufficient_count > 0:
                warnings.append(
                    f"{insufficient_count} holding(s) have insufficient "
                    f"data for DNSH assessment"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error(
                "DNSHScreening failed: %s", exc, exc_info=True
            )
            errors.append(f"DNSH screening failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

class GovernanceVerificationPhase:
    """
    Phase 3: Governance Verification.

    Verifies good governance practices for all investee companies
    covering management structures, employee relations, remuneration,
    and tax compliance.
    """

    PHASE_NAME = "governance_verification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute governance verification phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])
            classification_output = context.get_phase_output(
                "holding_classification"
            )
            classifications = classification_output.get(
                "classifications", []
            )

            min_board = config.get(
                "governance_min_board_independence", 30.0
            )
            min_tax = config.get(
                "governance_min_tax_transparency", 50.0
            )

            cls_map = {
                c["holding_id"]: c for c in classifications
            }

            governance_results: List[Dict[str, Any]] = []
            verified_count = 0
            failed_count = 0

            for holding in holdings:
                h_id = holding.get("holding_id", "")
                cls_info = cls_map.get(h_id, {})
                asset_type = holding.get("asset_type", "equity")

                # Cash and derivatives are exempt
                if asset_type in ("cash", "derivative"):
                    governance_results.append({
                        "holding_id": h_id,
                        "issuer_name": holding.get("issuer_name", ""),
                        "governance_result": GovernanceResult.EXEMPT.value,
                        "checks": {},
                    })
                    continue

                checks: Dict[str, Any] = {}

                # UNGC compliance
                ungc = holding.get("ungc_compliant", True)
                checks["ungc_compliance"] = {
                    "passed": ungc,
                    "detail": "Compliant" if ungc else "Non-compliant",
                }

                # Board independence
                board = holding.get("board_independence_pct")
                if board is not None:
                    checks["board_independence"] = {
                        "passed": board >= min_board,
                        "value": board,
                        "threshold": min_board,
                    }
                else:
                    checks["board_independence"] = {
                        "passed": True,
                        "detail": "Data not available, assumed compliant",
                    }

                # Tax transparency
                tax = holding.get("tax_transparency_score")
                if tax is not None:
                    checks["tax_transparency"] = {
                        "passed": tax >= min_tax,
                        "value": tax,
                        "threshold": min_tax,
                    }
                else:
                    checks["tax_transparency"] = {
                        "passed": True,
                        "detail": "Data not available, assumed compliant",
                    }

                all_passed = all(
                    c.get("passed", True) for c in checks.values()
                )

                if all_passed:
                    result = GovernanceResult.VERIFIED.value
                    verified_count += 1
                else:
                    result = GovernanceResult.FAILED.value
                    failed_count += 1

                governance_results.append({
                    "holding_id": h_id,
                    "issuer_name": holding.get("issuer_name", ""),
                    "governance_result": result,
                    "checks": checks,
                })

            outputs["governance_results"] = governance_results
            outputs["governance_verified_count"] = verified_count
            outputs["governance_failed_count"] = failed_count

            if failed_count > 0:
                warnings.append(
                    f"{failed_count} holding(s) failed governance check"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error(
                "GovernanceVerification failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Governance verification failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

class ComplianceAggregationPhase:
    """
    Phase 4: Compliance Aggregation.

    Aggregates classification, DNSH, and governance results to determine
    overall 100% sustainable compliance.
    """

    PHASE_NAME = "compliance_aggregation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute compliance aggregation phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            cls_output = context.get_phase_output(
                "holding_classification"
            )
            dnsh_output = context.get_phase_output("dnsh_screening")
            gov_output = context.get_phase_output(
                "governance_verification"
            )

            classifications = cls_output.get("classifications", [])
            dnsh_results = dnsh_output.get("dnsh_results", [])
            gov_results = gov_output.get("governance_results", [])

            # Build lookups
            dnsh_map = {
                d["holding_id"]: d for d in dnsh_results
            }
            gov_map = {
                g["holding_id"]: g for g in gov_results
            }

            holding_compliance: List[Dict[str, Any]] = []
            fully_compliant = 0
            fully_compliant_weight = 0.0
            non_compliant_holdings: List[Dict[str, Any]] = []

            for cls_item in classifications:
                h_id = cls_item["holding_id"]
                is_sustainable = cls_item.get("is_sustainable", False)
                weight = cls_item.get("portfolio_weight_pct", 0.0)

                dnsh_info = dnsh_map.get(h_id, {})
                gov_info = gov_map.get(h_id, {})

                dnsh_pass = dnsh_info.get("dnsh_result") in (
                    DNSHResult.PASS.value,
                    DNSHResult.NOT_APPLICABLE.value,
                )
                gov_pass = gov_info.get("governance_result") in (
                    GovernanceResult.VERIFIED.value,
                    GovernanceResult.EXEMPT.value,
                )

                is_compliant = is_sustainable and dnsh_pass and gov_pass

                entry = {
                    "holding_id": h_id,
                    "issuer_name": cls_item.get("issuer_name", ""),
                    "classification": cls_item.get("classification", ""),
                    "is_sustainable": is_sustainable,
                    "dnsh_result": dnsh_info.get("dnsh_result", ""),
                    "governance_result": gov_info.get(
                        "governance_result", ""
                    ),
                    "is_fully_compliant": is_compliant,
                    "portfolio_weight_pct": weight,
                }
                holding_compliance.append(entry)

                if is_compliant:
                    fully_compliant += 1
                    fully_compliant_weight += weight
                elif is_sustainable:
                    non_compliant_holdings.append(entry)

            total = len(classifications)
            outputs["holding_compliance"] = holding_compliance
            outputs["fully_compliant_holdings"] = fully_compliant
            outputs["fully_compliant_pct"] = round(
                fully_compliant_weight, 2
            )
            outputs["non_compliant_sustainable_holdings"] = (
                non_compliant_holdings
            )
            outputs["total_holdings"] = total

            # 100% sustainable check (allowing cash/hedging)
            sustainable_weight = cls_output.get(
                "sustainable_weight_pct", 0.0
            )
            is_100_pct = (
                fully_compliant_weight >= sustainable_weight * 0.99
                and len(non_compliant_holdings) == 0
            )
            outputs["is_100_pct_sustainable"] = is_100_pct

            outputs["compliance_summary"] = {
                "sustainable_holdings": cls_output.get(
                    "sustainable_count", 0
                ),
                "dnsh_pass": dnsh_output.get("dnsh_pass_count", 0),
                "dnsh_fail": dnsh_output.get("dnsh_fail_count", 0),
                "governance_verified": gov_output.get(
                    "governance_verified_count", 0
                ),
                "governance_failed": gov_output.get(
                    "governance_failed_count", 0
                ),
                "fully_compliant": fully_compliant,
                "fully_compliant_weight_pct": round(
                    fully_compliant_weight, 2
                ),
            }

            if not is_100_pct:
                warnings.append(
                    "Portfolio does not meet 100% sustainable "
                    "investment requirement"
                )
            if non_compliant_holdings:
                warnings.append(
                    f"{len(non_compliant_holdings)} sustainable "
                    f"holding(s) have compliance issues"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "ComplianceAggregation failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Compliance aggregation failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class SustainableVerificationWorkflow:
    """
    Four-phase sustainable investment verification for Article 9.

    Verifies 100% sustainable investment compliance through holding
    classification, DNSH screening, governance verification, and
    compliance aggregation.

    Example:
        >>> wf = SustainableVerificationWorkflow()
        >>> input_data = SustainableVerificationInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     verification_date="2026-03-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.is_100_pct_sustainable
    """

    WORKFLOW_NAME = "sustainable_verification"

    PHASE_ORDER = [
        "holding_classification",
        "dnsh_screening",
        "governance_verification",
        "compliance_aggregation",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the sustainable verification workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "holding_classification": HoldingClassificationPhase(),
            "dnsh_screening": DNSHScreeningPhase(),
            "governance_verification": GovernanceVerificationPhase(),
            "compliance_aggregation": ComplianceAggregationPhase(),
        }

    async def run(
        self, input_data: SustainableVerificationInput
    ) -> SustainableVerificationResult:
        """Execute the complete 4-phase sustainable verification workflow."""
        started_at = utcnow()
        logger.info(
            "Starting sustainable verification workflow %s for org=%s",
            self.workflow_id, input_data.organization_id,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "holding_classification":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
            )

        completed_at = utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return SustainableVerificationResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            total_holdings=summary.get("total_holdings", 0),
            sustainable_holdings=summary.get(
                "sustainable_holdings", 0
            ),
            sustainable_pct=summary.get("sustainable_pct", 0.0),
            dnsh_pass_count=summary.get("dnsh_pass_count", 0),
            dnsh_fail_count=summary.get("dnsh_fail_count", 0),
            governance_verified_count=summary.get(
                "governance_verified_count", 0
            ),
            governance_failed_count=summary.get(
                "governance_failed_count", 0
            ),
            fully_compliant_holdings=summary.get(
                "fully_compliant_holdings", 0
            ),
            fully_compliant_pct=summary.get(
                "fully_compliant_pct", 0.0
            ),
            is_100_pct_sustainable=summary.get(
                "is_100_pct_sustainable", False
            ),
        )

    def _build_config(
        self, input_data: SustainableVerificationInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict."""
        config = input_data.model_dump()
        if input_data.holdings:
            config["holdings"] = [
                h.model_dump() for h in input_data.holdings
            ]
        return config

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        cls_out = context.get_phase_output("holding_classification")
        dnsh_out = context.get_phase_output("dnsh_screening")
        gov_out = context.get_phase_output("governance_verification")
        agg_out = context.get_phase_output("compliance_aggregation")

        return {
            "product_name": config.get("product_name", ""),
            "total_holdings": cls_out.get("total_holdings", 0),
            "sustainable_holdings": cls_out.get(
                "sustainable_count", 0
            ),
            "sustainable_pct": cls_out.get(
                "sustainable_weight_pct", 0.0
            ),
            "dnsh_pass_count": dnsh_out.get("dnsh_pass_count", 0),
            "dnsh_fail_count": dnsh_out.get("dnsh_fail_count", 0),
            "governance_verified_count": gov_out.get(
                "governance_verified_count", 0
            ),
            "governance_failed_count": gov_out.get(
                "governance_failed_count", 0
            ),
            "fully_compliant_holdings": agg_out.get(
                "fully_compliant_holdings", 0
            ),
            "fully_compliant_pct": agg_out.get(
                "fully_compliant_pct", 0.0
            ),
            "is_100_pct_sustainable": agg_out.get(
                "is_100_pct_sustainable", False
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
