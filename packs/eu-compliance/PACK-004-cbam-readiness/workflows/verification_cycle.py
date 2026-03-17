# -*- coding: utf-8 -*-
"""
Verification Cycle Workflow
============================

Five-phase verification engagement workflow for CBAM compliance. Manages the
end-to-end process of selecting an accredited verifier, defining scope,
preparing evidence, tracking verification execution, and resolving findings.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 8: Embedded emissions in imported goods must be verified by an
      accredited verifier before the annual CBAM declaration is submitted
    - Article 18: Verifiers must be accredited by a national accreditation
      body per Regulation (EC) No 765/2008
    - Implementing Regulation 2023/1773 Article 8: Verification must follow
      the principles of EU ETS verification (Regulation 2018/2067)

    Verification requirements escalate over time:
        - 2026-2027: Reasonable assurance for large importers, limited for small
        - 2028+: Mandatory reasonable assurance verification for all declarants

    Materiality thresholds:
        - Quantitative: typically 5% of total embedded emissions
        - Qualitative: any individual misstatement in methodology

Phases:
    1. Verifier selection - Select accredited verifier, check specialization
    2. Scope definition - Define verification scope, materiality thresholds
    3. Evidence preparation - Package emission data and calculation evidence
    4. Verification execution - Track progress, document reviews
    5. Finding resolution - Address findings, corrective actions, obtain statement

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class AssuranceLevel(str, Enum):
    """Verification assurance level."""
    LIMITED = "limited"
    REASONABLE = "reasonable"


class FindingSeverity(str, Enum):
    """Verification finding severity."""
    MAJOR = "major"             # Material misstatement requiring correction
    MINOR = "minor"             # Non-material issue, improvement recommended
    OBSERVATION = "observation"  # Advisory note, no action required


class FindingStatus(str, Enum):
    """Status of a verification finding resolution."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"  # Accepted by verifier as resolved


class VerificationOutcome(str, Enum):
    """Overall verification outcome."""
    VERIFIED = "verified"                      # Clean opinion
    VERIFIED_WITH_COMMENTS = "verified_with_comments"  # Qualified opinion
    NOT_VERIFIED = "not_verified"              # Adverse opinion
    PENDING = "pending"                        # Verification in progress


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class VerifierCandidate(BaseModel):
    """Accredited verifier candidate information."""
    verifier_id: str = Field(..., description="Verifier identifier")
    organization_name: str = Field(..., description="Verifier organization name")
    accreditation_body: str = Field(default="", description="National accreditation body")
    accreditation_number: str = Field(default="")
    sector_specializations: List[str] = Field(default_factory=list)
    assurance_levels: List[AssuranceLevel] = Field(default_factory=list)
    country: str = Field(default="")
    available_from: Optional[str] = Field(None, description="Earliest availability YYYY-MM-DD")
    estimated_fee_eur: Optional[float] = Field(None, ge=0)


class VerificationFinding(BaseModel):
    """A single verification finding."""
    finding_id: str = Field(...)
    severity: FindingSeverity = Field(...)
    title: str = Field(...)
    description: str = Field(default="")
    affected_area: str = Field(default="", description="e.g. 'cement emissions calculation'")
    corrective_action: str = Field(default="")
    status: FindingStatus = Field(default=FindingStatus.OPEN)
    response: str = Field(default="", description="Declarant's response")
    resolution_date: Optional[str] = Field(None)


class VerificationResult(BaseModel):
    """Complete result from the verification cycle workflow."""
    workflow_name: str = Field(default="verification_cycle")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    engagement_id: str = Field(..., description="Verification engagement identifier")
    verifier_selected: Optional[str] = Field(None, description="Selected verifier ID")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.REASONABLE)
    verification_outcome: VerificationOutcome = Field(default=VerificationOutcome.PENDING)
    findings_count: int = Field(default=0, ge=0)
    major_findings: int = Field(default=0, ge=0)
    minor_findings: int = Field(default=0, ge=0)
    all_findings_resolved: bool = Field(default=False)
    verification_statement_ref: Optional[str] = Field(None)
    materiality_threshold_pct: float = Field(default=5.0)
    evidence_packages_count: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# VERIFICATION CYCLE WORKFLOW
# =============================================================================


class VerificationCycleWorkflow:
    """
    Five-phase verification engagement workflow for CBAM compliance.

    Manages the complete verification process from verifier selection through
    to obtaining a verification statement. Tracks findings, corrective
    actions, and ensures all major findings are resolved before the
    verification opinion is finalized.

    Attributes:
        config: Optional configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.

    Example:
        >>> wf = VerificationCycleWorkflow()
        >>> result = await wf.execute(
        ...     config={"organization_id": "org-123"},
        ...     engagement_data={"year": 2026, "sectors": ["cement", "iron_steel"]},
        ... )
        >>> assert result.verification_outcome in (
        ...     VerificationOutcome.VERIFIED,
        ...     VerificationOutcome.VERIFIED_WITH_COMMENTS,
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the VerificationCycleWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.VerificationCycleWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Optional[Dict[str, Any]],
        engagement_data: Dict[str, Any],
    ) -> VerificationResult:
        """
        Execute the full 5-phase verification cycle workflow.

        Args:
            config: Execution-level config overrides.
            engagement_data: Verification engagement configuration including
                year, sectors, emission data references.

        Returns:
            VerificationResult with findings, outcome, and statement reference.
        """
        started_at = datetime.utcnow()
        merged_config = {**self.config, **(config or {})}
        engagement_id = f"VER-{self._execution_id[:12]}"

        self.logger.info(
            "Starting verification cycle execution_id=%s engagement=%s",
            self._execution_id, engagement_id,
        )

        context: Dict[str, Any] = {
            "config": merged_config,
            "engagement_data": engagement_data,
            "engagement_id": engagement_id,
            "execution_id": self._execution_id,
        }

        phase_handlers = [
            ("verifier_selection", self._phase_1_verifier_selection),
            ("scope_definition", self._phase_2_scope_definition),
            ("evidence_preparation", self._phase_3_evidence_preparation),
            ("verification_execution", self._phase_4_verification_execution),
            ("finding_resolution", self._phase_5_finding_resolution),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase_name, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase_name)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (
                    datetime.utcnow() - phase_start
                ).total_seconds()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase_name, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "verifier_selection":
                    break

        completed_at = datetime.utcnow()

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "engagement_id": engagement_id,
        })

        self.logger.info(
            "Verification cycle finished execution_id=%s outcome=%s",
            self._execution_id,
            context.get("verification_outcome", VerificationOutcome.PENDING).value
            if isinstance(context.get("verification_outcome"), VerificationOutcome)
            else context.get("verification_outcome", "pending"),
        )

        return VerificationResult(
            status=overall_status,
            phases=self._phase_results,
            engagement_id=engagement_id,
            verifier_selected=context.get("selected_verifier_id"),
            assurance_level=context.get("assurance_level", AssuranceLevel.REASONABLE),
            verification_outcome=context.get("verification_outcome", VerificationOutcome.PENDING),
            findings_count=context.get("findings_count", 0),
            major_findings=context.get("major_findings", 0),
            minor_findings=context.get("minor_findings", 0),
            all_findings_resolved=context.get("all_findings_resolved", False),
            verification_statement_ref=context.get("verification_statement_ref"),
            materiality_threshold_pct=context.get("materiality_threshold_pct", 5.0),
            evidence_packages_count=context.get("evidence_packages_count", 0),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Verifier Selection
    # -------------------------------------------------------------------------

    async def _phase_1_verifier_selection(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Select an accredited verifier with appropriate specialization.

        Evaluates verifier candidates based on:
            - Accreditation status (must be accredited per Regulation 765/2008)
            - Sector specialization match
            - Assurance level capability
            - Availability and pricing
        """
        phase_name = "verifier_selection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        engagement_data = context.get("engagement_data", {})
        required_sectors = engagement_data.get("sectors", [])
        required_assurance = AssuranceLevel(
            engagement_data.get("assurance_level", AssuranceLevel.REASONABLE.value)
        )

        # Fetch available verifiers
        candidates = await self._fetch_verifier_candidates(required_sectors, required_assurance)

        if not candidates:
            warnings.append(
                "No accredited verifiers found matching requirements. "
                "Expand search criteria or contact national accreditation body."
            )
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No suitable verifiers found", "candidates_count": 0},
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "found": 0}),
            )

        # Score and rank candidates
        ranked = self._rank_verifier_candidates(candidates, required_sectors, required_assurance)

        # Select top candidate
        selected = ranked[0]
        context["selected_verifier_id"] = selected.get("verifier_id")
        context["selected_verifier"] = selected
        context["assurance_level"] = required_assurance

        outputs["candidates_evaluated"] = len(candidates)
        outputs["selected_verifier_id"] = selected.get("verifier_id")
        outputs["selected_verifier_name"] = selected.get("organization_name", "")
        outputs["accreditation_number"] = selected.get("accreditation_number", "")
        outputs["estimated_fee_eur"] = selected.get("estimated_fee_eur", 0)
        outputs["assurance_level"] = required_assurance.value

        self.logger.info(
            "Phase 1 complete: selected verifier=%s from %d candidates",
            selected.get("verifier_id"), len(candidates),
        )

        provenance = self._hash({
            "phase": phase_name,
            "selected": selected.get("verifier_id"),
            "candidates": len(candidates),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Scope Definition
    # -------------------------------------------------------------------------

    async def _phase_2_scope_definition(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Define verification scope and materiality thresholds.

        Scope includes:
            - Goods categories and CN codes to be verified
            - Emission calculation methodologies to be reviewed
            - Materiality threshold (typically 5% of total emissions)
            - Sampling approach for supplier data
            - Period covered
        """
        phase_name = "scope_definition"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        engagement_data = context.get("engagement_data", {})
        year = engagement_data.get("year", datetime.utcnow().year)
        sectors = engagement_data.get("sectors", [])
        total_emissions = engagement_data.get("total_emissions_tco2e", 0)

        # Define materiality threshold
        materiality_pct = context.get("config", {}).get("materiality_threshold_pct", 5.0)
        materiality_absolute = total_emissions * (materiality_pct / 100)

        # Define scope document
        scope = {
            "engagement_id": context["engagement_id"],
            "year": year,
            "period_start": f"{year}-01-01",
            "period_end": f"{year}-12-31",
            "sectors_in_scope": sectors,
            "total_emissions_tco2e": total_emissions,
            "materiality_threshold_pct": materiality_pct,
            "materiality_threshold_tco2e": round(materiality_absolute, 4),
            "assurance_level": context.get("assurance_level", AssuranceLevel.REASONABLE).value,
            "scope_areas": [
                "Import data completeness and accuracy",
                "Emission factor selection and application",
                "Calculation methodology compliance",
                "Supplier data verification",
                "Free allocation and carbon price deductions",
                "CN code classification accuracy",
            ],
            "sampling_approach": "Risk-based sampling of supplier emission data",
            "key_risks": [
                "Use of default emission factors instead of actual data",
                "Incorrect CN code classification",
                "Incomplete supplier data coverage",
                "Calculation errors in embedded emissions",
            ],
        }

        context["verification_scope"] = scope
        context["materiality_threshold_pct"] = materiality_pct

        outputs["scope"] = scope
        outputs["sectors_count"] = len(sectors)
        outputs["materiality_pct"] = materiality_pct
        outputs["materiality_tco2e"] = round(materiality_absolute, 4)

        self.logger.info(
            "Phase 2 complete: %d sectors, materiality=%.1f%% (%.4f tCO2e)",
            len(sectors), materiality_pct, materiality_absolute,
        )

        provenance = self._hash({"phase": phase_name, "scope": scope})

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Evidence Preparation
    # -------------------------------------------------------------------------

    async def _phase_3_evidence_preparation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Package emission data, calculation evidence, and supplier data
        for the verifier.

        Evidence packages include:
            - Import records with CN codes and quantities
            - Emission calculation workbooks with formulas
            - Supplier emission data with verification status
            - Default factor usage documentation
            - Free allocation and carbon price deduction support
        """
        phase_name = "evidence_preparation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        engagement_data = context.get("engagement_data", {})
        scope = context.get("verification_scope", {})

        # Prepare evidence packages
        evidence_packages = await self._prepare_evidence_packages(engagement_data, scope)

        if not evidence_packages:
            warnings.append(
                "No evidence packages could be generated. "
                "Ensure emission data is available for the verification period."
            )

        # Validate evidence completeness
        completeness_checks: List[Dict[str, Any]] = []
        required_evidence = [
            "import_records",
            "emission_calculations",
            "supplier_data",
            "default_factor_documentation",
            "cn_code_classification",
        ]

        for evidence_type in required_evidence:
            present = any(
                e.get("type") == evidence_type for e in evidence_packages
            )
            completeness_checks.append({
                "evidence_type": evidence_type,
                "present": present,
                "status": "complete" if present else "missing",
            })
            if not present:
                warnings.append(f"Missing evidence: {evidence_type}")

        context["evidence_packages"] = evidence_packages
        context["evidence_packages_count"] = len(evidence_packages)

        outputs["evidence_packages_count"] = len(evidence_packages)
        outputs["completeness_checks"] = completeness_checks
        outputs["evidence_types"] = [e.get("type", "") for e in evidence_packages]

        self.logger.info(
            "Phase 3 complete: %d evidence packages prepared",
            len(evidence_packages),
        )

        provenance = self._hash({
            "phase": phase_name,
            "packages": len(evidence_packages),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Verification Execution
    # -------------------------------------------------------------------------

    async def _phase_4_verification_execution(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Track verification progress and document reviews.

        Monitors the verifier's work including:
            - Document review status
            - Data sampling outcomes
            - Preliminary findings
            - Verification timeline adherence
        """
        phase_name = "verification_execution"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        engagement_id = context["engagement_id"]

        # Fetch verification progress
        progress = await self._fetch_verification_progress(engagement_id)

        # Collect findings
        findings = await self._fetch_verification_findings(engagement_id)

        major_count = sum(1 for f in findings if f.get("severity") == FindingSeverity.MAJOR.value)
        minor_count = sum(1 for f in findings if f.get("severity") == FindingSeverity.MINOR.value)
        observation_count = sum(1 for f in findings if f.get("severity") == FindingSeverity.OBSERVATION.value)

        context["findings"] = findings
        context["findings_count"] = len(findings)
        context["major_findings"] = major_count
        context["minor_findings"] = minor_count

        outputs["verification_progress"] = progress
        outputs["findings_count"] = len(findings)
        outputs["major_findings"] = major_count
        outputs["minor_findings"] = minor_count
        outputs["observation_count"] = observation_count
        outputs["findings"] = findings

        if major_count > 0:
            warnings.append(
                f"{major_count} major finding(s) require corrective action "
                "before verification opinion can be issued"
            )

        self.logger.info(
            "Phase 4 complete: %d findings (%d major, %d minor, %d observations)",
            len(findings), major_count, minor_count, observation_count,
        )

        provenance = self._hash({
            "phase": phase_name,
            "findings": len(findings),
            "major": major_count,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Finding Resolution
    # -------------------------------------------------------------------------

    async def _phase_5_finding_resolution(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Address verification findings, implement corrective actions, and
        obtain the verification statement.

        For each finding:
            - Major: must be resolved before positive opinion
            - Minor: should be resolved; non-resolution noted in statement
            - Observation: advisory; no resolution required

        Once all major findings are resolved, the verifier issues a
        verification statement that accompanies the annual declaration.
        """
        phase_name = "finding_resolution"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        findings: List[Dict[str, Any]] = context.get("findings", [])
        engagement_id = context["engagement_id"]

        if not findings:
            # No findings: clean verification
            context["all_findings_resolved"] = True
            context["verification_outcome"] = VerificationOutcome.VERIFIED

            statement_ref = await self._obtain_verification_statement(engagement_id, "clean")
            context["verification_statement_ref"] = statement_ref

            outputs["resolution_status"] = "no_findings"
            outputs["verification_outcome"] = VerificationOutcome.VERIFIED.value
            outputs["statement_ref"] = statement_ref

            self.logger.info("Phase 5 complete: clean verification, no findings")

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs=outputs,
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "clean": True}),
            )

        # Process findings and apply corrective actions
        resolved_findings: List[Dict[str, Any]] = []
        unresolved_major: List[Dict[str, Any]] = []
        unresolved_minor: List[Dict[str, Any]] = []

        for finding in findings:
            severity = finding.get("severity", "")
            resolution = await self._resolve_finding(finding)

            if resolution.get("resolved", False):
                finding["status"] = FindingStatus.RESOLVED.value
                finding["resolution_date"] = datetime.utcnow().isoformat()
                finding["response"] = resolution.get("response", "")
                resolved_findings.append(finding)
            else:
                if severity == FindingSeverity.MAJOR.value:
                    unresolved_major.append(finding)
                elif severity == FindingSeverity.MINOR.value:
                    unresolved_minor.append(finding)

        all_major_resolved = len(unresolved_major) == 0
        all_resolved = all_major_resolved and len(unresolved_minor) == 0

        # Determine outcome
        if all_resolved:
            outcome = VerificationOutcome.VERIFIED
        elif all_major_resolved:
            outcome = VerificationOutcome.VERIFIED_WITH_COMMENTS
        else:
            outcome = VerificationOutcome.NOT_VERIFIED
            warnings.append(
                f"{len(unresolved_major)} major finding(s) remain unresolved. "
                "Verification cannot be completed."
            )

        # Obtain statement if possible
        statement_ref = None
        if outcome in (VerificationOutcome.VERIFIED, VerificationOutcome.VERIFIED_WITH_COMMENTS):
            opinion_type = "clean" if outcome == VerificationOutcome.VERIFIED else "qualified"
            statement_ref = await self._obtain_verification_statement(engagement_id, opinion_type)

        context["all_findings_resolved"] = all_resolved
        context["verification_outcome"] = outcome
        context["verification_statement_ref"] = statement_ref

        outputs["total_findings"] = len(findings)
        outputs["resolved"] = len(resolved_findings)
        outputs["unresolved_major"] = len(unresolved_major)
        outputs["unresolved_minor"] = len(unresolved_minor)
        outputs["verification_outcome"] = outcome.value
        outputs["statement_ref"] = statement_ref

        self.logger.info(
            "Phase 5 complete: outcome=%s resolved=%d/%d unresolved_major=%d",
            outcome.value, len(resolved_findings), len(findings), len(unresolved_major),
        )

        provenance = self._hash({
            "phase": phase_name,
            "outcome": outcome.value,
            "resolved": len(resolved_findings),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _rank_verifier_candidates(
        self,
        candidates: List[Dict[str, Any]],
        required_sectors: List[str],
        required_assurance: AssuranceLevel,
    ) -> List[Dict[str, Any]]:
        """Rank verifier candidates by sector match and capability."""
        scored: List[tuple] = []

        for candidate in candidates:
            score = 0.0
            specializations = candidate.get("sector_specializations", [])
            assurance_levels = candidate.get("assurance_levels", [])

            # Sector match score
            sector_matches = len(set(required_sectors) & set(specializations))
            if required_sectors:
                score += (sector_matches / len(required_sectors)) * 50

            # Assurance level match
            if required_assurance.value in assurance_levels:
                score += 30

            # Availability bonus
            if candidate.get("available_from"):
                score += 10

            # Cost efficiency (lower is better, normalize)
            fee = candidate.get("estimated_fee_eur", 50000)
            if fee > 0:
                score += max(0, 10 - (fee / 10000))

            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _fetch_verifier_candidates(
        self, sectors: List[str], assurance: AssuranceLevel
    ) -> List[Dict[str, Any]]:
        """Fetch available accredited verifier candidates."""
        await asyncio.sleep(0)
        return []

    async def _prepare_evidence_packages(
        self, engagement_data: Dict[str, Any], scope: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare evidence packages for the verifier."""
        await asyncio.sleep(0)
        return []

    async def _fetch_verification_progress(
        self, engagement_id: str
    ) -> Dict[str, Any]:
        """Fetch verification progress status."""
        await asyncio.sleep(0)
        return {"status": "in_progress", "completion_pct": 0}

    async def _fetch_verification_findings(
        self, engagement_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch verification findings from the verifier."""
        await asyncio.sleep(0)
        return []

    async def _resolve_finding(
        self, finding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to resolve a verification finding."""
        await asyncio.sleep(0)
        return {"resolved": False, "response": ""}

    async def _obtain_verification_statement(
        self, engagement_id: str, opinion_type: str
    ) -> Optional[str]:
        """Obtain verification statement from the verifier."""
        await asyncio.sleep(0)
        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
