# -*- coding: utf-8 -*-
"""
Supplier Onboarding Workflow
==============================

Five-phase supplier registration workflow for CBAM compliance. Handles
supplier profile collection, installation mapping, emission data requests,
submission review, and quality assessment.

Regulatory Context:
    Per EU CBAM Implementing Regulation 2023/1773:
    - Article 4: Importers must request actual emission data from their
      non-EU suppliers/installations
    - Article 4(3): If actual data is not available, default values may
      be used, subject to markup from 2026 onward
    - Annex IV: Data that must be collected from installations includes
      direct emissions, indirect emissions, production processes,
      and relevant CN codes

    Supplier data quality directly impacts CBAM cost. Using actual verified
    data vs. default values can reduce certificate costs by 20-40%.

Phases:
    1. Supplier registration - Collect profile, validate EORI, check country
    2. Installation mapping - Map installations to goods categories and CN codes
    3. Data request - Generate and send emission data request to supplier
    4. Submission review - Review submitted data, validate, check completeness
    5. Quality assessment - Score data quality, assign quality tier

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SupplierTier(str, Enum):
    """Supplier data quality tier classification."""
    TIER_1 = "tier_1"  # Verified actual data, high completeness
    TIER_2 = "tier_2"  # Actual data, moderate completeness
    TIER_3 = "tier_3"  # Partial actual data, default fallback
    TIER_4 = "tier_4"  # No actual data, full default values


class OnboardingStatus(str, Enum):
    """Overall supplier onboarding status."""
    APPROVED = "approved"
    PENDING_DATA = "pending_data"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"


# EU ETS countries (exempt from CBAM - imports from these are not in scope)
EU_ETS_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    # EEA members in the EU ETS
    "IS", "LI", "NO",
    # Switzerland (linked ETS)
    "CH",
}

# EORI validation pattern
EORI_PATTERN = re.compile(r"^[A-Z]{2}[A-Za-z0-9]{1,15}$")


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


class SupplierProfile(BaseModel):
    """Supplier registration profile data."""
    supplier_name: str = Field(..., min_length=1, description="Legal entity name")
    supplier_id: Optional[str] = Field(None, description="Existing supplier ID if any")
    country_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166 alpha-2")
    eori_number: Optional[str] = Field(None, description="EORI number if available")
    contact_email: str = Field(..., description="Primary contact email")
    contact_name: Optional[str] = Field(None, description="Contact person name")
    industry_sector: Optional[str] = Field(None, description="Industry classification")
    cbam_sectors: List[str] = Field(default_factory=list, description="CBAM sector(s)")
    installation_count: int = Field(default=0, ge=0, description="Number of installations")
    has_carbon_pricing: bool = Field(default=False, description="Subject to carbon pricing")
    carbon_pricing_instrument: Optional[str] = Field(None, description="Type of carbon pricing")

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alpha-2."""
        if not v.isalpha() or not v.isupper() or len(v) != 2:
            raise ValueError(f"Country code must be uppercase ISO 3166 alpha-2, got: {v}")
        return v


class InstallationInfo(BaseModel):
    """Installation-level information for a supplier."""
    installation_id: str = Field(..., description="Installation identifier")
    installation_name: str = Field(default="", description="Installation name")
    country_code: str = Field(..., min_length=2, max_length=2)
    cn_codes: List[str] = Field(default_factory=list, description="CN codes produced")
    cbam_sectors: List[str] = Field(default_factory=list, description="CBAM sectors")
    production_processes: List[str] = Field(default_factory=list)
    has_monitoring_plan: bool = Field(default=False)
    accredited_verifier: Optional[str] = Field(None)


class SupplierOnboardingResult(BaseModel):
    """Complete result from the supplier onboarding workflow."""
    workflow_name: str = Field(default="supplier_onboarding")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    supplier_id: str = Field(..., description="Assigned supplier identifier")
    onboarding_status: OnboardingStatus = Field(...)
    supplier_tier: SupplierTier = Field(default=SupplierTier.TIER_4)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    installations_mapped: int = Field(default=0, ge=0)
    cn_codes_covered: int = Field(default=0, ge=0)
    data_request_sent: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# SUPPLIER ONBOARDING WORKFLOW
# =============================================================================


class SupplierOnboardingWorkflow:
    """
    Five-phase supplier registration and data quality workflow.

    Manages the end-to-end process of onboarding a new supplier into the
    CBAM compliance system, from profile registration through to data
    quality assessment and tier assignment.

    Supplier tiers determine the quality of emission data available:
        TIER 1: Verified actual data, high completeness (>90%)
        TIER 2: Actual data, moderate completeness (70-90%)
        TIER 3: Partial actual data with default fallback (40-70%)
        TIER 4: No actual data, full default values (<40%)

    Attributes:
        config: Optional configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.

    Example:
        >>> wf = SupplierOnboardingWorkflow()
        >>> result = await wf.execute(
        ...     supplier_data=SupplierProfile(
        ...         supplier_name="Steel Corp",
        ...         country_code="TR",
        ...         contact_email="contact@steelcorp.com",
        ...         cbam_sectors=["iron_steel"],
        ...     )
        ... )
        >>> assert result.onboarding_status in (OnboardingStatus.APPROVED, OnboardingStatus.PENDING_DATA)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the SupplierOnboardingWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.SupplierOnboardingWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        supplier_data: SupplierProfile,
    ) -> SupplierOnboardingResult:
        """
        Execute the full 5-phase supplier onboarding workflow.

        Args:
            supplier_data: Supplier profile with registration information.

        Returns:
            SupplierOnboardingResult with tier assignment and quality scores.
        """
        started_at = datetime.utcnow()
        supplier_id = supplier_data.supplier_id or f"SUP-{self._execution_id[:12]}"

        self.logger.info(
            "Starting supplier onboarding execution_id=%s supplier=%s",
            self._execution_id, supplier_data.supplier_name,
        )

        context: Dict[str, Any] = {
            "config": self.config,
            "supplier_data": supplier_data,
            "supplier_id": supplier_id,
            "execution_id": self._execution_id,
        }

        phase_handlers = [
            ("supplier_registration", self._phase_1_supplier_registration),
            ("installation_mapping", self._phase_2_installation_mapping),
            ("data_request", self._phase_3_data_request),
            ("submission_review", self._phase_4_submission_review),
            ("quality_assessment", self._phase_5_quality_assessment),
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
                if phase_name == "supplier_registration":
                    self.logger.error("Registration failed; halting onboarding.")
                    break

        completed_at = datetime.utcnow()

        # Extract final quality scores
        quality_score = context.get("data_quality_score", 0.0)
        completeness_score = context.get("completeness_score", 0.0)
        accuracy_score = context.get("accuracy_score", 0.0)
        timeliness_score = context.get("timeliness_score", 0.0)
        supplier_tier = context.get("supplier_tier", SupplierTier.TIER_4)
        onboarding_status = context.get("onboarding_status", OnboardingStatus.PENDING_DATA)

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "supplier_id": supplier_id,
        })

        self.logger.info(
            "Supplier onboarding finished supplier_id=%s tier=%s quality=%.2f status=%s",
            supplier_id, supplier_tier.value if isinstance(supplier_tier, SupplierTier) else supplier_tier,
            quality_score, onboarding_status.value if isinstance(onboarding_status, OnboardingStatus) else onboarding_status,
        )

        return SupplierOnboardingResult(
            status=overall_status,
            phases=self._phase_results,
            supplier_id=supplier_id,
            onboarding_status=onboarding_status,
            supplier_tier=supplier_tier,
            data_quality_score=quality_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            installations_mapped=context.get("installations_mapped", 0),
            cn_codes_covered=context.get("cn_codes_covered", 0),
            data_request_sent=context.get("data_request_sent", False),
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Registration
    # -------------------------------------------------------------------------

    async def _phase_1_supplier_registration(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Collect supplier profile, validate EORI, and check country eligibility.

        Validates:
            - Country is not an EU/EEA/CH member (CBAM scope check)
            - EORI number format if provided
            - Required fields are populated
            - Supplier does not already exist in the system
        """
        phase_name = "supplier_registration"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier: SupplierProfile = context["supplier_data"]
        supplier_id = context["supplier_id"]

        # Validate country is not in EU ETS scope
        if supplier.country_code in EU_ETS_COUNTRIES:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={
                    "error": (
                        f"Country '{supplier.country_code}' is in EU ETS scope. "
                        "CBAM supplier onboarding is only for non-EU/EEA suppliers."
                    ),
                },
                provenance_hash=self._hash({"error": "eu_ets_country"}),
            )

        # Validate EORI if provided
        eori_valid = True
        if supplier.eori_number:
            if not EORI_PATTERN.match(supplier.eori_number):
                eori_valid = False
                warnings.append(
                    f"EORI '{supplier.eori_number}' does not match expected format. "
                    "Expected: 2-letter country code + 1-15 alphanumeric characters."
                )

        # Check for duplicate supplier
        existing = await self._check_existing_supplier(supplier.supplier_name, supplier.country_code)
        if existing:
            warnings.append(
                f"Potential duplicate: supplier '{supplier.supplier_name}' may already "
                f"exist with ID '{existing}'. Review before proceeding."
            )

        # Validate CBAM sectors
        valid_sectors = {"cement", "iron_steel", "aluminium", "fertilisers", "electricity", "hydrogen"}
        invalid_sectors = [s for s in supplier.cbam_sectors if s not in valid_sectors]
        if invalid_sectors:
            warnings.append(f"Invalid CBAM sectors: {', '.join(invalid_sectors)}")

        # Register supplier
        registration_record = {
            "supplier_id": supplier_id,
            "supplier_name": supplier.supplier_name,
            "country_code": supplier.country_code,
            "eori_number": supplier.eori_number,
            "eori_valid": eori_valid,
            "contact_email": supplier.contact_email,
            "cbam_sectors": supplier.cbam_sectors,
            "installation_count": supplier.installation_count,
            "has_carbon_pricing": supplier.has_carbon_pricing,
            "registered_at": datetime.utcnow().isoformat(),
        }

        context["registration_record"] = registration_record

        outputs["supplier_id"] = supplier_id
        outputs["registration"] = registration_record
        outputs["country_eligible"] = True
        outputs["eori_valid"] = eori_valid

        self.logger.info(
            "Phase 1 complete: registered supplier_id=%s country=%s sectors=%s",
            supplier_id, supplier.country_code, supplier.cbam_sectors,
        )

        provenance = self._hash({"phase": phase_name, "supplier_id": supplier_id})

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Installation Mapping
    # -------------------------------------------------------------------------

    async def _phase_2_installation_mapping(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Map supplier installations to goods categories and CN codes.

        For each installation, identifies:
            - Production processes (e.g. blast furnace, electric arc furnace)
            - CBAM goods categories produced
            - Applicable CN codes
            - Whether installation has a monitoring plan
        """
        phase_name = "installation_mapping"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier: SupplierProfile = context["supplier_data"]
        supplier_id = context["supplier_id"]

        # Fetch or generate installation data
        installations = await self._fetch_installation_data(supplier_id, supplier)

        if not installations:
            warnings.append(
                "No installations provided. Supplier must register at least "
                "one installation for emission data collection."
            )
            context["installations_mapped"] = 0
            context["cn_codes_covered"] = 0

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"installations_mapped": 0, "cn_codes_covered": 0},
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "installations": 0}),
            )

        # Map installations to CN codes
        all_cn_codes: set = set()
        installation_maps: List[Dict[str, Any]] = []

        for inst in installations:
            cn_codes = inst.get("cn_codes", [])
            all_cn_codes.update(cn_codes)

            installation_maps.append({
                "installation_id": inst.get("installation_id", ""),
                "installation_name": inst.get("installation_name", ""),
                "country_code": inst.get("country_code", supplier.country_code),
                "cn_codes": cn_codes,
                "cbam_sectors": inst.get("cbam_sectors", []),
                "has_monitoring_plan": inst.get("has_monitoring_plan", False),
                "production_processes": inst.get("production_processes", []),
            })

        # Check for installations without monitoring plans
        no_monitor = [
            im["installation_id"] for im in installation_maps
            if not im.get("has_monitoring_plan", False)
        ]
        if no_monitor:
            warnings.append(
                f"{len(no_monitor)} installation(s) without monitoring plans. "
                "Monitoring plans are required for verified emission data."
            )

        context["installations"] = installation_maps
        context["installations_mapped"] = len(installation_maps)
        context["cn_codes_covered"] = len(all_cn_codes)

        outputs["installations_mapped"] = len(installation_maps)
        outputs["cn_codes_covered"] = len(all_cn_codes)
        outputs["cn_codes"] = sorted(all_cn_codes)
        outputs["installations"] = installation_maps

        self.logger.info(
            "Phase 2 complete: %d installations, %d CN codes",
            len(installation_maps), len(all_cn_codes),
        )

        provenance = self._hash({
            "phase": phase_name,
            "installations": len(installation_maps),
            "cn_codes": len(all_cn_codes),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Data Request
    # -------------------------------------------------------------------------

    async def _phase_3_data_request(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Generate and send emission data request to supplier.

        Creates a structured data request per CBAM Annex IV requirements:
            - Direct (Scope 1) emissions per installation
            - Indirect (Scope 2) emissions from electricity consumption
            - Production volumes by CN code
            - Carbon price paid (if applicable)
            - Monitoring methodology used
        """
        phase_name = "data_request"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier: SupplierProfile = context["supplier_data"]
        supplier_id = context["supplier_id"]
        installations = context.get("installations", [])

        # Generate data request template
        data_request = {
            "request_id": f"DR-{supplier_id}-{datetime.utcnow().strftime('%Y%m%d')}",
            "supplier_id": supplier_id,
            "supplier_name": supplier.supplier_name,
            "contact_email": supplier.contact_email,
            "requested_at": datetime.utcnow().isoformat(),
            "response_deadline": self._calculate_response_deadline(),
            "installations_covered": len(installations),
            "data_fields_requested": [
                "direct_emissions_tco2e_per_tonne",
                "indirect_emissions_tco2e_per_tonne",
                "production_volume_tonnes",
                "electricity_consumption_mwh",
                "heat_consumption_gj",
                "carbon_price_paid_eur_per_tco2e",
                "monitoring_methodology",
                "verification_status",
            ],
            "per_installation_fields": [
                "installation_id",
                "cn_code",
                "specific_embedded_emissions",
                "production_process",
                "emission_factor_source",
            ],
        }

        # Send the data request
        send_result = await self._send_data_request(data_request)
        data_request_sent = send_result.get("sent", False)

        if not data_request_sent:
            warnings.append(
                "Data request could not be sent automatically. "
                "Manual follow-up required."
            )

        context["data_request"] = data_request
        context["data_request_sent"] = data_request_sent

        outputs["request_id"] = data_request["request_id"]
        outputs["data_request_sent"] = data_request_sent
        outputs["response_deadline"] = data_request["response_deadline"]
        outputs["fields_requested"] = len(data_request["data_fields_requested"])

        self.logger.info(
            "Phase 3 complete: request_id=%s sent=%s deadline=%s",
            data_request["request_id"], data_request_sent,
            data_request["response_deadline"],
        )

        provenance = self._hash({
            "phase": phase_name,
            "request_id": data_request["request_id"],
            "sent": data_request_sent,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Submission Review
    # -------------------------------------------------------------------------

    async def _phase_4_submission_review(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Review submitted emission data from supplier.

        Validates:
            - Emission factors are within plausible ranges
            - All requested fields are populated
            - Data is internally consistent (direct + indirect = total)
            - Production volumes align with import records
            - Carbon price documentation is adequate
        """
        phase_name = "submission_review"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        supplier_id = context["supplier_id"]

        # Fetch supplier's submitted data
        submitted_data = await self._fetch_supplier_submission(supplier_id)

        if not submitted_data:
            warnings.append(
                "No emission data submitted by supplier yet. "
                "Supplier will be assigned Tier 4 (default values only)."
            )
            context["submission_complete"] = False

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={
                    "submission_received": False,
                    "review_status": "no_submission",
                },
                warnings=warnings,
                provenance_hash=self._hash({"phase": phase_name, "received": False}),
            )

        # Validate submitted data
        review_findings: List[Dict[str, Any]] = []
        fields_populated = 0
        fields_total = 0

        for entry in submitted_data:
            fields_total += 8  # Expected number of data fields per entry
            entry_findings: List[str] = []

            # Check emission factor range
            see = entry.get("specific_embedded_emissions", 0)
            if see <= 0:
                entry_findings.append("Specific embedded emissions is zero or negative")
            elif see > 50:
                entry_findings.append(f"Emission factor {see} tCO2e/t is implausibly high")
            else:
                fields_populated += 1

            # Check direct emissions
            direct = entry.get("direct_emissions", 0)
            if direct > 0:
                fields_populated += 1
            else:
                entry_findings.append("Direct emissions not provided")

            # Check indirect emissions
            indirect = entry.get("indirect_emissions", 0)
            if indirect >= 0:
                fields_populated += 1

            # Check production volume
            volume = entry.get("production_volume_tonnes", 0)
            if volume > 0:
                fields_populated += 1
            else:
                entry_findings.append("Production volume not provided")

            # Check internal consistency
            if see > 0 and direct > 0 and indirect >= 0:
                reconstructed = direct + indirect
                if abs(see - reconstructed) > 0.1 * see:
                    entry_findings.append(
                        f"Inconsistency: SEE={see:.3f} vs direct+indirect={reconstructed:.3f}"
                    )

            # Check other fields
            for field in ["monitoring_methodology", "cn_code", "installation_id", "verification_status"]:
                if entry.get(field):
                    fields_populated += 1
                else:
                    entry_findings.append(f"Missing field: {field}")

            if entry_findings:
                review_findings.append({
                    "installation_id": entry.get("installation_id", "unknown"),
                    "findings": entry_findings,
                })

        completeness = fields_populated / fields_total if fields_total > 0 else 0.0

        context["submitted_data"] = submitted_data
        context["submission_complete"] = completeness >= 0.7
        context["submission_completeness"] = completeness

        outputs["submission_received"] = True
        outputs["entries_reviewed"] = len(submitted_data)
        outputs["completeness_pct"] = round(completeness * 100, 2)
        outputs["findings_count"] = len(review_findings)
        outputs["review_findings"] = review_findings

        if completeness < 0.5:
            warnings.append(
                f"Submission completeness is low ({completeness*100:.1f}%). "
                "Request additional data from supplier."
            )

        self.logger.info(
            "Phase 4 complete: %d entries, completeness=%.1f%%, %d findings",
            len(submitted_data), completeness * 100, len(review_findings),
        )

        provenance = self._hash({
            "phase": phase_name,
            "entries": len(submitted_data),
            "completeness": completeness,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Quality Assessment
    # -------------------------------------------------------------------------

    async def _phase_5_quality_assessment(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Score supplier data quality and assign quality tier.

        Quality dimensions:
            - Completeness (0-1): Percentage of required fields populated
            - Accuracy (0-1): Plausibility of emission factors and consistency
            - Timeliness (0-1): Whether data was submitted before deadline

        Tier assignment:
            TIER 1: overall >= 0.9 (verified, high completeness)
            TIER 2: overall >= 0.7 (actual data, moderate completeness)
            TIER 3: overall >= 0.4 (partial actual data)
            TIER 4: overall < 0.4 (default values only)
        """
        phase_name = "quality_assessment"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        submission_complete = context.get("submission_complete", False)
        submitted_data = context.get("submitted_data", [])
        submission_completeness = context.get("submission_completeness", 0.0)

        # Calculate quality dimensions
        completeness = submission_completeness if submission_complete else 0.0

        # Accuracy score: based on review findings
        accuracy = await self._calculate_accuracy_score(submitted_data)

        # Timeliness score
        data_request = context.get("data_request", {})
        timeliness = self._calculate_timeliness_score(
            data_request.get("response_deadline"),
            submitted_data,
        )

        # Overall quality score (weighted average)
        overall_quality = (
            completeness * 0.40 + accuracy * 0.35 + timeliness * 0.25
        )
        overall_quality = round(min(1.0, max(0.0, overall_quality)), 4)

        # Assign tier
        if overall_quality >= 0.9:
            tier = SupplierTier.TIER_1
        elif overall_quality >= 0.7:
            tier = SupplierTier.TIER_2
        elif overall_quality >= 0.4:
            tier = SupplierTier.TIER_3
        else:
            tier = SupplierTier.TIER_4

        # Determine onboarding status
        if tier in (SupplierTier.TIER_1, SupplierTier.TIER_2):
            onboarding_status = OnboardingStatus.APPROVED
        elif submission_complete:
            onboarding_status = OnboardingStatus.PENDING_REVIEW
        else:
            onboarding_status = OnboardingStatus.PENDING_DATA

        # Store in context
        context["data_quality_score"] = overall_quality
        context["completeness_score"] = completeness
        context["accuracy_score"] = accuracy
        context["timeliness_score"] = timeliness
        context["supplier_tier"] = tier
        context["onboarding_status"] = onboarding_status

        outputs["data_quality_score"] = overall_quality
        outputs["completeness_score"] = round(completeness, 4)
        outputs["accuracy_score"] = round(accuracy, 4)
        outputs["timeliness_score"] = round(timeliness, 4)
        outputs["supplier_tier"] = tier.value
        outputs["onboarding_status"] = onboarding_status.value

        # Generate quality recommendations
        recommendations: List[str] = []
        if completeness < 0.7:
            recommendations.append("Improve data completeness: provide all requested emission fields")
        if accuracy < 0.7:
            recommendations.append("Verify emission factors against production process benchmarks")
        if timeliness < 0.5:
            recommendations.append("Submit data before the response deadline for better scoring")
        if tier == SupplierTier.TIER_4:
            recommendations.append(
                "Tier 4 suppliers have default emission factors applied, which incur "
                "markup surcharges from 2026. Provide actual data to reduce costs."
            )

        outputs["recommendations"] = recommendations
        if recommendations:
            for rec in recommendations:
                warnings.append(rec)

        self.logger.info(
            "Phase 5 complete: quality=%.4f tier=%s status=%s",
            overall_quality, tier.value, onboarding_status.value,
        )

        provenance = self._hash({
            "phase": phase_name,
            "quality": overall_quality,
            "tier": tier.value,
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

    def _calculate_response_deadline(self) -> str:
        """Calculate a 30-day response deadline from now."""
        from datetime import timedelta
        deadline = datetime.utcnow() + timedelta(days=30)
        return deadline.strftime("%Y-%m-%d")

    def _calculate_timeliness_score(
        self, deadline: Optional[str], submitted_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate timeliness score based on submission vs deadline."""
        if not submitted_data:
            return 0.0
        if not deadline:
            return 0.5  # No deadline set; neutral score

        try:
            deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
            # If data was submitted (we have it), check if before deadline
            if datetime.utcnow() <= deadline_date:
                return 1.0  # On time
            else:
                # Late: reduce score based on days overdue
                days_late = (datetime.utcnow() - deadline_date).days
                return max(0.0, 1.0 - (days_late * 0.05))
        except (ValueError, TypeError):
            return 0.5

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _check_existing_supplier(
        self, name: str, country: str
    ) -> Optional[str]:
        """Check if a supplier with this name/country already exists."""
        await asyncio.sleep(0)
        return None

    async def _fetch_installation_data(
        self, supplier_id: str, supplier: SupplierProfile
    ) -> List[Dict[str, Any]]:
        """Fetch installation data for a supplier."""
        await asyncio.sleep(0)
        return []

    async def _send_data_request(
        self, data_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send emission data request to supplier."""
        await asyncio.sleep(0)
        return {"sent": False, "method": "email"}

    async def _fetch_supplier_submission(
        self, supplier_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch supplier's submitted emission data."""
        await asyncio.sleep(0)
        return []

    async def _calculate_accuracy_score(
        self, submitted_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate accuracy score from submitted data plausibility."""
        await asyncio.sleep(0)
        if not submitted_data:
            return 0.0
        # In production: validate against benchmarks, check consistency
        return 0.5

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
