# -*- coding: utf-8 -*-
"""
CutoffDateEngine - PACK-006 EUDR Starter Engine 6
====================================================

December 31, 2020 cutoff date verification engine for EUDR compliance.
Verifies that commodities and products placed on the EU market have been
produced on land that was not subject to deforestation or forest
degradation after the EUDR cutoff date.

Key Capabilities:
    - Cutoff date compliance verification per Article 3
    - Deforestation-free status assessment
    - Temporal evidence collection and evaluation
    - Land use history analysis
    - Cutoff compliance declaration generation
    - Exemption checking for specific product categories
    - Batch verification across multiple plots
    - Status summary reporting

EUDR Cutoff Date:
    - December 31, 2020 (Article 2(8))
    - Products must be "deforestation-free" relative to this date
    - Agricultural use established before cutoff = compliant
    - Forest conversion after cutoff = non-compliant

Compliance Categories:
    - COMPLIANT: Plot confirmed deforestation-free after cutoff
    - NON_COMPLIANT: Evidence of deforestation after cutoff date
    - INSUFFICIENT_EVIDENCE: Unable to determine status
    - EXEMPTED: Product/plot exempt from cutoff requirements

Zero-Hallucination:
    - All verifications use deterministic date comparison logic
    - No LLM involvement in any compliance determination path
    - SHA-256 provenance hashing on every output
    - Pydantic validation at all input/output boundaries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-006 EUDR Starter
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# EUDR Cutoff Date: December 31, 2020
CUTOFF_DATE = datetime(2020, 12, 31, tzinfo=timezone.utc)
CUTOFF_DATE_DATE = date(2020, 12, 31)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _parse_date(value: Any) -> Optional[datetime]:
    """Parse a date value to a timezone-aware datetime.

    Args:
        value: A datetime, date, or ISO format string.

    Returns:
        Timezone-aware datetime or None.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CutoffComplianceStatus(str, Enum):
    """Cutoff date compliance status categories."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EXEMPTED = "EXEMPTED"

class LandUseType(str, Enum):
    """Types of land use for temporal analysis."""

    FOREST = "FOREST"
    AGRICULTURAL = "AGRICULTURAL"
    GRASSLAND = "GRASSLAND"
    PLANTATION = "PLANTATION"
    URBAN = "URBAN"
    WETLAND = "WETLAND"
    BARREN = "BARREN"
    UNKNOWN = "UNKNOWN"

class EvidenceSource(str, Enum):
    """Sources of temporal evidence."""

    SATELLITE_IMAGERY = "SATELLITE_IMAGERY"
    LAND_REGISTRY = "LAND_REGISTRY"
    GOVERNMENT_RECORDS = "GOVERNMENT_RECORDS"
    CERTIFICATION_BODY = "CERTIFICATION_BODY"
    SUPPLIER_DECLARATION = "SUPPLIER_DECLARATION"
    THIRD_PARTY_AUDIT = "THIRD_PARTY_AUDIT"
    GPS_SURVEY = "GPS_SURVEY"
    AERIAL_PHOTOGRAPHY = "AERIAL_PHOTOGRAPHY"
    OTHER = "OTHER"

class EvidenceStrength(str, Enum):
    """Strength/reliability classification of evidence."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"

class ExemptionReason(str, Enum):
    """Reasons for cutoff date exemption."""

    RECYCLED_MATERIAL = "RECYCLED_MATERIAL"
    PRE_REGULATION_STOCK = "PRE_REGULATION_STOCK"
    PERSONAL_USE = "PERSONAL_USE"
    FORCE_MAJEURE = "FORCE_MAJEURE"
    NON_COVERED_PRODUCT = "NON_COVERED_PRODUCT"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TemporalEvidence(BaseModel):
    """A piece of temporal evidence for cutoff date verification."""

    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence identifier")
    source: EvidenceSource = Field(..., description="Evidence source type")
    observation_date: datetime = Field(..., description="Date of the observation")
    land_use_observed: LandUseType = Field(..., description="Land use type observed")
    is_pre_cutoff: bool = Field(default=False, description="Whether observation is before cutoff")
    is_post_cutoff: bool = Field(default=False, description="Whether observation is after cutoff")
    description: str = Field(default="", description="Evidence description")
    strength: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE, description="Evidence strength"
    )
    issuing_authority: Optional[str] = Field(None, description="Authority that provided evidence")
    document_reference: Optional[str] = Field(None, description="Reference to supporting document")
    confidence_score: float = Field(default=0.5, ge=0, le=1.0, description="Confidence 0-1")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CutoffVerification(BaseModel):
    """Result of cutoff date compliance verification."""

    verification_id: str = Field(default_factory=_new_uuid, description="Verification identifier")
    plot_id: str = Field(default="", description="Plot identifier")
    status: CutoffComplianceStatus = Field(..., description="Compliance status")
    cutoff_date: datetime = Field(default=CUTOFF_DATE, description="EUDR cutoff date")
    evidence_count: int = Field(default=0, description="Number of evidence pieces evaluated")
    pre_cutoff_evidence: int = Field(default=0, description="Pre-cutoff evidence count")
    post_cutoff_evidence: int = Field(default=0, description="Post-cutoff evidence count")
    land_use_pre_cutoff: Optional[LandUseType] = Field(None, description="Land use before cutoff")
    land_use_post_cutoff: Optional[LandUseType] = Field(None, description="Land use after cutoff")
    deforestation_detected: bool = Field(default=False, description="Whether deforestation detected")
    confidence_score: float = Field(default=0.0, ge=0, le=1.0, description="Confidence 0-1")
    reasoning: List[str] = Field(default_factory=list, description="Verification reasoning")
    article_reference: str = Field(default="Article 2(8), 3(a)", description="EUDR article reference")
    verified_at: datetime = Field(default_factory=utcnow, description="Verification timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DeforestationFreeResult(BaseModel):
    """Result of deforestation-free assessment."""

    plot_id: str = Field(default="", description="Plot identifier")
    is_deforestation_free: bool = Field(default=False, description="Whether plot is deforestation-free")
    cutoff_date: datetime = Field(default=CUTOFF_DATE, description="Reference cutoff date")
    assessment_period_start: Optional[datetime] = Field(None, description="Assessment period start")
    assessment_period_end: Optional[datetime] = Field(None, description="Assessment period end")
    forest_status_pre_cutoff: Optional[LandUseType] = Field(None, description="Forest status pre-cutoff")
    forest_status_current: Optional[LandUseType] = Field(None, description="Current forest/land status")
    change_detected: bool = Field(default=False, description="Whether land use change detected")
    change_description: Optional[str] = Field(None, description="Description of change")
    confidence_score: float = Field(default=0.0, ge=0, le=1.0, description="Confidence 0-1")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class LandUseChange(BaseModel):
    """A single land use change observation."""

    observation_date: datetime = Field(..., description="Date of observation")
    from_type: LandUseType = Field(..., description="Previous land use")
    to_type: LandUseType = Field(..., description="New land use")
    is_deforestation: bool = Field(default=False, description="Whether this constitutes deforestation")
    is_post_cutoff: bool = Field(default=False, description="Whether change occurred after cutoff")

class LandUseHistory(BaseModel):
    """Land use history for a plot over a time period."""

    plot_id: str = Field(default="", description="Plot identifier")
    start_date: datetime = Field(..., description="History start date")
    end_date: datetime = Field(..., description="History end date")
    observations: List[LandUseChange] = Field(default_factory=list, description="Land use changes")
    dominant_use_pre_cutoff: Optional[LandUseType] = Field(None, description="Dominant use pre-cutoff")
    dominant_use_post_cutoff: Optional[LandUseType] = Field(None, description="Dominant use post-cutoff")
    deforestation_events: int = Field(default=0, description="Number of deforestation events")
    post_cutoff_deforestation: int = Field(default=0, description="Post-cutoff deforestation events")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CutoffDeclaration(BaseModel):
    """Formal cutoff date compliance declaration."""

    declaration_id: str = Field(default_factory=_new_uuid, description="Declaration identifier")
    plot_id: str = Field(..., description="Plot identifier")
    status: CutoffComplianceStatus = Field(..., description="Compliance status")
    cutoff_date: datetime = Field(default=CUTOFF_DATE, description="EUDR cutoff date")
    declaration_text: str = Field(default="", description="Formal declaration text")
    evidence_summary: str = Field(default="", description="Summary of supporting evidence")
    evidence_count: int = Field(default=0, description="Number of evidence items")
    declared_by: Optional[str] = Field(None, description="Person/entity making declaration")
    declared_at: datetime = Field(default_factory=utcnow, description="Declaration timestamp")
    valid_until: Optional[datetime] = Field(None, description="Declaration validity period end")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ExemptionResult(BaseModel):
    """Result of exemption check for cutoff date requirements."""

    is_exempt: bool = Field(default=False, description="Whether product/plot is exempt")
    exemption_reason: Optional[ExemptionReason] = Field(None, description="Reason for exemption")
    exemption_description: str = Field(default="", description="Exemption details")
    article_reference: Optional[str] = Field(None, description="EUDR article reference")
    conditions: List[str] = Field(default_factory=list, description="Exemption conditions")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class BatchCutoffResult(BaseModel):
    """Result of batch cutoff verification across multiple plots."""

    batch_id: str = Field(default_factory=_new_uuid, description="Batch identifier")
    total_plots: int = Field(default=0, description="Total plots verified")
    compliant_count: int = Field(default=0, description="Compliant plots")
    non_compliant_count: int = Field(default=0, description="Non-compliant plots")
    insufficient_evidence_count: int = Field(default=0, description="Insufficient evidence plots")
    exempted_count: int = Field(default=0, description="Exempted plots")
    results: List[CutoffVerification] = Field(default_factory=list, description="Per-plot results")
    overall_compliance_rate: float = Field(default=0.0, description="% of compliant plots")
    verified_at: datetime = Field(default_factory=utcnow, description="Batch verification timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CutoffSummary(BaseModel):
    """Summary of cutoff compliance status across plots."""

    total_plots: int = Field(default=0, description="Total plots assessed")
    compliant: int = Field(default=0, description="Compliant count")
    non_compliant: int = Field(default=0, description="Non-compliant count")
    insufficient_evidence: int = Field(default=0, description="Insufficient evidence count")
    exempted: int = Field(default=0, description="Exempted count")
    compliance_rate: float = Field(default=0.0, description="Compliance rate %")
    high_risk_plots: List[str] = Field(default_factory=list, description="Plot IDs needing attention")
    summary_text: str = Field(default="", description="Human-readable summary")
    generated_at: datetime = Field(default_factory=utcnow, description="Summary timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CutoffDateEngine:
    """
    December 31, 2020 Cutoff Date Verification Engine.

    Verifies that EUDR-regulated commodities were produced on land that
    was not subject to deforestation or forest degradation after the
    EUDR cutoff date of December 31, 2020.

    All verifications are deterministic date comparisons with evidence
    evaluation. No LLM involvement in any compliance determination path.

    Attributes:
        config: Optional engine configuration
        cutoff_date: The EUDR cutoff date
        _verification_count: Counter for verifications performed

    Example:
        >>> engine = CutoffDateEngine()
        >>> result = engine.verify_cutoff_compliance(plot_data, evidence_list)
        >>> assert result.status in list(CutoffComplianceStatus)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CutoffDateEngine.

        Args:
            config: Optional configuration dictionary with keys:
                - cutoff_date: Override cutoff date (default: 2020-12-31)
                - min_evidence_count: Minimum evidence for COMPLIANT (default: 2)
                - declaration_validity_days: Declaration validity period (default: 365)
        """
        self.config = config or {}
        custom_cutoff = self.config.get("cutoff_date")
        if custom_cutoff:
            self.cutoff_date = _parse_date(custom_cutoff) or CUTOFF_DATE
        else:
            self.cutoff_date = CUTOFF_DATE
        self._min_evidence: int = self.config.get("min_evidence_count", 2)
        self._declaration_validity: int = self.config.get("declaration_validity_days", 365)
        self._verification_count: int = 0
        logger.info(
            "CutoffDateEngine initialized (version=%s, cutoff=%s)",
            _MODULE_VERSION, self.cutoff_date.date().isoformat(),
        )

    # -------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------

    def verify_cutoff_compliance(
        self, plot: Dict[str, Any], evidence: List[Dict[str, Any]]
    ) -> CutoffVerification:
        """Verify cutoff date compliance for a production plot.

        Evaluates temporal evidence to determine whether a plot was
        deforestation-free after the EUDR cutoff date.

        Args:
            plot: Plot information with keys: plot_id, country, area_hectares.
            evidence: List of temporal evidence dictionaries with keys:
                - source, observation_date, land_use_observed
                - strength, confidence_score, description

        Returns:
            CutoffVerification with compliance status and reasoning.
        """
        plot_id = plot.get("plot_id", _new_uuid())
        logger.info("Verifying cutoff compliance for plot %s", plot_id)

        parsed_evidence = self._parse_evidence(evidence)
        pre_cutoff = [e for e in parsed_evidence if e.is_pre_cutoff]
        post_cutoff = [e for e in parsed_evidence if e.is_post_cutoff]

        reasoning: List[str] = []
        deforestation_detected = False

        # Determine land use before and after cutoff
        land_use_pre = self._determine_dominant_land_use(pre_cutoff)
        land_use_post = self._determine_dominant_land_use(post_cutoff)

        # Check for deforestation (forest -> non-forest after cutoff)
        if land_use_pre == LandUseType.FOREST and land_use_post in (
            LandUseType.AGRICULTURAL, LandUseType.PLANTATION,
            LandUseType.GRASSLAND, LandUseType.BARREN,
        ):
            deforestation_detected = True
            reasoning.append(
                f"Deforestation detected: land use changed from {land_use_pre.value} "
                f"to {land_use_post.value} after cutoff date {self.cutoff_date.date()}"
            )

        # Determine compliance status
        status, confidence = self._determine_compliance_status(
            parsed_evidence, pre_cutoff, post_cutoff,
            deforestation_detected, land_use_pre, land_use_post,
            reasoning,
        )

        result = CutoffVerification(
            plot_id=plot_id,
            status=status,
            cutoff_date=self.cutoff_date,
            evidence_count=len(parsed_evidence),
            pre_cutoff_evidence=len(pre_cutoff),
            post_cutoff_evidence=len(post_cutoff),
            land_use_pre_cutoff=land_use_pre,
            land_use_post_cutoff=land_use_post,
            deforestation_detected=deforestation_detected,
            confidence_score=round(confidence, 2),
            reasoning=reasoning,
        )
        result.provenance_hash = _compute_hash(result)
        self._verification_count += 1
        return result

    def check_deforestation_free(
        self, plot: Dict[str, Any], cutoff_date: Optional[str] = None
    ) -> DeforestationFreeResult:
        """Check whether a plot is deforestation-free after the cutoff date.

        Args:
            plot: Plot information with keys:
                - plot_id, land_use_pre_cutoff, land_use_current
                - forest_status_pre_cutoff (optional)
                - change_events (list of dicts with date, from_type, to_type)
            cutoff_date: Optional override cutoff date string (ISO format).

        Returns:
            DeforestationFreeResult with status and details.
        """
        plot_id = plot.get("plot_id", _new_uuid())
        cutoff = _parse_date(cutoff_date) if cutoff_date else self.cutoff_date

        pre_type_str = plot.get("land_use_pre_cutoff", "UNKNOWN")
        current_type_str = plot.get("land_use_current", "UNKNOWN")

        try:
            pre_type = LandUseType(pre_type_str)
        except ValueError:
            pre_type = LandUseType.UNKNOWN

        try:
            current_type = LandUseType(current_type_str)
        except ValueError:
            current_type = LandUseType.UNKNOWN

        # Check for deforestation
        change_detected = False
        change_description = None
        is_deforestation_free = True

        change_events = plot.get("change_events", [])
        for event in change_events:
            event_date = _parse_date(event.get("date"))
            if event_date and event_date > cutoff:
                from_type_str = event.get("from_type", "UNKNOWN")
                to_type_str = event.get("to_type", "UNKNOWN")
                try:
                    from_type = LandUseType(from_type_str)
                except ValueError:
                    from_type = LandUseType.UNKNOWN
                try:
                    to_type = LandUseType(to_type_str)
                except ValueError:
                    to_type = LandUseType.UNKNOWN

                if from_type == LandUseType.FOREST and to_type != LandUseType.FOREST:
                    change_detected = True
                    is_deforestation_free = False
                    change_description = (
                        f"Forest conversion detected on {event_date.date()}: "
                        f"{from_type.value} -> {to_type.value}"
                    )
                    break

        # If pre-cutoff was forest and current is non-forest, flag
        if (
            not change_detected
            and pre_type == LandUseType.FOREST
            and current_type not in (LandUseType.FOREST, LandUseType.UNKNOWN)
        ):
            change_detected = True
            is_deforestation_free = False
            change_description = (
                f"Land use changed from {pre_type.value} to {current_type.value}"
            )

        confidence = 0.8 if change_events else 0.5

        result = DeforestationFreeResult(
            plot_id=plot_id,
            is_deforestation_free=is_deforestation_free,
            cutoff_date=cutoff,
            forest_status_pre_cutoff=pre_type,
            forest_status_current=current_type,
            change_detected=change_detected,
            change_description=change_description,
            confidence_score=confidence,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def collect_temporal_evidence(
        self, plot_id: str
    ) -> List[TemporalEvidence]:
        """Collect and structure temporal evidence for a plot.

        Returns a template list of evidence categories needed for
        cutoff date verification.

        Args:
            plot_id: Plot identifier.

        Returns:
            List of TemporalEvidence templates for collection.
        """
        templates: List[TemporalEvidence] = []

        evidence_requirements = [
            (EvidenceSource.SATELLITE_IMAGERY, "Pre-cutoff satellite imagery (before 2021-01-01)",
             EvidenceStrength.STRONG),
            (EvidenceSource.SATELLITE_IMAGERY, "Post-cutoff satellite imagery (after 2020-12-31)",
             EvidenceStrength.STRONG),
            (EvidenceSource.LAND_REGISTRY, "Land registry records showing historical use",
             EvidenceStrength.STRONG),
            (EvidenceSource.GOVERNMENT_RECORDS, "Government land use classification records",
             EvidenceStrength.MODERATE),
            (EvidenceSource.SUPPLIER_DECLARATION, "Supplier declaration on land use history",
             EvidenceStrength.WEAK),
            (EvidenceSource.CERTIFICATION_BODY, "Certification body land use audit",
             EvidenceStrength.MODERATE),
        ]

        for source, description, strength in evidence_requirements:
            ev = TemporalEvidence(
                source=source,
                observation_date=utcnow(),
                land_use_observed=LandUseType.UNKNOWN,
                description=description,
                strength=strength,
                confidence_score=0.0,
            )
            ev.provenance_hash = _compute_hash(ev)
            templates.append(ev)

        return templates

    def assess_land_use_history(
        self, plot: Dict[str, Any], start_date: str, end_date: str
    ) -> LandUseHistory:
        """Assess land use history for a plot over a time period.

        Args:
            plot: Plot information with keys:
                - plot_id
                - observations (list of dicts with date, land_use)
                - change_events (list of dicts with date, from_type, to_type)
            start_date: Analysis start date (ISO format).
            end_date: Analysis end date (ISO format).

        Returns:
            LandUseHistory with analysis results.
        """
        plot_id = plot.get("plot_id", _new_uuid())
        start = _parse_date(start_date) or datetime(2015, 1, 1, tzinfo=timezone.utc)
        end = _parse_date(end_date) or utcnow()

        changes: List[LandUseChange] = []
        deforestation_events = 0
        post_cutoff_deforestation = 0

        for event in plot.get("change_events", []):
            event_date = _parse_date(event.get("date"))
            if not event_date:
                continue
            if event_date < start or event_date > end:
                continue

            try:
                from_type = LandUseType(event.get("from_type", "UNKNOWN"))
            except ValueError:
                from_type = LandUseType.UNKNOWN
            try:
                to_type = LandUseType(event.get("to_type", "UNKNOWN"))
            except ValueError:
                to_type = LandUseType.UNKNOWN

            is_deforestation = (
                from_type == LandUseType.FOREST
                and to_type != LandUseType.FOREST
            )
            is_post_cutoff = event_date > self.cutoff_date

            if is_deforestation:
                deforestation_events += 1
                if is_post_cutoff:
                    post_cutoff_deforestation += 1

            changes.append(LandUseChange(
                observation_date=event_date,
                from_type=from_type,
                to_type=to_type,
                is_deforestation=is_deforestation,
                is_post_cutoff=is_post_cutoff,
            ))

        changes.sort(key=lambda c: c.observation_date)

        # Determine dominant use before/after cutoff
        pre_cutoff_obs = [c for c in changes if not c.is_post_cutoff]
        post_cutoff_obs = [c for c in changes if c.is_post_cutoff]
        dominant_pre = pre_cutoff_obs[-1].to_type if pre_cutoff_obs else None
        dominant_post = post_cutoff_obs[-1].to_type if post_cutoff_obs else None

        result = LandUseHistory(
            plot_id=plot_id,
            start_date=start,
            end_date=end,
            observations=changes,
            dominant_use_pre_cutoff=dominant_pre,
            dominant_use_post_cutoff=dominant_post,
            deforestation_events=deforestation_events,
            post_cutoff_deforestation=post_cutoff_deforestation,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def generate_cutoff_declaration(
        self, plot: Dict[str, Any]
    ) -> CutoffDeclaration:
        """Generate a formal cutoff date compliance declaration for a plot.

        Args:
            plot: Plot information with keys:
                - plot_id, status (CutoffComplianceStatus value)
                - evidence_count, declared_by
                - evidence_summary (optional)

        Returns:
            CutoffDeclaration with formal declaration text.
        """
        plot_id = plot.get("plot_id", _new_uuid())
        status_str = plot.get("status", "INSUFFICIENT_EVIDENCE")

        try:
            status = CutoffComplianceStatus(status_str)
        except ValueError:
            status = CutoffComplianceStatus.INSUFFICIENT_EVIDENCE

        evidence_count = plot.get("evidence_count", 0)
        declared_by = plot.get("declared_by")
        evidence_summary = plot.get("evidence_summary", "")

        # Generate declaration text based on status
        if status == CutoffComplianceStatus.COMPLIANT:
            declaration_text = (
                f"I hereby declare that the production plot (ID: {plot_id}) has been "
                f"verified as deforestation-free after the EUDR cutoff date of "
                f"{self.cutoff_date.date().isoformat()}. This determination is based on "
                f"{evidence_count} pieces of temporal evidence confirming that no "
                f"deforestation or forest degradation occurred on this plot after "
                f"31 December 2020, in accordance with Article 3(a) of Regulation "
                f"(EU) 2023/1115."
            )
        elif status == CutoffComplianceStatus.NON_COMPLIANT:
            declaration_text = (
                f"Based on available evidence, the production plot (ID: {plot_id}) "
                f"has been determined as NON-COMPLIANT with the EUDR cutoff date of "
                f"{self.cutoff_date.date().isoformat()}. Evidence indicates deforestation "
                f"or forest degradation occurred after 31 December 2020. Products from "
                f"this plot may not be placed on or exported from the EU market per "
                f"Article 3 of Regulation (EU) 2023/1115."
            )
        elif status == CutoffComplianceStatus.EXEMPTED:
            declaration_text = (
                f"The production plot (ID: {plot_id}) or associated product has been "
                f"determined as EXEMPT from the EUDR cutoff date requirement. "
                f"Exemption details: {evidence_summary or 'See attached documentation'}."
            )
        else:
            declaration_text = (
                f"Insufficient evidence is available to determine the cutoff date "
                f"compliance status of production plot (ID: {plot_id}) relative to "
                f"the EUDR cutoff date of {self.cutoff_date.date().isoformat()}. "
                f"Additional temporal evidence is required before a compliance "
                f"determination can be made."
            )

        valid_until = utcnow() + timedelta(days=self._declaration_validity)

        declaration = CutoffDeclaration(
            plot_id=plot_id,
            status=status,
            cutoff_date=self.cutoff_date,
            declaration_text=declaration_text,
            evidence_summary=evidence_summary,
            evidence_count=evidence_count,
            declared_by=declared_by,
            valid_until=valid_until,
        )
        declaration.provenance_hash = _compute_hash(declaration)
        return declaration

    def check_exemptions(
        self, product: Dict[str, Any], plot: Dict[str, Any]
    ) -> ExemptionResult:
        """Check whether a product or plot qualifies for cutoff date exemption.

        Args:
            product: Product information with keys:
                - is_recycled (bool), is_personal_use (bool)
                - import_date (str/datetime), product_type (str)
            plot: Plot information with keys:
                - country (str), is_force_majeure (bool)

        Returns:
            ExemptionResult with exemption status and details.
        """
        is_exempt = False
        exemption_reason = None
        description = ""
        article_ref = None
        conditions: List[str] = []

        # Check recycled material exemption
        if product.get("is_recycled", False):
            is_exempt = True
            exemption_reason = ExemptionReason.RECYCLED_MATERIAL
            description = "Product is made from recycled materials"
            article_ref = "Article 2(3)"
            conditions.append("Product must be entirely recycled content")
            conditions.append("Recycling documentation required")

        # Check personal use exemption
        elif product.get("is_personal_use", False):
            is_exempt = True
            exemption_reason = ExemptionReason.PERSONAL_USE
            description = "Product is for personal non-commercial use"
            article_ref = "Article 1(3)"
            conditions.append("Must be for personal consumption only")
            conditions.append("Must not be placed on the market")

        # Check force majeure
        elif plot.get("is_force_majeure", False):
            is_exempt = True
            exemption_reason = ExemptionReason.FORCE_MAJEURE
            description = "Force majeure circumstances apply"
            article_ref = "Article 38"
            conditions.append("Force majeure must be documented and verified")
            conditions.append("Competent authority notification required")

        # Check pre-regulation stock
        elif self._is_pre_regulation_stock(product):
            is_exempt = True
            exemption_reason = ExemptionReason.PRE_REGULATION_STOCK
            description = "Product was in stock before regulation application date"
            article_ref = "Article 38(2)"
            conditions.append("Stock records must demonstrate pre-regulation import")

        result = ExemptionResult(
            is_exempt=is_exempt,
            exemption_reason=exemption_reason,
            exemption_description=description,
            article_reference=article_ref,
            conditions=conditions,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def batch_verify(
        self, plots: List[Dict[str, Any]]
    ) -> BatchCutoffResult:
        """Verify cutoff compliance for multiple plots in batch.

        Each plot dictionary should contain plot data and an 'evidence' key
        with the evidence list.

        Args:
            plots: List of plot dictionaries with keys:
                - plot_id, country, area_hectares
                - evidence (list of evidence dicts)

        Returns:
            BatchCutoffResult with per-plot and aggregate results.
        """
        logger.info("Batch verifying cutoff compliance for %d plots", len(plots))
        results: List[CutoffVerification] = []
        compliant = 0
        non_compliant = 0
        insufficient = 0
        exempted = 0

        for plot_data in plots:
            evidence = plot_data.get("evidence", [])
            result = self.verify_cutoff_compliance(plot_data, evidence)
            results.append(result)

            if result.status == CutoffComplianceStatus.COMPLIANT:
                compliant += 1
            elif result.status == CutoffComplianceStatus.NON_COMPLIANT:
                non_compliant += 1
            elif result.status == CutoffComplianceStatus.EXEMPTED:
                exempted += 1
            else:
                insufficient += 1

        total = len(plots)
        compliance_rate = round((compliant / total) * 100.0, 2) if total > 0 else 0.0

        batch = BatchCutoffResult(
            total_plots=total,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            insufficient_evidence_count=insufficient,
            exempted_count=exempted,
            results=results,
            overall_compliance_rate=compliance_rate,
        )
        batch.provenance_hash = _compute_hash(batch)
        return batch

    def get_cutoff_status_summary(
        self, plots: List[Dict[str, Any]]
    ) -> CutoffSummary:
        """Get a summary of cutoff compliance status across plots.

        Args:
            plots: List of plot dictionaries with keys:
                - plot_id, status (CutoffComplianceStatus value)

        Returns:
            CutoffSummary with aggregate status counts.
        """
        compliant = 0
        non_compliant = 0
        insufficient = 0
        exempted = 0
        high_risk_plots: List[str] = []

        for plot in plots:
            status_str = plot.get("status", "INSUFFICIENT_EVIDENCE")
            plot_id = plot.get("plot_id", "unknown")

            try:
                status = CutoffComplianceStatus(status_str)
            except ValueError:
                status = CutoffComplianceStatus.INSUFFICIENT_EVIDENCE

            if status == CutoffComplianceStatus.COMPLIANT:
                compliant += 1
            elif status == CutoffComplianceStatus.NON_COMPLIANT:
                non_compliant += 1
                high_risk_plots.append(plot_id)
            elif status == CutoffComplianceStatus.EXEMPTED:
                exempted += 1
            else:
                insufficient += 1
                high_risk_plots.append(plot_id)

        total = len(plots)
        rate = round((compliant / total) * 100.0, 2) if total > 0 else 0.0

        summary_text = (
            f"Cutoff compliance summary: {total} plots assessed. "
            f"{compliant} compliant ({rate:.1f}%), {non_compliant} non-compliant, "
            f"{insufficient} insufficient evidence, {exempted} exempted. "
            f"{len(high_risk_plots)} plots require attention."
        )

        summary = CutoffSummary(
            total_plots=total,
            compliant=compliant,
            non_compliant=non_compliant,
            insufficient_evidence=insufficient,
            exempted=exempted,
            compliance_rate=rate,
            high_risk_plots=high_risk_plots,
            summary_text=summary_text,
        )
        summary.provenance_hash = _compute_hash(summary)
        return summary

    # -------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------

    def _parse_evidence(
        self, evidence_list: List[Dict[str, Any]]
    ) -> List[TemporalEvidence]:
        """Parse raw evidence dictionaries into TemporalEvidence models.

        Args:
            evidence_list: List of evidence dictionaries.

        Returns:
            List of parsed TemporalEvidence objects.
        """
        results: List[TemporalEvidence] = []

        for ev in evidence_list:
            obs_date = _parse_date(ev.get("observation_date"))
            if not obs_date:
                continue

            try:
                source = EvidenceSource(ev.get("source", "OTHER"))
            except ValueError:
                source = EvidenceSource.OTHER

            try:
                land_use = LandUseType(ev.get("land_use_observed", "UNKNOWN"))
            except ValueError:
                land_use = LandUseType.UNKNOWN

            try:
                strength = EvidenceStrength(ev.get("strength", "MODERATE"))
            except ValueError:
                strength = EvidenceStrength.MODERATE

            is_pre = obs_date <= self.cutoff_date
            is_post = obs_date > self.cutoff_date

            parsed = TemporalEvidence(
                source=source,
                observation_date=obs_date,
                land_use_observed=land_use,
                is_pre_cutoff=is_pre,
                is_post_cutoff=is_post,
                description=ev.get("description", ""),
                strength=strength,
                issuing_authority=ev.get("issuing_authority"),
                document_reference=ev.get("document_reference"),
                confidence_score=float(ev.get("confidence_score", 0.5)),
            )
            parsed.provenance_hash = _compute_hash(parsed)
            results.append(parsed)

        return results

    def _determine_dominant_land_use(
        self, evidence: List[TemporalEvidence]
    ) -> Optional[LandUseType]:
        """Determine dominant land use from a set of evidence.

        Selects the land use type observed in the strongest evidence,
        breaking ties by most recent observation.

        Args:
            evidence: List of temporal evidence.

        Returns:
            Dominant LandUseType or None if no evidence.
        """
        if not evidence:
            return None

        # Weight by evidence strength
        strength_weights = {
            EvidenceStrength.STRONG: 3,
            EvidenceStrength.MODERATE: 2,
            EvidenceStrength.WEAK: 1,
            EvidenceStrength.INSUFFICIENT: 0,
        }

        type_scores: Dict[LandUseType, float] = {}
        for ev in evidence:
            weight = strength_weights.get(ev.strength, 1) * ev.confidence_score
            current = type_scores.get(ev.land_use_observed, 0.0)
            type_scores[ev.land_use_observed] = current + weight

        if not type_scores:
            return None

        return max(type_scores, key=lambda k: type_scores[k])

    def _determine_compliance_status(
        self,
        all_evidence: List[TemporalEvidence],
        pre_cutoff: List[TemporalEvidence],
        post_cutoff: List[TemporalEvidence],
        deforestation_detected: bool,
        land_use_pre: Optional[LandUseType],
        land_use_post: Optional[LandUseType],
        reasoning: List[str],
    ) -> tuple:
        """Determine compliance status from evidence analysis.

        Args:
            all_evidence: All evidence pieces.
            pre_cutoff: Evidence from before cutoff.
            post_cutoff: Evidence from after cutoff.
            deforestation_detected: Whether deforestation was detected.
            land_use_pre: Pre-cutoff land use.
            land_use_post: Post-cutoff land use.
            reasoning: Reasoning list to append to.

        Returns:
            Tuple of (CutoffComplianceStatus, confidence_score).
        """
        if deforestation_detected:
            reasoning.append("Status: NON_COMPLIANT - deforestation after cutoff date")
            return CutoffComplianceStatus.NON_COMPLIANT, 0.85

        if len(all_evidence) < self._min_evidence:
            reasoning.append(
                f"Status: INSUFFICIENT_EVIDENCE - only {len(all_evidence)} evidence pieces "
                f"(minimum {self._min_evidence} required)"
            )
            return CutoffComplianceStatus.INSUFFICIENT_EVIDENCE, 0.3

        if not post_cutoff:
            reasoning.append(
                "Status: INSUFFICIENT_EVIDENCE - no post-cutoff evidence available"
            )
            return CutoffComplianceStatus.INSUFFICIENT_EVIDENCE, 0.4

        # Check for consistent agricultural/plantation use
        if land_use_pre in (LandUseType.AGRICULTURAL, LandUseType.PLANTATION):
            if land_use_post in (LandUseType.AGRICULTURAL, LandUseType.PLANTATION):
                reasoning.append(
                    f"Status: COMPLIANT - consistent {land_use_pre.value} use "
                    f"before and after cutoff date"
                )
                return CutoffComplianceStatus.COMPLIANT, 0.9

        # Post-cutoff shows no forest and pre-cutoff was not forest
        if land_use_pre != LandUseType.FOREST and land_use_post != LandUseType.FOREST:
            reasoning.append(
                "Status: COMPLIANT - no forest cover detected before or after cutoff"
            )
            return CutoffComplianceStatus.COMPLIANT, 0.8

        # Post-cutoff shows forest maintained
        if land_use_post == LandUseType.FOREST:
            reasoning.append(
                "Status: COMPLIANT - forest cover maintained after cutoff date"
            )
            return CutoffComplianceStatus.COMPLIANT, 0.85

        # Default: not enough clarity
        reasoning.append(
            "Status: INSUFFICIENT_EVIDENCE - unable to conclusively determine compliance"
        )
        return CutoffComplianceStatus.INSUFFICIENT_EVIDENCE, 0.5

    def _is_pre_regulation_stock(self, product: Dict[str, Any]) -> bool:
        """Check if a product qualifies as pre-regulation stock.

        Args:
            product: Product dictionary with import_date field.

        Returns:
            True if product was imported before EUDR application date.
        """
        import_date = _parse_date(product.get("import_date"))
        if not import_date:
            return False
        # EUDR application date: December 30, 2024 for large operators
        eudr_application = datetime(2024, 12, 30, tzinfo=timezone.utc)
        return import_date < eudr_application
