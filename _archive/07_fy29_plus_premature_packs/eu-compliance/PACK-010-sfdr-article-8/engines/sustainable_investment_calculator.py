# -*- coding: utf-8 -*-
"""
SustainableInvestmentCalculatorEngine - PACK-010 SFDR Article 8 Engine 6

Calculate and classify sustainable investments per SFDR Article 2(17).
Implements the three-step classification test (DNSH, good governance,
sustainable objective contribution) and proportion calculation for
pre-contractual and periodic disclosure reporting.

SFDR Article 2(17) Definition:
    A sustainable investment is an investment in an economic activity that:
    (a) contributes to an environmental objective (measured by key resource
        efficiency indicators) or a social objective (tackling inequality,
        fostering social cohesion, or investing in human capital); and
    (b) does not significantly harm any of those objectives (DNSH); and
    (c) the investee company follows good governance practices.

Classification Hierarchy:
    1. TAXONOMY_ALIGNED   - Meets EU Taxonomy criteria (highest confidence)
    2. OTHER_ENVIRONMENTAL - Contributes to environmental objectives but
                             not Taxonomy-aligned
    3. SOCIAL              - Contributes to social objectives
    4. NOT_SUSTAINABLE     - Does not meet Article 2(17) criteria

Proportion Calculation:
    sustainable_pct = (sustainable_NAV / total_NAV) * 100
    taxonomy_aligned_pct = (taxonomy_aligned_NAV / total_NAV) * 100

Zero-Hallucination:
    - All classification steps use deterministic rule evaluation
    - Proportion calculations are pure arithmetic on NAV figures
    - No LLM involvement in classification or calculation paths
    - SHA-256 provenance hash on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage value or 0.0.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InvestmentClassificationType(str, Enum):
    """Classification of an investment per SFDR Article 2(17)."""
    TAXONOMY_ALIGNED = "taxonomy_aligned"
    OTHER_ENVIRONMENTAL = "other_environmental"
    SOCIAL = "social"
    NOT_SUSTAINABLE = "not_sustainable"

class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment status."""
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_ASSESSED = "not_assessed"

class GovernanceStatus(str, Enum):
    """Good governance assessment status."""
    GOOD = "good"
    INADEQUATE = "inadequate"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_ASSESSED = "not_assessed"

class ObjectiveContribution(str, Enum):
    """Type of sustainable objective contribution."""
    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"
    SOCIAL_INEQUALITY = "social_inequality"
    SOCIAL_COHESION = "social_cohesion"
    HUMAN_CAPITAL = "human_capital"
    NONE = "none"

class AdherenceStatus(str, Enum):
    """Status of adherence to minimum commitment."""
    MEETING = "meeting"
    EXCEEDING = "exceeding"
    BELOW = "below"
    AT_RISK = "at_risk"

# ---------------------------------------------------------------------------
# DNSH Criteria Reference Data
# ---------------------------------------------------------------------------

DNSH_PAI_MAPPING: Dict[str, List[str]] = {
    ObjectiveContribution.CLIMATE_MITIGATION: [
        "ghg_emissions", "carbon_footprint", "ghg_intensity",
        "fossil_fuel_exposure", "non_renewable_energy",
    ],
    ObjectiveContribution.CLIMATE_ADAPTATION: [
        "ghg_emissions", "carbon_footprint", "biodiversity_impact",
    ],
    ObjectiveContribution.WATER_MARINE: [
        "water_emissions", "water_recycling", "hazardous_waste",
    ],
    ObjectiveContribution.CIRCULAR_ECONOMY: [
        "hazardous_waste", "waste_recycling",
    ],
    ObjectiveContribution.POLLUTION_PREVENTION: [
        "water_emissions", "hazardous_waste",
    ],
    ObjectiveContribution.BIODIVERSITY: [
        "biodiversity_impact", "deforestation",
    ],
    ObjectiveContribution.SOCIAL_INEQUALITY: [
        "gender_pay_gap", "board_gender_diversity", "human_rights_violations",
    ],
    ObjectiveContribution.SOCIAL_COHESION: [
        "human_rights_violations", "controversies",
    ],
    ObjectiveContribution.HUMAN_CAPITAL: [
        "gender_pay_gap", "board_gender_diversity",
    ],
}

GOVERNANCE_CRITERIA: List[str] = [
    "sound_management_structures",
    "employee_relations",
    "remuneration_compliance",
    "tax_compliance",
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DNSHAssessment(BaseModel):
    """Do No Significant Harm assessment for an investment.

    Per SFDR Article 2(17)(b), a sustainable investment must not
    significantly harm any environmental or social objective.
    """
    assessment_id: str = Field(default_factory=_new_uuid, description="Unique assessment identifier")
    investment_id: str = Field(description="Assessed investment identifier")
    status: DNSHStatus = Field(default=DNSHStatus.NOT_ASSESSED, description="Overall DNSH status")
    pai_indicators_checked: List[str] = Field(
        default_factory=list, description="PAI indicators evaluated"
    )
    pai_indicators_passed: List[str] = Field(
        default_factory=list, description="PAI indicators that passed"
    )
    pai_indicators_failed: List[str] = Field(
        default_factory=list, description="PAI indicators that failed"
    )
    coverage_pct: float = Field(default=0.0, description="Percentage of required PAI indicators covered")
    notes: str = Field(default="", description="Assessment notes")
    assessed_at: datetime = Field(default_factory=utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class GovernanceAssessment(BaseModel):
    """Good governance assessment for an investee company.

    Per SFDR Article 2(17), investee companies must follow good
    governance practices with respect to sound management structures,
    employee relations, remuneration, and tax compliance.
    """
    assessment_id: str = Field(default_factory=_new_uuid, description="Unique assessment identifier")
    investment_id: str = Field(description="Assessed investment identifier")
    company_name: str = Field(default="", description="Investee company name")
    status: GovernanceStatus = Field(
        default=GovernanceStatus.NOT_ASSESSED, description="Overall governance status"
    )
    management_structures: bool = Field(default=False, description="Sound management structures")
    employee_relations: bool = Field(default=False, description="Adequate employee relations")
    remuneration_compliance: bool = Field(default=False, description="Compliant remuneration of staff")
    tax_compliance: bool = Field(default=False, description="Tax compliance")
    criteria_met: int = Field(default=0, description="Number of governance criteria met")
    criteria_total: int = Field(default=4, description="Total governance criteria")
    ungc_signatory: bool = Field(default=False, description="UN Global Compact signatory")
    controversies_flag: bool = Field(default=False, description="Active controversies flag")
    notes: str = Field(default="", description="Assessment notes")
    assessed_at: datetime = Field(default_factory=utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class InvestmentData(BaseModel):
    """Input data for a single investment holding.

    Represents the data needed to classify an investment under
    SFDR Article 2(17).
    """
    investment_id: str = Field(default_factory=_new_uuid, description="Unique investment identifier")
    company_name: str = Field(default="", description="Investee company name")
    isin: str = Field(default="", description="ISIN of the security")
    nav_value: float = Field(default=0.0, description="Net Asset Value of this holding in EUR")
    weight_pct: float = Field(default=0.0, description="Portfolio weight as percentage")
    sector: str = Field(default="", description="Sector classification (NACE/GICS)")
    country: str = Field(default="", description="Country of domicile (ISO 3166)")
    taxonomy_eligible: bool = Field(default=False, description="EU Taxonomy eligible activity")
    taxonomy_aligned_pct: float = Field(
        default=0.0, description="Percentage of revenue Taxonomy-aligned"
    )
    environmental_contribution: Optional[ObjectiveContribution] = Field(
        default=None, description="Primary environmental objective contribution"
    )
    social_contribution: Optional[ObjectiveContribution] = Field(
        default=None, description="Primary social objective contribution"
    )
    # PAI indicator data for DNSH
    pai_data: Dict[str, float] = Field(
        default_factory=dict, description="PAI indicator values"
    )
    # Governance data
    governance_data: Dict[str, bool] = Field(
        default_factory=dict, description="Governance criteria flags"
    )

class InvestmentClassification(BaseModel):
    """Classification result for a single investment.

    Contains the final classification, confidence level, and evidence
    from the three-step assessment process.

from greenlang.schemas import utcnow
    """
    classification_id: str = Field(default_factory=_new_uuid, description="Unique classification identifier")
    investment_id: str = Field(description="Classified investment identifier")
    company_name: str = Field(default="", description="Investee company name")
    classification: InvestmentClassificationType = Field(
        description="Final classification per Article 2(17)"
    )
    confidence: float = Field(
        default=0.0, description="Classification confidence (0.0 to 1.0)"
    )
    evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting classification"
    )
    dnsh_assessment: Optional[DNSHAssessment] = Field(
        default=None, description="DNSH assessment result"
    )
    governance_assessment: Optional[GovernanceAssessment] = Field(
        default=None, description="Governance assessment result"
    )
    objective_contribution: Optional[ObjectiveContribution] = Field(
        default=None, description="Primary objective contribution identified"
    )
    taxonomy_aligned: bool = Field(default=False, description="Whether Taxonomy-aligned")
    nav_value: float = Field(default=0.0, description="NAV value of this holding")
    classified_at: datetime = Field(default_factory=utcnow, description="Classification timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ProportionResult(BaseModel):
    """Result of sustainable investment proportion calculation.

    Breaks down the portfolio into sustainable categories as percentages
    of total NAV, for use in pre-contractual and periodic disclosures.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    total_nav: float = Field(description="Total portfolio NAV in EUR")
    total_sustainable_pct: float = Field(
        description="Total sustainable investments as % of NAV"
    )
    taxonomy_aligned_pct: float = Field(
        description="Taxonomy-aligned investments as % of NAV"
    )
    other_environmental_pct: float = Field(
        description="Other environmental sustainable investments as % of NAV"
    )
    social_pct: float = Field(
        description="Social sustainable investments as % of NAV"
    )
    not_sustainable_pct: float = Field(
        description="Non-sustainable investments as % of NAV"
    )
    taxonomy_aligned_nav: float = Field(default=0.0, description="Taxonomy-aligned NAV in EUR")
    other_environmental_nav: float = Field(default=0.0, description="Other environmental NAV in EUR")
    social_nav: float = Field(default=0.0, description="Social NAV in EUR")
    not_sustainable_nav: float = Field(default=0.0, description="Non-sustainable NAV in EUR")
    total_holdings: int = Field(default=0, description="Total number of holdings")
    sustainable_holdings: int = Field(default=0, description="Number of sustainable holdings")
    coverage_ratio: float = Field(
        default=0.0, description="Percentage of holdings with sufficient data"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CommitmentAdherence(BaseModel):
    """Result of checking adherence to minimum sustainability commitment.

    Article 8 products with sustainable investments must disclose a
    minimum proportion committed to in pre-contractual documents.
    """
    adherence_id: str = Field(default_factory=_new_uuid, description="Unique adherence identifier")
    committed_minimum_pct: float = Field(
        description="Minimum sustainable investment % from pre-contractual disclosure"
    )
    actual_proportion_pct: float = Field(
        description="Actual sustainable investment % currently"
    )
    buffer_pct: float = Field(
        description="Buffer above minimum (actual - committed)"
    )
    adherence_status: AdherenceStatus = Field(description="Current adherence status")
    taxonomy_committed_pct: float = Field(
        default=0.0, description="Committed minimum Taxonomy-aligned %"
    )
    taxonomy_actual_pct: float = Field(
        default=0.0, description="Actual Taxonomy-aligned %"
    )
    taxonomy_adherence: AdherenceStatus = Field(
        default=AdherenceStatus.MEETING, description="Taxonomy commitment adherence"
    )
    at_risk_threshold_pct: float = Field(
        default=5.0, description="Buffer below which status becomes AT_RISK"
    )
    checked_at: datetime = Field(default_factory=utcnow, description="Check timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ClassificationSummary(BaseModel):
    """Summary of all investment classifications.

    Provides aggregate statistics for disclosure and reporting.
    """
    summary_id: str = Field(default_factory=_new_uuid, description="Unique summary identifier")
    total_investments: int = Field(default=0, description="Total investments classified")
    taxonomy_aligned_count: int = Field(default=0, description="Taxonomy-aligned count")
    other_environmental_count: int = Field(default=0, description="Other environmental count")
    social_count: int = Field(default=0, description="Social sustainable count")
    not_sustainable_count: int = Field(default=0, description="Not sustainable count")
    average_confidence: float = Field(default=0.0, description="Average classification confidence")
    by_sector: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Classification breakdown by sector"
    )
    by_country: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Classification breakdown by country"
    )
    dnsh_pass_rate: float = Field(default=0.0, description="DNSH pass rate (%)")
    governance_pass_rate: float = Field(default=0.0, description="Governance pass rate (%)")
    classifications: List[InvestmentClassification] = Field(
        default_factory=list, description="Individual classification results"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class SustainableInvestmentConfig(BaseModel):
    """Configuration for the SustainableInvestmentCalculatorEngine.

    Controls classification criteria, minimum proportions, and
    threshold settings.
    """
    minimum_sustainable_pct: float = Field(
        default=0.0, description="Minimum committed sustainable investment %"
    )
    minimum_taxonomy_pct: float = Field(
        default=0.0, description="Minimum committed Taxonomy-aligned %"
    )
    dnsh_coverage_threshold: float = Field(
        default=50.0, description="Minimum PAI coverage % to pass DNSH"
    )
    governance_min_criteria: int = Field(
        default=3, description="Minimum governance criteria to pass (out of 4)"
    )
    taxonomy_alignment_threshold: float = Field(
        default=0.0, description="Minimum Taxonomy alignment % to classify as aligned"
    )
    at_risk_buffer_pct: float = Field(
        default=5.0, description="Buffer % below which adherence is AT_RISK"
    )
    pai_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "ghg_emissions": 1000000.0,
            "carbon_footprint": 500.0,
            "ghg_intensity": 800.0,
            "fossil_fuel_exposure": 10.0,
            "non_renewable_energy": 50.0,
            "water_emissions": 5.0,
            "hazardous_waste": 10.0,
            "biodiversity_impact": 0.0,
            "gender_pay_gap": 15.0,
            "board_gender_diversity": 30.0,
            "human_rights_violations": 0.0,
            "controversies": 0.0,
            "deforestation": 0.0,
            "water_recycling": 50.0,
            "waste_recycling": 40.0,
        },
        description="PAI indicator thresholds for DNSH assessment"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

SustainableInvestmentConfig.model_rebuild()
DNSHAssessment.model_rebuild()
GovernanceAssessment.model_rebuild()
InvestmentData.model_rebuild()
InvestmentClassification.model_rebuild()
ProportionResult.model_rebuild()
CommitmentAdherence.model_rebuild()
ClassificationSummary.model_rebuild()

# ---------------------------------------------------------------------------
# SustainableInvestmentCalculatorEngine
# ---------------------------------------------------------------------------

class SustainableInvestmentCalculatorEngine:
    """
    Sustainable investment classification and proportion calculator.

    Implements the three-step classification test per SFDR Article 2(17):
    (1) DNSH check against PAI indicators,
    (2) Good governance assessment,
    (3) Sustainable objective contribution evaluation.

    Then calculates the proportion of the portfolio that qualifies as
    sustainable, including Taxonomy-aligned, other environmental, and
    social sub-categories.

    Attributes:
        config: Engine configuration parameters.
        _classifications: Stored classification results.
        _investments: Input investment data.

    Example:
        >>> engine = SustainableInvestmentCalculatorEngine({
        ...     "minimum_sustainable_pct": 20.0
        ... })
        >>> investments = [InvestmentData(nav_value=1000, ...)]
        >>> results = engine.classify_investments(investments)
        >>> proportion = engine.calculate_proportion()
        >>> assert proportion.total_sustainable_pct >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SustainableInvestmentCalculatorEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = SustainableInvestmentConfig(**config)
        elif config and isinstance(config, SustainableInvestmentConfig):
            self.config = config
        else:
            self.config = SustainableInvestmentConfig()

        self._investments: List[InvestmentData] = []
        self._classifications: Dict[str, InvestmentClassification] = {}

        logger.info(
            "SustainableInvestmentCalculatorEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_investments(
        self,
        investments: List[InvestmentData],
    ) -> List[InvestmentClassification]:
        """Classify a list of investments per SFDR Article 2(17).

        Applies the three-step test to each investment:
        1. DNSH check (does not significantly harm any objective)
        2. Good governance check (sound management, employee relations, etc.)
        3. Sustainable objective contribution (environmental or social)

        Args:
            investments: List of InvestmentData to classify.

        Returns:
            List of InvestmentClassification results.
        """
        start = utcnow()
        self._investments = investments
        results: List[InvestmentClassification] = []

        for inv in investments:
            classification = self._classify_single(inv)
            self._classifications[inv.investment_id] = classification
            results.append(classification)

        sustainable_count = sum(
            1 for r in results
            if r.classification != InvestmentClassificationType.NOT_SUSTAINABLE
        )
        logger.info(
            "Classified %d investments (%d sustainable) in %dms",
            len(results),
            sustainable_count,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return results

    def _classify_single(self, inv: InvestmentData) -> InvestmentClassification:
        """Classify a single investment through the three-step test.

        Args:
            inv: Investment data to classify.

        Returns:
            InvestmentClassification result.
        """
        evidence: List[str] = []

        # Step 1: DNSH assessment
        dnsh = self._assess_dnsh(inv)
        if dnsh.status == DNSHStatus.PASSED:
            evidence.append("DNSH: Passed - does not significantly harm any objective")
        elif dnsh.status == DNSHStatus.FAILED:
            evidence.append(f"DNSH: Failed - indicators failed: {dnsh.pai_indicators_failed}")
        else:
            evidence.append(f"DNSH: {dnsh.status.value}")

        # Step 2: Good governance assessment
        governance = self._assess_governance(inv)
        if governance.status == GovernanceStatus.GOOD:
            evidence.append(
                f"Governance: Good ({governance.criteria_met}/{governance.criteria_total} criteria met)"
            )
        else:
            evidence.append(f"Governance: {governance.status.value}")

        # Step 3: Objective contribution
        contribution = self._assess_contribution(inv)
        if contribution != ObjectiveContribution.NONE:
            evidence.append(f"Contribution: {contribution.value}")

        # Determine final classification
        classification_type, confidence = self._determine_classification(
            inv, dnsh, governance, contribution
        )

        result = InvestmentClassification(
            investment_id=inv.investment_id,
            company_name=inv.company_name,
            classification=classification_type,
            confidence=round(confidence, 3),
            evidence=evidence,
            dnsh_assessment=dnsh,
            governance_assessment=governance,
            objective_contribution=contribution,
            taxonomy_aligned=(classification_type == InvestmentClassificationType.TAXONOMY_ALIGNED),
            nav_value=inv.nav_value,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_dnsh(self, inv: InvestmentData) -> DNSHAssessment:
        """Assess whether an investment passes the DNSH test.

        Checks PAI indicator values against configured thresholds.

        Args:
            inv: Investment data with PAI indicator values.

        Returns:
            DNSHAssessment result.
        """
        # Determine which PAI indicators to check based on contribution
        required_indicators: List[str] = []
        contribution = inv.environmental_contribution or inv.social_contribution
        if contribution and contribution in DNSH_PAI_MAPPING:
            required_indicators = DNSH_PAI_MAPPING[contribution]
        else:
            # Check all available indicators
            required_indicators = list(self.config.pai_thresholds.keys())

        checked: List[str] = []
        passed: List[str] = []
        failed: List[str] = []

        for indicator in required_indicators:
            if indicator in inv.pai_data:
                checked.append(indicator)
                threshold = self.config.pai_thresholds.get(indicator)
                value = inv.pai_data[indicator]

                if threshold is not None and self._pai_passes(indicator, value, threshold):
                    passed.append(indicator)
                elif threshold is not None:
                    failed.append(indicator)
                else:
                    passed.append(indicator)  # No threshold = pass by default

        coverage = _safe_pct(len(checked), len(required_indicators)) if required_indicators else 100.0

        if len(failed) > 0:
            status = DNSHStatus.FAILED
        elif coverage < self.config.dnsh_coverage_threshold:
            status = DNSHStatus.INSUFFICIENT_DATA
        elif len(checked) > 0:
            status = DNSHStatus.PASSED
        else:
            status = DNSHStatus.INSUFFICIENT_DATA

        assessment = DNSHAssessment(
            investment_id=inv.investment_id,
            status=status,
            pai_indicators_checked=checked,
            pai_indicators_passed=passed,
            pai_indicators_failed=failed,
            coverage_pct=round(coverage, 2),
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    def _assess_governance(self, inv: InvestmentData) -> GovernanceAssessment:
        """Assess whether an investee company follows good governance.

        Checks four governance criteria per SFDR Article 2(17):
        sound management structures, employee relations,
        remuneration compliance, and tax compliance.

        Args:
            inv: Investment data with governance flags.

        Returns:
            GovernanceAssessment result.
        """
        gov = inv.governance_data
        mgmt = gov.get("sound_management_structures", False)
        employee = gov.get("employee_relations", False)
        remuneration = gov.get("remuneration_compliance", False)
        tax = gov.get("tax_compliance", False)

        criteria_met = sum([mgmt, employee, remuneration, tax])
        has_controversies = gov.get("controversies", False)
        ungc = gov.get("ungc_signatory", False)

        if criteria_met >= self.config.governance_min_criteria and not has_controversies:
            status = GovernanceStatus.GOOD
        elif len(gov) == 0:
            status = GovernanceStatus.INSUFFICIENT_DATA
        else:
            status = GovernanceStatus.INADEQUATE

        assessment = GovernanceAssessment(
            investment_id=inv.investment_id,
            company_name=inv.company_name,
            status=status,
            management_structures=mgmt,
            employee_relations=employee,
            remuneration_compliance=remuneration,
            tax_compliance=tax,
            criteria_met=criteria_met,
            ungc_signatory=ungc,
            controversies_flag=has_controversies,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        return assessment

    def _assess_contribution(self, inv: InvestmentData) -> ObjectiveContribution:
        """Assess which sustainable objective an investment contributes to.

        Args:
            inv: Investment data with contribution flags.

        Returns:
            ObjectiveContribution enum value.
        """
        if inv.environmental_contribution and inv.environmental_contribution != ObjectiveContribution.NONE:
            return inv.environmental_contribution
        if inv.social_contribution and inv.social_contribution != ObjectiveContribution.NONE:
            return inv.social_contribution
        return ObjectiveContribution.NONE

    def _determine_classification(
        self,
        inv: InvestmentData,
        dnsh: DNSHAssessment,
        governance: GovernanceAssessment,
        contribution: ObjectiveContribution,
    ) -> Tuple[InvestmentClassificationType, float]:
        """Determine final classification and confidence.

        Classification logic:
        - All three steps must pass for sustainable classification
        - Taxonomy alignment requires additional Taxonomy criteria
        - Confidence based on data completeness and assessment strength

        Args:
            inv: Investment data.
            dnsh: DNSH assessment result.
            governance: Governance assessment result.
            contribution: Objective contribution.

        Returns:
            Tuple of (classification_type, confidence).
        """
        # Must pass DNSH and governance to be sustainable
        passes_dnsh = dnsh.status == DNSHStatus.PASSED
        passes_governance = governance.status == GovernanceStatus.GOOD
        has_contribution = contribution != ObjectiveContribution.NONE

        if not (passes_dnsh and passes_governance and has_contribution):
            # Calculate confidence for not-sustainable
            confidence = self._calculate_confidence(dnsh, governance, contribution)
            return InvestmentClassificationType.NOT_SUSTAINABLE, confidence

        # Determine specific sustainable category
        confidence = self._calculate_confidence(dnsh, governance, contribution)

        # Check Taxonomy alignment
        if (
            inv.taxonomy_eligible
            and inv.taxonomy_aligned_pct > self.config.taxonomy_alignment_threshold
            and contribution in (
                ObjectiveContribution.CLIMATE_MITIGATION,
                ObjectiveContribution.CLIMATE_ADAPTATION,
                ObjectiveContribution.WATER_MARINE,
                ObjectiveContribution.CIRCULAR_ECONOMY,
                ObjectiveContribution.POLLUTION_PREVENTION,
                ObjectiveContribution.BIODIVERSITY,
            )
        ):
            return InvestmentClassificationType.TAXONOMY_ALIGNED, confidence

        # Environmental but not Taxonomy-aligned
        if contribution in (
            ObjectiveContribution.CLIMATE_MITIGATION,
            ObjectiveContribution.CLIMATE_ADAPTATION,
            ObjectiveContribution.WATER_MARINE,
            ObjectiveContribution.CIRCULAR_ECONOMY,
            ObjectiveContribution.POLLUTION_PREVENTION,
            ObjectiveContribution.BIODIVERSITY,
        ):
            return InvestmentClassificationType.OTHER_ENVIRONMENTAL, confidence

        # Social contribution
        if contribution in (
            ObjectiveContribution.SOCIAL_INEQUALITY,
            ObjectiveContribution.SOCIAL_COHESION,
            ObjectiveContribution.HUMAN_CAPITAL,
        ):
            return InvestmentClassificationType.SOCIAL, confidence

        return InvestmentClassificationType.NOT_SUSTAINABLE, confidence

    def _calculate_confidence(
        self,
        dnsh: DNSHAssessment,
        governance: GovernanceAssessment,
        contribution: ObjectiveContribution,
    ) -> float:
        """Calculate classification confidence based on data completeness.

        Confidence factors:
        - DNSH coverage (40% weight)
        - Governance data completeness (30% weight)
        - Contribution clarity (30% weight)

        Args:
            dnsh: DNSH assessment.
            governance: Governance assessment.
            contribution: Objective contribution.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        # DNSH coverage component (0-1)
        dnsh_score = dnsh.coverage_pct / 100.0

        # Governance completeness component (0-1)
        gov_score = governance.criteria_met / governance.criteria_total

        # Contribution clarity component (0 or 1)
        contrib_score = 1.0 if contribution != ObjectiveContribution.NONE else 0.0

        # Weighted average
        confidence = (dnsh_score * 0.4) + (gov_score * 0.3) + (contrib_score * 0.3)
        return min(confidence, 1.0)

    def _pai_passes(
        self,
        indicator: str,
        value: float,
        threshold: float,
    ) -> bool:
        """Check if a PAI indicator value passes the threshold.

        For most indicators, lower is better (emissions, waste, etc.).
        For some indicators, higher is better (recycling rates, diversity).

        Args:
            indicator: PAI indicator name.
            value: Measured value.
            threshold: Threshold value.

        Returns:
            True if the indicator passes.
        """
        higher_is_better = {"board_gender_diversity", "water_recycling", "waste_recycling"}
        if indicator in higher_is_better:
            return value >= threshold
        else:
            return value <= threshold

    # ------------------------------------------------------------------
    # Proportion Calculation
    # ------------------------------------------------------------------

    def calculate_proportion(
        self,
        classifications: Optional[List[InvestmentClassification]] = None,
    ) -> ProportionResult:
        """Calculate sustainable investment proportions as % of total NAV.

        Aggregates classified investments into SFDR disclosure categories
        and calculates percentage breakdowns.

        Formula:
            sustainable_pct = (sustainable_NAV / total_NAV) * 100

        Args:
            classifications: Optional list of classifications
                (uses stored classifications if not provided).

        Returns:
            ProportionResult with percentage breakdowns.
        """
        start = utcnow()
        if classifications is None:
            classifications = list(self._classifications.values())

        if not classifications:
            logger.warning("No classifications available for proportion calculation")
            return ProportionResult(
                total_nav=0.0,
                total_sustainable_pct=0.0,
                taxonomy_aligned_pct=0.0,
                other_environmental_pct=0.0,
                social_pct=0.0,
                not_sustainable_pct=100.0,
            )

        total_nav = sum(c.nav_value for c in classifications)
        taxonomy_nav = sum(
            c.nav_value for c in classifications
            if c.classification == InvestmentClassificationType.TAXONOMY_ALIGNED
        )
        other_env_nav = sum(
            c.nav_value for c in classifications
            if c.classification == InvestmentClassificationType.OTHER_ENVIRONMENTAL
        )
        social_nav = sum(
            c.nav_value for c in classifications
            if c.classification == InvestmentClassificationType.SOCIAL
        )
        not_sustainable_nav = sum(
            c.nav_value for c in classifications
            if c.classification == InvestmentClassificationType.NOT_SUSTAINABLE
        )

        sustainable_nav = taxonomy_nav + other_env_nav + social_nav
        sustainable_count = sum(
            1 for c in classifications
            if c.classification != InvestmentClassificationType.NOT_SUSTAINABLE
        )

        # Coverage ratio: holdings with sufficient confidence
        high_confidence = sum(1 for c in classifications if c.confidence >= 0.5)
        coverage = _safe_pct(high_confidence, len(classifications))

        result = ProportionResult(
            total_nav=round(total_nav, 2),
            total_sustainable_pct=round(_safe_pct(sustainable_nav, total_nav), 2),
            taxonomy_aligned_pct=round(_safe_pct(taxonomy_nav, total_nav), 2),
            other_environmental_pct=round(_safe_pct(other_env_nav, total_nav), 2),
            social_pct=round(_safe_pct(social_nav, total_nav), 2),
            not_sustainable_pct=round(_safe_pct(not_sustainable_nav, total_nav), 2),
            taxonomy_aligned_nav=round(taxonomy_nav, 2),
            other_environmental_nav=round(other_env_nav, 2),
            social_nav=round(social_nav, 2),
            not_sustainable_nav=round(not_sustainable_nav, 2),
            total_holdings=len(classifications),
            sustainable_holdings=sustainable_count,
            coverage_ratio=round(coverage, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Proportion calculated: %.1f%% sustainable (%.1f%% taxonomy, "
            "%.1f%% other env, %.1f%% social) from %d holdings in %dms",
            result.total_sustainable_pct,
            result.taxonomy_aligned_pct,
            result.other_environmental_pct,
            result.social_pct,
            len(classifications),
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    def breakdown_sustainable(
        self,
        classifications: Optional[List[InvestmentClassification]] = None,
    ) -> Dict[str, List[InvestmentClassification]]:
        """Break down classifications by sustainable category.

        Args:
            classifications: Optional list (uses stored if not provided).

        Returns:
            Dictionary mapping classification type to list of classifications.
        """
        if classifications is None:
            classifications = list(self._classifications.values())

        breakdown: Dict[str, List[InvestmentClassification]] = {
            InvestmentClassificationType.TAXONOMY_ALIGNED.value: [],
            InvestmentClassificationType.OTHER_ENVIRONMENTAL.value: [],
            InvestmentClassificationType.SOCIAL.value: [],
            InvestmentClassificationType.NOT_SUSTAINABLE.value: [],
        }

        for c in classifications:
            breakdown[c.classification.value].append(c)

        return breakdown

    # ------------------------------------------------------------------
    # Commitment Adherence
    # ------------------------------------------------------------------

    def check_minimum_commitment(
        self,
        proportion: Optional[ProportionResult] = None,
    ) -> CommitmentAdherence:
        """Check whether the portfolio meets minimum sustainability commitments.

        Compares actual proportions against the minimum commitments
        declared in pre-contractual disclosures.

        Args:
            proportion: Optional ProportionResult (calculates if not provided).

        Returns:
            CommitmentAdherence result.
        """
        start = utcnow()
        if proportion is None:
            proportion = self.calculate_proportion()

        committed = self.config.minimum_sustainable_pct
        actual = proportion.total_sustainable_pct
        buffer = actual - committed

        # Overall adherence
        if buffer < 0:
            status = AdherenceStatus.BELOW
        elif buffer < self.config.at_risk_buffer_pct:
            status = AdherenceStatus.AT_RISK
        elif actual > committed:
            status = AdherenceStatus.EXCEEDING
        else:
            status = AdherenceStatus.MEETING

        # Taxonomy adherence
        taxonomy_committed = self.config.minimum_taxonomy_pct
        taxonomy_actual = proportion.taxonomy_aligned_pct
        taxonomy_buffer = taxonomy_actual - taxonomy_committed

        if taxonomy_buffer < 0:
            tax_status = AdherenceStatus.BELOW
        elif taxonomy_buffer < self.config.at_risk_buffer_pct:
            tax_status = AdherenceStatus.AT_RISK
        elif taxonomy_actual > taxonomy_committed:
            tax_status = AdherenceStatus.EXCEEDING
        else:
            tax_status = AdherenceStatus.MEETING

        result = CommitmentAdherence(
            committed_minimum_pct=committed,
            actual_proportion_pct=round(actual, 2),
            buffer_pct=round(buffer, 2),
            adherence_status=status,
            taxonomy_committed_pct=taxonomy_committed,
            taxonomy_actual_pct=round(taxonomy_actual, 2),
            taxonomy_adherence=tax_status,
            at_risk_threshold_pct=self.config.at_risk_buffer_pct,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Commitment check: %s (actual=%.1f%%, committed=%.1f%%, buffer=%.1f%%) in %dms",
            status.value,
            actual,
            committed,
            buffer,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_classification_summary(
        self,
        classifications: Optional[List[InvestmentClassification]] = None,
    ) -> ClassificationSummary:
        """Generate a summary of all investment classifications.

        Provides aggregate statistics by sector, country, and classification
        type for disclosure and reporting purposes.

        Args:
            classifications: Optional list (uses stored if not provided).

        Returns:
            ClassificationSummary with aggregate statistics.
        """
        start = utcnow()
        if classifications is None:
            classifications = list(self._classifications.values())

        if not classifications:
            summary = ClassificationSummary()
            summary.provenance_hash = _compute_hash(summary)
            return summary

        # Counts by type
        type_counts: Dict[str, int] = defaultdict(int)
        for c in classifications:
            type_counts[c.classification.value] += 1

        # Average confidence
        avg_conf = sum(c.confidence for c in classifications) / len(classifications)

        # By sector breakdown
        by_sector: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_country: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for c in classifications:
            inv = self._find_investment(c.investment_id)
            sector = inv.sector if inv else "unknown"
            country = inv.country if inv else "unknown"
            by_sector[sector][c.classification.value] += 1
            by_country[country][c.classification.value] += 1

        # DNSH and governance pass rates
        dnsh_passed = sum(
            1 for c in classifications
            if c.dnsh_assessment and c.dnsh_assessment.status == DNSHStatus.PASSED
        )
        gov_passed = sum(
            1 for c in classifications
            if c.governance_assessment and c.governance_assessment.status == GovernanceStatus.GOOD
        )

        summary = ClassificationSummary(
            total_investments=len(classifications),
            taxonomy_aligned_count=type_counts.get(InvestmentClassificationType.TAXONOMY_ALIGNED.value, 0),
            other_environmental_count=type_counts.get(InvestmentClassificationType.OTHER_ENVIRONMENTAL.value, 0),
            social_count=type_counts.get(InvestmentClassificationType.SOCIAL.value, 0),
            not_sustainable_count=type_counts.get(InvestmentClassificationType.NOT_SUSTAINABLE.value, 0),
            average_confidence=round(avg_conf, 3),
            by_sector=dict(by_sector),
            by_country=dict(by_country),
            dnsh_pass_rate=round(_safe_pct(dnsh_passed, len(classifications)), 2),
            governance_pass_rate=round(_safe_pct(gov_passed, len(classifications)), 2),
            classifications=classifications,
        )
        summary.provenance_hash = _compute_hash(summary)

        logger.info(
            "Classification summary: %d total (%d taxonomy, %d other_env, %d social, %d not_sustainable) in %dms",
            summary.total_investments,
            summary.taxonomy_aligned_count,
            summary.other_environmental_count,
            summary.social_count,
            summary.not_sustainable_count,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return summary

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _find_investment(self, investment_id: str) -> Optional[InvestmentData]:
        """Find an investment by ID in the stored investments.

        Args:
            investment_id: Target investment identifier.

        Returns:
            InvestmentData if found, None otherwise.
        """
        for inv in self._investments:
            if inv.investment_id == investment_id:
                return inv
        return None
