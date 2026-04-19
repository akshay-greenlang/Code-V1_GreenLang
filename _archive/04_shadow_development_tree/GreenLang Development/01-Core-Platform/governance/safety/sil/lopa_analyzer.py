"""
LOPAAnalyzer - Layer of Protection Analysis Calculator

This module implements Layer of Protection Analysis (LOPA) per IEC 61511
for determining Safety Integrity Level (SIL) targets for Safety Instrumented
Functions (SIFs).

LOPA is a simplified risk assessment method that:
1. Identifies hazard scenarios
2. Quantifies initiating event frequencies
3. Credits Independent Protection Layers (IPLs)
4. Determines residual risk and required SIL

Reference: IEC 61511-3 Annex F - LOPA

Example:
    >>> from greenlang.safety.sil.lopa_analyzer import LOPAAnalyzer, LOPAScenario
    >>> analyzer = LOPAAnalyzer()
    >>> scenario = LOPAScenario(
    ...     scenario_id="SCN-001",
    ...     description="High pressure in reactor vessel",
    ...     initiating_event_frequency=0.1,  # per year
    ...     consequence_severity="fatality",
    ...     ipls=[
    ...         {"name": "BPCS", "pfd": 0.1},
    ...         {"name": "Alarm", "pfd": 0.1},
    ...         {"name": "Relief Valve", "pfd": 0.01}
    ...     ]
    ... )
    >>> result = analyzer.analyze(scenario)
    >>> print(f"Required SIF PFD: {result.required_sif_pfd}")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import hashlib
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class ConsequenceSeverity(str, Enum):
    """Consequence severity categories per IEC 61511."""

    MINOR = "minor"  # First aid injury
    SERIOUS = "serious"  # Lost time injury
    SEVERE = "severe"  # Permanent disability
    FATALITY = "fatality"  # Single fatality potential
    MULTIPLE_FATALITIES = "multiple_fatalities"  # Multiple fatality potential
    CATASTROPHIC = "catastrophic"  # Community-wide impact


class IPLDefinition(BaseModel):
    """Independent Protection Layer definition."""

    name: str = Field(..., description="Name of the IPL")
    pfd: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of Failure on Demand (0-1)"
    )
    description: Optional[str] = Field(None, description="IPL description")
    is_sis: bool = Field(False, description="Is this a Safety Instrumented System?")

    @field_validator('pfd')
    @classmethod
    def validate_pfd(cls, v: float) -> float:
        """Validate PFD is within acceptable range for IPL."""
        if v < 1e-5:
            logger.warning(f"PFD {v} is very low. Verify IPL credit is justified.")
        if v > 0.1 and not cls.model_fields.get('is_sis'):
            logger.warning(f"PFD {v} exceeds typical IPL credit of 0.1")
        return v


class LOPAScenario(BaseModel):
    """LOPA Scenario definition for analysis."""

    scenario_id: str = Field(..., description="Unique scenario identifier")
    description: str = Field(..., description="Scenario description")
    initiating_event: str = Field(
        default="",
        description="Description of initiating event"
    )
    initiating_event_frequency: float = Field(
        ...,
        gt=0,
        description="Initiating event frequency (per year)"
    )
    consequence_severity: ConsequenceSeverity = Field(
        ...,
        description="Consequence severity category"
    )
    consequence_description: Optional[str] = Field(
        None,
        description="Detailed consequence description"
    )
    ipls: List[IPLDefinition] = Field(
        default_factory=list,
        description="List of Independent Protection Layers"
    )
    conditional_modifiers: Dict[str, float] = Field(
        default_factory=dict,
        description="Conditional modifiers (probability factors)"
    )
    target_mitigated_frequency: Optional[float] = Field(
        None,
        description="Target mitigated event frequency (per year)"
    )

    @field_validator('ipls')
    @classmethod
    def validate_ipls(cls, v: List[IPLDefinition]) -> List[IPLDefinition]:
        """Validate IPL list for independence."""
        names = [ipl.name for ipl in v]
        if len(names) != len(set(names)):
            raise ValueError("IPL names must be unique")
        return v


class LOPAResult(BaseModel):
    """LOPA Analysis result."""

    scenario_id: str = Field(..., description="Scenario identifier")
    initiating_event_frequency: float = Field(
        ...,
        description="Initiating event frequency (per year)"
    )
    total_ipl_pfd: float = Field(
        ...,
        description="Combined PFD of all IPLs"
    )
    conditional_modifier_product: float = Field(
        ...,
        description="Product of all conditional modifiers"
    )
    unmitigated_frequency: float = Field(
        ...,
        description="Frequency before IPL credit"
    )
    mitigated_frequency: float = Field(
        ...,
        description="Frequency after IPL credit"
    )
    target_frequency: float = Field(
        ...,
        description="Target tolerable frequency"
    )
    risk_gap: float = Field(
        ...,
        description="Gap between mitigated and target frequency"
    )
    sif_required: bool = Field(
        ...,
        description="Is a SIF required?"
    )
    required_sif_pfd: Optional[float] = Field(
        None,
        description="Required SIF PFD if SIF needed"
    )
    recommended_sil: Optional[int] = Field(
        None,
        description="Recommended SIL level (1-4)"
    )
    ipls_credited: List[str] = Field(
        default_factory=list,
        description="List of credited IPLs"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of calculation"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Analysis warnings"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LOPAAnalyzer:
    """
    Layer of Protection Analysis (LOPA) Analyzer.

    Implements LOPA methodology per IEC 61511-3 Annex F for determining
    SIL targets for Safety Instrumented Functions.

    The analyzer follows zero-hallucination principles:
    - All calculations are deterministic
    - No LLM involvement in numeric computations
    - Complete audit trail with provenance hashing

    Attributes:
        target_frequencies: Dict mapping severity to target frequency

    Example:
        >>> analyzer = LOPAAnalyzer()
        >>> result = analyzer.analyze(scenario)
        >>> print(f"SIF Required: {result.sif_required}")
        >>> print(f"Required SIL: {result.recommended_sil}")
    """

    # Default target frequencies per consequence severity (per year)
    # Based on typical ALARP criteria
    DEFAULT_TARGET_FREQUENCIES: Dict[ConsequenceSeverity, float] = {
        ConsequenceSeverity.MINOR: 1.0,
        ConsequenceSeverity.SERIOUS: 0.1,
        ConsequenceSeverity.SEVERE: 0.01,
        ConsequenceSeverity.FATALITY: 1e-4,
        ConsequenceSeverity.MULTIPLE_FATALITIES: 1e-5,
        ConsequenceSeverity.CATASTROPHIC: 1e-6,
    }

    # SIL PFD ranges per IEC 61511
    SIL_PFD_RANGES: Dict[int, tuple] = {
        4: (1e-5, 1e-4),
        3: (1e-4, 1e-3),
        2: (1e-3, 1e-2),
        1: (1e-2, 1e-1),
    }

    def __init__(
        self,
        target_frequencies: Optional[Dict[ConsequenceSeverity, float]] = None
    ):
        """
        Initialize LOPAAnalyzer.

        Args:
            target_frequencies: Optional custom target frequencies per severity.
                               Uses IEC 61511 defaults if not provided.
        """
        self.target_frequencies = (
            target_frequencies or self.DEFAULT_TARGET_FREQUENCIES.copy()
        )
        logger.info("LOPAAnalyzer initialized with target frequencies")

    def analyze(self, scenario: LOPAScenario) -> LOPAResult:
        """
        Perform LOPA analysis on a scenario.

        Args:
            scenario: LOPAScenario to analyze

        Returns:
            LOPAResult with analysis results

        Raises:
            ValueError: If scenario data is invalid
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []

        logger.info(f"Starting LOPA analysis for scenario: {scenario.scenario_id}")

        try:
            # Step 1: Calculate conditional modifier product
            conditional_product = self._calculate_conditional_product(
                scenario.conditional_modifiers
            )

            # Step 2: Calculate unmitigated frequency
            unmitigated_frequency = (
                scenario.initiating_event_frequency * conditional_product
            )

            # Step 3: Calculate total IPL PFD
            total_ipl_pfd, ipl_warnings = self._calculate_total_ipl_pfd(
                scenario.ipls
            )
            warnings.extend(ipl_warnings)

            # Step 4: Calculate mitigated frequency
            mitigated_frequency = unmitigated_frequency * total_ipl_pfd

            # Step 5: Get target frequency
            target_frequency = self._get_target_frequency(
                scenario.consequence_severity,
                scenario.target_mitigated_frequency
            )

            # Step 6: Calculate risk gap
            risk_gap = mitigated_frequency / target_frequency

            # Step 7: Determine if SIF is required
            sif_required = risk_gap > 1.0

            # Step 8: Calculate required SIF PFD if needed
            required_sif_pfd = None
            recommended_sil = None

            if sif_required:
                required_sif_pfd = target_frequency / (
                    unmitigated_frequency * total_ipl_pfd
                )
                # Apply safety factor (typically 0.5 to account for uncertainty)
                required_sif_pfd *= 0.5
                recommended_sil = self._pfd_to_sil(required_sif_pfd)

                if recommended_sil is None:
                    warnings.append(
                        f"Required PFD {required_sif_pfd:.2e} exceeds SIL 4 capability. "
                        "Consider additional IPLs or inherently safer design."
                    )
                    recommended_sil = 4  # Cap at SIL 4

            # Step 9: Generate provenance hash
            provenance_hash = self._calculate_provenance(
                scenario, mitigated_frequency, target_frequency
            )

            # Build result
            result = LOPAResult(
                scenario_id=scenario.scenario_id,
                initiating_event_frequency=scenario.initiating_event_frequency,
                total_ipl_pfd=total_ipl_pfd,
                conditional_modifier_product=conditional_product,
                unmitigated_frequency=unmitigated_frequency,
                mitigated_frequency=mitigated_frequency,
                target_frequency=target_frequency,
                risk_gap=risk_gap,
                sif_required=sif_required,
                required_sif_pfd=required_sif_pfd,
                recommended_sil=recommended_sil,
                ipls_credited=[ipl.name for ipl in scenario.ipls],
                calculation_timestamp=start_time,
                provenance_hash=provenance_hash,
                warnings=warnings,
            )

            logger.info(
                f"LOPA analysis complete for {scenario.scenario_id}. "
                f"SIF Required: {sif_required}, Recommended SIL: {recommended_sil}"
            )

            return result

        except Exception as e:
            logger.error(
                f"LOPA analysis failed for {scenario.scenario_id}: {str(e)}",
                exc_info=True
            )
            raise

    def _calculate_conditional_product(
        self,
        modifiers: Dict[str, float]
    ) -> float:
        """
        Calculate product of conditional modifiers.

        Common modifiers include:
        - Probability of ignition
        - Probability of personnel in area
        - Probability of weather conditions

        Args:
            modifiers: Dict of modifier name to probability

        Returns:
            Product of all modifiers (1.0 if none)
        """
        if not modifiers:
            return 1.0

        product = 1.0
        for name, value in modifiers.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Conditional modifier '{name}' value {value} "
                    "must be between 0 and 1"
                )
            product *= value
            logger.debug(f"Applied conditional modifier '{name}': {value}")

        return product

    def _calculate_total_ipl_pfd(
        self,
        ipls: List[IPLDefinition]
    ) -> tuple:
        """
        Calculate combined PFD of all IPLs.

        IPLs are assumed independent per IEC 61511.
        Combined PFD = PFD1 * PFD2 * ... * PFDn

        Args:
            ipls: List of IPL definitions

        Returns:
            Tuple of (total_pfd, warnings)
        """
        warnings: List[str] = []

        if not ipls:
            logger.debug("No IPLs credited")
            return 1.0, warnings

        total_pfd = 1.0
        for ipl in ipls:
            total_pfd *= ipl.pfd

            # Check for excessive IPL credit
            if ipl.pfd < 0.01 and not ipl.is_sis:
                warnings.append(
                    f"IPL '{ipl.name}' has PFD {ipl.pfd} < 0.01. "
                    "Ensure this credit is justified for non-SIS IPL."
                )

        logger.debug(f"Total IPL PFD: {total_pfd:.2e}")
        return total_pfd, warnings

    def _get_target_frequency(
        self,
        severity: ConsequenceSeverity,
        override: Optional[float] = None
    ) -> float:
        """
        Get target tolerable frequency.

        Args:
            severity: Consequence severity category
            override: Optional override value

        Returns:
            Target frequency (per year)
        """
        if override is not None:
            logger.debug(f"Using override target frequency: {override}")
            return override

        return self.target_frequencies[severity]

    def _pfd_to_sil(self, pfd: float) -> Optional[int]:
        """
        Convert PFD to SIL level.

        Args:
            pfd: Probability of Failure on Demand

        Returns:
            SIL level (1-4) or None if beyond SIL 4
        """
        for sil, (lower, upper) in self.SIL_PFD_RANGES.items():
            if lower <= pfd < upper:
                return sil

        if pfd < 1e-5:
            return None  # Beyond SIL 4

        if pfd >= 0.1:
            return None  # Below SIL 1

        return None

    def _calculate_provenance(
        self,
        scenario: LOPAScenario,
        mitigated_freq: float,
        target_freq: float
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            scenario: Input scenario
            mitigated_freq: Calculated mitigated frequency
            target_freq: Target frequency used

        Returns:
            SHA-256 hex digest
        """
        provenance_str = (
            f"{scenario.scenario_id}|"
            f"{scenario.initiating_event_frequency}|"
            f"{mitigated_freq}|"
            f"{target_freq}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def batch_analyze(
        self,
        scenarios: List[LOPAScenario]
    ) -> List[LOPAResult]:
        """
        Analyze multiple scenarios.

        Args:
            scenarios: List of scenarios to analyze

        Returns:
            List of LOPAResults
        """
        results = []
        for scenario in scenarios:
            try:
                result = self.analyze(scenario)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to analyze scenario {scenario.scenario_id}: {e}"
                )
                raise

        return results

    def generate_summary_report(
        self,
        results: List[LOPAResult]
    ) -> Dict[str, Any]:
        """
        Generate summary report for multiple LOPA analyses.

        Args:
            results: List of LOPAResults

        Returns:
            Summary report dictionary
        """
        sif_required_count = sum(1 for r in results if r.sif_required)
        sil_distribution = {1: 0, 2: 0, 3: 0, 4: 0}

        for result in results:
            if result.recommended_sil:
                sil_distribution[result.recommended_sil] += 1

        return {
            "total_scenarios": len(results),
            "sif_required_count": sif_required_count,
            "sil_distribution": sil_distribution,
            "max_risk_gap": max(r.risk_gap for r in results) if results else 0,
            "scenarios_exceeding_target": [
                r.scenario_id for r in results if r.risk_gap > 1.0
            ],
            "generated_at": datetime.utcnow().isoformat(),
        }
