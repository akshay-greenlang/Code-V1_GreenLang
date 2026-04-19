"""
SILClassifier - Safety Integrity Level Classification

This module implements SIL classification per IEC 61511 based on:
- PFD (Probability of Failure on Demand) for low-demand mode
- PFH (Probability of Failure per Hour) for high/continuous demand mode
- Risk matrix calibration tables
- LOPA results

Reference: IEC 61511-1 Table 4, IEC 61508-1 Table 2

Example:
    >>> from greenlang.safety.sil.sil_classifier import SILClassifier, SILLevel
    >>> classifier = SILClassifier()
    >>> result = classifier.classify_from_pfd(pfd_avg=1e-3)
    >>> print(f"SIL Level: {result.sil_level.value}")
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum, IntEnum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SILLevel(IntEnum):
    """Safety Integrity Level per IEC 61511."""

    SIL_0 = 0  # Below SIL 1 (not safety-rated)
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4  # Highest SIL (rarely used in process industry)


class DemandMode(str, Enum):
    """Demand mode classification per IEC 61511."""

    LOW_DEMAND = "low_demand"  # < 1 demand per year
    HIGH_DEMAND = "high_demand"  # >= 1 demand per year
    CONTINUOUS = "continuous"  # Continuous safety function


class ConsequenceCategory(str, Enum):
    """Consequence severity categories for risk matrix."""

    C1 = "C1"  # Minor injury
    C2 = "C2"  # Serious permanent injury to one person
    C3 = "C3"  # Death of one person
    C4 = "C4"  # Death of several persons


class FrequencyCategory(str, Enum):
    """Exposure frequency categories for risk matrix."""

    F1 = "F1"  # Rare exposure
    F2 = "F2"  # Frequent exposure


class ProbabilityCategory(str, Enum):
    """Probability of avoiding hazard categories."""

    P1 = "P1"  # Possible under certain conditions
    P2 = "P2"  # Almost impossible


class DemandRateCategory(str, Enum):
    """Demand rate categories for risk graph."""

    W1 = "W1"  # Very low demand rate
    W2 = "W2"  # Low demand rate
    W3 = "W3"  # High demand rate


class SILClassificationResult(BaseModel):
    """Result of SIL classification."""

    sil_level: SILLevel = Field(
        ...,
        description="Determined SIL level"
    )
    demand_mode: DemandMode = Field(
        ...,
        description="Demand mode used for classification"
    )
    pfd_avg: Optional[float] = Field(
        None,
        description="PFDavg value if low demand mode"
    )
    pfh: Optional[float] = Field(
        None,
        description="PFH value if high/continuous demand mode"
    )
    risk_reduction_factor: float = Field(
        ...,
        description="Required Risk Reduction Factor"
    )
    classification_method: str = Field(
        ...,
        description="Method used for classification"
    )
    meets_target: bool = Field(
        ...,
        description="Does achieved SIL meet target?"
    )
    target_sil: Optional[SILLevel] = Field(
        None,
        description="Target SIL if specified"
    )
    margin: Optional[float] = Field(
        None,
        description="Safety margin (ratio of achieved to required)"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of classification"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Classification notes and recommendations"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RiskGraphInput(BaseModel):
    """Input for risk graph SIL determination per IEC 61508-5."""

    consequence: ConsequenceCategory = Field(
        ...,
        description="Consequence severity category"
    )
    frequency: FrequencyCategory = Field(
        ...,
        description="Exposure frequency category"
    )
    probability: ProbabilityCategory = Field(
        ...,
        description="Probability of avoiding hazard"
    )
    demand_rate: DemandRateCategory = Field(
        ...,
        description="Demand rate category"
    )


class SILClassifier:
    """
    Safety Integrity Level Classifier.

    Implements SIL classification methods per IEC 61511 and IEC 61508:
    - Classification from PFD/PFH values
    - Risk graph method
    - Risk matrix method
    - LOPA-based classification

    The classifier follows zero-hallucination principles with
    deterministic table lookups only.

    Attributes:
        pfd_ranges: SIL classification ranges for low demand mode
        pfh_ranges: SIL classification ranges for high/continuous demand

    Example:
        >>> classifier = SILClassifier()
        >>> result = classifier.classify_from_pfd(1e-3)
        >>> print(f"Achieved: SIL {result.sil_level}")
    """

    # PFD ranges for low demand mode per IEC 61511-1 Table 4
    PFD_RANGES: Dict[SILLevel, Tuple[float, float]] = {
        SILLevel.SIL_4: (1e-5, 1e-4),
        SILLevel.SIL_3: (1e-4, 1e-3),
        SILLevel.SIL_2: (1e-3, 1e-2),
        SILLevel.SIL_1: (1e-2, 1e-1),
        SILLevel.SIL_0: (1e-1, 1.0),
    }

    # PFH ranges for high demand/continuous mode per IEC 61511-1 Table 4
    PFH_RANGES: Dict[SILLevel, Tuple[float, float]] = {
        SILLevel.SIL_4: (1e-9, 1e-8),
        SILLevel.SIL_3: (1e-8, 1e-7),
        SILLevel.SIL_2: (1e-7, 1e-6),
        SILLevel.SIL_1: (1e-6, 1e-5),
        SILLevel.SIL_0: (1e-5, 1.0),
    }

    # Risk graph lookup table per IEC 61508-5 Figure D.1
    # Key: (C, F, P, W) -> SIL requirement
    RISK_GRAPH_TABLE: Dict[Tuple[str, str, str, str], Optional[int]] = {
        # C1 combinations
        ("C1", "F1", "P1", "W1"): None,  # No SIL required
        ("C1", "F1", "P1", "W2"): None,
        ("C1", "F1", "P1", "W3"): None,
        ("C1", "F1", "P2", "W1"): None,
        ("C1", "F1", "P2", "W2"): None,
        ("C1", "F1", "P2", "W3"): 1,
        ("C1", "F2", "P1", "W1"): None,
        ("C1", "F2", "P1", "W2"): None,
        ("C1", "F2", "P1", "W3"): 1,
        ("C1", "F2", "P2", "W1"): None,
        ("C1", "F2", "P2", "W2"): 1,
        ("C1", "F2", "P2", "W3"): 2,
        # C2 combinations
        ("C2", "F1", "P1", "W1"): None,
        ("C2", "F1", "P1", "W2"): None,
        ("C2", "F1", "P1", "W3"): 1,
        ("C2", "F1", "P2", "W1"): None,
        ("C2", "F1", "P2", "W2"): 1,
        ("C2", "F1", "P2", "W3"): 2,
        ("C2", "F2", "P1", "W1"): None,
        ("C2", "F2", "P1", "W2"): 1,
        ("C2", "F2", "P1", "W3"): 2,
        ("C2", "F2", "P2", "W1"): 1,
        ("C2", "F2", "P2", "W2"): 2,
        ("C2", "F2", "P2", "W3"): 3,
        # C3 combinations
        ("C3", "F1", "P1", "W1"): None,
        ("C3", "F1", "P1", "W2"): 1,
        ("C3", "F1", "P1", "W3"): 2,
        ("C3", "F1", "P2", "W1"): 1,
        ("C3", "F1", "P2", "W2"): 2,
        ("C3", "F1", "P2", "W3"): 3,
        ("C3", "F2", "P1", "W1"): 1,
        ("C3", "F2", "P1", "W2"): 2,
        ("C3", "F2", "P1", "W3"): 3,
        ("C3", "F2", "P2", "W1"): 2,
        ("C3", "F2", "P2", "W2"): 3,
        ("C3", "F2", "P2", "W3"): 4,
        # C4 combinations
        ("C4", "F1", "P1", "W1"): 1,
        ("C4", "F1", "P1", "W2"): 2,
        ("C4", "F1", "P1", "W3"): 3,
        ("C4", "F1", "P2", "W1"): 2,
        ("C4", "F1", "P2", "W2"): 3,
        ("C4", "F1", "P2", "W3"): 4,
        ("C4", "F2", "P1", "W1"): 2,
        ("C4", "F2", "P1", "W2"): 3,
        ("C4", "F2", "P1", "W3"): 4,
        ("C4", "F2", "P2", "W1"): 3,
        ("C4", "F2", "P2", "W2"): 4,
        ("C4", "F2", "P2", "W3"): 4,  # SIL 4 max, may need additional measures
    }

    def __init__(self):
        """Initialize SILClassifier."""
        logger.info("SILClassifier initialized")

    def classify_from_pfd(
        self,
        pfd_avg: float,
        target_sil: Optional[SILLevel] = None
    ) -> SILClassificationResult:
        """
        Classify SIL from PFDavg value (low demand mode).

        Args:
            pfd_avg: Average Probability of Failure on Demand
            target_sil: Optional target SIL to compare against

        Returns:
            SILClassificationResult with classification details

        Raises:
            ValueError: If pfd_avg is out of valid range
        """
        logger.info(f"Classifying SIL from PFDavg: {pfd_avg:.2e}")

        if not 0 < pfd_avg <= 1.0:
            raise ValueError(f"PFDavg must be between 0 and 1, got {pfd_avg}")

        # Determine SIL level from PFD ranges
        sil_level = SILLevel.SIL_0
        for sil, (lower, upper) in self.PFD_RANGES.items():
            if lower <= pfd_avg < upper:
                sil_level = sil
                break

        # Handle better than SIL 4
        if pfd_avg < 1e-5:
            sil_level = SILLevel.SIL_4

        # Calculate risk reduction factor
        rrf = 1.0 / pfd_avg

        # Check against target
        meets_target = True
        margin = None
        if target_sil is not None:
            meets_target = sil_level >= target_sil
            target_pfd_upper = self.PFD_RANGES[target_sil][1]
            margin = target_pfd_upper / pfd_avg

        # Generate notes
        notes = self._generate_classification_notes(
            sil_level, pfd_avg, target_sil, "pfd"
        )

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(
            "pfd", pfd_avg, sil_level.value
        )

        result = SILClassificationResult(
            sil_level=sil_level,
            demand_mode=DemandMode.LOW_DEMAND,
            pfd_avg=pfd_avg,
            pfh=None,
            risk_reduction_factor=rrf,
            classification_method="PFD table lookup per IEC 61511-1 Table 4",
            meets_target=meets_target,
            target_sil=target_sil,
            margin=margin,
            provenance_hash=provenance_hash,
            notes=notes,
        )

        logger.info(
            f"SIL classification complete: SIL {sil_level.value}, "
            f"RRF: {rrf:.0f}"
        )

        return result

    def classify_from_pfh(
        self,
        pfh: float,
        target_sil: Optional[SILLevel] = None
    ) -> SILClassificationResult:
        """
        Classify SIL from PFH value (high demand/continuous mode).

        Args:
            pfh: Probability of Failure per Hour
            target_sil: Optional target SIL to compare against

        Returns:
            SILClassificationResult with classification details

        Raises:
            ValueError: If pfh is out of valid range
        """
        logger.info(f"Classifying SIL from PFH: {pfh:.2e}")

        if not 0 < pfh <= 1.0:
            raise ValueError(f"PFH must be between 0 and 1, got {pfh}")

        # Determine SIL level from PFH ranges
        sil_level = SILLevel.SIL_0
        for sil, (lower, upper) in self.PFH_RANGES.items():
            if lower <= pfh < upper:
                sil_level = sil
                break

        # Handle better than SIL 4
        if pfh < 1e-9:
            sil_level = SILLevel.SIL_4

        # Calculate risk reduction factor (per hour basis)
        rrf = 1.0 / pfh

        # Check against target
        meets_target = True
        margin = None
        if target_sil is not None:
            meets_target = sil_level >= target_sil
            target_pfh_upper = self.PFH_RANGES[target_sil][1]
            margin = target_pfh_upper / pfh

        # Generate notes
        notes = self._generate_classification_notes(
            sil_level, pfh, target_sil, "pfh"
        )

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(
            "pfh", pfh, sil_level.value
        )

        result = SILClassificationResult(
            sil_level=sil_level,
            demand_mode=DemandMode.HIGH_DEMAND,
            pfd_avg=None,
            pfh=pfh,
            risk_reduction_factor=rrf,
            classification_method="PFH table lookup per IEC 61511-1 Table 4",
            meets_target=meets_target,
            target_sil=target_sil,
            margin=margin,
            provenance_hash=provenance_hash,
            notes=notes,
        )

        logger.info(
            f"SIL classification complete: SIL {sil_level.value}"
        )

        return result

    def classify_from_risk_graph(
        self,
        risk_input: RiskGraphInput
    ) -> SILClassificationResult:
        """
        Determine SIL using risk graph method per IEC 61508-5.

        Args:
            risk_input: Risk graph input parameters

        Returns:
            SILClassificationResult with target SIL

        Raises:
            ValueError: If combination not found in table
        """
        logger.info(
            f"Classifying SIL using risk graph: "
            f"C={risk_input.consequence}, F={risk_input.frequency}, "
            f"P={risk_input.probability}, W={risk_input.demand_rate}"
        )

        # Look up in risk graph table
        key = (
            risk_input.consequence.value,
            risk_input.frequency.value,
            risk_input.probability.value,
            risk_input.demand_rate.value
        )

        if key not in self.RISK_GRAPH_TABLE:
            raise ValueError(f"Risk graph combination not found: {key}")

        sil_value = self.RISK_GRAPH_TABLE[key]

        if sil_value is None:
            sil_level = SILLevel.SIL_0
            notes = ["No SIL requirement determined. Non-SIS measures may suffice."]
        else:
            sil_level = SILLevel(sil_value)
            notes = [f"SIL {sil_value} required based on risk graph analysis."]

        # Estimate required RRF from SIL
        if sil_level == SILLevel.SIL_0:
            rrf = 1.0
        else:
            # Use midpoint of PFD range
            lower, upper = self.PFD_RANGES[sil_level]
            pfd_mid = (lower * upper) ** 0.5  # Geometric mean
            rrf = 1.0 / pfd_mid

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(
            "risk_graph", str(key), sil_level.value
        )

        result = SILClassificationResult(
            sil_level=sil_level,
            demand_mode=DemandMode.LOW_DEMAND,  # Default assumption
            pfd_avg=None,
            pfh=None,
            risk_reduction_factor=rrf,
            classification_method="Risk graph per IEC 61508-5 Figure D.1",
            meets_target=True,  # This IS the target
            target_sil=sil_level,
            margin=1.0,
            provenance_hash=provenance_hash,
            notes=notes,
        )

        logger.info(f"Risk graph classification: SIL {sil_level.value}")

        return result

    def determine_demand_mode(
        self,
        demands_per_year: float
    ) -> DemandMode:
        """
        Determine demand mode based on demand frequency.

        Per IEC 61511-1 clause 3.2.52:
        - Low demand: < 1 demand per year
        - High demand: >= 1 demand per year

        Args:
            demands_per_year: Expected demands per year

        Returns:
            DemandMode classification
        """
        if demands_per_year < 1.0:
            mode = DemandMode.LOW_DEMAND
        else:
            mode = DemandMode.HIGH_DEMAND

        logger.debug(
            f"Demand mode for {demands_per_year} demands/year: {mode.value}"
        )
        return mode

    def get_pfd_range(self, sil: SILLevel) -> Tuple[float, float]:
        """
        Get PFD range for a SIL level.

        Args:
            sil: SIL level

        Returns:
            Tuple of (lower, upper) PFD bounds
        """
        return self.PFD_RANGES[sil]

    def get_pfh_range(self, sil: SILLevel) -> Tuple[float, float]:
        """
        Get PFH range for a SIL level.

        Args:
            sil: SIL level

        Returns:
            Tuple of (lower, upper) PFH bounds
        """
        return self.PFH_RANGES[sil]

    def get_target_pfd(
        self,
        sil: SILLevel,
        safety_margin: float = 2.0
    ) -> float:
        """
        Get target PFD for a SIL level with safety margin.

        Args:
            sil: Target SIL level
            safety_margin: Safety margin factor (default 2x)

        Returns:
            Target PFD value
        """
        lower, upper = self.PFD_RANGES[sil]
        # Target is upper bound divided by safety margin
        target = upper / safety_margin
        return target

    def _generate_classification_notes(
        self,
        sil_level: SILLevel,
        value: float,
        target_sil: Optional[SILLevel],
        method: str
    ) -> List[str]:
        """Generate classification notes and recommendations."""
        notes = []

        if sil_level == SILLevel.SIL_0:
            notes.append(
                "SIL 0: Does not meet minimum SIL 1 requirements. "
                "Consider design improvements or additional protection layers."
            )
        elif sil_level == SILLevel.SIL_4:
            notes.append(
                "SIL 4: Highest SIL level. Rarely claimed in process industry. "
                "Verify with independent assessment."
            )

        if target_sil is not None:
            if sil_level < target_sil:
                notes.append(
                    f"WARNING: Achieved SIL {sil_level.value} does not meet "
                    f"target SIL {target_sil.value}. Design improvements required."
                )
            elif sil_level > target_sil:
                notes.append(
                    f"Achieved SIL {sil_level.value} exceeds target SIL {target_sil.value}. "
                    f"May allow optimization of proof test intervals."
                )

        return notes

    def _calculate_provenance(
        self,
        method: str,
        input_value: Any,
        sil_value: int
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_str = (
            f"{method}|{input_value}|{sil_value}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def validate_sil_for_application(
        self,
        achieved_sil: SILLevel,
        target_sil: SILLevel,
        pfd_avg: float,
        hft: int = 0
    ) -> Dict[str, Any]:
        """
        Validate achieved SIL meets requirements including HFT.

        Args:
            achieved_sil: SIL achieved from PFD calculation
            target_sil: Required SIL from risk assessment
            pfd_avg: Achieved PFDavg
            hft: Hardware Fault Tolerance achieved

        Returns:
            Validation result dictionary
        """
        # Minimum HFT requirements per IEC 61511-1 Table 6
        min_hft_type_a = {
            SILLevel.SIL_1: 0,
            SILLevel.SIL_2: 1,
            SILLevel.SIL_3: 2,
            SILLevel.SIL_4: 3,
        }

        min_hft_type_b = {
            SILLevel.SIL_1: 1,
            SILLevel.SIL_2: 2,
            SILLevel.SIL_3: 3,
            SILLevel.SIL_4: 4,  # SIL 4 not achievable with Type B
        }

        required_hft = min_hft_type_a.get(target_sil, 0)

        validation = {
            "pfd_meets_target": achieved_sil >= target_sil,
            "achieved_sil": achieved_sil.value,
            "target_sil": target_sil.value,
            "pfd_avg": pfd_avg,
            "achieved_hft": hft,
            "required_hft_type_a": required_hft,
            "hft_meets_requirement": hft >= required_hft,
            "overall_valid": (achieved_sil >= target_sil) and (hft >= required_hft),
            "recommendations": []
        }

        if not validation["pfd_meets_target"]:
            validation["recommendations"].append(
                f"Improve PFD to achieve SIL {target_sil.value}. "
                f"Consider increased redundancy or reduced proof test interval."
            )

        if not validation["hft_meets_requirement"]:
            validation["recommendations"].append(
                f"Increase HFT to {required_hft} for SIL {target_sil.value}. "
                f"Current HFT is {hft}."
            )

        return validation
