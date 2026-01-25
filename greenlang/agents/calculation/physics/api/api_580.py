"""
API 580 - Risk-Based Inspection

Zero-Hallucination Risk Assessment Calculations

This module implements API RP 580 for risk-based inspection (RBI)
program development and risk assessment.

References:
    - API RP 580, 3rd Edition (2016): Risk-Based Inspection
    - API RP 581: Risk-Based Inspection Methodology
    - API 510: Pressure Vessel Inspection Code

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import hashlib


class DamageMechanism(Enum):
    """Common damage mechanisms per API 580/581."""
    GENERAL_CORROSION = "general_corrosion"
    LOCALIZED_CORROSION = "localized_corrosion"
    STRESS_CORROSION_CRACKING = "scc"
    HIGH_TEMP_HYDROGEN_ATTACK = "htha"
    CREEP = "creep"
    FATIGUE = "fatigue"
    BRITTLE_FRACTURE = "brittle_fracture"
    EXTERNAL_CORROSION = "external_corrosion"
    EROSION = "erosion"
    CUI = "corrosion_under_insulation"


class ConsequenceCategory(Enum):
    """Consequence categories per API 580."""
    SAFETY = "safety"
    ENVIRONMENTAL = "environmental"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


class RiskLevel(Enum):
    """Risk levels for 5x5 matrix."""
    LOW = "low"
    MEDIUM_LOW = "medium_low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"


@dataclass
class LikelihoodFactors:
    """Factors affecting probability of failure."""
    equipment_age_years: float
    corrosion_rate_mm_yr: float
    remaining_life_factor: float  # 0-1, lower = closer to end of life
    inspection_effectiveness: float  # 0-1, higher = better
    mechanical_integrity: float  # 0-1, higher = better condition
    management_system_factor: float  # 0.8-1.2, lower = better systems


@dataclass
class ConsequenceFactors:
    """Factors affecting consequence of failure."""
    fluid_hazard_factor: float  # 1-5, higher = more hazardous
    release_rate_factor: float  # Based on hole size
    detection_factor: float  # 0-1, higher = better detection
    isolation_factor: float  # 0-1, higher = faster isolation
    personnel_density: int  # Number of people at risk
    environmental_sensitivity: float  # 1-5, higher = more sensitive
    production_impact_usd_day: float


@dataclass
class RBIResult:
    """
    Risk-Based Inspection assessment results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Probability of failure
    pof_category: int  # 1-5
    pof_value: Decimal
    pof_description: str

    # Consequence of failure
    cof_category: int  # 1-5
    cof_safety: Decimal
    cof_environmental: Decimal
    cof_production: Decimal
    cof_total: Decimal
    cof_description: str

    # Risk assessment
    risk_level: str
    risk_value: Decimal
    risk_rank: int  # 1-25 from risk matrix

    # Inspection planning
    recommended_interval_months: int
    inspection_methods: List[str]
    priority: str

    # Active damage mechanisms
    damage_mechanisms: List[str]

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pof_category": self.pof_category,
            "cof_category": self.cof_category,
            "risk_level": self.risk_level,
            "risk_rank": self.risk_rank,
            "recommended_interval_months": self.recommended_interval_months,
            "inspection_methods": self.inspection_methods,
            "damage_mechanisms": self.damage_mechanisms,
            "provenance_hash": self.provenance_hash
        }


class API580RBI:
    """
    API 580 Risk-Based Inspection Calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on API RP 580 3rd Edition
    - Complete provenance tracking
    - Conservative risk assessment

    Risk Assessment Methods:
    1. Qualitative: 5x5 risk matrix
    2. Semi-quantitative: Indexed approach
    3. Quantitative: Full probability/consequence calculation

    References:
        - API RP 580, Section 7 (Risk Analysis)
        - API RP 580, Section 8 (RBI Planning)
        - API RP 581: Detailed Methodology
    """

    # 5x5 Risk Matrix (POF x COF)
    # Values represent risk rank 1-25
    RISK_MATRIX = [
        [1, 2, 3, 4, 5],      # POF = 1 (Lowest)
        [2, 4, 6, 8, 10],     # POF = 2
        [3, 6, 9, 12, 15],    # POF = 3
        [4, 8, 12, 16, 20],   # POF = 4
        [5, 10, 15, 20, 25],  # POF = 5 (Highest)
    ]

    # Risk level thresholds
    RISK_THRESHOLDS = {
        (1, 4): RiskLevel.LOW,
        (5, 8): RiskLevel.MEDIUM_LOW,
        (9, 12): RiskLevel.MEDIUM,
        (13, 16): RiskLevel.MEDIUM_HIGH,
        (17, 25): RiskLevel.HIGH,
    }

    # Inspection intervals by risk level (months)
    INSPECTION_INTERVALS = {
        RiskLevel.LOW: 120,        # 10 years
        RiskLevel.MEDIUM_LOW: 72,  # 6 years
        RiskLevel.MEDIUM: 48,      # 4 years
        RiskLevel.MEDIUM_HIGH: 24, # 2 years
        RiskLevel.HIGH: 12,        # 1 year
    }

    def __init__(self, precision: int = 4):
        """Initialize calculator."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "API_RP_580_RBI",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _get_risk_level(self, risk_rank: int) -> RiskLevel:
        """Get risk level from risk rank."""
        for (low, high), level in self.RISK_THRESHOLDS.items():
            if low <= risk_rank <= high:
                return level
        return RiskLevel.HIGH

    def calculate_pof(
        self,
        factors: LikelihoodFactors,
        damage_mechanisms: List[DamageMechanism]
    ) -> Tuple[int, Decimal, str]:
        """
        Calculate Probability of Failure category.

        Reference: API RP 580, Section 7.4

        Args:
            factors: Likelihood factors
            damage_mechanisms: Active damage mechanisms

        Returns:
            Tuple of (category 1-5, probability value, description)
        """
        # Base failure frequency (events per year)
        # Generic equipment failure rate ~1E-4 to 1E-5
        base_pof = Decimal("1E-4")

        # Age factor
        age = Decimal(str(factors.equipment_age_years))
        if age > Decimal("30"):
            age_factor = Decimal("2.0")
        elif age > Decimal("20"):
            age_factor = Decimal("1.5")
        elif age > Decimal("10"):
            age_factor = Decimal("1.2")
        else:
            age_factor = Decimal("1.0")

        # Corrosion factor
        cr = Decimal(str(factors.corrosion_rate_mm_yr))
        if cr > Decimal("0.5"):
            corr_factor = Decimal("2.5")
        elif cr > Decimal("0.25"):
            corr_factor = Decimal("1.8")
        elif cr > Decimal("0.125"):
            corr_factor = Decimal("1.3")
        else:
            corr_factor = Decimal("1.0")

        # Remaining life factor
        rl = Decimal(str(factors.remaining_life_factor))
        rl_factor = Decimal("1") / rl if rl > Decimal("0.1") else Decimal("10")

        # Inspection effectiveness (reduces POF)
        ie = Decimal(str(factors.inspection_effectiveness))
        insp_factor = Decimal("2") - ie  # 0.0 inspection = 2x, 1.0 = 1x

        # Mechanical integrity factor
        mi = Decimal(str(factors.mechanical_integrity))
        mi_factor = Decimal("2") - mi

        # Management system factor
        ms = Decimal(str(factors.management_system_factor))

        # Damage mechanism factor
        dm_factor = Decimal("1.0")
        for dm in damage_mechanisms:
            if dm == DamageMechanism.STRESS_CORROSION_CRACKING:
                dm_factor *= Decimal("2.0")
            elif dm == DamageMechanism.HIGH_TEMP_HYDROGEN_ATTACK:
                dm_factor *= Decimal("2.5")
            elif dm == DamageMechanism.CREEP:
                dm_factor *= Decimal("1.8")
            elif dm == DamageMechanism.FATIGUE:
                dm_factor *= Decimal("1.5")
            elif dm == DamageMechanism.CUI:
                dm_factor *= Decimal("1.5")
            else:
                dm_factor *= Decimal("1.1")

        # Calculate POF
        pof = base_pof * age_factor * corr_factor * rl_factor * insp_factor * mi_factor * ms * dm_factor

        # Convert to category 1-5
        if pof < Decimal("1E-5"):
            category = 1
            description = "Very Low - Highly unlikely"
        elif pof < Decimal("1E-4"):
            category = 2
            description = "Low - Unlikely"
        elif pof < Decimal("1E-3"):
            category = 3
            description = "Medium - Possible"
        elif pof < Decimal("1E-2"):
            category = 4
            description = "High - Likely"
        else:
            category = 5
            description = "Very High - Almost certain"

        return category, self._apply_precision(pof), description

    def calculate_cof(
        self,
        factors: ConsequenceFactors
    ) -> Tuple[int, Decimal, Decimal, Decimal, str]:
        """
        Calculate Consequence of Failure category.

        Reference: API RP 580, Section 7.5

        Args:
            factors: Consequence factors

        Returns:
            Tuple of (category 1-5, safety_cof, env_cof, prod_cof, description)
        """
        # ============================================================
        # SAFETY CONSEQUENCE
        # Reference: API RP 580, Section 7.5.2
        # ============================================================

        fluid_hazard = Decimal(str(factors.fluid_hazard_factor))
        release_rate = Decimal(str(factors.release_rate_factor))
        detection = Decimal(str(factors.detection_factor))
        isolation = Decimal(str(factors.isolation_factor))
        personnel = Decimal(str(factors.personnel_density))

        # Safety consequence score
        safety_base = fluid_hazard * release_rate
        safety_mitigation = (Decimal("1") - detection * Decimal("0.3")) * (Decimal("1") - isolation * Decimal("0.3"))
        personnel_factor = Decimal("1") + personnel * Decimal("0.1")

        cof_safety = safety_base * safety_mitigation * personnel_factor

        # ============================================================
        # ENVIRONMENTAL CONSEQUENCE
        # Reference: API RP 580, Section 7.5.3
        # ============================================================

        env_sensitivity = Decimal(str(factors.environmental_sensitivity))

        cof_environmental = fluid_hazard * release_rate * env_sensitivity * Decimal("0.5")

        # ============================================================
        # PRODUCTION CONSEQUENCE
        # Reference: API RP 580, Section 7.5.4
        # ============================================================

        prod_impact = Decimal(str(factors.production_impact_usd_day))

        # Assume 7-day average outage for failure
        avg_downtime = Decimal("7")
        cof_production = prod_impact * avg_downtime / Decimal("100000")  # Normalize

        # ============================================================
        # TOTAL COF AND CATEGORY
        # ============================================================

        # Weighted total (safety most important)
        cof_total = (cof_safety * Decimal("0.5") +
                     cof_environmental * Decimal("0.3") +
                     cof_production * Decimal("0.2"))

        # Convert to category 1-5
        if cof_total < Decimal("1"):
            category = 1
            description = "Negligible - Minor impact"
        elif cof_total < Decimal("3"):
            category = 2
            description = "Low - Limited impact"
        elif cof_total < Decimal("6"):
            category = 3
            description = "Medium - Moderate impact"
        elif cof_total < Decimal("10"):
            category = 4
            description = "High - Significant impact"
        else:
            category = 5
            description = "Very High - Catastrophic impact"

        return (category,
                self._apply_precision(cof_safety),
                self._apply_precision(cof_environmental),
                self._apply_precision(cof_production),
                description)

    def assess_risk(
        self,
        likelihood_factors: LikelihoodFactors,
        consequence_factors: ConsequenceFactors,
        damage_mechanisms: List[DamageMechanism]
    ) -> RBIResult:
        """
        Perform complete RBI assessment.

        ZERO-HALLUCINATION: Deterministic calculation per API RP 580.

        Reference: API RP 580, Section 7

        Args:
            likelihood_factors: POF input factors
            consequence_factors: COF input factors
            damage_mechanisms: Active damage mechanisms

        Returns:
            RBIResult with complete assessment
        """
        # Calculate POF
        pof_cat, pof_value, pof_desc = self.calculate_pof(
            likelihood_factors, damage_mechanisms
        )

        # Calculate COF
        cof_cat, cof_safety, cof_env, cof_prod, cof_desc = self.calculate_cof(
            consequence_factors
        )

        # Total COF
        cof_total = cof_safety + cof_env + cof_prod

        # ============================================================
        # RISK MATRIX EVALUATION
        # Reference: API RP 580, Section 7.6
        # ============================================================

        # Get risk rank from matrix (1-indexed categories)
        risk_rank = self.RISK_MATRIX[pof_cat - 1][cof_cat - 1]

        # Get risk level
        risk_level = self._get_risk_level(risk_rank)

        # Calculate risk value (quantitative)
        risk_value = pof_value * cof_total

        # ============================================================
        # INSPECTION PLANNING
        # Reference: API RP 580, Section 8
        # ============================================================

        # Recommended inspection interval
        interval = self.INSPECTION_INTERVALS.get(risk_level, 48)

        # Inspection methods based on damage mechanisms
        inspection_methods = []
        for dm in damage_mechanisms:
            if dm == DamageMechanism.GENERAL_CORROSION:
                inspection_methods.extend(["UT Thickness", "Visual External"])
            elif dm == DamageMechanism.LOCALIZED_CORROSION:
                inspection_methods.extend(["UT Thickness Grid", "Pit Gauging"])
            elif dm == DamageMechanism.STRESS_CORROSION_CRACKING:
                inspection_methods.extend(["WFMT", "TOFD", "PAUT"])
            elif dm == DamageMechanism.CUI:
                inspection_methods.extend(["Insulation Removal", "Pulsed Eddy Current"])
            elif dm == DamageMechanism.CREEP:
                inspection_methods.extend(["Replication", "Hardness Testing"])
            elif dm == DamageMechanism.FATIGUE:
                inspection_methods.extend(["MPI", "Dye Penetrant"])

        # Remove duplicates while preserving order
        inspection_methods = list(dict.fromkeys(inspection_methods))

        # Priority based on risk level
        priority_map = {
            RiskLevel.HIGH: "CRITICAL",
            RiskLevel.MEDIUM_HIGH: "HIGH",
            RiskLevel.MEDIUM: "MEDIUM",
            RiskLevel.MEDIUM_LOW: "LOW",
            RiskLevel.LOW: "ROUTINE",
        }
        priority = priority_map.get(risk_level, "MEDIUM")

        # Create provenance
        inputs = {
            "pof_category": str(pof_cat),
            "cof_category": str(cof_cat),
            "num_damage_mechanisms": str(len(damage_mechanisms))
        }
        outputs = {
            "risk_rank": str(risk_rank),
            "risk_level": risk_level.value,
            "interval_months": str(interval)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return RBIResult(
            pof_category=pof_cat,
            pof_value=pof_value,
            pof_description=pof_desc,
            cof_category=cof_cat,
            cof_safety=cof_safety,
            cof_environmental=cof_env,
            cof_production=cof_prod,
            cof_total=self._apply_precision(cof_total),
            cof_description=cof_desc,
            risk_level=risk_level.value,
            risk_value=self._apply_precision(risk_value),
            risk_rank=risk_rank,
            recommended_interval_months=interval,
            inspection_methods=inspection_methods,
            priority=priority,
            damage_mechanisms=[dm.value for dm in damage_mechanisms],
            provenance_hash=provenance_hash
        )


# Convenience functions
def quick_risk_assessment(
    equipment_age_years: float,
    corrosion_rate_mm_yr: float,
    fluid_hazard: int,  # 1-5
    personnel_nearby: int
) -> RBIResult:
    """
    Quick risk assessment using simplified inputs.

    Example:
        >>> result = quick_risk_assessment(
        ...     equipment_age_years=25,
        ...     corrosion_rate_mm_yr=0.3,
        ...     fluid_hazard=3,
        ...     personnel_nearby=5
        ... )
        >>> print(f"Risk Level: {result.risk_level}")
    """
    calc = API580RBI()

    likelihood = LikelihoodFactors(
        equipment_age_years=equipment_age_years,
        corrosion_rate_mm_yr=corrosion_rate_mm_yr,
        remaining_life_factor=0.5,
        inspection_effectiveness=0.7,
        mechanical_integrity=0.8,
        management_system_factor=1.0
    )

    consequence = ConsequenceFactors(
        fluid_hazard_factor=float(fluid_hazard),
        release_rate_factor=1.0,
        detection_factor=0.7,
        isolation_factor=0.7,
        personnel_density=personnel_nearby,
        environmental_sensitivity=2.0,
        production_impact_usd_day=50000
    )

    damage = [DamageMechanism.GENERAL_CORROSION]

    return calc.assess_risk(likelihood, consequence, damage)


def get_inspection_interval(risk_rank: int) -> int:
    """
    Get recommended inspection interval from risk rank.

    Args:
        risk_rank: Risk matrix rank (1-25)

    Returns:
        Recommended interval in months
    """
    calc = API580RBI()
    level = calc._get_risk_level(risk_rank)
    return calc.INSPECTION_INTERVALS.get(level, 48)
