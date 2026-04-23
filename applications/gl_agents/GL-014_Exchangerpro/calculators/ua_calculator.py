"""
GL-014 EXCHANGERPRO - UA Calculator

Deterministic calculation of Overall Heat Transfer Coefficient times Area (UA)
using both LMTD and NTU methods. Includes fouling resistance tracking and
UA degradation analysis for performance monitoring.

Fundamental Equations:
    From LMTD:  UA = Q / (F * LMTD)
    From NTU:   UA = NTU * C_min

    With fouling:
        1/U = 1/U_clean + R_f_hot + R_f_cold
        U = U_clean / (1 + U_clean * (R_f_hot + R_f_cold))

TEMA Compliance:
    - Fouling resistances per TEMA Standards Table RGP-T-2.4
    - UA tracking for exchanger performance monitoring
    - Cleanliness factor calculation

Reference:
    - TEMA Standards, 10th Edition
    - ASME PTC 12.5-2000
    - Heat Exchanger Design Handbook (HEDH)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math


# =============================================================================
# Constants
# =============================================================================

# Minimum LMTD for calculation validity (K)
MIN_LMTD = 0.1

# Minimum UA value (W/K)
MIN_UA = 1.0

# Warning threshold for cleanliness factor
CLEANLINESS_WARNING_THRESHOLD = 0.75


# =============================================================================
# Enums
# =============================================================================

class UACalculationMethod(str, Enum):
    """Method used for UA calculation."""
    FROM_LMTD = "from_lmtd"          # UA = Q / (F * LMTD)
    FROM_NTU = "from_ntu"            # UA = NTU * C_min
    FROM_RESISTANCES = "from_resistances"  # 1/UA = sum of resistances


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FoulingResistance:
    """
    Fouling resistance data per TEMA standards.

    Units: m^2*K/W

    Reference values from TEMA Table RGP-T-2.4:
    - Clean water (< 50C): 0.0001
    - Cooling tower water: 0.0002-0.0003
    - Fuel oil: 0.0009
    - Heavy crude: 0.0035

    The fouling resistance (Rf) represents the additional thermal
    resistance due to deposit buildup on heat transfer surfaces.
    """
    Rf_hot_m2K_W: float = 0.0       # Hot side fouling resistance
    Rf_cold_m2K_W: float = 0.0      # Cold side fouling resistance
    description_hot: str = ""       # Description of hot-side fluid/fouling
    description_cold: str = ""      # Description of cold-side fluid/fouling
    measurement_date: Optional[datetime] = None

    @property
    def Rf_total_m2K_W(self) -> float:
        """Total fouling resistance."""
        return self.Rf_hot_m2K_W + self.Rf_cold_m2K_W

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "Rf_hot_m2K_W": self.Rf_hot_m2K_W,
            "Rf_cold_m2K_W": self.Rf_cold_m2K_W,
            "Rf_total_m2K_W": self.Rf_total_m2K_W,
            "description_hot": self.description_hot,
            "description_cold": self.description_cold,
            "measurement_date": self.measurement_date.isoformat() if self.measurement_date else None,
        }


@dataclass
class UAInputs:
    """
    Inputs for UA calculation.

    Provide either:
    - Q_W and LMTD_K (and F_factor) for LMTD method
    - NTU and C_min_W_K for NTU method
    - Individual resistances for resistance method
    """
    # For LMTD method
    Q_W: Optional[float] = None              # Heat duty [W]
    LMTD_K: Optional[float] = None           # Log mean temperature difference [K]
    F_factor: float = 1.0                    # LMTD correction factor

    # For NTU method
    NTU: Optional[float] = None              # Number of transfer units
    C_min_W_K: Optional[float] = None        # Minimum heat capacity rate [W/K]

    # For resistance method (all in m^2*K/W, assuming area A_m2 provided)
    R_hot_film_m2K_W: Optional[float] = None     # Hot-side film resistance
    R_cold_film_m2K_W: Optional[float] = None    # Cold-side film resistance
    R_wall_m2K_W: Optional[float] = None         # Tube wall resistance
    A_m2: Optional[float] = None                 # Heat transfer area [m^2]

    # Fouling data
    fouling: Optional[FoulingResistance] = None

    # Reference clean UA for degradation tracking
    UA_clean_W_K: Optional[float] = None

    # Design UA for comparison
    UA_design_W_K: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing."""
        return {
            "Q_W": self.Q_W,
            "LMTD_K": self.LMTD_K,
            "F_factor": self.F_factor,
            "NTU": self.NTU,
            "C_min_W_K": self.C_min_W_K,
            "R_hot_film_m2K_W": self.R_hot_film_m2K_W,
            "R_cold_film_m2K_W": self.R_cold_film_m2K_W,
            "R_wall_m2K_W": self.R_wall_m2K_W,
            "A_m2": self.A_m2,
            "fouling": self.fouling.to_dict() if self.fouling else None,
            "UA_clean_W_K": self.UA_clean_W_K,
            "UA_design_W_K": self.UA_design_W_K,
        }


@dataclass
class UADegradationAnalysis:
    """
    UA degradation analysis for fouling monitoring.

    Tracks how actual UA compares to clean and design values.
    """
    UA_current_W_K: float              # Current operating UA
    UA_clean_W_K: Optional[float]      # Clean (unfouled) UA
    UA_design_W_K: Optional[float]     # Design UA

    # Derived metrics
    cleanliness_factor: Optional[float]     # UA_current / UA_clean (0 to 1)
    design_margin: Optional[float]          # UA_design / UA_current (> 1 means overdesign)
    degradation_percent: Optional[float]    # (1 - cleanliness) * 100

    # Fouling proxy
    Rf_apparent_m2K_W: Optional[float]      # Apparent fouling resistance

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "UA_current_W_K": round(self.UA_current_W_K, 3),
            "UA_clean_W_K": round(self.UA_clean_W_K, 3) if self.UA_clean_W_K else None,
            "UA_design_W_K": round(self.UA_design_W_K, 3) if self.UA_design_W_K else None,
            "cleanliness_factor": round(self.cleanliness_factor, 4) if self.cleanliness_factor else None,
            "design_margin": round(self.design_margin, 4) if self.design_margin else None,
            "degradation_percent": round(self.degradation_percent, 2) if self.degradation_percent else None,
            "Rf_apparent_m2K_W": round(self.Rf_apparent_m2K_W, 8) if self.Rf_apparent_m2K_W else None,
        }


@dataclass
class UAResult:
    """
    Complete UA calculation result.
    """
    # Core result
    UA_W_K: float                      # Overall heat transfer coefficient * Area [W/K]
    UA_kW_K: float                     # Same in kW/K

    # Calculation method used
    calculation_method: UACalculationMethod

    # Input values used
    Q_W: Optional[float]
    LMTD_K: Optional[float]
    F_factor: float
    NTU: Optional[float]
    C_min_W_K: Optional[float]

    # Derived values
    effective_LMTD_K: Optional[float]  # F * LMTD

    # If area is known
    U_W_m2K: Optional[float] = None    # Overall coefficient [W/(m^2*K)]
    A_m2: Optional[float] = None       # Area [m^2]

    # Fouling analysis
    fouling_applied: bool = False
    fouling: Optional[FoulingResistance] = None

    # Degradation analysis
    degradation: Optional[UADegradationAnalysis] = None

    # Validation
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    # Calculation trace
    calculation_steps: List[str] = field(default_factory=list)

    # Provenance
    inputs_hash: str = ""
    outputs_hash: str = ""
    computation_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    calculator_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "UA_W_K": round(self.UA_W_K, 3),
            "UA_kW_K": round(self.UA_kW_K, 6),
            "calculation_method": self.calculation_method.value,
            "Q_W": round(self.Q_W, 3) if self.Q_W else None,
            "LMTD_K": round(self.LMTD_K, 4) if self.LMTD_K else None,
            "F_factor": round(self.F_factor, 4),
            "effective_LMTD_K": round(self.effective_LMTD_K, 4) if self.effective_LMTD_K else None,
            "NTU": round(self.NTU, 6) if self.NTU else None,
            "C_min_W_K": round(self.C_min_W_K, 3) if self.C_min_W_K else None,
            "U_W_m2K": round(self.U_W_m2K, 3) if self.U_W_m2K else None,
            "A_m2": round(self.A_m2, 3) if self.A_m2 else None,
            "fouling_applied": self.fouling_applied,
            "fouling": self.fouling.to_dict() if self.fouling else None,
            "degradation": self.degradation.to_dict() if self.degradation else None,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "calculation_steps": self.calculation_steps,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "computation_hash": self.computation_hash,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": round(self.execution_time_ms, 3),
            "calculator_version": self.calculator_version,
        }


# =============================================================================
# UA Calculator
# =============================================================================

class UACalculator:
    """
    Deterministic UA (Overall Heat Transfer Coefficient * Area) Calculator.

    Supports three calculation methods:
    1. LMTD Method:  UA = Q / (F * LMTD)
    2. NTU Method:   UA = NTU * C_min
    3. Resistance Method: 1/UA = (R_hot + R_cold + R_wall + R_f) / A

    Features:
    - Automatic method selection based on available inputs
    - Fouling resistance incorporation
    - UA degradation tracking for performance monitoring
    - Cleanliness factor calculation
    - Apparent fouling resistance proxy (Rf)

    Zero-Hallucination Guarantee:
        All calculations are deterministic. Same inputs produce
        bit-perfect identical outputs. No LLM involvement.

    Example (LMTD Method):
        >>> calc = UACalculator()
        >>> inputs = UAInputs(
        ...     Q_W=500000,
        ...     LMTD_K=45.0,
        ...     F_factor=0.95
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"UA = {result.UA_kW_K:.2f} kW/K")

    Example (NTU Method):
        >>> inputs = UAInputs(
        ...     NTU=2.5,
        ...     C_min_W_K=8360
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"UA = {result.UA_W_K:.0f} W/K")

    Example (With Fouling Tracking):
        >>> fouling = FoulingResistance(Rf_hot_m2K_W=0.0002, Rf_cold_m2K_W=0.0001)
        >>> inputs = UAInputs(
        ...     Q_W=500000,
        ...     LMTD_K=45.0,
        ...     fouling=fouling,
        ...     UA_clean_W_K=12000  # Reference clean UA
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"Cleanliness: {result.degradation.cleanliness_factor:.2%}")
    """

    NAME = "UACalculator"
    VERSION = "1.0.0"
    AGENT_ID = "GL-014"

    def __init__(
        self,
        min_lmtd: float = MIN_LMTD,
        min_ua: float = MIN_UA,
    ):
        """
        Initialize UA Calculator.

        Args:
            min_lmtd: Minimum LMTD for valid calculation [K]
            min_ua: Minimum UA value [W/K]
        """
        self.min_lmtd = min_lmtd
        self.min_ua = min_ua

    def calculate(self, inputs: UAInputs) -> UAResult:
        """
        Calculate UA using the most appropriate method.

        Automatically selects method based on available inputs:
        1. If Q and LMTD provided: LMTD method
        2. If NTU and C_min provided: NTU method
        3. If resistances and area provided: Resistance method

        Args:
            inputs: UA calculation inputs

        Returns:
            UAResult with UA and provenance
        """
        start_time = datetime.now(timezone.utc)
        warnings: List[str] = []
        calculation_steps: List[str] = []

        # Determine calculation method
        method = self._determine_method(inputs)
        calculation_steps.append(f"Calculation method: {method.value}")

        # Validate inputs for selected method
        validation_errors = self._validate_inputs(inputs, method)
        if validation_errors:
            warnings.extend(validation_errors)

        # Calculate UA
        UA_W_K = 0.0
        Q_W = inputs.Q_W
        LMTD_K = inputs.LMTD_K
        effective_LMTD_K = None
        NTU = inputs.NTU
        C_min_W_K = inputs.C_min_W_K

        if method == UACalculationMethod.FROM_LMTD:
            # UA = Q / (F * LMTD)
            effective_LMTD_K = inputs.F_factor * inputs.LMTD_K
            calculation_steps.append(
                f"Effective LMTD = F * LMTD = {inputs.F_factor:.4f} * {inputs.LMTD_K:.2f} = {effective_LMTD_K:.2f} K"
            )

            if effective_LMTD_K >= self.min_lmtd:
                UA_W_K = inputs.Q_W / effective_LMTD_K
                calculation_steps.append(
                    f"UA = Q / (F*LMTD) = {inputs.Q_W:.0f} / {effective_LMTD_K:.2f} = {UA_W_K:.2f} W/K"
                )
            else:
                warnings.append(f"Effective LMTD ({effective_LMTD_K:.3f} K) below minimum ({self.min_lmtd} K)")
                UA_W_K = 0.0

        elif method == UACalculationMethod.FROM_NTU:
            # UA = NTU * C_min
            UA_W_K = inputs.NTU * inputs.C_min_W_K
            calculation_steps.append(
                f"UA = NTU * C_min = {inputs.NTU:.4f} * {inputs.C_min_W_K:.2f} = {UA_W_K:.2f} W/K"
            )

        elif method == UACalculationMethod.FROM_RESISTANCES:
            # 1/UA = (R_hot + R_cold + R_wall + Rf) / A
            # UA = A / (R_hot + R_cold + R_wall + Rf)
            R_total = (
                (inputs.R_hot_film_m2K_W or 0.0) +
                (inputs.R_cold_film_m2K_W or 0.0) +
                (inputs.R_wall_m2K_W or 0.0)
            )

            if inputs.fouling:
                R_total += inputs.fouling.Rf_total_m2K_W

            calculation_steps.append(f"Total thermal resistance = {R_total:.6f} m^2*K/W")

            if R_total > 0 and inputs.A_m2 > 0:
                UA_W_K = inputs.A_m2 / R_total
                calculation_steps.append(
                    f"UA = A / R_total = {inputs.A_m2:.2f} / {R_total:.6f} = {UA_W_K:.2f} W/K"
                )
            else:
                warnings.append("Invalid resistance or area values")
                UA_W_K = 0.0

        else:
            warnings.append("Could not determine calculation method from inputs")

        # Calculate U if area is known
        U_W_m2K = None
        A_m2 = inputs.A_m2
        if A_m2 is not None and A_m2 > 0 and UA_W_K > 0:
            U_W_m2K = UA_W_K / A_m2
            calculation_steps.append(f"U = UA / A = {UA_W_K:.2f} / {A_m2:.2f} = {U_W_m2K:.2f} W/(m^2*K)")

        # Degradation analysis
        degradation = None
        if inputs.UA_clean_W_K is not None or inputs.UA_design_W_K is not None:
            degradation = self._analyze_degradation(
                UA_current=UA_W_K,
                UA_clean=inputs.UA_clean_W_K,
                UA_design=inputs.UA_design_W_K,
                A_m2=A_m2,
            )

            if degradation.cleanliness_factor is not None:
                calculation_steps.append(f"Cleanliness factor = {degradation.cleanliness_factor:.4f}")

                if degradation.cleanliness_factor < CLEANLINESS_WARNING_THRESHOLD:
                    warnings.append(
                        f"Cleanliness factor ({degradation.cleanliness_factor:.2%}) is below "
                        f"threshold ({CLEANLINESS_WARNING_THRESHOLD:.0%}). Consider cleaning."
                    )

        # Compute provenance
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        inputs_hash = self._compute_hash(inputs.to_dict())

        result_data = {
            "UA_W_K": UA_W_K,
            "method": method.value,
        }
        outputs_hash = self._compute_hash(result_data)

        computation_hash = self._compute_hash({
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "calculator": self.NAME,
            "version": self.VERSION,
        })

        return UAResult(
            UA_W_K=UA_W_K,
            UA_kW_K=UA_W_K / 1000.0,
            calculation_method=method,
            Q_W=Q_W,
            LMTD_K=LMTD_K,
            F_factor=inputs.F_factor,
            effective_LMTD_K=effective_LMTD_K,
            NTU=NTU,
            C_min_W_K=C_min_W_K,
            U_W_m2K=U_W_m2K,
            A_m2=A_m2,
            fouling_applied=inputs.fouling is not None,
            fouling=inputs.fouling,
            degradation=degradation,
            is_valid=len(validation_errors) == 0 and UA_W_K >= self.min_ua,
            warnings=warnings,
            calculation_steps=calculation_steps,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            computation_hash=computation_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            calculator_version=self.VERSION,
        )

    def calculate_from_lmtd(
        self,
        Q_W: float,
        LMTD_K: float,
        F_factor: float = 1.0,
    ) -> float:
        """
        Quick UA calculation from LMTD (no provenance).

        Args:
            Q_W: Heat duty [W]
            LMTD_K: Log mean temperature difference [K]
            F_factor: LMTD correction factor

        Returns:
            UA [W/K]
        """
        effective_LMTD = F_factor * LMTD_K
        if effective_LMTD < self.min_lmtd:
            return 0.0
        return Q_W / effective_LMTD

    def calculate_from_ntu(
        self,
        NTU: float,
        C_min_W_K: float,
    ) -> float:
        """
        Quick UA calculation from NTU (no provenance).

        Args:
            NTU: Number of transfer units
            C_min_W_K: Minimum heat capacity rate [W/K]

        Returns:
            UA [W/K]
        """
        return NTU * C_min_W_K

    def estimate_fouling_resistance(
        self,
        UA_current_W_K: float,
        UA_clean_W_K: float,
        A_m2: float,
    ) -> float:
        """
        Estimate apparent fouling resistance from UA degradation.

        Rf_apparent = A * (1/UA_current - 1/UA_clean)

        Args:
            UA_current_W_K: Current operating UA [W/K]
            UA_clean_W_K: Clean (unfouled) UA [W/K]
            A_m2: Heat transfer area [m^2]

        Returns:
            Apparent fouling resistance [m^2*K/W]
        """
        if UA_current_W_K <= 0 or UA_clean_W_K <= 0 or A_m2 <= 0:
            return 0.0

        return A_m2 * (1.0 / UA_current_W_K - 1.0 / UA_clean_W_K)

    def _determine_method(self, inputs: UAInputs) -> UACalculationMethod:
        """Determine calculation method from available inputs."""
        # Priority: LMTD > NTU > Resistances

        if inputs.Q_W is not None and inputs.LMTD_K is not None:
            return UACalculationMethod.FROM_LMTD

        if inputs.NTU is not None and inputs.C_min_W_K is not None:
            return UACalculationMethod.FROM_NTU

        if (inputs.R_hot_film_m2K_W is not None and
            inputs.R_cold_film_m2K_W is not None and
            inputs.A_m2 is not None):
            return UACalculationMethod.FROM_RESISTANCES

        # Default to LMTD if partially available
        return UACalculationMethod.FROM_LMTD

    def _validate_inputs(
        self,
        inputs: UAInputs,
        method: UACalculationMethod,
    ) -> List[str]:
        """Validate inputs for selected method."""
        errors: List[str] = []

        if method == UACalculationMethod.FROM_LMTD:
            if inputs.Q_W is None or inputs.Q_W <= 0:
                errors.append(f"Heat duty Q must be positive: {inputs.Q_W}")
            if inputs.LMTD_K is None or inputs.LMTD_K <= 0:
                errors.append(f"LMTD must be positive: {inputs.LMTD_K}")
            if inputs.F_factor <= 0 or inputs.F_factor > 1:
                errors.append(f"F-factor must be in (0, 1]: {inputs.F_factor}")

        elif method == UACalculationMethod.FROM_NTU:
            if inputs.NTU is None or inputs.NTU <= 0:
                errors.append(f"NTU must be positive: {inputs.NTU}")
            if inputs.C_min_W_K is None or inputs.C_min_W_K <= 0:
                errors.append(f"C_min must be positive: {inputs.C_min_W_K}")

        elif method == UACalculationMethod.FROM_RESISTANCES:
            if inputs.A_m2 is None or inputs.A_m2 <= 0:
                errors.append(f"Area must be positive: {inputs.A_m2}")

        return errors

    def _analyze_degradation(
        self,
        UA_current: float,
        UA_clean: Optional[float],
        UA_design: Optional[float],
        A_m2: Optional[float],
    ) -> UADegradationAnalysis:
        """Analyze UA degradation for performance monitoring."""
        cleanliness_factor = None
        design_margin = None
        degradation_percent = None
        Rf_apparent = None

        if UA_clean is not None and UA_clean > 0 and UA_current > 0:
            cleanliness_factor = UA_current / UA_clean
            degradation_percent = (1.0 - cleanliness_factor) * 100.0

            # Estimate apparent fouling resistance
            if A_m2 is not None and A_m2 > 0:
                Rf_apparent = A_m2 * (1.0 / UA_current - 1.0 / UA_clean)

        if UA_design is not None and UA_design > 0 and UA_current > 0:
            design_margin = UA_design / UA_current

        return UADegradationAnalysis(
            UA_current_W_K=UA_current,
            UA_clean_W_K=UA_clean,
            UA_design_W_K=UA_design,
            cleanliness_factor=cleanliness_factor,
            design_margin=design_margin,
            degradation_percent=degradation_percent,
            Rf_apparent_m2K_W=Rf_apparent,
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        normalized = self._normalize_for_hash(data)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _normalize_for_hash(self, obj: Any) -> Any:
        """Normalize for consistent hashing."""
        if obj is None:
            return None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            return round(obj, 10)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._normalize_for_hash(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._normalize_for_hash(v) for k, v in sorted(obj.items())}
        else:
            return str(obj)


# =============================================================================
# TEMA Fouling Resistance Tables
# =============================================================================

# Standard fouling resistances per TEMA Table RGP-T-2.4 (m^2*K/W)
TEMA_FOULING_RESISTANCES = {
    # Water - low temperature (< 50C)
    "water_distilled": 0.00009,
    "water_treated_boiler_feedwater": 0.00009,
    "water_treated_cooling_tower_makeup": 0.00018,
    "water_river": 0.00035,
    "water_muddy_or_silty": 0.00053,
    "water_seawater": 0.00018,
    "water_brackish": 0.00035,

    # Water - high temperature (> 50C)
    "water_treated_boiler_feedwater_ht": 0.00018,
    "water_treated_cooling_tower_ht": 0.00035,
    "water_river_ht": 0.00053,

    # Steam
    "steam_clean": 0.00009,
    "steam_exhaust_oil_bearing": 0.00027,

    # Liquids
    "refrigerant": 0.00018,
    "hydraulic_fluid": 0.00018,
    "transformer_oil": 0.00018,
    "lubricating_oil": 0.00018,
    "fuel_oil_light": 0.00035,
    "fuel_oil_heavy": 0.00053,
    "crude_oil_dry": 0.00053,
    "crude_oil_with_salts": 0.00088,
    "asphalt_bitumen": 0.00176,

    # Gases
    "air_clean": 0.00018,
    "air_industrial": 0.00035,
    "natural_gas": 0.00018,
    "flue_gas_clean": 0.00018,
    "flue_gas_with_soot": 0.00088,
}


def get_tema_fouling_resistance(fluid_type: str) -> float:
    """
    Get TEMA standard fouling resistance for a fluid type.

    Args:
        fluid_type: Key from TEMA_FOULING_RESISTANCES

    Returns:
        Fouling resistance in m^2*K/W (0 if not found)
    """
    return TEMA_FOULING_RESISTANCES.get(fluid_type.lower().replace(" ", "_"), 0.0)
