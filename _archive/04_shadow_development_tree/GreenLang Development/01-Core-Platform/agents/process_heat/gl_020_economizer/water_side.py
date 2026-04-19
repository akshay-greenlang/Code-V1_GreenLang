"""
GL-020 ECONOPULSE - Water-Side Fouling Analyzer

Analyzes water-side scaling and fouling in economizers through:
- Pressure drop analysis
- Heat transfer degradation
- Water chemistry compliance
- Scale composition estimation

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - ASME Guidelines for Feedwater Chemistry

Zero-Hallucination: All calculations use deterministic formulas with full provenance.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Thermal conductivity of common scale deposits (BTU/hr-ft-F)
SCALE_THERMAL_CONDUCTIVITIES = {
    "calcium_carbonate": 1.5,  # CaCO3
    "calcium_sulfate": 1.0,   # CaSO4
    "silica": 0.8,            # SiO2
    "iron_oxide": 2.5,        # Fe2O3/Fe3O4
    "copper_oxide": 10.0,     # CuO
    "mixed": 1.2,             # Mixed deposit
}

# Default thermal conductivity for unknown scale
DEFAULT_SCALE_CONDUCTIVITY = 1.2

# Water chemistry limits per ASME guidelines (high-pressure boilers)
CHEMISTRY_LIMITS_HP = {
    "hardness_ppm": 0.5,      # as CaCO3
    "silica_ppm": 0.02,
    "iron_ppm": 0.01,
    "copper_ppm": 0.005,
    "oxygen_ppb": 7.0,
    "ph_min": 9.0,
    "ph_max": 9.5,
}

# Water chemistry limits for medium-pressure boilers
CHEMISTRY_LIMITS_MP = {
    "hardness_ppm": 2.0,
    "silica_ppm": 0.1,
    "iron_ppm": 0.05,
    "copper_ppm": 0.01,
    "oxygen_ppb": 10.0,
    "ph_min": 8.8,
    "ph_max": 9.6,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WaterChemistryData:
    """Water chemistry measurement data."""
    ph: Optional[float] = None
    hardness_ppm: Optional[float] = None  # as CaCO3
    silica_ppm: Optional[float] = None
    iron_ppm: Optional[float] = None
    copper_ppm: Optional[float] = None
    oxygen_ppb: Optional[float] = None
    conductivity_umhos: Optional[float] = None
    total_dissolved_solids_ppm: Optional[float] = None
    alkalinity_ppm: Optional[float] = None  # as CaCO3
    phosphate_ppm: Optional[float] = None


@dataclass
class WaterSideFoulingInput:
    """Input data for water-side fouling analysis."""

    # Current pressure drop
    current_dp_psi: float
    design_dp_psi: float

    # Flow conditions
    current_water_flow_lb_hr: float
    design_water_flow_lb_hr: float

    # Temperatures
    water_inlet_temp_f: float
    water_outlet_temp_f: float

    # Design fouling factor
    design_fouling_factor: float = 0.001  # hr-ft2-F/BTU

    # Heat transfer values
    current_ua_btu_hr_f: Optional[float] = None
    design_ua_btu_hr_f: Optional[float] = None

    # Water chemistry
    chemistry: Optional[WaterChemistryData] = None

    # Pressure class for chemistry limits
    pressure_class: str = "high"  # high, medium, low

    # Historical data for trending
    dp_history: List[Tuple[datetime, float]] = field(default_factory=list)
    chemistry_history: List[Tuple[datetime, WaterChemistryData]] = field(
        default_factory=list
    )

    # Thresholds
    dp_warning_ratio: float = 1.2
    dp_alarm_ratio: float = 1.4


@dataclass
class WaterSideFoulingResult:
    """Result of water-side fouling analysis."""

    # Fouling detection
    fouling_detected: bool
    fouling_severity: str  # none, light, moderate, severe, critical
    fouling_type: str  # scale, deposit, corrosion, none

    # Pressure drop analysis
    current_dp_psi: float
    design_dp_psi: float
    corrected_dp_psi: float
    dp_ratio: float

    # Fouling factor
    fouling_factor_hr_ft2_f_btu: float
    design_fouling_factor: float
    fouling_factor_ratio: float

    # Scale estimate
    estimated_scale_thickness_mils: Optional[float]
    scale_composition: Optional[str]

    # Chemistry compliance
    chemistry_compliant: bool
    chemistry_deviations: List[str]

    # Recommendations
    cleaning_status: str
    recommended_cleaning_method: Optional[str]

    # Provenance
    calculation_method: str
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# WATER-SIDE FOULING ANALYZER
# =============================================================================

class WaterSideFoulingAnalyzer:
    """
    Analyzes water-side scaling and fouling in economizers.

    Water-side fouling mechanisms:
    1. Scale formation (calcium, silica deposits)
    2. Corrosion product deposition (iron/copper oxides)
    3. Biological fouling (rare in high-temp economizers)

    Detection methods:
    - Pressure drop increase (tube blockage)
    - Heat transfer coefficient degradation
    - Water chemistry analysis

    Scale prevention relies on proper water treatment:
    - pH control (9.0-9.5 for high-pressure systems)
    - Hardness removal (softening, dealkalization)
    - Oxygen removal (deaeration, chemical scavenging)

    Reference: ASME Consensus Operating Guidelines
    """

    def __init__(
        self,
        scale_conductivity: float = DEFAULT_SCALE_CONDUCTIVITY,
    ):
        """
        Initialize water-side fouling analyzer.

        Args:
            scale_conductivity: Default scale thermal conductivity (BTU/hr-ft-F)
        """
        self.scale_conductivity = scale_conductivity
        logger.info("WaterSideFoulingAnalyzer initialized")

    def analyze_pressure_drop(
        self,
        current_dp: float,
        design_dp: float,
        flow_ratio: float,
    ) -> Tuple[float, float]:
        """
        Analyze water-side fouling from pressure drop.

        Water-side DP follows Darcy-Weisbach:
        DP = f * (L/D) * (rho * V^2 / 2)

        For turbulent flow in tubes:
        DP is proportional to flow^1.8 to flow^2.0

        Using n=1.85 as typical exponent:
        DP_corrected = DP_actual / (flow_ratio)^1.85

        Args:
            current_dp: Current measured pressure drop (psi)
            design_dp: Design pressure drop at design flow (psi)
            flow_ratio: Current flow / design flow

        Returns:
            Tuple of (corrected_dp, dp_ratio)
        """
        # Ensure valid flow ratio
        if flow_ratio <= 0:
            flow_ratio = 0.01

        # Correct DP to design flow conditions
        # Using n=1.85 exponent for turbulent pipe flow
        flow_exponent = 1.85
        corrected_dp = current_dp / (flow_ratio ** flow_exponent)

        # Calculate fouling ratio
        if design_dp <= 0:
            design_dp = 0.1

        dp_ratio = corrected_dp / design_dp

        logger.debug(
            f"Water DP analysis: actual={current_dp:.2f} psi, "
            f"corrected={corrected_dp:.2f} psi, ratio={dp_ratio:.3f}"
        )

        return corrected_dp, dp_ratio

    def calculate_fouling_factor(
        self,
        current_ua: float,
        design_ua: float,
        heat_transfer_area_ft2: float,
    ) -> float:
        """
        Calculate water-side fouling factor from UA degradation.

        The fouling factor (Rf) relates to UA degradation:
        1/UA_actual = 1/UA_clean + Rf/A

        Therefore:
        Rf = A * (1/UA_actual - 1/UA_design)

        For water-side specific fouling:
        Assuming gas-side is clean, all degradation is water-side

        Args:
            current_ua: Current UA value (BTU/hr-F)
            design_ua: Design UA value (BTU/hr-F)
            heat_transfer_area_ft2: Heat transfer area (ft2)

        Returns:
            Fouling factor (hr-ft2-F/BTU)
        """
        if current_ua <= 0 or design_ua <= 0:
            return 0.0

        # Calculate fouling resistance per unit area
        fouling_factor = heat_transfer_area_ft2 * (1.0 / current_ua - 1.0 / design_ua)

        # Ensure non-negative
        return max(0.0, fouling_factor)

    def estimate_scale_thickness(
        self,
        fouling_factor: float,
        scale_type: str = "mixed",
    ) -> float:
        """
        Estimate scale thickness from fouling factor.

        For a scale layer:
        Rf = thickness / k_scale

        Therefore:
        thickness = Rf * k_scale

        Args:
            fouling_factor: Fouling factor (hr-ft2-F/BTU)
            scale_type: Type of scale deposit

        Returns:
            Estimated scale thickness in mils (1 mil = 0.001 inch)
        """
        conductivity = SCALE_THERMAL_CONDUCTIVITIES.get(
            scale_type, DEFAULT_SCALE_CONDUCTIVITY
        )

        # thickness (ft) = Rf * k
        thickness_ft = fouling_factor * conductivity

        # Convert to mils (1 ft = 12000 mils)
        thickness_mils = thickness_ft * 12000

        return thickness_mils

    def analyze_water_chemistry(
        self,
        chemistry: WaterChemistryData,
        pressure_class: str = "high",
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Analyze water chemistry for scale potential.

        Checks chemistry against ASME limits and identifies:
        - Out-of-spec conditions
        - Scale/corrosion risks
        - Probable deposit composition

        Args:
            chemistry: Water chemistry data
            pressure_class: Boiler pressure class (high, medium, low)

        Returns:
            Tuple of (compliant, deviations, probable_scale_type)
        """
        # Select limits based on pressure class
        if pressure_class == "high":
            limits = CHEMISTRY_LIMITS_HP
        else:
            limits = CHEMISTRY_LIMITS_MP

        deviations = []
        scale_indicators = []

        # Check pH
        if chemistry.ph is not None:
            if chemistry.ph < limits["ph_min"]:
                deviations.append(
                    f"pH low: {chemistry.ph:.1f} < {limits['ph_min']} (corrosion risk)"
                )
            elif chemistry.ph > limits["ph_max"]:
                deviations.append(
                    f"pH high: {chemistry.ph:.1f} > {limits['ph_max']} (caustic attack risk)"
                )

        # Check hardness
        if chemistry.hardness_ppm is not None:
            if chemistry.hardness_ppm > limits["hardness_ppm"]:
                deviations.append(
                    f"Hardness high: {chemistry.hardness_ppm:.2f} > "
                    f"{limits['hardness_ppm']} ppm (scale risk)"
                )
                scale_indicators.append("calcium_carbonate")

        # Check silica
        if chemistry.silica_ppm is not None:
            if chemistry.silica_ppm > limits["silica_ppm"]:
                deviations.append(
                    f"Silica high: {chemistry.silica_ppm:.3f} > "
                    f"{limits['silica_ppm']} ppm (silica scale risk)"
                )
                scale_indicators.append("silica")

        # Check iron
        if chemistry.iron_ppm is not None:
            if chemistry.iron_ppm > limits["iron_ppm"]:
                deviations.append(
                    f"Iron high: {chemistry.iron_ppm:.3f} > "
                    f"{limits['iron_ppm']} ppm (iron oxide deposition)"
                )
                scale_indicators.append("iron_oxide")

        # Check copper
        if chemistry.copper_ppm is not None:
            if chemistry.copper_ppm > limits["copper_ppm"]:
                deviations.append(
                    f"Copper high: {chemistry.copper_ppm:.4f} > "
                    f"{limits['copper_ppm']} ppm (copper deposition)"
                )
                scale_indicators.append("copper_oxide")

        # Check oxygen
        if chemistry.oxygen_ppb is not None:
            if chemistry.oxygen_ppb > limits["oxygen_ppb"]:
                deviations.append(
                    f"Oxygen high: {chemistry.oxygen_ppb:.1f} > "
                    f"{limits['oxygen_ppb']} ppb (corrosion risk)"
                )

        # Determine compliance
        compliant = len(deviations) == 0

        # Determine probable scale type
        if scale_indicators:
            if len(scale_indicators) == 1:
                probable_scale = scale_indicators[0]
            else:
                probable_scale = "mixed"
        else:
            probable_scale = None

        return compliant, deviations, probable_scale

    def determine_fouling_severity(
        self,
        dp_ratio: float,
        fouling_factor_ratio: float,
        chemistry_deviations: int = 0,
    ) -> str:
        """
        Determine water-side fouling severity.

        Args:
            dp_ratio: Corrected DP ratio
            fouling_factor_ratio: Actual/design fouling factor ratio
            chemistry_deviations: Number of chemistry deviations

        Returns:
            Severity level: none, light, moderate, severe, critical
        """
        # Severity from DP
        if dp_ratio >= 1.6:
            dp_severity = 4  # critical
        elif dp_ratio >= 1.4:
            dp_severity = 3  # severe
        elif dp_ratio >= 1.2:
            dp_severity = 2  # moderate
        elif dp_ratio >= 1.1:
            dp_severity = 1  # light
        else:
            dp_severity = 0  # none

        # Severity from fouling factor
        if fouling_factor_ratio >= 3.0:
            ff_severity = 4
        elif fouling_factor_ratio >= 2.0:
            ff_severity = 3
        elif fouling_factor_ratio >= 1.5:
            ff_severity = 2
        elif fouling_factor_ratio >= 1.2:
            ff_severity = 1
        else:
            ff_severity = 0

        # Add weight for chemistry issues
        chemistry_severity = min(chemistry_deviations, 2)

        # Overall severity
        severity = max(dp_severity, ff_severity) + chemistry_severity // 2
        severity = min(severity, 4)

        severity_map = {0: "none", 1: "light", 2: "moderate", 3: "severe", 4: "critical"}
        return severity_map.get(severity, "none")

    def determine_fouling_type(
        self,
        dp_ratio: float,
        fouling_factor_ratio: float,
        chemistry_deviations: List[str],
    ) -> str:
        """
        Determine primary fouling type.

        Args:
            dp_ratio: DP ratio
            fouling_factor_ratio: Fouling factor ratio
            chemistry_deviations: List of chemistry deviations

        Returns:
            Fouling type: scale, deposit, corrosion, none
        """
        if dp_ratio < 1.1 and fouling_factor_ratio < 1.2:
            return "none"

        # Check for scale indicators in chemistry deviations
        scale_keywords = ["hardness", "silica", "scale"]
        corrosion_keywords = ["oxygen", "pH low", "corrosion"]
        deposit_keywords = ["iron", "copper"]

        has_scale = any(
            any(kw in dev.lower() for kw in scale_keywords)
            for dev in chemistry_deviations
        )
        has_corrosion = any(
            any(kw in dev.lower() for kw in corrosion_keywords)
            for dev in chemistry_deviations
        )
        has_deposit = any(
            any(kw in dev.lower() for kw in deposit_keywords)
            for dev in chemistry_deviations
        )

        if has_scale and not has_corrosion:
            return "scale"
        elif has_corrosion:
            return "corrosion"
        elif has_deposit:
            return "deposit"
        else:
            # Default based on indicators
            if fouling_factor_ratio > 1.5:
                return "scale"
            elif dp_ratio > 1.3:
                return "deposit"
            else:
                return "none"

    def recommend_cleaning_method(
        self,
        fouling_type: str,
        scale_composition: Optional[str],
        severity: str,
    ) -> Optional[str]:
        """
        Recommend appropriate cleaning method.

        Args:
            fouling_type: Type of fouling
            scale_composition: Probable scale composition
            severity: Fouling severity

        Returns:
            Recommended cleaning method or None
        """
        if severity in ("none", "light"):
            return None

        # Base recommendation on fouling type and composition
        if fouling_type == "scale":
            if scale_composition == "calcium_carbonate":
                return "Acid cleaning with inhibited HCl (5-10%) - safe for carbon steel"
            elif scale_composition == "silica":
                return "Alkaline cleaning with NaOH/Na2CO3 - silica requires high pH"
            elif scale_composition == "calcium_sulfate":
                return "EDTA chelant cleaning - sulfate scales are acid-resistant"
            else:
                return "Chemical cleaning consultation required - determine scale composition first"

        elif fouling_type == "deposit":
            if scale_composition == "iron_oxide":
                return "Acid cleaning with citric acid and inhibitor"
            elif scale_composition == "copper_oxide":
                return "Ammonium citrate cleaning - dissolves copper without re-deposition"
            else:
                return "Mechanical cleaning or hydro-blasting if accessible"

        elif fouling_type == "corrosion":
            return "Address water chemistry first, then evaluate for chemical cleaning"

        return "Inspect and determine appropriate cleaning method"

    def determine_cleaning_status(
        self,
        dp_ratio: float,
        fouling_factor_ratio: float,
        severity: str,
    ) -> str:
        """
        Determine cleaning status and urgency.

        Args:
            dp_ratio: DP ratio
            fouling_factor_ratio: Fouling factor ratio
            severity: Fouling severity

        Returns:
            Cleaning status: not_required, monitor, recommended, required, urgent
        """
        if severity == "critical":
            return "urgent"
        elif severity == "severe":
            return "required"
        elif severity == "moderate":
            return "recommended"
        elif severity == "light":
            return "monitor"
        else:
            return "not_required"

    def analyze(self, input_data: WaterSideFoulingInput) -> WaterSideFoulingResult:
        """
        Perform complete water-side fouling analysis.

        Args:
            input_data: WaterSideFoulingInput with all required data

        Returns:
            WaterSideFoulingResult with analysis results
        """
        start_time = datetime.now(timezone.utc)

        # Calculate flow ratio
        flow_ratio = input_data.current_water_flow_lb_hr / input_data.design_water_flow_lb_hr

        # Analyze pressure drop
        corrected_dp, dp_ratio = self.analyze_pressure_drop(
            input_data.current_dp_psi,
            input_data.design_dp_psi,
            flow_ratio,
        )

        # Calculate fouling factor
        if input_data.current_ua_btu_hr_f and input_data.design_ua_btu_hr_f:
            fouling_factor = self.calculate_fouling_factor(
                input_data.current_ua_btu_hr_f,
                input_data.design_ua_btu_hr_f,
                5000.0,  # Default area - would be from config in real system
            )
        else:
            # Estimate fouling factor from DP ratio
            # Higher DP ratio suggests more fouling
            fouling_factor = input_data.design_fouling_factor * (dp_ratio ** 0.5)

        fouling_factor_ratio = fouling_factor / input_data.design_fouling_factor

        # Analyze water chemistry
        chemistry_compliant = True
        chemistry_deviations = []
        scale_composition = None

        if input_data.chemistry:
            chemistry_compliant, chemistry_deviations, scale_composition = (
                self.analyze_water_chemistry(
                    input_data.chemistry,
                    input_data.pressure_class,
                )
            )

        # Estimate scale thickness
        scale_thickness = None
        if fouling_factor > 0:
            scale_type = scale_composition if scale_composition else "mixed"
            scale_thickness = self.estimate_scale_thickness(fouling_factor, scale_type)

        # Determine severity and type
        severity = self.determine_fouling_severity(
            dp_ratio,
            fouling_factor_ratio,
            len(chemistry_deviations),
        )

        fouling_type = self.determine_fouling_type(
            dp_ratio,
            fouling_factor_ratio,
            chemistry_deviations,
        )

        # Determine cleaning status and method
        cleaning_status = self.determine_cleaning_status(
            dp_ratio,
            fouling_factor_ratio,
            severity,
        )

        cleaning_method = self.recommend_cleaning_method(
            fouling_type,
            scale_composition,
            severity,
        )

        # Build result
        result_data = {
            "fouling_detected": severity != "none",
            "fouling_severity": severity,
            "fouling_type": fouling_type,
            "current_dp_psi": round(input_data.current_dp_psi, 2),
            "design_dp_psi": round(input_data.design_dp_psi, 2),
            "corrected_dp_psi": round(corrected_dp, 2),
            "dp_ratio": round(dp_ratio, 3),
            "fouling_factor_hr_ft2_f_btu": round(fouling_factor, 6),
            "design_fouling_factor": round(input_data.design_fouling_factor, 6),
            "fouling_factor_ratio": round(fouling_factor_ratio, 2),
            "estimated_scale_thickness_mils": round(scale_thickness, 2) if scale_thickness else None,
            "scale_composition": scale_composition,
            "chemistry_compliant": chemistry_compliant,
            "chemistry_deviations": chemistry_deviations,
            "cleaning_status": cleaning_status,
            "recommended_cleaning_method": cleaning_method,
            "calculation_method": "ASME_PTC_4.1",
        }

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            json.dumps(result_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        result = WaterSideFoulingResult(
            **result_data,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Water-side fouling analysis complete: severity={severity}, "
            f"type={fouling_type}, dp_ratio={dp_ratio:.3f}"
        )

        return result


def create_water_side_fouling_analyzer(
    scale_conductivity: float = DEFAULT_SCALE_CONDUCTIVITY,
) -> WaterSideFoulingAnalyzer:
    """Factory function to create WaterSideFoulingAnalyzer."""
    return WaterSideFoulingAnalyzer(scale_conductivity=scale_conductivity)
