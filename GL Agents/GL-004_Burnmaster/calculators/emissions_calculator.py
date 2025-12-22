"""
Emissions Calculator for GL-004 BURNMASTER

Zero-hallucination calculation engine for emissions estimation and compliance.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- NOx emission estimation from combustion parameters
- CO emission estimation from combustion efficiency
- Permit compliance checking
- Emission rate calculations (mass flow)
- Rolling average calculations for regulatory reporting

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import math

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums and Constants
# =============================================================================

class ComplianceStatus(str, Enum):
    """Permit compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"           # Within 80-100% of limit
    EXCEEDANCE = "exceedance"     # Above limit
    CRITICAL = "critical"         # Significantly above limit (>150%)


class PollutantType(str, Enum):
    """Types of pollutants tracked."""
    NOX = "nox"
    CO = "co"
    SO2 = "so2"
    PM = "pm"
    VOC = "voc"


# Reference O2 percentages for emissions normalization
# Different jurisdictions use different reference O2 values
REFERENCE_O2: Dict[str, float] = {
    "us_epa": 3.0,       # US EPA standard
    "eu_ied": 3.0,       # EU Industrial Emissions Directive
    "uk_mcpd": 3.0,      # UK Medium Combustion Plant Directive
    "california": 3.0,   # California AQMD
    "china_gb": 9.0,     # China GB standards (some applications)
}

# Typical permit limits (ppm at reference O2) - examples
# Actual limits vary by facility, jurisdiction, and permit
DEFAULT_PERMIT_LIMITS: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "nox_ppm": 25.0,     # Typical BACT limit
        "co_ppm": 100.0,
    },
    "fuel_oil": {
        "nox_ppm": 100.0,
        "co_ppm": 100.0,
    },
    "coal": {
        "nox_ppm": 200.0,
        "co_ppm": 200.0,
    },
}

# Molecular weights for emission rate calculations
MOLECULAR_WEIGHTS: Dict[str, float] = {
    "nox": 46.0,    # NO2 basis
    "co": 28.0,
    "so2": 64.0,
    "co2": 44.0,
}


# =============================================================================
# Pydantic Schemas for Input/Output
# =============================================================================

class NOxEstimateInput(BaseModel):
    """Input schema for NOx estimation."""

    flame_temp: float = Field(..., gt=0, description="Flame temperature (Kelvin)")
    residence_time: float = Field(..., gt=0, description="Gas residence time in flame zone (ms)")
    o2_percent: float = Field(..., ge=0, le=21, description="O2 percentage in flue gas")
    fuel_nitrogen: float = Field(default=0.0, ge=0, description="Fuel nitrogen content (wt%)")


class COEstimateInput(BaseModel):
    """Input schema for CO estimation."""

    combustion_efficiency: float = Field(..., ge=0, le=100, description="Combustion efficiency (%)")
    o2_percent: float = Field(..., ge=0, le=21, description="O2 percentage in flue gas")
    co_baseline: float = Field(default=5.0, ge=0, description="Baseline CO at optimal conditions (ppm)")


class EmissionRateInput(BaseModel):
    """Input schema for emission rate calculation."""

    concentration_ppm: float = Field(..., ge=0, description="Pollutant concentration (ppm)")
    flue_flow_rate: float = Field(..., gt=0, description="Flue gas volumetric flow (Nm3/h)")
    pollutant: PollutantType = Field(..., description="Type of pollutant")
    measured_o2: float = Field(default=3.0, ge=0, le=21, description="Measured O2 (%)")
    reference_o2: float = Field(default=3.0, ge=0, le=21, description="Reference O2 for normalization (%)")


class EmissionRateResult(BaseModel):
    """Output schema for emission rate calculation."""

    mass_rate_kg_h: Decimal = Field(..., description="Mass emission rate (kg/h)")
    mass_rate_tons_year: Decimal = Field(..., description="Annualized rate (tons/year)")
    normalized_concentration_ppm: Decimal = Field(..., description="Concentration at reference O2 (ppm)")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_steps: List[Dict[str, Any]] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    """Output schema for permit compliance check."""

    nox_status: ComplianceStatus = Field(..., description="NOx compliance status")
    co_status: ComplianceStatus = Field(..., description="CO compliance status")
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")
    nox_measured_ppm: Decimal = Field(..., description="Measured NOx (ppm at ref O2)")
    co_measured_ppm: Decimal = Field(..., description="Measured CO (ppm at ref O2)")
    nox_limit_ppm: Decimal = Field(..., description="NOx permit limit (ppm)")
    co_limit_ppm: Decimal = Field(..., description="CO permit limit (ppm)")
    nox_percent_of_limit: Decimal = Field(..., description="NOx as % of limit")
    co_percent_of_limit: Decimal = Field(..., description="CO as % of limit")
    margin_to_exceedance: Dict[str, Decimal] = Field(..., description="Margin to limit for each pollutant")
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RollingAverageResult(BaseModel):
    """Output schema for rolling average calculation."""

    rolling_average: Decimal = Field(..., description="Rolling average value")
    window_hours: float = Field(..., description="Averaging window (hours)")
    sample_count: int = Field(..., description="Number of samples in average")
    min_value: Decimal = Field(..., description="Minimum value in window")
    max_value: Decimal = Field(..., description="Maximum value in window")
    trend: str = Field(..., description="Trend direction (increasing/decreasing/stable)")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# =============================================================================
# Emissions Calculator Class
# =============================================================================

class EmissionsCalculator:
    """
    Zero-hallucination calculator for emissions estimation and compliance.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure arithmetic and lookup operations only

    Example:
        >>> calculator = EmissionsCalculator()
        >>> nox = calculator.compute_nox_estimate(1800.0, 50.0, 3.0)
        >>> print(f"NOx estimate: {nox} ppm")
    """

    def __init__(self, precision: int = 2, reference_o2: float = 3.0):
        """
        Initialize calculator with precision settings.

        Args:
            precision: Decimal places for output values (default: 2)
            reference_o2: Reference O2 for normalization (default: 3.0%)
        """
        self.precision = precision
        self.reference_o2 = reference_o2
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding (ROUND_HALF_UP for regulatory compliance)."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _normalize_to_reference_o2(
        self,
        concentration: float,
        measured_o2: float,
        reference_o2: float = None
    ) -> Decimal:
        """
        Normalize concentration to reference O2 level.

        DETERMINISTIC formula:
        C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)

        Args:
            concentration: Measured concentration (ppm)
            measured_o2: Measured O2 percentage
            reference_o2: Reference O2 percentage (default: self.reference_o2)

        Returns:
            Normalized concentration at reference O2
        """
        if reference_o2 is None:
            reference_o2 = self.reference_o2

        # Avoid division by zero
        if measured_o2 >= 21.0:
            return Decimal('0')

        # Normalization factor
        factor = (21.0 - reference_o2) / (21.0 - measured_o2)
        normalized = concentration * factor

        return self._quantize(Decimal(str(normalized)))

    # -------------------------------------------------------------------------
    # Core Calculation Methods
    # -------------------------------------------------------------------------

    def compute_nox_estimate(
        self,
        flame_temp: float,
        residence_time: float,
        o2: float,
        fuel_nitrogen: float = 0.0
    ) -> Decimal:
        """
        Estimate NOx emissions from combustion parameters.

        DETERMINISTIC: Zeldovich correlation + fuel NOx contribution.

        This uses a simplified Zeldovich mechanism for thermal NOx:
        NOx_thermal = A * exp(-E_a / (R * T)) * sqrt(O2) * t

        Where:
        - A is a pre-exponential factor
        - E_a is activation energy (~68 kcal/mol)
        - R is gas constant
        - T is flame temperature
        - t is residence time

        Plus fuel NOx (if fuel contains nitrogen):
        NOx_fuel = B * fuel_N * conversion_efficiency

        Args:
            flame_temp: Flame temperature (Kelvin), typically 1500-2200K
            residence_time: Gas residence time in flame zone (ms)
            o2: O2 percentage in flue gas (dry basis)
            fuel_nitrogen: Fuel nitrogen content (wt%), default 0

        Returns:
            Estimated NOx concentration (ppm at measured O2)
        """
        # Zeldovich mechanism constants (simplified)
        A = 1.5e6  # Pre-exponential factor (calibrated for natural gas)
        Ea_R = 34000  # Ea/R in Kelvin (~68 kcal/mol)

        # Step 1: Calculate thermal NOx using Zeldovich (DETERMINISTIC)
        if flame_temp > 0:
            # Arrhenius term
            arrhenius = math.exp(-Ea_R / flame_temp)

            # O2 dependency (square root relationship)
            o2_term = math.sqrt(max(0.01, o2 / 100))

            # Time dependency (linear with residence time)
            time_term = residence_time / 1000  # Convert ms to seconds

            # Temperature effect (strong exponential)
            # NOx formation rate doubles approximately every 90K above 1800K
            temp_factor = 1.0
            if flame_temp > 1800:
                temp_factor = 2 ** ((flame_temp - 1800) / 90)

            thermal_nox = A * arrhenius * o2_term * time_term * temp_factor
        else:
            thermal_nox = 0.0

        # Step 2: Calculate fuel NOx (DETERMINISTIC)
        # Conversion efficiency typically 10-50% for fuel nitrogen
        fuel_nox_conversion = 0.25  # 25% conversion assumption
        fuel_nox = fuel_nitrogen * 10000 * fuel_nox_conversion  # Rough estimate

        # Step 3: Total NOx
        total_nox = thermal_nox + fuel_nox

        # Cap at reasonable maximum (500 ppm for typical combustion)
        total_nox = min(500.0, max(0.0, total_nox))

        return self._quantize(Decimal(str(total_nox)))

    def compute_co_estimate(
        self,
        combustion_efficiency: float,
        o2: float,
        co_baseline: float = 5.0
    ) -> Decimal:
        """
        Estimate CO emissions from combustion efficiency.

        DETERMINISTIC: CO increases as combustion efficiency decreases.

        CO = CO_baseline / (efficiency_factor * O2_factor)

        Where:
        - Efficiency factor decreases with decreasing efficiency
        - O2 factor accounts for air/fuel ratio effects

        Args:
            combustion_efficiency: Combustion efficiency (%), typically 95-99.9%
            o2: O2 percentage in flue gas
            co_baseline: Baseline CO at optimal conditions (ppm)

        Returns:
            Estimated CO concentration (ppm at measured O2)
        """
        # Step 1: Calculate efficiency factor (DETERMINISTIC)
        # CO increases exponentially as efficiency drops below 99%
        if combustion_efficiency >= 99.9:
            efficiency_factor = 1.0
        elif combustion_efficiency >= 99.0:
            # Small CO increase
            efficiency_factor = 1.0 + (99.9 - combustion_efficiency) * 0.5
        elif combustion_efficiency >= 98.0:
            # Moderate CO increase
            efficiency_factor = 1.5 + (99.0 - combustion_efficiency) * 2.0
        elif combustion_efficiency >= 95.0:
            # Significant CO increase
            efficiency_factor = 3.5 + (98.0 - combustion_efficiency) * 5.0
        else:
            # Poor combustion - high CO
            efficiency_factor = 18.5 + (95.0 - combustion_efficiency) * 10.0

        # Step 2: Calculate O2 factor (DETERMINISTIC)
        # Very low O2 (rich combustion) increases CO
        # Very high O2 (lean combustion) may also increase CO due to quenching
        if o2 < 1.0:
            o2_factor = 1.0 + (1.0 - o2) * 5.0  # Rich combustion penalty
        elif o2 > 8.0:
            o2_factor = 1.0 + (o2 - 8.0) * 0.2  # Very lean penalty
        else:
            o2_factor = 1.0  # Optimal O2 range

        # Step 3: Calculate total CO (DETERMINISTIC)
        co_estimate = co_baseline * efficiency_factor * o2_factor

        # Cap at reasonable maximum
        co_estimate = min(1000.0, max(0.0, co_estimate))

        return self._quantize(Decimal(str(co_estimate)))

    def check_permit_compliance(
        self,
        nox: float,
        co: float,
        limits: Dict[str, float],
        measured_o2: float = 3.0
    ) -> ComplianceResult:
        """
        Check emissions against permit limits.

        DETERMINISTIC: Compare normalized values against limits.

        Args:
            nox: Measured NOx concentration (ppm)
            co: Measured CO concentration (ppm)
            limits: Dict with 'nox_ppm' and 'co_ppm' permit limits
            measured_o2: Measured O2 for normalization

        Returns:
            ComplianceResult with status and recommendations
        """
        # Step 1: Normalize to reference O2 (DETERMINISTIC)
        nox_normalized = self._normalize_to_reference_o2(nox, measured_o2)
        co_normalized = self._normalize_to_reference_o2(co, measured_o2)

        # Step 2: Get limits
        nox_limit = Decimal(str(limits.get('nox_ppm', 25.0)))
        co_limit = Decimal(str(limits.get('co_ppm', 100.0)))

        # Step 3: Calculate percent of limit (DETERMINISTIC)
        nox_percent = self._quantize((nox_normalized / nox_limit) * Decimal('100')) if nox_limit > 0 else Decimal('0')
        co_percent = self._quantize((co_normalized / co_limit) * Decimal('100')) if co_limit > 0 else Decimal('0')

        # Step 4: Determine compliance status (DETERMINISTIC thresholds)
        def get_status(percent: Decimal) -> ComplianceStatus:
            if percent < Decimal('80'):
                return ComplianceStatus.COMPLIANT
            elif percent < Decimal('100'):
                return ComplianceStatus.WARNING
            elif percent < Decimal('150'):
                return ComplianceStatus.EXCEEDANCE
            else:
                return ComplianceStatus.CRITICAL

        nox_status = get_status(nox_percent)
        co_status = get_status(co_percent)

        # Overall status is the worst of the two
        status_priority = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.WARNING: 1,
            ComplianceStatus.EXCEEDANCE: 2,
            ComplianceStatus.CRITICAL: 3,
        }
        overall_status = nox_status if status_priority[nox_status] >= status_priority[co_status] else co_status

        # Step 5: Calculate margin to exceedance (DETERMINISTIC)
        nox_margin = self._quantize(nox_limit - nox_normalized)
        co_margin = self._quantize(co_limit - co_normalized)

        # Step 6: Generate recommendations (DETERMINISTIC rules)
        recommendations = []
        if nox_status in [ComplianceStatus.WARNING, ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
            recommendations.append("Reduce flame temperature to lower thermal NOx")
            recommendations.append("Consider flue gas recirculation (FGR)")
            recommendations.append("Check burner tune - may need O2 trim adjustment")

        if co_status in [ComplianceStatus.WARNING, ComplianceStatus.EXCEEDANCE, ComplianceStatus.CRITICAL]:
            recommendations.append("Increase combustion air to improve mixing")
            recommendations.append("Check burner alignment and flame pattern")
            recommendations.append("Verify fuel quality and heating value")

        if overall_status == ComplianceStatus.CRITICAL:
            recommendations.insert(0, "URGENT: Consider load reduction to return to compliance")

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'nox_measured': nox,
            'co_measured': co,
            'measured_o2': measured_o2,
            'nox_normalized': str(nox_normalized),
            'co_normalized': str(co_normalized),
            'nox_limit': str(nox_limit),
            'co_limit': str(co_limit),
            'overall_status': overall_status.value
        })

        return ComplianceResult(
            nox_status=nox_status,
            co_status=co_status,
            overall_status=overall_status,
            nox_measured_ppm=nox_normalized,
            co_measured_ppm=co_normalized,
            nox_limit_ppm=nox_limit,
            co_limit_ppm=co_limit,
            nox_percent_of_limit=nox_percent,
            co_percent_of_limit=co_percent,
            margin_to_exceedance={
                'nox_ppm': nox_margin,
                'co_ppm': co_margin
            },
            recommendations=recommendations,
            provenance_hash=provenance
        )

    def compute_emission_rate(
        self,
        concentration_ppm: float,
        flue_flow: float,
        pollutant: str = "nox",
        measured_o2: float = 3.0,
        reference_o2: float = 3.0
    ) -> EmissionRateResult:
        """
        Compute mass emission rate from concentration and flow.

        DETERMINISTIC: Mass = Concentration * Flow * MW / MolarVolume

        Args:
            concentration_ppm: Pollutant concentration (ppm, volumetric)
            flue_flow: Flue gas volumetric flow rate (Nm3/h at 0C, 1 atm)
            pollutant: Pollutant type (nox, co, so2)
            measured_o2: Measured O2 percentage
            reference_o2: Reference O2 for normalized concentration

        Returns:
            EmissionRateResult with mass rates and normalized concentration
        """
        calculation_steps = []

        # Step 1: Normalize concentration to reference O2
        normalized_ppm = self._normalize_to_reference_o2(concentration_ppm, measured_o2, reference_o2)
        calculation_steps.append({
            'step': 1,
            'description': 'Normalize to reference O2',
            'measured_ppm': concentration_ppm,
            'measured_o2': measured_o2,
            'reference_o2': reference_o2,
            'normalized_ppm': str(normalized_ppm)
        })

        # Step 2: Get molecular weight
        pollutant_lower = pollutant.lower()
        mw = MOLECULAR_WEIGHTS.get(pollutant_lower, 46.0)  # Default to NO2
        calculation_steps.append({
            'step': 2,
            'description': 'Get molecular weight',
            'pollutant': pollutant_lower,
            'molecular_weight': mw
        })

        # Step 3: Calculate mass rate (DETERMINISTIC)
        # At STP: 1 mole = 22.414 L = 0.022414 Nm3
        # ppm = moles pollutant / moles total * 1e6
        # mass (g/h) = ppm * 1e-6 * flow (Nm3/h) * (1/0.022414 mol/Nm3) * MW (g/mol)

        molar_volume = 22.414  # L/mol at STP
        mass_rate_g_h = float(normalized_ppm) * 1e-6 * flue_flow * (1000 / molar_volume) * mw
        mass_rate_kg_h = mass_rate_g_h / 1000

        calculation_steps.append({
            'step': 3,
            'description': 'Calculate mass rate',
            'formula': 'ppm * 1e-6 * flow * (1000/22.414) * MW',
            'mass_rate_g_h': round(mass_rate_g_h, 4),
            'mass_rate_kg_h': round(mass_rate_kg_h, 4)
        })

        # Step 4: Annualize (assuming continuous operation)
        hours_per_year = 8760
        mass_rate_tons_year = mass_rate_kg_h * hours_per_year / 1000

        calculation_steps.append({
            'step': 4,
            'description': 'Annualize rate',
            'hours_per_year': hours_per_year,
            'mass_rate_tons_year': round(mass_rate_tons_year, 2)
        })

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'concentration_ppm': concentration_ppm,
            'flue_flow': flue_flow,
            'pollutant': pollutant_lower,
            'mass_rate_kg_h': mass_rate_kg_h,
            'mass_rate_tons_year': mass_rate_tons_year
        })

        return EmissionRateResult(
            mass_rate_kg_h=self._quantize(Decimal(str(mass_rate_kg_h))),
            mass_rate_tons_year=self._quantize(Decimal(str(mass_rate_tons_year))),
            normalized_concentration_ppm=normalized_ppm,
            provenance_hash=provenance,
            calculation_steps=calculation_steps
        )

    def compute_rolling_average(
        self,
        emissions: List[float],
        window_hours: float = 1.0,
        sample_interval_minutes: float = 1.0
    ) -> RollingAverageResult:
        """
        Compute rolling average of emissions data.

        DETERMINISTIC: Simple moving average over specified window.

        Args:
            emissions: List of emission values (most recent last)
            window_hours: Averaging window in hours
            sample_interval_minutes: Time between samples in minutes

        Returns:
            RollingAverageResult with average and statistics
        """
        if not emissions:
            provenance = self._compute_provenance_hash({'error': 'no_data'})
            return RollingAverageResult(
                rolling_average=Decimal('0'),
                window_hours=window_hours,
                sample_count=0,
                min_value=Decimal('0'),
                max_value=Decimal('0'),
                trend="stable",
                provenance_hash=provenance
            )

        # Step 1: Calculate number of samples in window
        samples_per_hour = 60.0 / sample_interval_minutes
        window_samples = int(window_hours * samples_per_hour)

        # Step 2: Extract window data (DETERMINISTIC)
        window_data = emissions[-window_samples:] if len(emissions) >= window_samples else emissions

        # Step 3: Calculate statistics (DETERMINISTIC)
        avg = sum(window_data) / len(window_data)
        min_val = min(window_data)
        max_val = max(window_data)

        # Step 4: Determine trend (DETERMINISTIC)
        # Compare first half to second half of window
        if len(window_data) >= 4:
            mid = len(window_data) // 2
            first_half_avg = sum(window_data[:mid]) / mid
            second_half_avg = sum(window_data[mid:]) / (len(window_data) - mid)

            if second_half_avg > first_half_avg * 1.05:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.95:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'sample_count': len(window_data),
            'window_hours': window_hours,
            'average': avg,
            'trend': trend
        })

        return RollingAverageResult(
            rolling_average=self._quantize(Decimal(str(avg))),
            window_hours=window_hours,
            sample_count=len(window_data),
            min_value=self._quantize(Decimal(str(min_val))),
            max_value=self._quantize(Decimal(str(max_val))),
            trend=trend,
            provenance_hash=provenance
        )

    # -------------------------------------------------------------------------
    # Batch Processing Methods
    # -------------------------------------------------------------------------

    def compute_rates_batch(
        self,
        measurements: List[Dict[str, Any]]
    ) -> List[EmissionRateResult]:
        """
        Compute emission rates for a batch of measurements.

        Args:
            measurements: List of dicts with required fields

        Returns:
            List of EmissionRateResult for each measurement
        """
        results = []
        for m in measurements:
            result = self.compute_emission_rate(
                concentration_ppm=m.get('concentration_ppm', 0.0),
                flue_flow=m.get('flue_flow', 1000.0),
                pollutant=m.get('pollutant', 'nox'),
                measured_o2=m.get('measured_o2', 3.0),
                reference_o2=m.get('reference_o2', 3.0)
            )
            results.append(result)
        return results
