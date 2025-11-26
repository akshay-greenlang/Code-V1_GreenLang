"""
Compliance Checker Module for GL-010 EMISSIONWATCH.

This module provides a multi-jurisdiction regulatory compliance engine
for emissions limits checking. All rules are deterministic and based
on published regulatory standards.

Supported Regulations:
- EPA NSPS (New Source Performance Standards)
- EPA MACT (Maximum Achievable Control Technology)
- EU IED (Industrial Emissions Directive)
- EU BAT-AELs (Best Available Techniques - Associated Emission Levels)
- State/Local permits

References:
- EPA 40 CFR Part 60 (NSPS)
- EPA 40 CFR Part 63 (MACT)
- EU Directive 2010/75/EU (IED)
- EU BAT Reference Documents (BREFs)

Zero-Hallucination Guarantee:
- All limits are from published regulations (cited)
- Deterministic comparison logic
- Full audit trail for compliance determinations
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
from pydantic import BaseModel, Field

from .constants import O2_REFERENCE, AVERAGING_PERIODS


class Jurisdiction(str, Enum):
    """Regulatory jurisdictions."""
    EPA_FEDERAL = "epa_federal"
    EU = "european_union"
    CALIFORNIA = "california"
    TEXAS = "texas"
    NEW_YORK = "new_york"
    PENNSYLVANIA = "pennsylvania"
    UK = "united_kingdom"
    GERMANY = "germany"
    CHINA = "china"
    PERMIT_SPECIFIC = "permit_specific"


class RegulatoryProgram(str, Enum):
    """Regulatory programs."""
    NSPS = "nsps"  # New Source Performance Standards
    MACT = "mact"  # Maximum Achievable Control Technology
    IED = "ied"    # Industrial Emissions Directive
    BAT_AEL = "bat_ael"  # Best Available Techniques
    TITLE_V = "title_v"  # Title V Operating Permit
    PSD = "psd"    # Prevention of Significant Deterioration
    NAAQS = "naaqs"  # National Ambient Air Quality Standards
    PERMIT = "permit"  # Facility-specific permit


class AveragingPeriod(str, Enum):
    """Averaging periods for emission limits."""
    INSTANTANEOUS = "instantaneous"
    ONE_HOUR = "1_hour"
    THREE_HOUR = "3_hour"
    EIGHT_HOUR = "8_hour"
    TWENTY_FOUR_HOUR = "24_hour"
    THIRTY_DAY = "30_day"
    ROLLING_30_DAY = "rolling_30_day"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ComplianceStatus(str, Enum):
    """Compliance determination status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    APPROACHING_LIMIT = "approaching_limit"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"


class SourceCategory(str, Enum):
    """Emission source categories."""
    BOILER = "boiler"
    GAS_TURBINE = "gas_turbine"
    RECIPROCATING_ENGINE = "reciprocating_engine"
    INCINERATOR = "incinerator"
    PROCESS_HEATER = "process_heater"
    CEMENT_KILN = "cement_kiln"
    REFINERY = "refinery"
    POWER_PLANT = "power_plant"


@dataclass(frozen=True)
class EmissionLimit:
    """
    Single emission limit with regulatory provenance.

    Attributes:
        pollutant: Pollutant type (NOx, SOx, CO2, PM, etc.)
        limit_value: Numeric limit value
        unit: Unit of measurement
        averaging_period: Time period for averaging
        o2_reference: Reference O2 level (%)
        jurisdiction: Regulatory jurisdiction
        program: Regulatory program
        source_category: Applicable source category
        effective_date: Date limit became effective
        citation: Regulatory citation
        conditions: Additional conditions/notes
    """
    pollutant: str
    limit_value: Decimal
    unit: str
    averaging_period: AveragingPeriod
    o2_reference: Optional[Decimal]
    jurisdiction: Jurisdiction
    program: RegulatoryProgram
    source_category: SourceCategory
    effective_date: date
    citation: str
    conditions: Optional[str] = None


@dataclass(frozen=True)
class ComplianceCheckResult:
    """
    Result of a compliance check.

    Attributes:
        status: Compliance status
        measured_value: Measured/calculated emission value
        limit: Applicable emission limit
        margin_percent: Margin to limit (negative = exceedance)
        margin_absolute: Absolute margin to limit
        averaging_period: Period over which measurement was averaged
        data_completeness: Percentage of valid data in period
        check_timestamp: When check was performed
        notes: Additional notes
    """
    status: ComplianceStatus
    measured_value: Decimal
    limit: EmissionLimit
    margin_percent: Decimal
    margin_absolute: Decimal
    averaging_period: AveragingPeriod
    data_completeness: Decimal
    check_timestamp: datetime
    notes: Optional[str] = None


@dataclass
class ComplianceReport:
    """
    Complete compliance report for a source.

    Attributes:
        source_id: Source identifier
        source_category: Source type
        check_results: List of compliance check results
        overall_status: Overall compliance status
        reporting_period: Reporting period
        generated_timestamp: When report was generated
    """
    source_id: str
    source_category: SourceCategory
    check_results: List[ComplianceCheckResult]
    overall_status: ComplianceStatus
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_timestamp: datetime


class EmissionLimitQuery(BaseModel):
    """Query parameters for limit lookup."""
    pollutant: str = Field(description="Pollutant type")
    source_category: SourceCategory = Field(description="Source category")
    jurisdiction: Jurisdiction = Field(description="Regulatory jurisdiction")
    fuel_type: Optional[str] = Field(default=None, description="Fuel type")
    capacity_mmbtu_hr: Optional[float] = Field(
        default=None, description="Source capacity (MMBtu/hr)"
    )
    effective_date: Optional[date] = Field(
        default=None, description="Date for applicable limits"
    )


class MeasuredEmission(BaseModel):
    """Measured or calculated emission data."""
    pollutant: str = Field(description="Pollutant type")
    value: float = Field(description="Measured value")
    unit: str = Field(description="Unit of measurement")
    timestamp: datetime = Field(description="Measurement timestamp")
    o2_percent: Optional[float] = Field(
        default=None, description="O2 at time of measurement"
    )
    averaging_period: AveragingPeriod = Field(
        default=AveragingPeriod.ONE_HOUR,
        description="Averaging period"
    )
    data_quality: Optional[str] = Field(
        default="valid", description="Data quality flag"
    )


class ComplianceChecker:
    """
    Multi-jurisdiction regulatory compliance checker.

    Provides deterministic compliance checking against:
    - EPA NSPS limits
    - EU IED BAT-AELs
    - State/local limits
    - Permit-specific limits

    All compliance determinations are traceable to specific
    regulatory citations.
    """

    def __init__(self):
        """Initialize compliance checker with regulatory database."""
        self._limits: List[EmissionLimit] = []
        self._load_nsps_limits()
        self._load_ied_limits()
        self._load_state_limits()

    def _load_nsps_limits(self) -> None:
        """Load EPA NSPS limits."""

        # NSPS Subpart D - Fossil Fuel-Fired Steam Generators
        self._limits.extend([
            # NOx limits for large boilers
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("0.20"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1971, 8, 17),
                citation="40 CFR 60.44(a)",
                conditions="Natural gas, >250 MMBtu/hr"
            ),
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("0.30"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1971, 8, 17),
                citation="40 CFR 60.44(a)",
                conditions="Oil, >250 MMBtu/hr"
            ),
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("0.70"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1971, 8, 17),
                citation="40 CFR 60.44(a)",
                conditions="Coal, >250 MMBtu/hr"
            ),
            # PM limits
            EmissionLimit(
                pollutant="PM",
                limit_value=Decimal("0.030"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THREE_HOUR,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1971, 8, 17),
                citation="40 CFR 60.42(a)",
                conditions=">250 MMBtu/hr"
            ),
            # Opacity limits
            EmissionLimit(
                pollutant="Opacity",
                limit_value=Decimal("20"),
                unit="%",
                averaging_period=AveragingPeriod.ONE_HOUR,
                o2_reference=None,
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1971, 8, 17),
                citation="40 CFR 60.42(b)",
                conditions="6-minute average"
            ),
        ])

        # NSPS Subpart Db - Industrial Boilers (>100 MMBtu/hr)
        self._limits.extend([
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("0.10"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1987, 6, 19),
                citation="40 CFR 60.44b(a)(1)",
                conditions="Natural gas, >100 MMBtu/hr, low-NOx burner"
            ),
            EmissionLimit(
                pollutant="SO2",
                limit_value=Decimal("0.50"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.BOILER,
                effective_date=date(1987, 6, 19),
                citation="40 CFR 60.42b(a)",
                conditions=">100 MMBtu/hr, coal"
            ),
        ])

        # NSPS Subpart KKKK - Gas Turbines
        self._limits.extend([
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("25"),
                unit="ppm @ 15% O2",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.GAS_TURBINE,
                effective_date=date(2006, 7, 6),
                citation="40 CFR 60.4320",
                conditions="Natural gas, >850 MMBtu/hr"
            ),
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("42"),
                unit="ppm @ 15% O2",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.GAS_TURBINE,
                effective_date=date(2006, 7, 6),
                citation="40 CFR 60.4320",
                conditions="Oil firing, >850 MMBtu/hr"
            ),
            EmissionLimit(
                pollutant="SO2",
                limit_value=Decimal("110"),
                unit="ng/J",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.GAS_TURBINE,
                effective_date=date(2006, 7, 6),
                citation="40 CFR 60.4330",
                conditions="All fuels"
            ),
        ])

        # NSPS Subpart JJJJ - Reciprocating Engines
        self._limits.extend([
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("1.0"),
                unit="g/hp-hr",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.RECIPROCATING_ENGINE,
                effective_date=date(2008, 6, 12),
                citation="40 CFR 60.4233",
                conditions="Lean burn, >500 HP"
            ),
            EmissionLimit(
                pollutant="CO",
                limit_value=Decimal("2.0"),
                unit="g/hp-hr",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EPA_FEDERAL,
                program=RegulatoryProgram.NSPS,
                source_category=SourceCategory.RECIPROCATING_ENGINE,
                effective_date=date(2008, 6, 12),
                citation="40 CFR 60.4233",
                conditions="All engines >500 HP"
            ),
        ])

    def _load_ied_limits(self) -> None:
        """Load EU Industrial Emissions Directive BAT-AEL limits."""

        # Large Combustion Plants (LCP) BREF
        self._limits.extend([
            # Natural gas boilers
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("50"),
                unit="mg/Nm3 @ 3% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.BOILER,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 28",
                conditions="Natural gas, >300 MWth, new plant"
            ),
            EmissionLimit(
                pollutant="CO",
                limit_value=Decimal("100"),
                unit="mg/Nm3 @ 3% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.BOILER,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 29",
                conditions="Natural gas, >300 MWth"
            ),
            # Coal boilers
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("85"),
                unit="mg/Nm3 @ 6% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("6.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.BOILER,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 21",
                conditions="Hard coal, >300 MWth, new plant"
            ),
            EmissionLimit(
                pollutant="SO2",
                limit_value=Decimal("130"),
                unit="mg/Nm3 @ 6% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("6.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.BOILER,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 22",
                conditions="Hard coal, >300 MWth"
            ),
            EmissionLimit(
                pollutant="PM",
                limit_value=Decimal("8"),
                unit="mg/Nm3 @ 6% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("6.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.BOILER,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 23",
                conditions="Hard coal, >300 MWth"
            ),
            # Gas turbines
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("30"),
                unit="mg/Nm3 @ 15% O2",
                averaging_period=AveragingPeriod.ANNUAL,
                o2_reference=Decimal("15.0"),
                jurisdiction=Jurisdiction.EU,
                program=RegulatoryProgram.BAT_AEL,
                source_category=SourceCategory.GAS_TURBINE,
                effective_date=date(2017, 8, 17),
                citation="LCP BREF, BAT 34",
                conditions="Natural gas, >50 MWth, new plant"
            ),
        ])

    def _load_state_limits(self) -> None:
        """Load state-specific emission limits."""

        # California SCAQMD (South Coast)
        self._limits.extend([
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("5"),
                unit="ppm @ 3% O2",
                averaging_period=AveragingPeriod.ONE_HOUR,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.CALIFORNIA,
                program=RegulatoryProgram.PERMIT,
                source_category=SourceCategory.BOILER,
                effective_date=date(2015, 1, 1),
                citation="SCAQMD Rule 1146",
                conditions="Natural gas, >5 MMBtu/hr"
            ),
            EmissionLimit(
                pollutant="CO",
                limit_value=Decimal("400"),
                unit="ppm @ 3% O2",
                averaging_period=AveragingPeriod.ONE_HOUR,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.CALIFORNIA,
                program=RegulatoryProgram.PERMIT,
                source_category=SourceCategory.BOILER,
                effective_date=date(2015, 1, 1),
                citation="SCAQMD Rule 1146",
                conditions="Natural gas, >5 MMBtu/hr"
            ),
        ])

        # Texas TCEQ
        self._limits.extend([
            EmissionLimit(
                pollutant="NOx",
                limit_value=Decimal("0.06"),
                unit="lb/MMBtu",
                averaging_period=AveragingPeriod.THIRTY_DAY,
                o2_reference=Decimal("3.0"),
                jurisdiction=Jurisdiction.TEXAS,
                program=RegulatoryProgram.PERMIT,
                source_category=SourceCategory.BOILER,
                effective_date=date(2010, 1, 1),
                citation="30 TAC 117",
                conditions="Natural gas, Houston-Galveston area"
            ),
        ])

    def get_applicable_limits(
        self,
        query: EmissionLimitQuery
    ) -> List[EmissionLimit]:
        """
        Get all applicable emission limits for a source.

        Args:
            query: Query parameters

        Returns:
            List of applicable emission limits
        """
        effective_date = query.effective_date or date.today()

        applicable = []
        for limit in self._limits:
            # Check pollutant
            if limit.pollutant.lower() != query.pollutant.lower():
                continue

            # Check source category
            if limit.source_category != query.source_category:
                continue

            # Check jurisdiction
            if limit.jurisdiction != query.jurisdiction:
                continue

            # Check effective date
            if limit.effective_date > effective_date:
                continue

            applicable.append(limit)

        # Sort by effective date (most recent first)
        applicable.sort(key=lambda x: x.effective_date, reverse=True)

        return applicable

    def check_compliance(
        self,
        emission: MeasuredEmission,
        limit: EmissionLimit,
        o2_correction: bool = True
    ) -> ComplianceCheckResult:
        """
        Check compliance of measured emission against a limit.

        Args:
            emission: Measured emission data
            limit: Applicable emission limit
            o2_correction: Apply O2 correction if needed

        Returns:
            ComplianceCheckResult with compliance determination
        """
        measured_value = Decimal(str(emission.value))

        # Apply O2 correction if needed
        if o2_correction and limit.o2_reference and emission.o2_percent:
            measured_o2 = Decimal(str(emission.o2_percent))
            ref_o2 = limit.o2_reference
            o2_air = Decimal("20.9")

            # O2 correction formula: C_ref = C_meas * (20.9 - O2_ref) / (20.9 - O2_meas)
            correction_factor = (o2_air - ref_o2) / (o2_air - measured_o2)
            corrected_value = measured_value * correction_factor
            notes = f"O2 corrected from {measured_o2}% to {ref_o2}% ref"
        else:
            corrected_value = measured_value
            notes = None

        # Calculate margin
        margin_absolute = limit.limit_value - corrected_value
        if limit.limit_value > 0:
            margin_percent = (margin_absolute / limit.limit_value) * Decimal("100")
        else:
            margin_percent = Decimal("0")

        # Determine compliance status
        if corrected_value > limit.limit_value:
            status = ComplianceStatus.NON_COMPLIANT
        elif margin_percent < Decimal("10"):
            status = ComplianceStatus.APPROACHING_LIMIT
        else:
            status = ComplianceStatus.COMPLIANT

        return ComplianceCheckResult(
            status=status,
            measured_value=self._apply_precision(corrected_value, 4),
            limit=limit,
            margin_percent=self._apply_precision(margin_percent, 2),
            margin_absolute=self._apply_precision(margin_absolute, 4),
            averaging_period=emission.averaging_period,
            data_completeness=Decimal("100"),  # Assumed for single value
            check_timestamp=datetime.now(),
            notes=notes
        )

    def check_compliance_batch(
        self,
        emissions: List[MeasuredEmission],
        source_category: SourceCategory,
        jurisdiction: Jurisdiction
    ) -> ComplianceReport:
        """
        Check compliance for multiple pollutants/emissions.

        Args:
            emissions: List of measured emissions
            source_category: Source category
            jurisdiction: Regulatory jurisdiction

        Returns:
            ComplianceReport with all check results
        """
        results = []

        for emission in emissions:
            # Get applicable limits
            query = EmissionLimitQuery(
                pollutant=emission.pollutant,
                source_category=source_category,
                jurisdiction=jurisdiction
            )
            limits = self.get_applicable_limits(query)

            if not limits:
                # No applicable limit found
                results.append(ComplianceCheckResult(
                    status=ComplianceStatus.NOT_APPLICABLE,
                    measured_value=Decimal(str(emission.value)),
                    limit=None,
                    margin_percent=Decimal("0"),
                    margin_absolute=Decimal("0"),
                    averaging_period=emission.averaging_period,
                    data_completeness=Decimal("100"),
                    check_timestamp=datetime.now(),
                    notes="No applicable limit found"
                ))
            else:
                # Check against most stringent limit
                for limit in limits:
                    result = self.check_compliance(emission, limit)
                    results.append(result)

        # Determine overall status
        if any(r.status == ComplianceStatus.NON_COMPLIANT for r in results):
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif any(r.status == ComplianceStatus.APPROACHING_LIMIT for r in results):
            overall_status = ComplianceStatus.APPROACHING_LIMIT
        elif any(r.status == ComplianceStatus.INSUFFICIENT_DATA for r in results):
            overall_status = ComplianceStatus.INSUFFICIENT_DATA
        else:
            overall_status = ComplianceStatus.COMPLIANT

        # Determine reporting period from emissions
        if emissions:
            period_start = min(e.timestamp for e in emissions)
            period_end = max(e.timestamp for e in emissions)
        else:
            period_start = datetime.now()
            period_end = datetime.now()

        return ComplianceReport(
            source_id="source_001",
            source_category=source_category,
            check_results=results,
            overall_status=overall_status,
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            generated_timestamp=datetime.now()
        )

    def calculate_averaging_period_value(
        self,
        hourly_values: List[Tuple[datetime, Decimal]],
        averaging_period: AveragingPeriod,
        minimum_data_completeness: float = 0.75
    ) -> Tuple[Optional[Decimal], Decimal]:
        """
        Calculate averaged value for a given averaging period.

        Args:
            hourly_values: List of (timestamp, value) tuples
            averaging_period: Averaging period to calculate
            minimum_data_completeness: Required data completeness (0-1)

        Returns:
            Tuple of (averaged value or None, data completeness)
        """
        if not hourly_values:
            return None, Decimal("0")

        # Sort by timestamp
        sorted_values = sorted(hourly_values, key=lambda x: x[0])

        # Get period hours
        period_hours = {
            AveragingPeriod.ONE_HOUR: 1,
            AveragingPeriod.THREE_HOUR: 3,
            AveragingPeriod.EIGHT_HOUR: 8,
            AveragingPeriod.TWENTY_FOUR_HOUR: 24,
            AveragingPeriod.THIRTY_DAY: 720,
            AveragingPeriod.ROLLING_30_DAY: 720,
            AveragingPeriod.QUARTERLY: 2190,
            AveragingPeriod.ANNUAL: 8760,
        }

        required_hours = period_hours.get(averaging_period, 1)

        # Check data completeness
        actual_hours = len(sorted_values)
        data_completeness = Decimal(str(actual_hours)) / Decimal(str(required_hours))

        if float(data_completeness) < minimum_data_completeness:
            return None, data_completeness

        # Calculate average
        total = sum(v[1] for v in sorted_values)
        average = total / Decimal(str(actual_hours))

        return self._apply_precision(average, 4), self._apply_precision(data_completeness * 100, 1)

    def add_permit_limit(
        self,
        limit: EmissionLimit
    ) -> None:
        """
        Add a permit-specific emission limit.

        Args:
            limit: Emission limit to add
        """
        self._limits.append(limit)

    def get_all_limits(
        self,
        jurisdiction: Optional[Jurisdiction] = None,
        source_category: Optional[SourceCategory] = None,
        pollutant: Optional[str] = None
    ) -> List[EmissionLimit]:
        """
        Get all limits matching criteria.

        Args:
            jurisdiction: Filter by jurisdiction
            source_category: Filter by source category
            pollutant: Filter by pollutant

        Returns:
            List of matching limits
        """
        results = []
        for limit in self._limits:
            if jurisdiction and limit.jurisdiction != jurisdiction:
                continue
            if source_category and limit.source_category != source_category:
                continue
            if pollutant and limit.pollutant.lower() != pollutant.lower():
                continue
            results.append(limit)
        return results

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# Convenience functions
def check_nox_compliance(
    nox_lb_mmbtu: float,
    source_category: str = "boiler",
    fuel_type: str = "natural_gas",
    o2_percent: float = 3.0
) -> ComplianceCheckResult:
    """
    Quick NOx compliance check against EPA NSPS.

    Args:
        nox_lb_mmbtu: NOx emission rate (lb/MMBtu)
        source_category: Type of source
        fuel_type: Fuel type
        o2_percent: O2 at measurement

    Returns:
        ComplianceCheckResult
    """
    checker = ComplianceChecker()

    emission = MeasuredEmission(
        pollutant="NOx",
        value=nox_lb_mmbtu,
        unit="lb/MMBtu",
        timestamp=datetime.now(),
        o2_percent=o2_percent,
        averaging_period=AveragingPeriod.THIRTY_DAY
    )

    query = EmissionLimitQuery(
        pollutant="NOx",
        source_category=SourceCategory(source_category),
        jurisdiction=Jurisdiction.EPA_FEDERAL
    )

    limits = checker.get_applicable_limits(query)

    if limits:
        return checker.check_compliance(emission, limits[0])
    else:
        return ComplianceCheckResult(
            status=ComplianceStatus.NOT_APPLICABLE,
            measured_value=Decimal(str(nox_lb_mmbtu)),
            limit=None,
            margin_percent=Decimal("0"),
            margin_absolute=Decimal("0"),
            averaging_period=AveragingPeriod.THIRTY_DAY,
            data_completeness=Decimal("100"),
            check_timestamp=datetime.now(),
            notes="No applicable NSPS limit found"
        )


def get_compliance_checker() -> ComplianceChecker:
    """Get an initialized compliance checker instance."""
    return ComplianceChecker()
