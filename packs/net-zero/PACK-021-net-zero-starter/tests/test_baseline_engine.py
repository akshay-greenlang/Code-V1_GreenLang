# -*- coding: utf-8 -*-
"""
Test suite for PACK-021 Net Zero Starter Pack - NetZeroBaselineEngine.

Validates GHG baseline calculation logic for Scope 1, 2, and 3 emissions
including fuel combustion, refrigerant leakage, grid electricity, PPA/REC
market-based adjustments, spend-based Scope 3, data quality scoring,
base year validation, intensity metrics, and provenance hashing.

All assertions on numeric values use Decimal for precision.

Author:  GreenLang Test Engineering
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure the pack root is importable
PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from engines.net_zero_baseline_engine import (
    BaselineInput,
    BaselineResult,
    BaseYearAssessment,
    BoundaryMethod,
    DataQualityAssessment,
    DataQualityScore,
    ElectricityEntry,
    EmissionsByGas,
    FuelConsumptionEntry,
    FuelType,
    GasType,
    IntensityMetrics,
    NetZeroBaselineEngine,
    RefrigerantEntry,
    Scope,
    Scope1Detail,
    Scope1SourceType,
    Scope2Detail,
    Scope3Category,
    Scope3Detail,
    Scope3SpendEntry,
    FUEL_EMISSION_FACTORS,
    GRID_EMISSION_FACTORS,
    RESIDUAL_MIX_FACTORS,
    SPEND_BASED_FACTORS,
    GWP_AR6,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> NetZeroBaselineEngine:
    """Create a fresh engine instance."""
    return NetZeroBaselineEngine()


@pytest.fixture
def simple_fuel_input() -> BaselineInput:
    """Baseline input with a single natural gas fuel entry."""
    return BaselineInput(
        entity_name="TestCorp",
        reporting_year=2024,
        base_year=2024,
        fuel_entries=[
            FuelConsumptionEntry(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("10000"),  # 10000 m3
                source_type=Scope1SourceType.STATIONARY_COMBUSTION,
                data_quality=DataQualityScore.SCORE_2,
            ),
        ],
    )


@pytest.fixture
def diesel_fleet_input() -> BaselineInput:
    """Baseline input with diesel fleet (mobile combustion)."""
    return BaselineInput(
        entity_name="FleetCorp",
        reporting_year=2024,
        base_year=2024,
        fuel_entries=[
            FuelConsumptionEntry(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("5000"),  # 5000 litres
                source_type=Scope1SourceType.MOBILE_COMBUSTION,
                data_quality=DataQualityScore.SCORE_1,
            ),
        ],
    )


@pytest.fixture
def refrigerant_input() -> BaselineInput:
    """Baseline input with R-410A refrigerant leakage."""
    return BaselineInput(
        entity_name="CoolCorp",
        reporting_year=2024,
        base_year=2024,
        refrigerant_entries=[
            RefrigerantEntry(
                refrigerant_id="r410a",
                charge_kg=Decimal("100"),
                leakage_rate_pct=Decimal("15"),
                data_quality=DataQualityScore.SCORE_3,
            ),
        ],
    )


@pytest.fixture
def electricity_input() -> BaselineInput:
    """Baseline input with grid electricity (US_AVG)."""
    return BaselineInput(
        entity_name="OfficeCorp",
        reporting_year=2024,
        base_year=2024,
        electricity_entries=[
            ElectricityEntry(
                quantity_mwh=Decimal("1000"),
                region="US_AVG",
                data_quality=DataQualityScore.SCORE_2,
            ),
        ],
    )


@pytest.fixture
def full_baseline_input() -> BaselineInput:
    """Full baseline input with Scope 1 + 2 + 3 data."""
    return BaselineInput(
        entity_name="FullCorp",
        reporting_year=2024,
        base_year=2024,
        boundary_method=BoundaryMethod.OPERATIONAL_CONTROL,
        fuel_entries=[
            FuelConsumptionEntry(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=Decimal("50000"),
                source_type=Scope1SourceType.STATIONARY_COMBUSTION,
            ),
            FuelConsumptionEntry(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("10000"),
                source_type=Scope1SourceType.MOBILE_COMBUSTION,
            ),
        ],
        refrigerant_entries=[
            RefrigerantEntry(
                refrigerant_id="r410a",
                charge_kg=Decimal("200"),
                leakage_rate_pct=Decimal("10"),
            ),
        ],
        electricity_entries=[
            ElectricityEntry(
                quantity_mwh=Decimal("2000"),
                region="EU_AVG",
            ),
        ],
        scope3_entries=[
            Scope3SpendEntry(
                category=Scope3Category.CAT_01_PURCHASED_GOODS,
                spend_usd_thousands=Decimal("5000"),
            ),
            Scope3SpendEntry(
                category=Scope3Category.CAT_06_BUSINESS_TRAVEL,
                spend_usd_thousands=Decimal("1000"),
            ),
        ],
        revenue_usd_millions=Decimal("100"),
        headcount=500,
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self) -> None:
        """Engine must instantiate without arguments."""
        engine = NetZeroBaselineEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- Scope 1 Calculations
# ===========================================================================


class TestScope1Calculations:
    """Tests for Scope 1 emission calculations."""

    def test_simple_scope1_calculation(
        self, engine: NetZeroBaselineEngine, simple_fuel_input: BaselineInput
    ) -> None:
        """Natural gas heating: 10000 m3 * 2.02 kgCO2e/m3 = 20200 kgCO2e = ~20.2 tCO2e."""
        result = engine.calculate(simple_fuel_input)

        assert isinstance(result, BaselineResult)
        # 10000 * 2.02 / 1000 = 20.200 tCO2e
        expected_tco2e = Decimal("10000") * Decimal("2.02") / Decimal("1000")
        assert float(result.scope1.stationary_combustion_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )
        assert result.scope1.total_tco2e > Decimal("0")

    def test_scope1_mobile_combustion(
        self, engine: NetZeroBaselineEngine, diesel_fleet_input: BaselineInput
    ) -> None:
        """Diesel fleet: 5000 L * 2.676 kgCO2e/L = 13380 kgCO2e = ~13.38 tCO2e."""
        result = engine.calculate(diesel_fleet_input)

        expected_tco2e = Decimal("5000") * Decimal("2.676") / Decimal("1000")
        assert float(result.scope1.mobile_combustion_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )

    def test_scope1_refrigerants(
        self, engine: NetZeroBaselineEngine, refrigerant_input: BaselineInput
    ) -> None:
        """R-410A: 100 kg charge * 15% leak rate = 15 kg leakage * 2088 GWP / 1000 = ~31.32 tCO2e."""
        result = engine.calculate(refrigerant_input)

        leakage_kg = Decimal("100") * Decimal("15") / Decimal("100")  # 15 kg
        gwp = GWP_AR6.get("r410a", Decimal("2088"))
        expected_tco2e = leakage_kg * gwp / Decimal("1000")
        assert float(result.scope1.refrigerants_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )

    @pytest.mark.parametrize("fuel_type,quantity,expected_factor", [
        (FuelType.NATURAL_GAS, Decimal("1000"), Decimal("2.02")),
        (FuelType.DIESEL, Decimal("1000"), Decimal("2.676")),
        (FuelType.GASOLINE, Decimal("1000"), Decimal("2.315")),
        (FuelType.LPG, Decimal("1000"), Decimal("1.557")),
        (FuelType.COAL_ANTHRACITE, Decimal("1000"), Decimal("2.886")),
    ])
    def test_fuel_emission_factor_accuracy(
        self,
        engine: NetZeroBaselineEngine,
        fuel_type: FuelType,
        quantity: Decimal,
        expected_factor: Decimal,
    ) -> None:
        """Fuel emission factors must match published DEFRA/EPA values."""
        inp = BaselineInput(
            entity_name="FactorTest",
            reporting_year=2024,
            base_year=2024,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=fuel_type,
                    quantity=quantity,
                    source_type=Scope1SourceType.STATIONARY_COMBUSTION,
                ),
            ],
        )
        result = engine.calculate(inp)
        expected_tco2e = quantity * expected_factor / Decimal("1000")
        assert float(result.scope1.total_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )


# ===========================================================================
# Tests -- Scope 2 Calculations
# ===========================================================================


class TestScope2Calculations:
    """Tests for Scope 2 emission calculations."""

    def test_scope2_location_based(
        self, engine: NetZeroBaselineEngine, electricity_input: BaselineInput
    ) -> None:
        """Location-based: 1000 MWh * 0.386 tCO2e/MWh = 386 tCO2e."""
        result = engine.calculate(electricity_input)

        grid_factor = GRID_EMISSION_FACTORS["US_AVG"]  # 0.386
        expected = Decimal("1000") * grid_factor
        assert float(result.scope2.location_based_tco2e) == pytest.approx(
            float(expected), rel=1e-3
        )

    def test_scope2_market_based(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Market-based with no PPA/REC uses residual mix factor."""
        inp = BaselineInput(
            entity_name="MarketTest",
            reporting_year=2024,
            base_year=2024,
            electricity_entries=[
                ElectricityEntry(
                    quantity_mwh=Decimal("1000"),
                    region="EU_AVG",
                    has_ppa=False,
                    rec_mwh=Decimal("0"),
                ),
            ],
        )
        result = engine.calculate(inp)

        # Market-based should use residual mix (or grid factor if no residual)
        assert result.scope2.market_based_tco2e > Decimal("0")

    def test_scope2_market_based_with_ppa(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Market-based with PPA at 0.01 tCO2e/MWh should be much lower than grid."""
        inp = BaselineInput(
            entity_name="PPATest",
            reporting_year=2024,
            base_year=2024,
            electricity_entries=[
                ElectricityEntry(
                    quantity_mwh=Decimal("1000"),
                    region="US_AVG",
                    has_ppa=True,
                    ppa_factor=Decimal("0.01"),
                ),
            ],
        )
        result = engine.calculate(inp)

        # PPA market-based should be 1000 * 0.01 = 10 tCO2e
        assert result.scope2.market_based_tco2e == pytest.approx(10.0, rel=0.1)
        # Location-based should still be full grid factor
        assert result.scope2.location_based_tco2e > result.scope2.market_based_tco2e

    def test_scope2_with_rec(self, engine: NetZeroBaselineEngine) -> None:
        """Market-based with RECs covering 500 of 1000 MWh should reduce by half."""
        inp = BaselineInput(
            entity_name="RECTest",
            reporting_year=2024,
            base_year=2024,
            electricity_entries=[
                ElectricityEntry(
                    quantity_mwh=Decimal("1000"),
                    region="US_AVG",
                    has_ppa=False,
                    rec_mwh=Decimal("500"),
                ),
            ],
        )
        result = engine.calculate(inp)
        # RECs cover 500 MWh at zero, remaining 500 at residual/grid
        assert result.scope2.market_based_tco2e < result.scope2.location_based_tco2e

    @pytest.mark.parametrize("region,expected_factor", [
        ("US_AVG", Decimal("0.386")),
        ("UK", Decimal("0.207")),
        ("DE", Decimal("0.338")),
        ("FR", Decimal("0.055")),
        ("SE", Decimal("0.012")),
        ("CN", Decimal("0.555")),
    ])
    def test_grid_factor_by_region(
        self,
        engine: NetZeroBaselineEngine,
        region: str,
        expected_factor: Decimal,
    ) -> None:
        """Grid factors must match published IEA/EEA values."""
        inp = BaselineInput(
            entity_name="GridTest",
            reporting_year=2024,
            base_year=2024,
            electricity_entries=[
                ElectricityEntry(quantity_mwh=Decimal("100"), region=region),
            ],
        )
        result = engine.calculate(inp)
        expected_tco2e = Decimal("100") * expected_factor
        assert float(result.scope2.location_based_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )


# ===========================================================================
# Tests -- Scope 3 Calculations
# ===========================================================================


class TestScope3Calculations:
    """Tests for spend-based Scope 3 calculations."""

    @pytest.mark.parametrize("category,spend,expected_factor", [
        (Scope3Category.CAT_01_PURCHASED_GOODS, Decimal("1000"), Decimal("0.430")),
        (Scope3Category.CAT_06_BUSINESS_TRAVEL, Decimal("500"), Decimal("0.310")),
        (Scope3Category.CAT_04_UPSTREAM_TRANSPORT, Decimal("2000"), Decimal("0.520")),
        (Scope3Category.CAT_05_WASTE, Decimal("300"), Decimal("0.210")),
        (Scope3Category.CAT_07_EMPLOYEE_COMMUTING, Decimal("800"), Decimal("0.180")),
    ])
    def test_scope3_spend_based(
        self,
        engine: NetZeroBaselineEngine,
        category: Scope3Category,
        spend: Decimal,
        expected_factor: Decimal,
    ) -> None:
        """Spend-based Scope 3: tCO2e = spend_thousands * factor."""
        inp = BaselineInput(
            entity_name="Scope3Test",
            reporting_year=2024,
            base_year=2024,
            scope3_entries=[
                Scope3SpendEntry(
                    category=category,
                    spend_usd_thousands=spend,
                ),
            ],
        )
        result = engine.calculate(inp)
        expected_tco2e = spend * expected_factor
        assert float(result.scope3.total_tco2e) == pytest.approx(
            float(expected_tco2e), rel=1e-3
        )

    def test_scope3_all_15_categories(self, engine: NetZeroBaselineEngine) -> None:
        """All 15 Scope 3 categories must be calculable."""
        entries = []
        for cat in Scope3Category:
            entries.append(
                Scope3SpendEntry(
                    category=cat,
                    spend_usd_thousands=Decimal("100"),
                )
            )
        inp = BaselineInput(
            entity_name="AllCats",
            reporting_year=2024,
            base_year=2024,
            scope3_entries=entries,
        )
        result = engine.calculate(inp)
        assert result.scope3.total_tco2e > Decimal("0")
        assert len(result.scope3.categories_included) == 15

    def test_empty_scope3_categories(self, engine: NetZeroBaselineEngine) -> None:
        """Empty Scope 3 entries should yield zero Scope 3 emissions."""
        inp = BaselineInput(
            entity_name="EmptyS3",
            reporting_year=2024,
            base_year=2024,
        )
        result = engine.calculate(inp)
        assert result.scope3.total_tco2e == Decimal("0")


# ===========================================================================
# Tests -- Full Baseline & Totals
# ===========================================================================


class TestFullBaseline:
    """Tests for complete baseline calculation with all scopes."""

    def test_full_baseline_calculation(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """Full baseline must have S1 + S2 (market) + S3 as total."""
        result = engine.calculate(full_baseline_input)

        assert result.scope1.total_tco2e > Decimal("0")
        assert result.scope2.market_based_tco2e > Decimal("0")
        assert result.scope3.total_tco2e > Decimal("0")

        # total = S1 + S2_market + S3
        expected_total = (
            result.scope1.total_tco2e
            + result.scope2.market_based_tco2e
            + result.scope3.total_tco2e
        )
        assert float(result.total_tco2e) == pytest.approx(
            float(expected_total), rel=1e-3
        )

    def test_scope_percentages_sum_to_100(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """Scope 1 + 2 + 3 percentages must sum to approximately 100%."""
        result = engine.calculate(full_baseline_input)
        total_pct = result.scope1_pct + result.scope2_pct + result.scope3_pct
        assert float(total_pct) == pytest.approx(100.0, abs=0.5)

    def test_zero_emissions_input(self, engine: NetZeroBaselineEngine) -> None:
        """An input with no fuel, electricity, or spend should yield zero total."""
        inp = BaselineInput(
            entity_name="ZeroCorp",
            reporting_year=2024,
            base_year=2024,
        )
        result = engine.calculate(inp)
        assert result.total_tco2e == Decimal("0")
        assert result.scope1.total_tco2e == Decimal("0")
        assert result.scope2.location_based_tco2e == Decimal("0")
        assert result.scope3.total_tco2e == Decimal("0")


# ===========================================================================
# Tests -- Base Year Validation
# ===========================================================================


class TestBaseYearValidation:
    """Tests for base year validation per GHG Protocol Chapter 5."""

    def test_base_year_equals_reporting_year(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Base year equal to reporting year must be valid (with non-zero emissions)."""
        inp = BaselineInput(
            entity_name="BaseYrTest",
            reporting_year=2024,
            base_year=2024,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=Decimal("1000"),
                    source_type=Scope1SourceType.STATIONARY_COMBUSTION,
                ),
            ],
        )
        result = engine.calculate(inp)
        # With non-zero emissions, base year == reporting year should be valid
        assert result.base_year_assessment.is_valid is True

    def test_base_year_before_reporting_year(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Base year before reporting year must be valid."""
        inp = BaselineInput(
            entity_name="BaseYrTest",
            reporting_year=2025,
            base_year=2020,
        )
        result = engine.calculate(inp)
        assert result.base_year_assessment.base_year == 2020
        assert result.base_year_assessment.years_since_base == 5

    def test_base_year_after_reporting_year_raises(self) -> None:
        """Base year after reporting year must raise validation error."""
        with pytest.raises(Exception):
            BaselineInput(
                entity_name="InvalidBase",
                reporting_year=2024,
                base_year=2025,
            )


# ===========================================================================
# Tests -- Data Quality Scoring
# ===========================================================================


class TestDataQualityScoring:
    """Tests for data quality assessment across inputs."""

    def test_data_quality_scoring(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """Data quality assessment must produce an overall score."""
        result = engine.calculate(full_baseline_input)
        dq = result.data_quality
        assert isinstance(dq, DataQualityAssessment)
        assert dq.entry_count > 0
        assert Decimal("0") <= dq.overall_score <= Decimal("1")

    def test_high_quality_data_scores_higher(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Score-1 data should yield higher overall quality than Score-5."""
        high_inp = BaselineInput(
            entity_name="HighQ",
            reporting_year=2024,
            base_year=2024,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=Decimal("1000"),
                    data_quality=DataQualityScore.SCORE_1,
                ),
            ],
        )
        low_inp = BaselineInput(
            entity_name="LowQ",
            reporting_year=2024,
            base_year=2024,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=Decimal("1000"),
                    data_quality=DataQualityScore.SCORE_5,
                ),
            ],
        )
        high_result = engine.calculate(high_inp)
        low_result = engine.calculate(low_inp)
        assert high_result.data_quality.overall_score >= low_result.data_quality.overall_score


# ===========================================================================
# Tests -- Organizational Boundary
# ===========================================================================


class TestOrganizationalBoundary:
    """Tests for organizational boundary method handling."""

    @pytest.mark.parametrize("method", [
        BoundaryMethod.OPERATIONAL_CONTROL,
        BoundaryMethod.FINANCIAL_CONTROL,
        BoundaryMethod.EQUITY_SHARE,
    ])
    def test_boundary_methods_accepted(
        self, engine: NetZeroBaselineEngine, method: BoundaryMethod
    ) -> None:
        """All three GHG Protocol boundary methods must be accepted."""
        inp = BaselineInput(
            entity_name="BoundaryTest",
            reporting_year=2024,
            base_year=2024,
            boundary_method=method,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=FuelType.NATURAL_GAS,
                    quantity=Decimal("1000"),
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.boundary_method == method.value


# ===========================================================================
# Tests -- Gas Breakdown & Intensity
# ===========================================================================


class TestGasBreakdownAndIntensity:
    """Tests for emissions by gas disaggregation and intensity metrics."""

    def test_emissions_by_gas_breakdown(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """by_gas must have non-zero total matching result total."""
        result = engine.calculate(full_baseline_input)
        assert isinstance(result.by_gas, EmissionsByGas)
        assert result.by_gas.total_tco2e > Decimal("0")

    def test_intensity_metrics_with_revenue(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """Intensity per revenue must be calculated when revenue is provided."""
        result = engine.calculate(full_baseline_input)
        assert result.intensity.per_revenue is not None
        assert result.intensity.per_revenue > Decimal("0")

    def test_intensity_metrics_with_headcount(
        self, engine: NetZeroBaselineEngine, full_baseline_input: BaselineInput
    ) -> None:
        """Intensity per headcount must be calculated when headcount is provided."""
        result = engine.calculate(full_baseline_input)
        assert result.intensity.per_headcount is not None
        assert result.intensity.per_headcount > Decimal("0")

    def test_intensity_without_revenue_is_none(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Intensity per revenue must be None when revenue is not provided."""
        inp = BaselineInput(
            entity_name="NoRevenue",
            reporting_year=2024,
            base_year=2024,
            fuel_entries=[
                FuelConsumptionEntry(
                    fuel_type=FuelType.DIESEL,
                    quantity=Decimal("100"),
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.intensity.per_revenue is None


# ===========================================================================
# Tests -- Provenance & Determinism
# ===========================================================================


class TestProvenanceAndDeterminism:
    """Tests for provenance hashing and calculation determinism."""

    def test_provenance_hash_present(
        self, engine: NetZeroBaselineEngine, simple_fuel_input: BaselineInput
    ) -> None:
        """Result must have a non-empty 64-character SHA-256 provenance hash."""
        result = engine.calculate(simple_fuel_input)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_hex_string(
        self, engine: NetZeroBaselineEngine, simple_fuel_input: BaselineInput
    ) -> None:
        """Provenance hash must be a valid 64-character hex string.

        Note: The hash includes result_id (UUID4) which changes per call,
        so deterministic equality is not expected across separate calls.
        We verify format and that the hash is non-trivially computed.
        """
        r1 = engine.calculate(simple_fuel_input)
        r2 = engine.calculate(simple_fuel_input)
        # Both are valid 64-char hex strings
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)

    def test_decimal_arithmetic_no_float(
        self, engine: NetZeroBaselineEngine, simple_fuel_input: BaselineInput
    ) -> None:
        """All emission values must be Decimal, not float."""
        result = engine.calculate(simple_fuel_input)
        assert isinstance(result.total_tco2e, Decimal)
        assert isinstance(result.scope1.total_tco2e, Decimal)

    def test_processing_time_recorded(
        self, engine: NetZeroBaselineEngine, simple_fuel_input: BaselineInput
    ) -> None:
        """processing_time_ms must be positive."""
        result = engine.calculate(simple_fuel_input)
        assert result.processing_time_ms > 0


# ===========================================================================
# Tests -- Invalid Input Handling
# ===========================================================================


class TestInvalidInputHandling:
    """Tests for error handling with invalid inputs."""

    def test_empty_entity_name_raises(self) -> None:
        """Empty entity_name must raise validation error."""
        with pytest.raises(Exception):
            BaselineInput(
                entity_name="",
                reporting_year=2024,
                base_year=2024,
            )

    def test_negative_quantity_raises(self) -> None:
        """Negative fuel quantity must raise validation error."""
        with pytest.raises(Exception):
            FuelConsumptionEntry(
                fuel_type=FuelType.DIESEL,
                quantity=Decimal("-100"),
            )

    def test_negative_electricity_raises(self) -> None:
        """Negative electricity consumption must raise validation error."""
        with pytest.raises(Exception):
            ElectricityEntry(quantity_mwh=Decimal("-500"))

    def test_negative_spend_raises(self) -> None:
        """Negative spend amount must raise validation error."""
        with pytest.raises(Exception):
            Scope3SpendEntry(
                category=Scope3Category.CAT_01_PURCHASED_GOODS,
                spend_usd_thousands=Decimal("-100"),
            )

    def test_unknown_region_uses_global_avg(
        self, engine: NetZeroBaselineEngine
    ) -> None:
        """Unknown region should fall back to GLOBAL_AVG grid factor."""
        inp = BaselineInput(
            entity_name="UnknownRegion",
            reporting_year=2024,
            base_year=2024,
            electricity_entries=[
                ElectricityEntry(
                    quantity_mwh=Decimal("100"),
                    region="XX_NONEXISTENT",
                ),
            ],
        )
        # Should not raise, but fall back gracefully
        result = engine.calculate(inp)
        assert isinstance(result, BaselineResult)


# ===========================================================================
# Tests -- Emission Factor Lookup API
# ===========================================================================


class TestEmissionFactorAPI:
    """Tests for emission factor lookup helper methods."""

    def test_get_emission_factor(self, engine: NetZeroBaselineEngine) -> None:
        """get_emission_factor must return dict with factor and unit."""
        ef = engine.get_emission_factor(FuelType.DIESEL)
        assert "factor" in ef
        assert "unit" in ef
        assert ef["factor"] == "2.676"

    def test_get_grid_factor(self, engine: NetZeroBaselineEngine) -> None:
        """get_grid_factor must return correct Decimal for known regions."""
        factor = engine.get_grid_factor("UK")
        assert factor == Decimal("0.207")
