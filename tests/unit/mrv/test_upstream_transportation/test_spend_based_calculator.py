"""
Unit tests for SpendBasedCalculatorEngine.

Tests spend-based emission calculation using EEIO models (USEEIO, EXIOBASE, DEFRA),
currency conversion, CPI deflation, margin removal, NAICS/NACE code resolution.

Tests:
- USEEIO calculations (trucking, air, rail, maritime, warehousing)
- EXIOBASE calculations (EU road, rail, US air)
- DEFRA calculations (road, air)
- Currency conversion (EUR→USD, GBP→USD)
- CPI deflation
- Margin removal (trucking 15%, air 20%)
- NAICS/NACE code resolution and mapping
- Spend classification
- Batch calculations
- Aggregations
- Data quality scoring
- Validation
- Uncertainty estimation
- Edge cases (zero spend, invalid codes, etc.)
"""

import pytest
from decimal import Decimal
from datetime import date
from typing import Dict, List, Any

from greenlang.mrv.upstream_transportation.engines.spend_based_calculator import (
    SpendBasedCalculatorEngine,
    SpendInput,
    SpendResult,
    EEIOModel,
    NAICSCode,
    NACECode,
    CurrencyCode,
)
from greenlang.mrv.upstream_transportation.models import (
    TransportMode,
    EmissionScope,
    DataQualityTier,
)


@pytest.fixture
def engine():
    """Create SpendBasedCalculatorEngine instance."""
    return SpendBasedCalculatorEngine()


@pytest.fixture
def useeio_trucking_input():
    """USEEIO trucking spend input."""
    return SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
        transport_mode=TransportMode.ROAD,
        description="Freight trucking services",
    )


@pytest.fixture
def useeio_air_freight_input():
    """USEEIO air freight spend input."""
    return SpendInput(
        amount=Decimal("50000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.AIR_FREIGHT,
        transport_mode=TransportMode.AIR,
        description="Air cargo services",
    )


@pytest.fixture
def exiobase_eu_road_input():
    """EXIOBASE EU road transport spend input."""
    return SpendInput(
        amount=Decimal("25000.00"),
        currency=CurrencyCode.EUR,
        year=2023,
        model=EEIOModel.EXIOBASE,
        nace_code=NACECode.ROAD_FREIGHT,
        transport_mode=TransportMode.ROAD,
        region="EU27",
        description="EU road freight",
    )


# ============================================================================
# USEEIO Calculations
# ============================================================================


def test_calculate_useeio_trucking(engine, useeio_trucking_input):
    """Test USEEIO trucking calculation."""
    result = engine.calculate(useeio_trucking_input)

    assert isinstance(result, SpendResult)
    assert result.co2_kg > Decimal("0")
    assert result.ch4_kg >= Decimal("0")
    assert result.n2o_kg >= Decimal("0")
    assert result.co2e_kg > Decimal("0")
    assert result.model == EEIOModel.USEEIO
    assert result.naics_code == NAICSCode.TRUCKING_LOCAL
    assert result.emission_factor_kg_per_usd > Decimal("0")
    # USEEIO trucking EF ~0.5-0.8 kg CO2e/USD
    assert Decimal("0.3") < result.emission_factor_kg_per_usd < Decimal("1.0")
    assert result.data_quality_tier == DataQualityTier.TIER_3  # Spend-based is Tier 3


def test_calculate_useeio_air_freight(engine, useeio_air_freight_input):
    """Test USEEIO air freight calculation."""
    result = engine.calculate(useeio_air_freight_input)

    assert result.co2e_kg > Decimal("0")
    assert result.naics_code == NAICSCode.AIR_FREIGHT
    # Air freight has higher EF than trucking
    assert result.emission_factor_kg_per_usd > Decimal("0.5")
    assert result.transport_mode == TransportMode.AIR


def test_calculate_useeio_rail(engine):
    """Test USEEIO rail calculation."""
    rail_input = SpendInput(
        amount=Decimal("15000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.RAIL_FREIGHT,
        transport_mode=TransportMode.RAIL,
    )
    result = engine.calculate(rail_input)

    assert result.co2e_kg > Decimal("0")
    # Rail has lower EF than trucking
    assert result.emission_factor_kg_per_usd < Decimal("0.5")


def test_calculate_useeio_deep_sea(engine):
    """Test USEEIO deep sea shipping calculation."""
    maritime_input = SpendInput(
        amount=Decimal("30000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.DEEP_SEA_FREIGHT,
        transport_mode=TransportMode.MARITIME,
    )
    result = engine.calculate(maritime_input)

    assert result.co2e_kg > Decimal("0")
    # Maritime has lowest EF per USD
    assert result.emission_factor_kg_per_usd < Decimal("0.4")


def test_calculate_useeio_warehousing(engine):
    """Test USEEIO warehousing calculation."""
    warehouse_input = SpendInput(
        amount=Decimal("8000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.WAREHOUSING_STORAGE,
        transport_mode=None,  # Warehousing has no mode
        description="Warehouse storage services",
    )
    result = engine.calculate(warehouse_input)

    assert result.co2e_kg > Decimal("0")
    assert result.naics_code == NAICSCode.WAREHOUSING_STORAGE
    assert result.transport_mode is None


# ============================================================================
# EXIOBASE Calculations
# ============================================================================


def test_calculate_exiobase_eu_road(engine, exiobase_eu_road_input):
    """Test EXIOBASE EU road transport calculation."""
    result = engine.calculate(exiobase_eu_road_input)

    assert result.co2e_kg > Decimal("0")
    assert result.model == EEIOModel.EXIOBASE
    assert result.nace_code == NACECode.ROAD_FREIGHT
    assert result.region == "EU27"
    # EXIOBASE uses EUR, should be converted to USD baseline
    assert result.currency_converted_to_usd is True


def test_calculate_exiobase_eu_rail(engine):
    """Test EXIOBASE EU rail calculation."""
    rail_input = SpendInput(
        amount=Decimal("18000.00"),
        currency=CurrencyCode.EUR,
        year=2023,
        model=EEIOModel.EXIOBASE,
        nace_code=NACECode.RAIL_FREIGHT,
        transport_mode=TransportMode.RAIL,
        region="EU27",
    )
    result = engine.calculate(rail_input)

    assert result.co2e_kg > Decimal("0")
    assert result.nace_code == NACECode.RAIL_FREIGHT


def test_calculate_exiobase_us_air(engine):
    """Test EXIOBASE US air transport calculation."""
    air_input = SpendInput(
        amount=Decimal("40000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.EXIOBASE,
        naics_code=NAICSCode.AIR_FREIGHT,
        transport_mode=TransportMode.AIR,
        region="US",
    )
    result = engine.calculate(air_input)

    assert result.co2e_kg > Decimal("0")
    assert result.region == "US"


# ============================================================================
# DEFRA Calculations
# ============================================================================


def test_calculate_defra_road(engine):
    """Test DEFRA road transport calculation."""
    road_input = SpendInput(
        amount=Decimal("12000.00"),
        currency=CurrencyCode.GBP,
        year=2023,
        model=EEIOModel.DEFRA,
        transport_mode=TransportMode.ROAD,
        region="UK",
    )
    result = engine.calculate(road_input)

    assert result.co2e_kg > Decimal("0")
    assert result.model == EEIOModel.DEFRA
    assert result.currency == CurrencyCode.GBP


def test_calculate_defra_air(engine):
    """Test DEFRA air freight calculation."""
    air_input = SpendInput(
        amount=Decimal("35000.00"),
        currency=CurrencyCode.GBP,
        year=2023,
        model=EEIOModel.DEFRA,
        transport_mode=TransportMode.AIR,
        region="UK",
    )
    result = engine.calculate(air_input)

    assert result.co2e_kg > Decimal("0")
    assert result.transport_mode == TransportMode.AIR


# ============================================================================
# Currency Conversion
# ============================================================================


def test_convert_currency_eur_to_usd(engine):
    """Test EUR to USD conversion."""
    eur_amount = Decimal("10000.00")
    usd_amount = engine.convert_currency(
        amount=eur_amount,
        from_currency=CurrencyCode.EUR,
        to_currency=CurrencyCode.USD,
        year=2023,
    )

    # 2023 average ~1.08 EUR/USD
    assert usd_amount > eur_amount  # USD > EUR
    assert Decimal("10500") < usd_amount < Decimal("11500")


def test_convert_currency_gbp_to_usd(engine):
    """Test GBP to USD conversion."""
    gbp_amount = Decimal("10000.00")
    usd_amount = engine.convert_currency(
        amount=gbp_amount,
        from_currency=CurrencyCode.GBP,
        to_currency=CurrencyCode.USD,
        year=2023,
    )

    # 2023 average ~1.24 GBP/USD
    assert usd_amount > gbp_amount  # USD > GBP
    assert Decimal("12000") < usd_amount < Decimal("13000")


# ============================================================================
# CPI Deflation
# ============================================================================


def test_get_cpi_deflator(engine):
    """Test CPI deflator retrieval."""
    # Deflate 2020 USD to 2023 USD
    deflator = engine.get_cpi_deflator(from_year=2020, to_year=2023, region="US")

    # ~3% inflation per year = ~9% over 3 years
    assert deflator > Decimal("1.05")
    assert deflator < Decimal("1.15")


# ============================================================================
# Margin Removal
# ============================================================================


def test_apply_margin_removal_trucking_15pct(engine):
    """Test margin removal for trucking (15%)."""
    spend = Decimal("10000.00")
    adjusted = engine.apply_margin_removal(
        amount=spend,
        naics_code=NAICSCode.TRUCKING_LOCAL,
    )

    # 15% margin removed
    expected = spend * Decimal("0.85")
    assert adjusted == expected


def test_apply_margin_removal_air_20pct(engine):
    """Test margin removal for air freight (20%)."""
    spend = Decimal("50000.00")
    adjusted = engine.apply_margin_removal(
        amount=spend,
        naics_code=NAICSCode.AIR_FREIGHT,
    )

    # 20% margin removed
    expected = spend * Decimal("0.80")
    assert adjusted == expected


# ============================================================================
# EEIO Factor Retrieval
# ============================================================================


def test_get_eeio_factor_by_naics(engine):
    """Test EEIO factor retrieval by NAICS code."""
    factor = engine.get_eeio_factor(
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
        year=2023,
    )

    assert factor > Decimal("0")
    assert factor < Decimal("2.0")  # kg CO2e/USD


# ============================================================================
# NAICS/NACE Code Resolution
# ============================================================================


def test_resolve_naics_code_trucking(engine):
    """Test NAICS code resolution for trucking."""
    naics = engine.resolve_naics_code(
        description="General freight trucking long distance",
        transport_mode=TransportMode.ROAD,
    )

    assert naics in [
        NAICSCode.TRUCKING_LOCAL,
        NAICSCode.TRUCKING_LONG_DISTANCE,
    ]


def test_resolve_naics_code_air(engine):
    """Test NAICS code resolution for air freight."""
    naics = engine.resolve_naics_code(
        description="Air cargo charter service",
        transport_mode=TransportMode.AIR,
    )

    assert naics == NAICSCode.AIR_FREIGHT


def test_map_naics_to_nace(engine):
    """Test NAICS to NACE mapping."""
    nace = engine.map_naics_to_nace(NAICSCode.TRUCKING_LOCAL)

    assert nace == NACECode.ROAD_FREIGHT


def test_map_nace_to_naics(engine):
    """Test NACE to NAICS mapping."""
    naics = engine.map_nace_to_naics(NACECode.ROAD_FREIGHT)

    assert naics in [
        NAICSCode.TRUCKING_LOCAL,
        NAICSCode.TRUCKING_LONG_DISTANCE,
    ]


# ============================================================================
# Spend Classification
# ============================================================================


def test_classify_transport_spend_trucking(engine):
    """Test spend classification for trucking."""
    classification = engine.classify_transport_spend(
        description="Freight transportation - truck delivery",
        amount=Decimal("5000.00"),
    )

    assert classification["transport_mode"] == TransportMode.ROAD
    assert classification["naics_code"] in [
        NAICSCode.TRUCKING_LOCAL,
        NAICSCode.TRUCKING_LONG_DISTANCE,
    ]
    assert classification["confidence"] > 0.7


def test_classify_transport_spend_air_freight(engine):
    """Test spend classification for air freight."""
    classification = engine.classify_transport_spend(
        description="Air cargo - express international shipping",
        amount=Decimal("25000.00"),
    )

    assert classification["transport_mode"] == TransportMode.AIR
    assert classification["naics_code"] == NAICSCode.AIR_FREIGHT


def test_classify_transport_spend_warehouse(engine):
    """Test spend classification for warehousing."""
    classification = engine.classify_transport_spend(
        description="Warehouse storage and handling fees",
        amount=Decimal("3000.00"),
    )

    assert classification["transport_mode"] is None
    assert classification["naics_code"] == NAICSCode.WAREHOUSING_STORAGE


# ============================================================================
# Batch Calculations
# ============================================================================


def test_batch_calculate(engine):
    """Test batch spend calculations."""
    inputs = [
        SpendInput(
            amount=Decimal("10000.00"),
            currency=CurrencyCode.USD,
            year=2023,
            model=EEIOModel.USEEIO,
            naics_code=NAICSCode.TRUCKING_LOCAL,
            transport_mode=TransportMode.ROAD,
        ),
        SpendInput(
            amount=Decimal("20000.00"),
            currency=CurrencyCode.USD,
            year=2023,
            model=EEIOModel.USEEIO,
            naics_code=NAICSCode.AIR_FREIGHT,
            transport_mode=TransportMode.AIR,
        ),
        SpendInput(
            amount=Decimal("5000.00"),
            currency=CurrencyCode.USD,
            year=2023,
            model=EEIOModel.USEEIO,
            naics_code=NAICSCode.WAREHOUSING_STORAGE,
        ),
    ]

    results = engine.batch_calculate(inputs)

    assert len(results) == 3
    assert all(isinstance(r, SpendResult) for r in results)
    assert all(r.co2e_kg > Decimal("0") for r in results)


# ============================================================================
# Aggregations
# ============================================================================


def test_aggregate_by_sector(engine):
    """Test aggregation by sector (NAICS code)."""
    results = [
        SpendResult(
            co2_kg=Decimal("500"),
            ch4_kg=Decimal("0.5"),
            n2o_kg=Decimal("0.05"),
            co2e_kg=Decimal("550"),
            naics_code=NAICSCode.TRUCKING_LOCAL,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("0.55"),
        ),
        SpendResult(
            co2_kg=Decimal("300"),
            ch4_kg=Decimal("0.3"),
            n2o_kg=Decimal("0.03"),
            co2e_kg=Decimal("330"),
            naics_code=NAICSCode.TRUCKING_LOCAL,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("0.55"),
        ),
        SpendResult(
            co2_kg=Decimal("1000"),
            ch4_kg=Decimal("1.0"),
            n2o_kg=Decimal("0.1"),
            co2e_kg=Decimal("1100"),
            naics_code=NAICSCode.AIR_FREIGHT,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("1.1"),
        ),
    ]

    aggregated = engine.aggregate_by_sector(results)

    assert NAICSCode.TRUCKING_LOCAL in aggregated
    assert NAICSCode.AIR_FREIGHT in aggregated
    assert aggregated[NAICSCode.TRUCKING_LOCAL]["co2e_kg"] == Decimal("880")
    assert aggregated[NAICSCode.AIR_FREIGHT]["co2e_kg"] == Decimal("1100")


def test_aggregate_by_mode(engine):
    """Test aggregation by transport mode."""
    results = [
        SpendResult(
            co2e_kg=Decimal("500"),
            transport_mode=TransportMode.ROAD,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("0.5"),
        ),
        SpendResult(
            co2e_kg=Decimal("300"),
            transport_mode=TransportMode.ROAD,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("0.5"),
        ),
        SpendResult(
            co2e_kg=Decimal("1200"),
            transport_mode=TransportMode.AIR,
            model=EEIOModel.USEEIO,
            emission_factor_kg_per_usd=Decimal("1.2"),
        ),
    ]

    aggregated = engine.aggregate_by_mode(results)

    assert TransportMode.ROAD in aggregated
    assert TransportMode.AIR in aggregated
    assert aggregated[TransportMode.ROAD]["co2e_kg"] == Decimal("800")
    assert aggregated[TransportMode.AIR]["co2e_kg"] == Decimal("1200")


# ============================================================================
# Data Quality
# ============================================================================


def test_data_quality_score_always_low(engine, useeio_trucking_input):
    """Test that spend-based always has low data quality."""
    result = engine.calculate(useeio_trucking_input)

    # Spend-based is Tier 3 (low quality)
    assert result.data_quality_tier == DataQualityTier.TIER_3
    assert result.data_quality_score < 40.0  # Low score


# ============================================================================
# Validation
# ============================================================================


def test_validate_spend_input_valid(engine, useeio_trucking_input):
    """Test validation of valid spend input."""
    is_valid, errors = engine.validate_input(useeio_trucking_input)

    assert is_valid is True
    assert len(errors) == 0


def test_validate_spend_input_negative(engine):
    """Test validation rejects negative spend."""
    invalid_input = SpendInput(
        amount=Decimal("-5000.00"),  # Negative
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
    )

    is_valid, errors = engine.validate_input(invalid_input)

    assert is_valid is False
    assert any("negative" in e.lower() for e in errors)


# ============================================================================
# Available Sectors
# ============================================================================


def test_get_available_sectors_useeio(engine):
    """Test getting available sectors for USEEIO."""
    sectors = engine.get_available_sectors(EEIOModel.USEEIO)

    assert len(sectors) > 0
    assert any(s["naics_code"] == NAICSCode.TRUCKING_LOCAL for s in sectors)
    assert any(s["naics_code"] == NAICSCode.AIR_FREIGHT for s in sectors)


# ============================================================================
# Uncertainty
# ============================================================================


def test_estimate_uncertainty_high(engine, useeio_trucking_input):
    """Test uncertainty estimation (spend-based has high uncertainty)."""
    result = engine.calculate(useeio_trucking_input)
    uncertainty = engine.estimate_uncertainty(result)

    # Spend-based has >50% uncertainty
    assert uncertainty > 50.0
    assert uncertainty < 150.0  # But not absurdly high


# ============================================================================
# Precision
# ============================================================================


def test_decimal_precision(engine, useeio_trucking_input):
    """Test Decimal precision maintained."""
    result = engine.calculate(useeio_trucking_input)

    assert isinstance(result.co2_kg, Decimal)
    assert isinstance(result.ch4_kg, Decimal)
    assert isinstance(result.n2o_kg, Decimal)
    assert isinstance(result.co2e_kg, Decimal)


# ============================================================================
# Edge Cases
# ============================================================================


def test_zero_spend_raises(engine):
    """Test zero spend raises error."""
    zero_input = SpendInput(
        amount=Decimal("0.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
    )

    with pytest.raises(ValueError, match="amount must be positive"):
        engine.calculate(zero_input)


def test_invalid_naics_raises(engine):
    """Test invalid NAICS code raises error."""
    invalid_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code="999999",  # Invalid
    )

    with pytest.raises(ValueError, match="invalid NAICS code"):
        engine.calculate(invalid_input)


def test_spend_higher_uncertainty_than_distance(engine):
    """Test spend-based has higher uncertainty than distance-based."""
    # This is a conceptual test - spend-based Tier 3 vs distance-based Tier 2
    spend_result = SpendResult(
        co2e_kg=Decimal("500"),
        data_quality_tier=DataQualityTier.TIER_3,
        model=EEIOModel.USEEIO,
        emission_factor_kg_per_usd=Decimal("0.5"),
    )

    spend_uncertainty = engine.estimate_uncertainty(spend_result)

    # Spend-based >50%, distance-based ~20-30%
    assert spend_uncertainty > 50.0


def test_future_year_uses_latest_factors(engine):
    """Test future year defaults to latest available factors."""
    future_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2030,  # Future
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
    )

    result = engine.calculate(future_input)

    # Should use 2023 factors (latest)
    assert result.factor_year == 2023


def test_old_year_deflated(engine):
    """Test old year spend is CPI-deflated."""
    old_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2015,  # Old
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
    )

    result = engine.calculate(old_input)

    # 2015 → 2023 deflation applied
    assert result.cpi_adjusted is True
    assert result.deflator > Decimal("1.0")


def test_exiobase_requires_region(engine):
    """Test EXIOBASE requires region specification."""
    no_region_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.EUR,
        year=2023,
        model=EEIOModel.EXIOBASE,
        nace_code=NACECode.ROAD_FREIGHT,
        # Missing region
    )

    with pytest.raises(ValueError, match="region required for EXIOBASE"):
        engine.calculate(no_region_input)


def test_mixed_naics_nace_raises(engine):
    """Test providing both NAICS and NACE raises error."""
    mixed_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
        nace_code=NACECode.ROAD_FREIGHT,  # Both provided
    )

    with pytest.raises(ValueError, match="provide NAICS or NACE, not both"):
        engine.calculate(mixed_input)


def test_useeio_with_nace_auto_converts(engine):
    """Test USEEIO with NACE code auto-converts to NAICS."""
    nace_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        nace_code=NACECode.ROAD_FREIGHT,  # NACE provided
        transport_mode=TransportMode.ROAD,
    )

    result = engine.calculate(nace_input)

    # Should auto-convert NACE → NAICS
    assert result.naics_code in [
        NAICSCode.TRUCKING_LOCAL,
        NAICSCode.TRUCKING_LONG_DISTANCE,
    ]


def test_very_large_spend(engine):
    """Test very large spend amount."""
    large_input = SpendInput(
        amount=Decimal("10000000.00"),  # $10M
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.AIR_FREIGHT,
    )

    result = engine.calculate(large_input)

    assert result.co2e_kg > Decimal("1000000")  # >1000 tonnes


def test_margin_removal_optional(engine):
    """Test margin removal can be disabled."""
    no_margin_input = SpendInput(
        amount=Decimal("10000.00"),
        currency=CurrencyCode.USD,
        year=2023,
        model=EEIOModel.USEEIO,
        naics_code=NAICSCode.TRUCKING_LOCAL,
        apply_margin_removal=False,
    )

    result = engine.calculate(no_margin_input)

    # Full amount used, no margin removed
    assert result.margin_removed is False


def test_classify_with_low_confidence_warns(engine):
    """Test classification with low confidence issues warning."""
    classification = engine.classify_transport_spend(
        description="Misc services",  # Ambiguous
        amount=Decimal("1000.00"),
    )

    # Low confidence
    assert classification["confidence"] < 0.6
    assert "warnings" in classification
