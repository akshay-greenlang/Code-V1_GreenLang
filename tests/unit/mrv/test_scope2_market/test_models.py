# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-010 Scope 2 Market-Based Emissions Agent Data Models.

Tests all 20 enumerations, 7 constant tables (GWP_VALUES, RESIDUAL_MIX_FACTORS,
ENERGY_SOURCE_EF, SUPPLIER_DEFAULT_EF, INSTRUMENT_QUALITY_WEIGHTS,
VINTAGE_VALIDITY_YEARS, UNIT_CONVERSIONS), module-level constants, and
20 Pydantic data models with field validators.

Target: 150+ tests, 85%+ coverage of models.py.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Conditional import guard
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope2_market.models import (
        # Enumerations (20)
        InstrumentType,
        InstrumentStatus,
        EnergySource,
        EnergyType,
        EnergyUnit,
        CalculationMethod,
        EmissionGas,
        GWPSource,
        QualityCriterion,
        TrackingSystem,
        ResidualMixSource,
        FacilityType,
        ComplianceStatus,
        CoverageStatus,
        ReportingPeriod,
        ContractType,
        DataQualityTier,
        DualReportingStatus,
        AllocationMethod,
        ConsumptionDataSource,
        # Constants
        GWP_VALUES,
        RESIDUAL_MIX_FACTORS,
        ENERGY_SOURCE_EF,
        SUPPLIER_DEFAULT_EF,
        INSTRUMENT_QUALITY_WEIGHTS,
        VINTAGE_VALIDITY_YEARS,
        UNIT_CONVERSIONS,
        # Module constants
        VERSION,
        MAX_CALCULATIONS_PER_BATCH,
        MAX_GASES_PER_RESULT,
        MAX_TRACE_STEPS,
        MAX_FACILITIES_PER_TENANT,
        MAX_ENERGY_PURCHASES_PER_CALC,
        MAX_INSTRUMENTS_PER_PURCHASE,
        DEFAULT_MONTE_CARLO_ITERATIONS,
        DEFAULT_CONFIDENCE_LEVEL,
        DEFAULT_QUALITY_THRESHOLD,
        TABLE_PREFIX,
        # Pydantic models
        ContractualInstrument,
        InstrumentQualityAssessment,
        SupplierEmissionFactor,
        ResidualMixFactor,
        EnergyPurchase,
        FacilityInfo,
        AllocationResult,
        CoveredEmissionResult,
        UncoveredEmissionResult,
        GasEmissionDetail,
        MarketBasedResult,
        DualReportingResult,
        CalculationRequest,
        BatchCalculationRequest,
        BatchCalculationResult,
        ComplianceCheckResult,
        InstrumentValidationResult,
        UncertaintyRequest,
        UncertaintyResult,
        AggregationResult,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(not MODELS_AVAILABLE, reason="models not available")


# ===========================================================================
# Enumeration Tests (20 enums)
# ===========================================================================


@_SKIP
class TestInstrumentType:
    """Tests for the InstrumentType enumeration."""

    def test_member_count(self):
        """InstrumentType has exactly 10 members."""
        assert len(InstrumentType) == 10

    @pytest.mark.parametrize("member,expected", [
        (InstrumentType.PPA, "ppa"),
        (InstrumentType.REC, "rec"),
        (InstrumentType.GO, "go"),
        (InstrumentType.REGO, "rego"),
        (InstrumentType.I_REC, "i_rec"),
        (InstrumentType.T_REC, "t_rec"),
        (InstrumentType.J_CREDIT, "j_credit"),
        (InstrumentType.LGC, "lgc"),
        (InstrumentType.GREEN_TARIFF, "green_tariff"),
        (InstrumentType.SUPPLIER_SPECIFIC, "supplier_specific"),
    ])
    def test_values(self, member, expected):
        """Each InstrumentType member has the correct string value."""
        assert member.value == expected

    def test_membership_from_string(self):
        """InstrumentType can be looked up from string value."""
        assert InstrumentType("rec") == InstrumentType.REC

    def test_all_are_str_enum(self):
        """All InstrumentType members are string-compatible."""
        for member in InstrumentType:
            assert isinstance(member.value, str)


@_SKIP
class TestInstrumentStatus:
    """Tests for the InstrumentStatus enumeration."""

    def test_member_count(self):
        """InstrumentStatus has exactly 5 members."""
        assert len(InstrumentStatus) == 5

    @pytest.mark.parametrize("member,expected", [
        (InstrumentStatus.ACTIVE, "active"),
        (InstrumentStatus.RETIRED, "retired"),
        (InstrumentStatus.EXPIRED, "expired"),
        (InstrumentStatus.CANCELLED, "cancelled"),
        (InstrumentStatus.PENDING, "pending"),
    ])
    def test_values(self, member, expected):
        """Each InstrumentStatus member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestEnergySource:
    """Tests for the EnergySource enumeration."""

    def test_member_count(self):
        """EnergySource has exactly 11 members."""
        assert len(EnergySource) == 11

    @pytest.mark.parametrize("member,expected", [
        (EnergySource.SOLAR, "solar"),
        (EnergySource.WIND, "wind"),
        (EnergySource.HYDRO, "hydro"),
        (EnergySource.NUCLEAR, "nuclear"),
        (EnergySource.BIOMASS, "biomass"),
        (EnergySource.GEOTHERMAL, "geothermal"),
        (EnergySource.NATURAL_GAS_CCGT, "natural_gas_ccgt"),
        (EnergySource.NATURAL_GAS_OCGT, "natural_gas_ocgt"),
        (EnergySource.COAL, "coal"),
        (EnergySource.OIL, "oil"),
        (EnergySource.MIXED, "mixed"),
    ])
    def test_values(self, member, expected):
        """Each EnergySource member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestEnergyType:
    """Tests for the EnergyType enumeration."""

    def test_member_count(self):
        """EnergyType has exactly 4 members."""
        assert len(EnergyType) == 4

    @pytest.mark.parametrize("member,expected", [
        (EnergyType.ELECTRICITY, "electricity"),
        (EnergyType.STEAM, "steam"),
        (EnergyType.HEATING, "heating"),
        (EnergyType.COOLING, "cooling"),
    ])
    def test_values(self, member, expected):
        """Each EnergyType member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestEnergyUnit:
    """Tests for the EnergyUnit enumeration."""

    def test_member_count(self):
        """EnergyUnit has exactly 5 members."""
        assert len(EnergyUnit) == 5

    @pytest.mark.parametrize("member,expected", [
        (EnergyUnit.KWH, "kwh"),
        (EnergyUnit.MWH, "mwh"),
        (EnergyUnit.GJ, "gj"),
        (EnergyUnit.MMBTU, "mmbtu"),
        (EnergyUnit.THERMS, "therms"),
    ])
    def test_values(self, member, expected):
        """Each EnergyUnit member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestCalculationMethod:
    """Tests for the CalculationMethod enumeration."""

    def test_member_count(self):
        """CalculationMethod has exactly 4 members."""
        assert len(CalculationMethod) == 4

    @pytest.mark.parametrize("member,expected", [
        (CalculationMethod.INSTRUMENT_BASED, "instrument_based"),
        (CalculationMethod.SUPPLIER_SPECIFIC, "supplier_specific"),
        (CalculationMethod.RESIDUAL_MIX, "residual_mix"),
        (CalculationMethod.HYBRID, "hybrid"),
    ])
    def test_values(self, member, expected):
        """Each CalculationMethod member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestEmissionGas:
    """Tests for the EmissionGas enumeration."""

    def test_member_count(self):
        """EmissionGas has exactly 3 members."""
        assert len(EmissionGas) == 3

    @pytest.mark.parametrize("member,expected", [
        (EmissionGas.CO2, "CO2"),
        (EmissionGas.CH4, "CH4"),
        (EmissionGas.N2O, "N2O"),
    ])
    def test_values(self, member, expected):
        """Each EmissionGas member has the correct uppercase value."""
        assert member.value == expected


@_SKIP
class TestGWPSource:
    """Tests for the GWPSource enumeration."""

    def test_member_count(self):
        """GWPSource has exactly 4 members."""
        assert len(GWPSource) == 4

    @pytest.mark.parametrize("member,expected", [
        (GWPSource.AR4, "AR4"),
        (GWPSource.AR5, "AR5"),
        (GWPSource.AR6, "AR6"),
        (GWPSource.AR6_20YR, "AR6_20YR"),
    ])
    def test_values(self, member, expected):
        """Each GWPSource member has the correct value."""
        assert member.value == expected


@_SKIP
class TestQualityCriterion:
    """Tests for the QualityCriterion enumeration."""

    def test_member_count(self):
        """QualityCriterion has exactly 7 members."""
        assert len(QualityCriterion) == 7

    @pytest.mark.parametrize("member,expected", [
        (QualityCriterion.UNIQUE_CLAIM, "unique_claim"),
        (QualityCriterion.ASSOCIATED_DELIVERY, "associated_delivery"),
        (QualityCriterion.TEMPORAL_MATCH, "temporal_match"),
        (QualityCriterion.GEOGRAPHIC_MATCH, "geographic_match"),
        (QualityCriterion.NO_DOUBLE_COUNT, "no_double_count"),
        (QualityCriterion.RECOGNIZED_REGISTRY, "recognized_registry"),
        (QualityCriterion.REPRESENTS_GENERATION, "represents_generation"),
    ])
    def test_values(self, member, expected):
        """Each QualityCriterion member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestTrackingSystem:
    """Tests for the TrackingSystem enumeration."""

    def test_member_count(self):
        """TrackingSystem has exactly 8 members."""
        assert len(TrackingSystem) == 8

    @pytest.mark.parametrize("member,expected", [
        (TrackingSystem.GREEN_E, "green_e"),
        (TrackingSystem.AIB_EECS, "aib_eecs"),
        (TrackingSystem.OFGEM, "ofgem"),
        (TrackingSystem.I_REC_STANDARD, "i_rec_standard"),
        (TrackingSystem.M_RETS, "m_rets"),
        (TrackingSystem.NAR, "nar"),
        (TrackingSystem.WREGIS, "wregis"),
        (TrackingSystem.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each TrackingSystem member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestResidualMixSource:
    """Tests for the ResidualMixSource enumeration."""

    def test_member_count(self):
        """ResidualMixSource has exactly 5 members."""
        assert len(ResidualMixSource) == 5

    @pytest.mark.parametrize("member,expected", [
        (ResidualMixSource.AIB, "aib"),
        (ResidualMixSource.GREEN_E, "green_e"),
        (ResidualMixSource.NATIONAL, "national"),
        (ResidualMixSource.ESTIMATED, "estimated"),
        (ResidualMixSource.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each ResidualMixSource member has the correct string value."""
        assert member.value == expected


@_SKIP
class TestFacilityType:
    """Tests for the FacilityType enumeration."""

    def test_member_count(self):
        """FacilityType has exactly 8 members."""
        assert len(FacilityType) == 8

    @pytest.mark.parametrize("member,expected", [
        (FacilityType.OFFICE, "office"),
        (FacilityType.WAREHOUSE, "warehouse"),
        (FacilityType.MANUFACTURING, "manufacturing"),
        (FacilityType.RETAIL, "retail"),
        (FacilityType.DATA_CENTER, "data_center"),
        (FacilityType.HOSPITAL, "hospital"),
        (FacilityType.SCHOOL, "school"),
        (FacilityType.OTHER, "other"),
    ])
    def test_values(self, member, expected):
        """Each FacilityType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestComplianceStatus:
    """Tests for the ComplianceStatus enumeration."""

    def test_member_count(self):
        """ComplianceStatus has exactly 4 members."""
        assert len(ComplianceStatus) == 4

    @pytest.mark.parametrize("member,expected", [
        (ComplianceStatus.COMPLIANT, "compliant"),
        (ComplianceStatus.NON_COMPLIANT, "non_compliant"),
        (ComplianceStatus.PARTIAL, "partial"),
        (ComplianceStatus.NOT_ASSESSED, "not_assessed"),
    ])
    def test_values(self, member, expected):
        """Each ComplianceStatus member has the correct value."""
        assert member.value == expected


@_SKIP
class TestCoverageStatus:
    """Tests for the CoverageStatus enumeration."""

    def test_member_count(self):
        """CoverageStatus has exactly 4 members."""
        assert len(CoverageStatus) == 4

    @pytest.mark.parametrize("member,expected", [
        (CoverageStatus.FULLY_COVERED, "fully_covered"),
        (CoverageStatus.PARTIALLY_COVERED, "partially_covered"),
        (CoverageStatus.UNCOVERED, "uncovered"),
        (CoverageStatus.OVER_COVERED, "over_covered"),
    ])
    def test_values(self, member, expected):
        """Each CoverageStatus member has the correct value."""
        assert member.value == expected


@_SKIP
class TestReportingPeriod:
    """Tests for the ReportingPeriod enumeration."""

    def test_member_count(self):
        """ReportingPeriod has exactly 4 members."""
        assert len(ReportingPeriod) == 4

    @pytest.mark.parametrize("member,expected", [
        (ReportingPeriod.ANNUAL, "annual"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each ReportingPeriod member has the correct value."""
        assert member.value == expected


@_SKIP
class TestContractType:
    """Tests for the ContractType enumeration."""

    def test_member_count(self):
        """ContractType has exactly 4 members."""
        assert len(ContractType) == 4

    @pytest.mark.parametrize("member,expected", [
        (ContractType.PHYSICAL_PPA, "physical_ppa"),
        (ContractType.VIRTUAL_PPA, "virtual_ppa"),
        (ContractType.SLEEVED_PPA, "sleeved_ppa"),
        (ContractType.DIRECT_PURCHASE, "direct_purchase"),
    ])
    def test_values(self, member, expected):
        """Each ContractType member has the correct value."""
        assert member.value == expected


@_SKIP
class TestDataQualityTier:
    """Tests for the DataQualityTier enumeration."""

    def test_member_count(self):
        """DataQualityTier has exactly 3 members."""
        assert len(DataQualityTier) == 3

    @pytest.mark.parametrize("member,expected", [
        (DataQualityTier.TIER_1, "tier_1"),
        (DataQualityTier.TIER_2, "tier_2"),
        (DataQualityTier.TIER_3, "tier_3"),
    ])
    def test_values(self, member, expected):
        """Each DataQualityTier member has the correct value."""
        assert member.value == expected


@_SKIP
class TestDualReportingStatus:
    """Tests for the DualReportingStatus enumeration."""

    def test_member_count(self):
        """DualReportingStatus has exactly 3 members."""
        assert len(DualReportingStatus) == 3

    @pytest.mark.parametrize("member,expected", [
        (DualReportingStatus.COMPLETE, "complete"),
        (DualReportingStatus.LOCATION_ONLY, "location_only"),
        (DualReportingStatus.MARKET_ONLY, "market_only"),
    ])
    def test_values(self, member, expected):
        """Each DualReportingStatus member has the correct value."""
        assert member.value == expected


@_SKIP
class TestAllocationMethod:
    """Tests for the AllocationMethod enumeration."""

    def test_member_count(self):
        """AllocationMethod has exactly 3 members."""
        assert len(AllocationMethod) == 3

    @pytest.mark.parametrize("member,expected", [
        (AllocationMethod.PRIORITY_BASED, "priority_based"),
        (AllocationMethod.PROPORTIONAL, "proportional"),
        (AllocationMethod.CUSTOM, "custom"),
    ])
    def test_values(self, member, expected):
        """Each AllocationMethod member has the correct value."""
        assert member.value == expected


@_SKIP
class TestConsumptionDataSource:
    """Tests for the ConsumptionDataSource enumeration."""

    def test_member_count(self):
        """ConsumptionDataSource has exactly 4 members."""
        assert len(ConsumptionDataSource) == 4

    @pytest.mark.parametrize("member,expected", [
        (ConsumptionDataSource.METER, "meter"),
        (ConsumptionDataSource.INVOICE, "invoice"),
        (ConsumptionDataSource.ESTIMATE, "estimate"),
        (ConsumptionDataSource.BENCHMARK, "benchmark"),
    ])
    def test_values(self, member, expected):
        """Each ConsumptionDataSource member has the correct value."""
        assert member.value == expected


# ===========================================================================
# Constant Table Tests
# ===========================================================================


@_SKIP
class TestGWPValues:
    """Tests for the GWP_VALUES constant table."""

    def test_has_all_four_gwp_sources(self):
        """GWP_VALUES covers all 4 GWP assessment report editions."""
        assert len(GWP_VALUES) == 4
        for key in ("AR4", "AR5", "AR6", "AR6_20YR"):
            assert key in GWP_VALUES

    def test_each_source_has_three_gases(self):
        """Each GWP source maps to exactly 3 gases (CO2, CH4, N2O)."""
        for source, gases in GWP_VALUES.items():
            assert len(gases) == 3, f"{source} has {len(gases)} gases"
            assert "CO2" in gases
            assert "CH4" in gases
            assert "N2O" in gases

    def test_co2_is_always_one(self):
        """CO2 GWP is 1 across all assessment reports."""
        for source, gases in GWP_VALUES.items():
            assert gases["CO2"] == Decimal("1"), f"{source} CO2 != 1"

    @pytest.mark.parametrize("source,gas,expected", [
        ("AR4", "CH4", Decimal("25")),
        ("AR5", "CH4", Decimal("28")),
        ("AR6", "CH4", Decimal("27.9")),
        ("AR6_20YR", "CH4", Decimal("81.2")),
        ("AR4", "N2O", Decimal("298")),
        ("AR5", "N2O", Decimal("265")),
        ("AR6", "N2O", Decimal("273")),
        ("AR6_20YR", "N2O", Decimal("273")),
    ])
    def test_specific_gwp_values(self, source, gas, expected):
        """Specific GWP values match published IPCC data."""
        assert GWP_VALUES[source][gas] == expected

    def test_all_values_are_decimal(self):
        """All GWP values are Decimal instances for deterministic arithmetic."""
        for source, gases in GWP_VALUES.items():
            for gas, value in gases.items():
                assert isinstance(value, Decimal), f"{source}/{gas} not Decimal"


@_SKIP
class TestResidualMixFactors:
    """Tests for the RESIDUAL_MIX_FACTORS constant table."""

    def test_has_more_than_56_regions(self):
        """RESIDUAL_MIX_FACTORS covers 56+ regions."""
        assert len(RESIDUAL_MIX_FACTORS) >= 56

    def test_all_values_are_decimal(self):
        """All residual mix factors are Decimal instances."""
        for region, factor in RESIDUAL_MIX_FACTORS.items():
            assert isinstance(factor, Decimal), f"{region} not Decimal"

    def test_all_values_positive(self):
        """All residual mix factors are positive."""
        for region, factor in RESIDUAL_MIX_FACTORS.items():
            assert factor > Decimal("0"), f"{region} factor is not positive"

    @pytest.mark.parametrize("region,expected", [
        ("US-CAMX", Decimal("0.285")),
        ("US-ERCT", Decimal("0.420")),
        ("EU-DE", Decimal("0.520")),
        ("EU-FR", Decimal("0.085")),
        ("EU-SE", Decimal("0.045")),
        ("APAC-AU", Decimal("0.750")),
        ("APAC-JP", Decimal("0.520")),
        ("AMER-CA", Decimal("0.145")),
        ("GLOBAL", Decimal("0.500")),
    ])
    def test_specific_residual_mix_values(self, region, expected):
        """Specific residual mix factors match published data."""
        assert RESIDUAL_MIX_FACTORS[region] == expected

    def test_has_us_subregions(self):
        """RESIDUAL_MIX_FACTORS includes US eGRID subregions."""
        us_regions = [k for k in RESIDUAL_MIX_FACTORS if k.startswith("US-")]
        assert len(us_regions) >= 18

    def test_has_eu_countries(self):
        """RESIDUAL_MIX_FACTORS includes EU countries."""
        eu_regions = [k for k in RESIDUAL_MIX_FACTORS if k.startswith("EU-")]
        assert len(eu_regions) >= 26

    def test_has_global_fallback(self):
        """RESIDUAL_MIX_FACTORS includes GLOBAL fallback."""
        assert "GLOBAL" in RESIDUAL_MIX_FACTORS


@_SKIP
class TestEnergySourceEF:
    """Tests for the ENERGY_SOURCE_EF constant table."""

    def test_has_11_sources(self):
        """ENERGY_SOURCE_EF covers all 11 energy sources."""
        assert len(ENERGY_SOURCE_EF) == 11

    def test_all_values_are_decimal(self):
        """All energy source EFs are Decimal instances."""
        for source, factor in ENERGY_SOURCE_EF.items():
            assert isinstance(factor, Decimal), f"{source} not Decimal"

    def test_renewables_are_zero(self):
        """Renewable energy sources have zero emission factors."""
        zero_sources = ["solar", "wind", "hydro", "nuclear", "biomass"]
        for source in zero_sources:
            assert ENERGY_SOURCE_EF[source] == Decimal("0.000"), (
                f"{source} should be 0.000"
            )

    def test_coal_highest(self):
        """Coal has the highest emission factor among all sources."""
        coal_ef = ENERGY_SOURCE_EF["coal"]
        for source, factor in ENERGY_SOURCE_EF.items():
            assert factor <= coal_ef, f"{source} exceeds coal"

    def test_gas_ccgt_less_than_ocgt(self):
        """CCGT natural gas EF is lower than OCGT."""
        assert ENERGY_SOURCE_EF["natural_gas_ccgt"] < ENERGY_SOURCE_EF["natural_gas_ocgt"]


@_SKIP
class TestSupplierDefaultEF:
    """Tests for the SUPPLIER_DEFAULT_EF constant table."""

    def test_has_52_countries(self):
        """SUPPLIER_DEFAULT_EF covers 52 countries."""
        assert len(SUPPLIER_DEFAULT_EF) == 52

    def test_all_values_are_decimal(self):
        """All supplier default EFs are Decimal instances."""
        for country, factor in SUPPLIER_DEFAULT_EF.items():
            assert isinstance(factor, Decimal), f"{country} not Decimal"

    def test_keys_are_uppercase_iso_alpha2(self):
        """All keys are 2-character uppercase ISO country codes."""
        for country in SUPPLIER_DEFAULT_EF:
            assert len(country) == 2, f"{country} not 2 chars"
            assert country == country.upper(), f"{country} not uppercase"

    @pytest.mark.parametrize("country,expected", [
        ("US", Decimal("0.390")),
        ("GB", Decimal("0.225")),
        ("DE", Decimal("0.350")),
        ("FR", Decimal("0.060")),
        ("ZA", Decimal("0.935")),
        ("SE", Decimal("0.010")),
        ("CN", Decimal("0.570")),
        ("JP", Decimal("0.470")),
    ])
    def test_specific_supplier_ef_values(self, country, expected):
        """Specific supplier EFs match national utility averages."""
        assert SUPPLIER_DEFAULT_EF[country] == expected


@_SKIP
class TestInstrumentQualityWeights:
    """Tests for the INSTRUMENT_QUALITY_WEIGHTS constant table."""

    def test_has_7_criteria(self):
        """INSTRUMENT_QUALITY_WEIGHTS covers all 7 quality criteria."""
        assert len(INSTRUMENT_QUALITY_WEIGHTS) == 7

    def test_all_values_are_decimal(self):
        """All quality weights are Decimal instances."""
        for criterion, weight in INSTRUMENT_QUALITY_WEIGHTS.items():
            assert isinstance(weight, Decimal), f"{criterion} not Decimal"

    def test_weights_sum_to_one(self):
        """Quality criterion weights sum to exactly 1.00."""
        total = sum(INSTRUMENT_QUALITY_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_all_weights_positive(self):
        """All quality weights are positive."""
        for criterion, weight in INSTRUMENT_QUALITY_WEIGHTS.items():
            assert weight > Decimal("0"), f"{criterion} weight not positive"

    def test_expected_criterion_names(self):
        """Quality weights contain all expected criterion names."""
        expected = [
            "unique_claim", "associated_delivery", "temporal_match",
            "geographic_match", "no_double_count", "recognized_registry",
            "represents_generation",
        ]
        for name in expected:
            assert name in INSTRUMENT_QUALITY_WEIGHTS, f"{name} missing"


@_SKIP
class TestVintageValidityYears:
    """Tests for the VINTAGE_VALIDITY_YEARS constant table."""

    def test_has_10_instrument_types(self):
        """VINTAGE_VALIDITY_YEARS covers 10 instrument types."""
        assert len(VINTAGE_VALIDITY_YEARS) == 10

    def test_all_values_are_positive_int(self):
        """All vintage validity years are positive integers."""
        for inst_type, years in VINTAGE_VALIDITY_YEARS.items():
            assert isinstance(years, int), f"{inst_type} not int"
            assert years > 0, f"{inst_type} years not positive"

    def test_ppa_has_longest_validity(self):
        """PPA instruments have the longest vintage validity."""
        ppa_years = VINTAGE_VALIDITY_YEARS["ppa"]
        for inst_type, years in VINTAGE_VALIDITY_YEARS.items():
            assert years <= ppa_years, (
                f"{inst_type} ({years}) exceeds PPA ({ppa_years})"
            )


@_SKIP
class TestUnitConversions:
    """Tests for the UNIT_CONVERSIONS constant table."""

    def test_has_eight_entries(self):
        """UNIT_CONVERSIONS has exactly 8 conversion factors."""
        assert len(UNIT_CONVERSIONS) == 8

    def test_all_values_are_decimal(self):
        """All unit conversion factors are Decimal instances."""
        for key, value in UNIT_CONVERSIONS.items():
            assert isinstance(value, Decimal), f"{key} not Decimal"

    @pytest.mark.parametrize("key,expected", [
        ("MWH_TO_GJ", Decimal("3.6")),
        ("GJ_TO_MWH", Decimal("0.277778")),
        ("MMBTU_TO_GJ", Decimal("1.05506")),
        ("KWH_TO_MWH", Decimal("0.001")),
        ("MWH_TO_KWH", Decimal("1000")),
    ])
    def test_specific_conversions(self, key, expected):
        """Specific unit conversion factors are correct."""
        assert UNIT_CONVERSIONS[key] == expected

    def test_kwh_mwh_inverse(self):
        """KWH_TO_MWH and MWH_TO_KWH are reciprocals."""
        product = UNIT_CONVERSIONS["KWH_TO_MWH"] * UNIT_CONVERSIONS["MWH_TO_KWH"]
        assert product == Decimal("1")


# ===========================================================================
# Module Constant Tests
# ===========================================================================


@_SKIP
class TestModuleConstants:
    """Tests for module-level constants."""

    def test_version(self):
        """VERSION is a valid semver string."""
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        """MAX_CALCULATIONS_PER_BATCH is 10000."""
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        """MAX_GASES_PER_RESULT is 10."""
        assert MAX_GASES_PER_RESULT == 10

    def test_max_trace_steps(self):
        """MAX_TRACE_STEPS is 200."""
        assert MAX_TRACE_STEPS == 200

    def test_max_facilities_per_tenant(self):
        """MAX_FACILITIES_PER_TENANT is 50000."""
        assert MAX_FACILITIES_PER_TENANT == 50_000

    def test_max_energy_purchases_per_calc(self):
        """MAX_ENERGY_PURCHASES_PER_CALC is 1000."""
        assert MAX_ENERGY_PURCHASES_PER_CALC == 1_000

    def test_max_instruments_per_purchase(self):
        """MAX_INSTRUMENTS_PER_PURCHASE is 100."""
        assert MAX_INSTRUMENTS_PER_PURCHASE == 100

    def test_default_monte_carlo_iterations(self):
        """DEFAULT_MONTE_CARLO_ITERATIONS is 10000."""
        assert DEFAULT_MONTE_CARLO_ITERATIONS == 10_000

    def test_default_confidence_level(self):
        """DEFAULT_CONFIDENCE_LEVEL is 0.95."""
        assert DEFAULT_CONFIDENCE_LEVEL == Decimal("0.95")

    def test_default_quality_threshold(self):
        """DEFAULT_QUALITY_THRESHOLD is 0.70."""
        assert DEFAULT_QUALITY_THRESHOLD == Decimal("0.70")

    def test_table_prefix(self):
        """TABLE_PREFIX is gl_s2m_."""
        assert TABLE_PREFIX == "gl_s2m_"


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


@_SKIP
class TestContractualInstrument:
    """Tests for the ContractualInstrument Pydantic model."""

    def _make_instrument(self, **overrides):
        """Helper to create a valid ContractualInstrument."""
        defaults = dict(
            instrument_type=InstrumentType.REC,
            quantity_mwh=Decimal("1000"),
            energy_source=EnergySource.WIND,
            ef_kgco2e_per_kwh=Decimal("0.000"),
            vintage_year=2025,
            tracking_system=TrackingSystem.GREEN_E,
            certificate_id="CERT-001",
            region="US-CAMX",
            delivery_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            delivery_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
            tenant_id="t1",
        )
        defaults.update(overrides)
        return ContractualInstrument(**defaults)

    def test_creation_with_required_fields(self):
        """ContractualInstrument can be created with required fields."""
        inst = self._make_instrument()
        assert inst.instrument_type == InstrumentType.REC
        assert inst.quantity_mwh == Decimal("1000")
        assert inst.energy_source == EnergySource.WIND

    def test_auto_generated_id(self):
        """ContractualInstrument generates a UUID instrument_id."""
        inst = self._make_instrument()
        uuid.UUID(inst.instrument_id)  # should not raise

    def test_default_status_is_active(self):
        """Default status is ACTIVE."""
        inst = self._make_instrument()
        assert inst.status == InstrumentStatus.ACTIVE

    def test_delivery_end_before_start_rejected(self):
        """delivery_end must be after delivery_start."""
        with pytest.raises(Exception):
            self._make_instrument(
                delivery_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
                delivery_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_negative_quantity_rejected(self):
        """Negative quantity_mwh is rejected."""
        with pytest.raises(Exception):
            self._make_instrument(quantity_mwh=Decimal("-100"))

    def test_zero_quantity_rejected(self):
        """Zero quantity_mwh is rejected (gt=0)."""
        with pytest.raises(Exception):
            self._make_instrument(quantity_mwh=Decimal("0"))

    def test_negative_ef_rejected(self):
        """Negative ef_kgco2e_per_kwh is rejected."""
        with pytest.raises(Exception):
            self._make_instrument(ef_kgco2e_per_kwh=Decimal("-0.1"))

    def test_model_is_frozen(self):
        """ContractualInstrument instances are immutable (frozen=True)."""
        inst = self._make_instrument()
        with pytest.raises(Exception):
            inst.quantity_mwh = Decimal("500")

    def test_optional_contract_type(self):
        """contract_type defaults to None."""
        inst = self._make_instrument()
        assert inst.contract_type is None

    def test_contract_type_set(self):
        """contract_type can be set explicitly."""
        inst = self._make_instrument(contract_type=ContractType.PHYSICAL_PPA)
        assert inst.contract_type == ContractType.PHYSICAL_PPA

    def test_serialization_round_trip(self):
        """ContractualInstrument can be serialized to dict."""
        inst = self._make_instrument()
        data = inst.model_dump()
        assert data["instrument_type"] == "rec"
        assert data["energy_source"] == "wind"


@_SKIP
class TestInstrumentQualityAssessment:
    """Tests for the InstrumentQualityAssessment Pydantic model."""

    def _make_assessment(self, **overrides):
        """Helper to create a valid InstrumentQualityAssessment."""
        defaults = dict(
            instrument_id="inst-001",
            unique_claim_score=Decimal("0.90"),
            associated_delivery_score=Decimal("0.85"),
            temporal_match_score=Decimal("1.00"),
            geographic_match_score=Decimal("0.80"),
            no_double_count_score=Decimal("1.00"),
            recognized_registry_score=Decimal("0.90"),
            represents_generation_score=Decimal("1.00"),
            overall_score=Decimal("0.92"),
            passes_threshold=True,
        )
        defaults.update(overrides)
        return InstrumentQualityAssessment(**defaults)

    def test_creation(self):
        """InstrumentQualityAssessment can be created with valid data."""
        assessment = self._make_assessment()
        assert assessment.overall_score == Decimal("0.92")
        assert assessment.passes_threshold is True

    def test_score_out_of_range_rejected(self):
        """Scores above 1.0 are rejected."""
        with pytest.raises(Exception):
            self._make_assessment(unique_claim_score=Decimal("1.5"))

    def test_negative_score_rejected(self):
        """Negative scores are rejected."""
        with pytest.raises(Exception):
            self._make_assessment(unique_claim_score=Decimal("-0.1"))

    def test_default_threshold(self):
        """Default threshold_used is DEFAULT_QUALITY_THRESHOLD (0.70)."""
        assessment = self._make_assessment()
        assert assessment.threshold_used == Decimal("0.70")


@_SKIP
class TestSupplierEmissionFactor:
    """Tests for the SupplierEmissionFactor Pydantic model."""

    def test_creation(self):
        """SupplierEmissionFactor can be created with valid data."""
        sef = SupplierEmissionFactor(
            name="Test Utility",
            country="us",
            ef_kgco2e_per_kwh=Decimal("0.350"),
            year=2024,
        )
        assert sef.name == "Test Utility"
        assert sef.ef_kgco2e_per_kwh == Decimal("0.350")

    def test_country_code_uppercased(self):
        """Country code is normalized to uppercase."""
        sef = SupplierEmissionFactor(
            name="Test",
            country="gb",
            ef_kgco2e_per_kwh=Decimal("0.200"),
            year=2024,
        )
        assert sef.country == "GB"

    def test_default_data_quality_tier(self):
        """Default data quality tier is TIER_2."""
        sef = SupplierEmissionFactor(
            name="Test",
            country="US",
            ef_kgco2e_per_kwh=Decimal("0.300"),
            year=2024,
        )
        assert sef.data_quality_tier == DataQualityTier.TIER_2

    def test_auto_generated_id(self):
        """SupplierEmissionFactor generates a UUID supplier_id."""
        sef = SupplierEmissionFactor(
            name="Test",
            country="US",
            ef_kgco2e_per_kwh=Decimal("0.300"),
            year=2024,
        )
        uuid.UUID(sef.supplier_id)


@_SKIP
class TestResidualMixFactor:
    """Tests for the ResidualMixFactor Pydantic model."""

    def test_creation(self):
        """ResidualMixFactor can be created with valid data."""
        rmf = ResidualMixFactor(
            region="US-CAMX",
            factor_kgco2e_per_kwh=Decimal("0.285"),
            source=ResidualMixSource.GREEN_E,
            year=2024,
        )
        assert rmf.region == "US-CAMX"
        assert rmf.factor_kgco2e_per_kwh == Decimal("0.285")

    def test_country_code_uppercased(self):
        """Country code is normalized to uppercase if provided."""
        rmf = ResidualMixFactor(
            region="EU-DE",
            factor_kgco2e_per_kwh=Decimal("0.520"),
            source=ResidualMixSource.AIB,
            year=2024,
            country_code="de",
        )
        assert rmf.country_code == "DE"

    def test_optional_country_code(self):
        """country_code defaults to None."""
        rmf = ResidualMixFactor(
            region="GLOBAL",
            factor_kgco2e_per_kwh=Decimal("0.500"),
            source=ResidualMixSource.ESTIMATED,
            year=2024,
        )
        assert rmf.country_code is None


@_SKIP
class TestEnergyPurchase:
    """Tests for the EnergyPurchase Pydantic model."""

    def _make_purchase(self, **overrides):
        """Helper to create a valid EnergyPurchase."""
        defaults = dict(
            facility_id="fac-001",
            energy_type=EnergyType.ELECTRICITY,
            quantity=Decimal("1000"),
            unit=EnergyUnit.MWH,
            region="US-CAMX",
            period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return EnergyPurchase(**defaults)

    def test_creation(self):
        """EnergyPurchase can be created with valid data."""
        purchase = self._make_purchase()
        assert purchase.quantity == Decimal("1000")
        assert purchase.data_source == ConsumptionDataSource.INVOICE

    def test_period_end_before_start_rejected(self):
        """period_end must be after period_start."""
        with pytest.raises(Exception):
            self._make_purchase(
                period_start=datetime(2025, 12, 31, tzinfo=timezone.utc),
                period_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    def test_zero_quantity_rejected(self):
        """Zero quantity is rejected (gt=0)."""
        with pytest.raises(Exception):
            self._make_purchase(quantity=Decimal("0"))

    def test_auto_generated_id(self):
        """EnergyPurchase generates a UUID purchase_id."""
        purchase = self._make_purchase()
        uuid.UUID(purchase.purchase_id)

    def test_empty_instruments_by_default(self):
        """instruments defaults to empty list."""
        purchase = self._make_purchase()
        assert purchase.instruments == []


@_SKIP
class TestFacilityInfo:
    """Tests for the FacilityInfo Pydantic model."""

    def test_creation_with_required_fields(self):
        """FacilityInfo can be created with all required fields."""
        facility = FacilityInfo(
            name="Test Office",
            facility_type=FacilityType.OFFICE,
            country_code="US",
            grid_region="US-CAMX",
            tenant_id="t1",
        )
        assert facility.name == "Test Office"
        assert facility.facility_type == FacilityType.OFFICE
        assert facility.country_code == "US"

    def test_auto_generated_facility_id(self):
        """FacilityInfo generates a UUID facility_id by default."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.WAREHOUSE,
            country_code="DE",
            grid_region="EU-DE",
            tenant_id="t1",
        )
        uuid.UUID(facility.facility_id)

    def test_country_code_uppercased(self):
        """Country code is normalized to uppercase."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.OFFICE,
            country_code="gb",
            grid_region="EU-GB",
            tenant_id="t1",
        )
        assert facility.country_code == "GB"

    def test_empty_name_rejected(self):
        """Empty name is rejected by min_length=1 constraint."""
        with pytest.raises(Exception):
            FacilityInfo(
                name="",
                facility_type=FacilityType.OFFICE,
                country_code="US",
                grid_region="US-CAMX",
                tenant_id="t1",
            )

    def test_model_is_frozen(self):
        """FacilityInfo instances are immutable (frozen=True)."""
        facility = FacilityInfo(
            name="Test",
            facility_type=FacilityType.OFFICE,
            country_code="US",
            grid_region="US-CAMX",
            tenant_id="t1",
        )
        with pytest.raises(Exception):
            facility.name = "Changed"

    def test_serialization_round_trip(self):
        """FacilityInfo can be serialized to dict."""
        facility = FacilityInfo(
            name="Test Office",
            facility_type=FacilityType.DATA_CENTER,
            country_code="US",
            grid_region="US-CAMX",
            tenant_id="t1",
        )
        data = facility.model_dump()
        assert data["name"] == "Test Office"
        assert data["facility_type"] == "data_center"
        assert data["country_code"] == "US"


@_SKIP
class TestAllocationResult:
    """Tests for the AllocationResult Pydantic model."""

    def test_creation(self):
        """AllocationResult can be created with valid data."""
        result = AllocationResult(
            purchase_id="pur-001",
            total_mwh=Decimal("1000"),
            covered_mwh=Decimal("500"),
            uncovered_mwh=Decimal("500"),
            coverage_pct=Decimal("50.0"),
            coverage_status=CoverageStatus.PARTIALLY_COVERED,
        )
        assert result.coverage_pct == Decimal("50.0")

    def test_default_allocation_method(self):
        """Default allocation method is PRIORITY_BASED."""
        result = AllocationResult(
            purchase_id="pur-001",
            total_mwh=Decimal("1000"),
            covered_mwh=Decimal("1000"),
            uncovered_mwh=Decimal("0"),
            coverage_pct=Decimal("100.0"),
            coverage_status=CoverageStatus.FULLY_COVERED,
        )
        assert result.allocation_method == AllocationMethod.PRIORITY_BASED


@_SKIP
class TestCoveredEmissionResult:
    """Tests for the CoveredEmissionResult Pydantic model."""

    def test_creation(self):
        """CoveredEmissionResult can be created with valid data."""
        result = CoveredEmissionResult(
            instrument_id="inst-001",
            instrument_type=InstrumentType.REC,
            mwh_covered=Decimal("500"),
            ef_kgco2e_per_kwh=Decimal("0.000"),
            emissions_kg=Decimal("0"),
            co2e_kg=Decimal("0"),
            energy_source=EnergySource.WIND,
        )
        assert result.co2e_kg == Decimal("0")
        assert result.quality_score == Decimal("0")


@_SKIP
class TestUncoveredEmissionResult:
    """Tests for the UncoveredEmissionResult Pydantic model."""

    def test_creation(self):
        """UncoveredEmissionResult can be created with valid data."""
        result = UncoveredEmissionResult(
            mwh_uncovered=Decimal("500"),
            region="US-CAMX",
            residual_mix_ef_kgco2e_per_kwh=Decimal("0.285"),
            emissions_kg=Decimal("142500"),
            co2e_kg=Decimal("142500"),
        )
        assert result.region == "US-CAMX"
        assert result.residual_mix_source == ResidualMixSource.ESTIMATED


@_SKIP
class TestGasEmissionDetail:
    """Tests for the GasEmissionDetail Pydantic model."""

    def test_creation(self):
        """GasEmissionDetail can be created with valid data."""
        detail = GasEmissionDetail(
            gas=EmissionGas.CO2,
            emission_kg=Decimal("225300"),
            gwp_factor=Decimal("1"),
            co2e_kg=Decimal("225300"),
        )
        assert detail.gas == EmissionGas.CO2
        assert detail.emission_kg == Decimal("225300")


@_SKIP
class TestMarketBasedResult:
    """Tests for the MarketBasedResult Pydantic model."""

    def test_creation(self):
        """MarketBasedResult can be created with valid data."""
        result = MarketBasedResult(
            calculation_id="calc-001",
            facility_id="fac-001",
            total_mwh=Decimal("1000"),
            covered_mwh=Decimal("500"),
            uncovered_mwh=Decimal("500"),
            coverage_pct=Decimal("50.0"),
            covered_emissions_tco2e=Decimal("0"),
            uncovered_emissions_tco2e=Decimal("142.5"),
            total_emissions_tco2e=Decimal("142.5"),
            provenance_hash="a" * 64,
        )
        assert result.total_emissions_tco2e == Decimal("142.5")
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_length_validation(self):
        """Provenance hash must be exactly 64 characters (SHA-256)."""
        with pytest.raises(Exception):
            MarketBasedResult(
                calculation_id="calc-001",
                facility_id="fac-001",
                total_mwh=Decimal("1000"),
                covered_mwh=Decimal("500"),
                uncovered_mwh=Decimal("500"),
                coverage_pct=Decimal("50.0"),
                covered_emissions_tco2e=Decimal("0"),
                uncovered_emissions_tco2e=Decimal("142.5"),
                total_emissions_tco2e=Decimal("142.5"),
                provenance_hash="short",
            )

    def test_default_calculation_method(self):
        """Default calculation method is HYBRID."""
        result = MarketBasedResult(
            calculation_id="calc-001",
            facility_id="fac-001",
            total_mwh=Decimal("0"),
            covered_mwh=Decimal("0"),
            uncovered_mwh=Decimal("0"),
            coverage_pct=Decimal("0"),
            covered_emissions_tco2e=Decimal("0"),
            uncovered_emissions_tco2e=Decimal("0"),
            total_emissions_tco2e=Decimal("0"),
            provenance_hash="b" * 64,
        )
        assert result.calculation_method == CalculationMethod.HYBRID

    def test_default_data_quality_tier(self):
        """Default data quality tier is TIER_1."""
        result = MarketBasedResult(
            calculation_id="calc-001",
            facility_id="fac-001",
            total_mwh=Decimal("0"),
            covered_mwh=Decimal("0"),
            uncovered_mwh=Decimal("0"),
            coverage_pct=Decimal("0"),
            covered_emissions_tco2e=Decimal("0"),
            uncovered_emissions_tco2e=Decimal("0"),
            total_emissions_tco2e=Decimal("0"),
            provenance_hash="c" * 64,
        )
        assert result.data_quality_tier == DataQualityTier.TIER_1


@_SKIP
class TestDualReportingResult:
    """Tests for the DualReportingResult Pydantic model."""

    def test_creation(self):
        """DualReportingResult can be created with valid data."""
        result = DualReportingResult(
            facility_id="fac-001",
            location_based_tco2e=Decimal("250.0"),
            market_based_tco2e=Decimal("142.5"),
            difference_tco2e=Decimal("-107.5"),
            difference_pct=Decimal("-43.0"),
            re_procurement_impact_tco2e=Decimal("107.5"),
        )
        assert result.reporting_status == DualReportingStatus.COMPLETE
        assert result.difference_tco2e == Decimal("-107.5")


@_SKIP
class TestCalculationRequest:
    """Tests for the CalculationRequest Pydantic model."""

    def test_creation(self):
        """CalculationRequest can be created with required fields."""
        req = CalculationRequest(
            tenant_id="t1",
            facility_id="fac-001",
        )
        assert req.gwp_source == GWPSource.AR6
        assert req.allocation_method == AllocationMethod.PRIORITY_BASED
        assert req.calculation_method == CalculationMethod.HYBRID

    def test_auto_generated_id(self):
        """CalculationRequest generates a UUID calculation_id."""
        req = CalculationRequest(
            tenant_id="t1",
            facility_id="fac-001",
        )
        uuid.UUID(req.calculation_id)

    def test_compliance_frameworks_default(self):
        """compliance_frameworks defaults to None."""
        req = CalculationRequest(
            tenant_id="t1",
            facility_id="fac-001",
        )
        assert req.compliance_frameworks is None


@_SKIP
class TestBatchCalculationRequest:
    """Tests for the BatchCalculationRequest Pydantic model."""

    def test_creation(self):
        """BatchCalculationRequest can be created with one request."""
        req = BatchCalculationRequest(
            tenant_id="t1",
            requests=[
                CalculationRequest(
                    tenant_id="t1",
                    facility_id="fac-001",
                ),
            ],
        )
        assert len(req.requests) == 1

    def test_empty_requests_rejected(self):
        """Empty requests list is rejected (min_length=1)."""
        with pytest.raises(Exception):
            BatchCalculationRequest(
                tenant_id="t1",
                requests=[],
            )


@_SKIP
class TestBatchCalculationResult:
    """Tests for the BatchCalculationResult Pydantic model."""

    def test_creation(self):
        """BatchCalculationResult can be created with valid data."""
        result = BatchCalculationResult(
            batch_id="batch-001",
            total_co2e_tonnes=Decimal("500"),
            facility_count=5,
            provenance_hash="d" * 64,
        )
        assert result.total_co2e_tonnes == Decimal("500")
        assert result.facility_count == 5
        assert len(result.provenance_hash) == 64


@_SKIP
class TestComplianceCheckResult:
    """Tests for the ComplianceCheckResult Pydantic model."""

    def test_creation(self):
        """ComplianceCheckResult can be created with valid data."""
        check = ComplianceCheckResult(
            calculation_id="calc-001",
            framework="GHG_PROTOCOL",
            status=ComplianceStatus.COMPLIANT,
        )
        assert check.framework == "GHG_PROTOCOL"
        assert check.status == ComplianceStatus.COMPLIANT
        assert check.findings == []
        assert check.recommendations == []

    def test_optional_coverage_fields(self):
        """Coverage fields default to None."""
        check = ComplianceCheckResult(
            calculation_id="calc-001",
            framework="RE100",
            status=ComplianceStatus.NON_COMPLIANT,
        )
        assert check.coverage_requirement is None
        assert check.actual_coverage is None


@_SKIP
class TestInstrumentValidationResult:
    """Tests for the InstrumentValidationResult Pydantic model."""

    def test_creation(self):
        """InstrumentValidationResult can be created with valid data."""
        assessment = InstrumentQualityAssessment(
            instrument_id="inst-001",
            unique_claim_score=Decimal("0.90"),
            associated_delivery_score=Decimal("0.85"),
            temporal_match_score=Decimal("1.00"),
            geographic_match_score=Decimal("0.80"),
            no_double_count_score=Decimal("1.00"),
            recognized_registry_score=Decimal("0.90"),
            represents_generation_score=Decimal("1.00"),
            overall_score=Decimal("0.92"),
            passes_threshold=True,
        )
        result = InstrumentValidationResult(
            instrument_id="inst-001",
            quality_assessment=assessment,
            is_valid=True,
        )
        assert result.is_valid is True
        assert result.vintage_valid is True
        assert result.status_valid is True


@_SKIP
class TestUncertaintyRequest:
    """Tests for the UncertaintyRequest Pydantic model."""

    def test_creation_monte_carlo(self):
        """UncertaintyRequest defaults to monte_carlo method."""
        req = UncertaintyRequest(calculation_id="calc-001")
        assert req.method == "monte_carlo"
        assert req.iterations == 10_000
        assert req.confidence_level == Decimal("0.95")

    def test_invalid_method_rejected(self):
        """Invalid uncertainty method is rejected."""
        with pytest.raises(Exception):
            UncertaintyRequest(
                calculation_id="calc-001",
                method="invalid_method",
            )

    def test_analytical_method(self):
        """Analytical method is accepted."""
        req = UncertaintyRequest(
            calculation_id="calc-001",
            method="analytical",
        )
        assert req.method == "analytical"


@_SKIP
class TestUncertaintyResult:
    """Tests for the UncertaintyResult Pydantic model."""

    def test_creation(self):
        """UncertaintyResult can be created with valid data."""
        result = UncertaintyResult(
            calculation_id="calc-001",
            method="monte_carlo",
            mean_co2e=Decimal("226.50"),
            std_dev=Decimal("15.30"),
            ci_lower=Decimal("196.50"),
            ci_upper=Decimal("256.50"),
            confidence_level=Decimal("0.95"),
            iterations=10000,
        )
        assert result.mean_co2e == Decimal("226.50")
        assert result.ci_lower < result.ci_upper


@_SKIP
class TestAggregationResult:
    """Tests for the AggregationResult Pydantic model."""

    def test_creation(self):
        """AggregationResult can be created with valid data."""
        agg = AggregationResult(
            group_by="instrument_type",
            period="2025",
            total_co2e_tonnes=Decimal("500"),
            total_covered_mwh=Decimal("5000"),
            total_uncovered_mwh=Decimal("2000"),
            average_coverage_pct=Decimal("71.4"),
            facility_count=5,
        )
        assert agg.total_co2e_tonnes == Decimal("500")
        assert agg.facility_count == 5
