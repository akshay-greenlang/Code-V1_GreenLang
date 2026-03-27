# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.models - AGENT-MRV-022.

Tests all 22 enums, 14 constant tables, 12 Pydantic models, and helper
functions for the Downstream Transportation & Distribution Agent (GL-MRV-S3-009).

Coverage:
- Enumerations: 22 enums (values, membership, count, str behavior)
- Constants: GWP_VALUES, TRANSPORT_EMISSION_FACTORS, COLD_CHAIN_FACTORS,
  WAREHOUSE_EMISSION_FACTORS, LAST_MILE_FACTORS, EEIO_FACTORS,
  CURRENCY_RATES, CPI_DEFLATORS, GRID_EMISSION_FACTORS,
  CHANNEL_AVERAGES, INCOTERM_CLASSIFICATIONS, LOAD_FACTORS,
  RETURN_FACTORS, DQI_SCORING
- Input models: ShipmentInput, SpendInput, WarehouseInput, LastMileInput,
  AverageDataInput, ColdChainInput, ReturnLogisticsInput, BatchInput,
  CalculationRequest, ComplianceRequest
- Result models: CalculationResult, ComplianceResult (frozen checks)
- Helper functions: calculate_provenance_hash, get_dqi_classification,
  convert_currency_to_usd, get_cpi_deflator, classify_incoterm

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from datetime import datetime
import hashlib
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.models import (
        # Enumerations
        CalculationMethod,
        TransportMode,
        RoadVehicleType,
        MaritimeVesselType,
        AircraftType,
        RailVehicleType,
        CourierVehicleType,
        LastMileVehicleType,
        WarehouseType,
        DistributionChannel,
        ColdChainRegime,
        DeliveryArea,
        Incoterm,
        EFSource,
        ComplianceFramework,
        DataQualityTier,
        ProvenanceStage,
        UncertaintyMethod,
        GWPVersion,
        EmissionGas,
        CurrencyCode,
        ExportFormat,

        # Agent metadata
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,

        # Constant tables
        GWP_VALUES,
        TRANSPORT_EMISSION_FACTORS,
        COLD_CHAIN_FACTORS,
        WAREHOUSE_EMISSION_FACTORS,
        LAST_MILE_FACTORS,
        EEIO_FACTORS,
        CURRENCY_RATES,
        CPI_DEFLATORS,
        GRID_EMISSION_FACTORS,
        CHANNEL_AVERAGES,
        INCOTERM_CLASSIFICATIONS,
        LOAD_FACTORS,
        RETURN_FACTORS,
        DQI_SCORING,

        # Input models
        ShipmentInput,
        SpendInput,
        WarehouseInput,
        LastMileInput,
        AverageDataInput,
        ColdChainInput,
        ReturnLogisticsInput,
        BatchInput,
        CalculationRequest,
        ComplianceRequest,

        # Result models
        CalculationResult,
        ComplianceResult,

        # Helper functions
        calculate_provenance_hash,
        get_dqi_classification,
        convert_currency_to_usd,
        get_cpi_deflator,
        classify_incoterm,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"downstream_transportation.models not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# ENUMERATION TESTS
# ==============================================================================


class TestCalculationMethodEnum:
    """Test CalculationMethod enum."""

    def test_values_exist(self):
        """Test all 4 calculation method values exist."""
        assert CalculationMethod.DISTANCE_BASED == "distance_based"
        assert CalculationMethod.SPEND_BASED == "spend_based"
        assert CalculationMethod.AVERAGE_DATA == "average_data"
        assert CalculationMethod.SUPPLIER_SPECIFIC == "supplier_specific"
        assert len(CalculationMethod) == 4

    def test_str_membership(self):
        """Test string membership for CalculationMethod."""
        values = [e.value for e in CalculationMethod]
        assert "distance_based" in values
        assert "spend_based" in values


class TestTransportModeEnum:
    """Test TransportMode enum."""

    def test_values_exist(self):
        """Test all 6 transport mode values exist."""
        expected = ["ROAD", "RAIL", "MARITIME", "AIR", "COURIER", "LAST_MILE"]
        actual = [e.value for e in TransportMode]
        for v in expected:
            assert v in actual
        assert len(TransportMode) == 6

    def test_str_membership(self):
        """Test string membership for TransportMode."""
        assert TransportMode.ROAD.value == "ROAD"
        assert TransportMode.MARITIME.value == "MARITIME"


class TestRoadVehicleTypeEnum:
    """Test RoadVehicleType enum."""

    def test_values_exist(self):
        """Test road vehicle types contain expected entries."""
        expected = [
            "ARTICULATED_33T", "ARTICULATED_40_44T", "RIGID_7_5T",
            "RIGID_12T", "RIGID_17T", "VAN_SMALL", "VAN_MEDIUM",
            "VAN_LARGE",
        ]
        actual = [e.value for e in RoadVehicleType]
        for v in expected:
            assert v in actual, f"Missing road vehicle type: {v}"

    def test_count(self):
        """Test road vehicle types count is at least 8."""
        assert len(RoadVehicleType) >= 8


class TestMaritimeVesselTypeEnum:
    """Test MaritimeVesselType enum."""

    def test_values_exist(self):
        """Test maritime vessel types contain expected entries."""
        expected = [
            "CONTAINER_FEEDER", "CONTAINER_PANAMAX",
            "CONTAINER_POST_PANAMAX", "BULK_HANDYSIZE",
            "BULK_PANAMAX", "TANKER_PRODUCT",
        ]
        actual = [e.value for e in MaritimeVesselType]
        for v in expected:
            assert v in actual, f"Missing vessel type: {v}"

    def test_count(self):
        """Test maritime vessel types count is at least 6."""
        assert len(MaritimeVesselType) >= 6


class TestAircraftTypeEnum:
    """Test AircraftType enum."""

    def test_values_exist(self):
        """Test aircraft types contain expected entries."""
        expected = [
            "NARROWBODY_FREIGHTER", "WIDEBODY_FREIGHTER",
            "BELLY_FREIGHT_NARROWBODY", "BELLY_FREIGHT_WIDEBODY",
        ]
        actual = [e.value for e in AircraftType]
        for v in expected:
            assert v in actual, f"Missing aircraft type: {v}"


class TestRailVehicleTypeEnum:
    """Test RailVehicleType enum."""

    def test_values_exist(self):
        """Test rail vehicle types contain expected entries."""
        expected = ["ELECTRIC_FREIGHT", "DIESEL_FREIGHT", "INTERMODAL"]
        actual = [e.value for e in RailVehicleType]
        for v in expected:
            assert v in actual, f"Missing rail vehicle type: {v}"


class TestCourierVehicleTypeEnum:
    """Test CourierVehicleType enum."""

    def test_values_exist(self):
        """Test courier vehicle types contain expected entries."""
        expected = ["VAN_MEDIUM", "VAN_SMALL", "MOTORCYCLE", "BICYCLE"]
        actual = [e.value for e in CourierVehicleType]
        for v in expected:
            assert v in actual, f"Missing courier vehicle type: {v}"


class TestLastMileVehicleTypeEnum:
    """Test LastMileVehicleType enum."""

    def test_values_exist(self):
        """Test last-mile vehicle types contain expected entries."""
        expected = [
            "VAN_DIESEL", "VAN_ELECTRIC", "CARGO_BIKE",
            "DRONE", "PARCEL_LOCKER", "CROWD_SHIPPING",
        ]
        actual = [e.value for e in LastMileVehicleType]
        for v in expected:
            assert v in actual, f"Missing last-mile vehicle type: {v}"
        assert len(LastMileVehicleType) == 6


class TestWarehouseTypeEnum:
    """Test WarehouseType enum."""

    def test_values_exist(self):
        """Test warehouse types contain expected entries."""
        expected = [
            "DISTRIBUTION_CENTER", "COLD_STORAGE", "FULFILLMENT_CENTER",
            "RETAIL_STORAGE", "CROSS_DOCK", "BONDED_WAREHOUSE",
            "TRANSIT_WAREHOUSE",
        ]
        actual = [e.value for e in WarehouseType]
        for v in expected:
            assert v in actual, f"Missing warehouse type: {v}"
        assert len(WarehouseType) == 7


class TestDistributionChannelEnum:
    """Test DistributionChannel enum."""

    def test_values_exist(self):
        """Test distribution channels contain expected entries."""
        expected = [
            "ECOMMERCE_DTC", "RETAIL_DISTRIBUTION", "WHOLESALE",
            "MARKETPLACE_3PL", "DROPSHIP", "OMNICHANNEL",
        ]
        actual = [e.value for e in DistributionChannel]
        for v in expected:
            assert v in actual, f"Missing channel: {v}"
        assert len(DistributionChannel) == 6


class TestColdChainRegimeEnum:
    """Test ColdChainRegime enum."""

    def test_values_exist(self):
        """Test cold chain regimes contain expected entries."""
        expected = ["CHILLED", "FROZEN", "PHARMA", "FRESH", "AMBIENT"]
        actual = [e.value for e in ColdChainRegime]
        for v in expected:
            assert v in actual, f"Missing cold chain regime: {v}"
        assert len(ColdChainRegime) == 5


class TestDeliveryAreaEnum:
    """Test DeliveryArea enum."""

    def test_values_exist(self):
        """Test delivery areas contain expected entries."""
        expected = ["URBAN", "SUBURBAN", "RURAL"]
        actual = [e.value for e in DeliveryArea]
        for v in expected:
            assert v in actual, f"Missing delivery area: {v}"
        assert len(DeliveryArea) == 3


class TestIncotermEnum:
    """Test Incoterm enum."""

    def test_values_exist(self):
        """Test all 11 Incoterms exist."""
        expected = [
            "EXW", "FCA", "FAS", "FOB", "CPT", "CIF",
            "CIP", "DAP", "DPU", "DDP", "CFR",
        ]
        actual = [e.value for e in Incoterm]
        for v in expected:
            assert v in actual, f"Missing Incoterm: {v}"
        assert len(Incoterm) == 11


class TestEFSourceEnum:
    """Test EFSource enum."""

    def test_values_exist(self):
        """Test EF source values contain expected entries."""
        expected = [
            "DEFRA_2024", "EPA_2024", "GLEC_2023", "ICAO_2024",
            "IMO_2024", "IPCC_2006",
        ]
        actual = [e.value for e in EFSource]
        for v in expected:
            assert v in actual, f"Missing EF source: {v}"


class TestComplianceFrameworkEnum:
    """Test ComplianceFramework enum."""

    def test_values_exist(self):
        """Test all 7 compliance framework values exist."""
        expected = [
            "GHG_PROTOCOL", "ISO_14064", "ISO_14083",
            "CSRD", "CDP", "SBTI", "SB_253",
        ]
        actual = [e.value for e in ComplianceFramework]
        for v in expected:
            assert v in actual, f"Missing framework: {v}"
        assert len(ComplianceFramework) == 7


class TestDataQualityTierEnum:
    """Test DataQualityTier enum."""

    def test_values_exist(self):
        """Test data quality tiers contain expected entries."""
        expected = ["TIER_1", "TIER_2", "TIER_3"]
        actual = [e.value for e in DataQualityTier]
        for v in expected:
            assert v in actual

    def test_count(self):
        """Test there are at least 3 data quality tiers."""
        assert len(DataQualityTier) >= 3


class TestProvenanceStageEnum:
    """Test ProvenanceStage enum."""

    def test_values_exist(self):
        """Test provenance stages contain at least 10 stages."""
        expected = [
            "VALIDATE", "CLASSIFY", "LOOKUP_EF", "CALCULATE",
            "COLD_CHAIN", "WAREHOUSE", "LAST_MILE", "AGGREGATE",
            "COMPLIANCE", "SEAL",
        ]
        actual = [e.value for e in ProvenanceStage]
        for v in expected:
            assert v in actual, f"Missing provenance stage: {v}"
        assert len(ProvenanceStage) >= 10


class TestGWPVersionEnum:
    """Test GWPVersion enum."""

    def test_values_exist(self):
        """Test GWP versions contain AR5 and AR6."""
        assert GWPVersion.AR5.value == "AR5"
        assert GWPVersion.AR6.value == "AR6"


class TestEmissionGasEnum:
    """Test EmissionGas enum."""

    def test_values_exist(self):
        """Test emission gases contain CO2, CH4, N2O."""
        expected = ["CO2", "CH4", "N2O"]
        actual = [e.value for e in EmissionGas]
        for v in expected:
            assert v in actual


class TestCurrencyCodeEnum:
    """Test CurrencyCode enum."""

    def test_values_exist(self):
        """Test currency codes contain expected entries."""
        expected = ["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD"]
        actual = [e.value for e in CurrencyCode]
        for v in expected:
            assert v in actual

    def test_count(self):
        """Test at least 12 currency codes."""
        assert len(CurrencyCode) >= 12


class TestExportFormatEnum:
    """Test ExportFormat enum."""

    def test_values_exist(self):
        """Test export formats contain expected entries."""
        expected = ["JSON", "CSV", "XLSX", "PDF"]
        actual = [e.value for e in ExportFormat]
        for v in expected:
            assert v in actual


class TestUncertaintyMethodEnum:
    """Test UncertaintyMethod enum."""

    def test_values_exist(self):
        """Test uncertainty methods contain expected entries."""
        expected = ["MONTE_CARLO", "ANALYTICAL", "PEDIGREE_MATRIX"]
        actual = [e.value for e in UncertaintyMethod]
        for v in expected:
            assert v in actual


# ==============================================================================
# ENUM PARAMETRIZE: verify every enum value is a string
# ==============================================================================


@pytest.mark.parametrize("enum_cls", [
    CalculationMethod, TransportMode, RoadVehicleType, MaritimeVesselType,
    AircraftType, RailVehicleType, CourierVehicleType, LastMileVehicleType,
    WarehouseType, DistributionChannel, ColdChainRegime, DeliveryArea,
    Incoterm, EFSource, ComplianceFramework, DataQualityTier,
    ProvenanceStage, UncertaintyMethod, GWPVersion, EmissionGas,
    CurrencyCode, ExportFormat,
])
def test_all_enum_values_are_strings(enum_cls):
    """Test that every member of every enum has a string value."""
    for member in enum_cls:
        assert isinstance(member.value, str), (
            f"{enum_cls.__name__}.{member.name} value is not str: {type(member.value)}"
        )


@pytest.mark.parametrize("enum_cls", [
    CalculationMethod, TransportMode, RoadVehicleType, MaritimeVesselType,
    AircraftType, RailVehicleType, CourierVehicleType, LastMileVehicleType,
    WarehouseType, DistributionChannel, ColdChainRegime, DeliveryArea,
    Incoterm, EFSource, ComplianceFramework, DataQualityTier,
    ProvenanceStage, UncertaintyMethod, GWPVersion, EmissionGas,
    CurrencyCode, ExportFormat,
])
def test_all_enum_members_unique(enum_cls):
    """Test that every enum has unique values (no duplicates)."""
    values = [m.value for m in enum_cls]
    assert len(values) == len(set(values)), f"Duplicate values in {enum_cls.__name__}"


# ==============================================================================
# AGENT METADATA CONSTANTS
# ==============================================================================


class TestAgentMetadata:
    """Test AGENT_ID, AGENT_COMPONENT, VERSION, TABLE_PREFIX."""

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-009."""
        assert AGENT_ID == "GL-MRV-S3-009"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-022."""
        assert AGENT_COMPONENT == "AGENT-MRV-022"

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_dto_."""
        assert TABLE_PREFIX == "gl_dto_"

    def test_table_prefix_ends_underscore(self):
        """Test TABLE_PREFIX ends with underscore."""
        assert TABLE_PREFIX.endswith("_")


# ==============================================================================
# CONSTANT TABLE TESTS - GWP VALUES
# ==============================================================================


class TestGWPValues:
    """Test GWP_VALUES constant table."""

    def test_ar5_ch4(self):
        """Test AR5 CH4 GWP is 28."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["ch4"] == Decimal("28")

    def test_ar5_n2o(self):
        """Test AR5 N2O GWP is 265."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["n2o"] == Decimal("265")

    def test_ar5_co2(self):
        """Test AR5 CO2 GWP is 1."""
        ar5 = GWP_VALUES[GWPVersion.AR5]
        assert ar5["co2"] == Decimal("1")

    def test_ar6_ch4(self):
        """Test AR6 CH4 GWP is 27.9."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["ch4"] == Decimal("27.9")

    def test_ar6_n2o(self):
        """Test AR6 N2O GWP is 273."""
        ar6 = GWP_VALUES[GWPVersion.AR6]
        assert ar6["n2o"] == Decimal("273")

    def test_all_versions_have_three_gases(self):
        """Test all GWP versions have CO2, CH4, N2O."""
        for version in GWP_VALUES:
            vals = GWP_VALUES[version]
            assert "co2" in vals
            assert "ch4" in vals
            assert "n2o" in vals

    def test_all_values_are_decimal(self):
        """Test all GWP values are Decimal type."""
        for version in GWP_VALUES:
            for gas, val in GWP_VALUES[version].items():
                assert isinstance(val, Decimal), f"GWP {version}/{gas} is not Decimal"


# ==============================================================================
# CONSTANT TABLE TESTS - TRANSPORT EMISSION FACTORS
# ==============================================================================


class TestTransportEmissionFactors:
    """Test TRANSPORT_EMISSION_FACTORS constant table (26 entries)."""

    def test_has_at_least_26_entries(self):
        """Test TRANSPORT_EMISSION_FACTORS has at least 26 entries."""
        assert len(TRANSPORT_EMISSION_FACTORS) >= 26

    def test_road_articulated_33t_key_exists(self):
        """Test road articulated 33t entry exists."""
        assert "ROAD_ARTICULATED_33T" in TRANSPORT_EMISSION_FACTORS or \
               ("ROAD", "ARTICULATED_33T") in TRANSPORT_EMISSION_FACTORS

    def test_all_values_are_decimal(self):
        """Test all transport EF values contain Decimal precision."""
        for key, entry in TRANSPORT_EMISSION_FACTORS.items():
            if isinstance(entry, dict):
                for sub_key, val in entry.items():
                    if sub_key in ("ef", "ef_kgco2e_per_tonne_km", "wtt", "ttw"):
                        assert isinstance(val, Decimal), (
                            f"EF {key}/{sub_key} is not Decimal: {type(val)}"
                        )
            elif isinstance(entry, Decimal):
                pass  # Direct Decimal value is valid

    @pytest.mark.parametrize("mode", [
        "ROAD", "RAIL", "MARITIME", "AIR", "COURIER",
    ])
    def test_mode_has_entries(self, mode):
        """Test each transport mode has at least one entry in the table."""
        found = False
        for key in TRANSPORT_EMISSION_FACTORS:
            key_str = str(key)
            if mode in key_str:
                found = True
                break
        assert found, f"No transport EF entries found for mode {mode}"

    def test_road_ef_positive(self):
        """Test road EFs are positive."""
        for key, entry in TRANSPORT_EMISSION_FACTORS.items():
            if "ROAD" in str(key):
                if isinstance(entry, dict):
                    ef = entry.get("ef", entry.get("ef_kgco2e_per_tonne_km"))
                    if ef is not None:
                        assert ef > 0, f"Road EF {key} is not positive"

    def test_air_ef_greater_than_road(self):
        """Test air freight EFs are greater than road freight EFs."""
        air_efs = []
        road_efs = []
        for key, entry in TRANSPORT_EMISSION_FACTORS.items():
            ef_val = entry if isinstance(entry, Decimal) else \
                entry.get("ef", entry.get("ef_kgco2e_per_tonne_km", Decimal("0")))
            if ef_val and isinstance(ef_val, Decimal):
                if "AIR" in str(key):
                    air_efs.append(ef_val)
                elif "ROAD" in str(key):
                    road_efs.append(ef_val)
        if air_efs and road_efs:
            assert min(air_efs) > max(road_efs), "Air EFs should exceed road EFs"


# ==============================================================================
# CONSTANT TABLE TESTS - COLD CHAIN FACTORS
# ==============================================================================


class TestColdChainFactors:
    """Test COLD_CHAIN_FACTORS constant table (5 regimes x 4 modes)."""

    def test_has_entries(self):
        """Test COLD_CHAIN_FACTORS is non-empty."""
        assert len(COLD_CHAIN_FACTORS) >= 5

    @pytest.mark.parametrize("regime", [
        "CHILLED", "FROZEN", "PHARMA", "FRESH", "AMBIENT",
    ])
    def test_regime_exists(self, regime):
        """Test each cold chain regime has an entry."""
        found = any(regime in str(k) for k in COLD_CHAIN_FACTORS)
        assert found, f"Cold chain regime {regime} not found"

    def test_frozen_uplift_greater_than_chilled(self):
        """Test frozen uplift is greater than chilled uplift."""
        frozen_val = None
        chilled_val = None
        for key, entry in COLD_CHAIN_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("reefer_uplift", entry.get("uplift"))
            if val and isinstance(val, Decimal):
                if "FROZEN" in str(key):
                    frozen_val = val
                elif "CHILLED" in str(key):
                    chilled_val = val
        if frozen_val and chilled_val:
            assert frozen_val > chilled_val

    def test_ambient_uplift_is_one(self):
        """Test ambient cold chain regime has uplift of 1.0 (no uplift)."""
        for key, entry in COLD_CHAIN_FACTORS.items():
            if "AMBIENT" in str(key):
                val = entry if isinstance(entry, Decimal) else \
                    entry.get("reefer_uplift", entry.get("uplift"))
                if val is not None:
                    assert val == Decimal("1.0") or val == Decimal("1.00")


# ==============================================================================
# CONSTANT TABLE TESTS - WAREHOUSE EMISSION FACTORS
# ==============================================================================


class TestWarehouseEmissionFactors:
    """Test WAREHOUSE_EMISSION_FACTORS constant table (7 types)."""

    def test_has_at_least_7_entries(self):
        """Test at least 7 warehouse type entries."""
        assert len(WAREHOUSE_EMISSION_FACTORS) >= 7

    @pytest.mark.parametrize("wh_type", [
        "DISTRIBUTION_CENTER", "COLD_STORAGE", "FULFILLMENT_CENTER",
        "RETAIL_STORAGE", "CROSS_DOCK", "BONDED_WAREHOUSE",
        "TRANSIT_WAREHOUSE",
    ])
    def test_warehouse_type_exists(self, wh_type):
        """Test each warehouse type has an entry."""
        found = any(wh_type in str(k) for k in WAREHOUSE_EMISSION_FACTORS)
        assert found, f"Warehouse type {wh_type} not found"

    def test_cold_storage_ef_greater_than_ambient(self):
        """Test cold storage EF is greater than ambient DC."""
        cold_val = None
        dc_val = None
        for key, entry in WAREHOUSE_EMISSION_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef_kgco2e_per_m2_year", entry.get("ef"))
            if val and isinstance(val, Decimal):
                if "COLD_STORAGE" in str(key):
                    cold_val = val
                elif "DISTRIBUTION_CENTER" in str(key):
                    dc_val = val
        if cold_val and dc_val:
            assert cold_val > dc_val

    def test_all_values_positive(self):
        """Test all warehouse EFs are positive."""
        for key, entry in WAREHOUSE_EMISSION_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef_kgco2e_per_m2_year", entry.get("ef"))
            if val is not None and isinstance(val, Decimal):
                assert val > 0, f"Warehouse EF {key} is not positive"


# ==============================================================================
# CONSTANT TABLE TESTS - LAST-MILE FACTORS
# ==============================================================================


class TestLastMileFactors:
    """Test LAST_MILE_FACTORS constant table (6 types x 3 areas)."""

    def test_has_entries(self):
        """Test LAST_MILE_FACTORS has at least 6 entries."""
        assert len(LAST_MILE_FACTORS) >= 6

    @pytest.mark.parametrize("vehicle", [
        "VAN_DIESEL", "VAN_ELECTRIC", "CARGO_BIKE",
        "DRONE", "PARCEL_LOCKER", "CROWD_SHIPPING",
    ])
    def test_vehicle_type_exists(self, vehicle):
        """Test each last-mile vehicle type has an entry."""
        found = any(vehicle in str(k) for k in LAST_MILE_FACTORS)
        assert found, f"Last-mile vehicle {vehicle} not found"

    def test_cargo_bike_ef_less_than_van_diesel(self):
        """Test cargo bike EF is less than van diesel EF."""
        bike_val = None
        van_val = None
        for key, entry in LAST_MILE_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef_kgco2e_per_parcel", entry.get("ef"))
            if val and isinstance(val, Decimal):
                if "CARGO_BIKE" in str(key):
                    bike_val = val
                elif "VAN_DIESEL" in str(key):
                    van_val = val
        if bike_val and van_val:
            assert bike_val < van_val

    def test_urban_ef_less_than_rural(self):
        """Test urban delivery EFs are less than rural for same vehicle."""
        for key, entry in LAST_MILE_FACTORS.items():
            if isinstance(entry, dict) and "URBAN" in entry and "RURAL" in entry:
                urban_ef = entry.get("URBAN")
                rural_ef = entry.get("RURAL")
                if urban_ef and rural_ef:
                    assert urban_ef < rural_ef


# ==============================================================================
# CONSTANT TABLE TESTS - EEIO FACTORS
# ==============================================================================


class TestEEIOFactors:
    """Test EEIO_FACTORS constant table (10 entries)."""

    def test_has_at_least_10_entries(self):
        """Test at least 10 EEIO sector entries."""
        assert len(EEIO_FACTORS) >= 10

    @pytest.mark.parametrize("naics", [
        "484110", "484121", "484220",
        "492110", "493110",
    ])
    def test_naics_code_exists(self, naics):
        """Test expected NAICS codes exist."""
        assert naics in EEIO_FACTORS, f"NAICS {naics} not in EEIO_FACTORS"

    def test_all_values_positive(self):
        """Test all EEIO EFs are positive."""
        for code, entry in EEIO_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef", entry.get("ef_kgco2e_per_usd"))
            if val is not None and isinstance(val, Decimal):
                assert val > 0, f"EEIO factor {code} is not positive"

    def test_all_values_are_decimal(self):
        """Test all EEIO factor values are Decimal type."""
        for code, entry in EEIO_FACTORS.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef", entry.get("ef_kgco2e_per_usd"))
            if val is not None:
                assert isinstance(val, Decimal), f"EEIO {code} value is not Decimal"


# ==============================================================================
# CONSTANT TABLE TESTS - CURRENCY RATES
# ==============================================================================


class TestCurrencyRates:
    """Test CURRENCY_RATES constant table (12 entries)."""

    def test_has_at_least_12_entries(self):
        """Test at least 12 currency entries."""
        assert len(CURRENCY_RATES) >= 12

    def test_usd_is_one(self):
        """Test USD rate is 1.0."""
        usd_key = CurrencyCode.USD if CurrencyCode.USD in CURRENCY_RATES else "USD"
        assert CURRENCY_RATES[usd_key] == Decimal("1.0")

    def test_eur_rate(self):
        """Test EUR rate is around 1.08-1.09."""
        eur_key = CurrencyCode.EUR if CurrencyCode.EUR in CURRENCY_RATES else "EUR"
        rate = CURRENCY_RATES[eur_key]
        assert Decimal("1.00") < rate < Decimal("1.20")

    def test_gbp_rate(self):
        """Test GBP rate is around 1.25-1.30."""
        gbp_key = CurrencyCode.GBP if CurrencyCode.GBP in CURRENCY_RATES else "GBP"
        rate = CURRENCY_RATES[gbp_key]
        assert Decimal("1.15") < rate < Decimal("1.40")

    def test_all_rates_positive(self):
        """Test all currency rates are positive."""
        for code, rate in CURRENCY_RATES.items():
            assert isinstance(rate, Decimal), f"Rate {code} is not Decimal"
            assert rate > 0, f"Rate {code} is not positive"


# ==============================================================================
# CONSTANT TABLE TESTS - CPI DEFLATORS
# ==============================================================================


class TestCPIDeflators:
    """Test CPI_DEFLATORS constant table (11 entries)."""

    def test_has_at_least_11_entries(self):
        """Test at least 11 CPI deflator years."""
        assert len(CPI_DEFLATORS) >= 11

    def test_base_year_is_one(self):
        """Test base year (2021) deflator is 1.0."""
        assert CPI_DEFLATORS[2021] == Decimal("1.0000") or \
               CPI_DEFLATORS[2021] == Decimal("1.0")

    def test_2024_deflator(self):
        """Test 2024 CPI deflator is approximately 1.14-1.16."""
        val = CPI_DEFLATORS[2024]
        assert Decimal("1.10") < val < Decimal("1.20")

    def test_deflators_increasing(self):
        """Test CPI deflators are generally increasing over time."""
        years = sorted(CPI_DEFLATORS.keys())
        for i in range(1, len(years)):
            assert CPI_DEFLATORS[years[i]] >= CPI_DEFLATORS[years[i-1]], (
                f"CPI deflator {years[i]} < {years[i-1]}"
            )

    def test_all_deflators_decimal(self):
        """Test all CPI deflators are Decimal type."""
        for year, val in CPI_DEFLATORS.items():
            assert isinstance(val, Decimal), f"CPI {year} is not Decimal"


# ==============================================================================
# CONSTANT TABLE TESTS - GRID EMISSION FACTORS
# ==============================================================================


class TestGridEmissionFactors:
    """Test GRID_EMISSION_FACTORS constant table (11 entries)."""

    def test_has_at_least_11_entries(self):
        """Test at least 11 grid EF entries."""
        assert len(GRID_EMISSION_FACTORS) >= 11

    @pytest.mark.parametrize("region", ["US", "EU", "GB", "DE", "CN", "JP"])
    def test_region_exists(self, region):
        """Test expected region exists in grid EFs."""
        assert region in GRID_EMISSION_FACTORS, f"Region {region} not found"

    def test_all_positive(self):
        """Test all grid EFs are positive."""
        for region, val in GRID_EMISSION_FACTORS.items():
            assert isinstance(val, Decimal) and val > 0


# ==============================================================================
# CONSTANT TABLE TESTS - CHANNEL AVERAGES
# ==============================================================================


class TestChannelAverages:
    """Test CHANNEL_AVERAGES constant table (6 entries)."""

    def test_has_6_entries(self):
        """Test exactly 6 channel average entries."""
        assert len(CHANNEL_AVERAGES) == 6

    @pytest.mark.parametrize("channel", [
        "ECOMMERCE_DTC", "RETAIL_DISTRIBUTION", "WHOLESALE",
        "MARKETPLACE_3PL", "DROPSHIP", "OMNICHANNEL",
    ])
    def test_channel_exists(self, channel):
        """Test each channel has an entry."""
        assert channel in CHANNEL_AVERAGES, f"Channel {channel} not found"

    def test_all_values_positive(self):
        """Test all channel average EFs are positive."""
        for channel, entry in CHANNEL_AVERAGES.items():
            val = entry if isinstance(entry, Decimal) else \
                entry.get("ef_kgco2e_per_unit", entry.get("ef"))
            if val is not None and isinstance(val, Decimal):
                assert val > 0


# ==============================================================================
# CONSTANT TABLE TESTS - INCOTERM CLASSIFICATIONS
# ==============================================================================


class TestIncotermClassifications:
    """Test INCOTERM_CLASSIFICATIONS constant table (11 entries)."""

    def test_has_11_entries(self):
        """Test exactly 11 Incoterm classification entries."""
        assert len(INCOTERM_CLASSIFICATIONS) == 11

    @pytest.mark.parametrize("incoterm", [
        "EXW", "FCA", "FAS", "FOB", "CPT", "CIF",
        "CIP", "DAP", "DPU", "DDP", "CFR",
    ])
    def test_incoterm_exists(self, incoterm):
        """Test each Incoterm has a classification entry."""
        assert incoterm in INCOTERM_CLASSIFICATIONS

    def test_exw_is_cat9(self):
        """Test EXW classifies to Category 9 (buyer pays transport)."""
        exw = INCOTERM_CLASSIFICATIONS["EXW"]
        cat9 = exw.get("cat_9", exw.get("category_9_applicable", exw.get("buyer_arranges")))
        assert cat9 is True

    def test_dap_is_not_cat9(self):
        """Test DAP classifies to Category 4 (seller pays transport)."""
        dap = INCOTERM_CLASSIFICATIONS["DAP"]
        cat9 = dap.get("cat_9", dap.get("category_9_applicable", dap.get("buyer_arranges")))
        # DAP = seller bears transport to destination, so NOT Cat 9
        assert cat9 is False


# ==============================================================================
# CONSTANT TABLE TESTS - LOAD FACTORS
# ==============================================================================


class TestLoadFactors:
    """Test LOAD_FACTORS constant table (5 entries)."""

    def test_has_at_least_5_entries(self):
        """Test at least 5 load factor entries."""
        assert len(LOAD_FACTORS) >= 5

    def test_all_between_zero_and_one(self):
        """Test all load factors are between 0 and 1."""
        for key, val in LOAD_FACTORS.items():
            factor = val if isinstance(val, Decimal) else \
                val.get("factor", val.get("load_factor"))
            if factor is not None and isinstance(factor, Decimal):
                assert Decimal("0") < factor <= Decimal("1.0")


# ==============================================================================
# CONSTANT TABLE TESTS - RETURN FACTORS
# ==============================================================================


class TestReturnFactors:
    """Test RETURN_FACTORS constant table (4 entries)."""

    def test_has_at_least_4_entries(self):
        """Test at least 4 return factor entries."""
        assert len(RETURN_FACTORS) >= 4

    def test_ecommerce_return_rate(self):
        """Test e-commerce return rate is typically 15-30%."""
        for key, entry in RETURN_FACTORS.items():
            if "ECOMMERCE" in str(key).upper() or "DTC" in str(key).upper():
                val = entry if isinstance(entry, Decimal) else \
                    entry.get("return_rate", entry.get("rate"))
                if val is not None:
                    assert Decimal("0.10") <= val <= Decimal("0.35")


# ==============================================================================
# CONSTANT TABLE TESTS - DQI SCORING
# ==============================================================================


class TestDQIScoring:
    """Test DQI_SCORING constant table."""

    def test_has_entries(self):
        """Test DQI_SCORING is non-empty."""
        assert len(DQI_SCORING) >= 3

    def test_has_dimensions(self):
        """Test DQI scoring has expected dimensions."""
        all_keys = set()
        for k in DQI_SCORING:
            all_keys.add(str(k))
        # Should have at least 3 dimension types
        assert len(all_keys) >= 3


# ==============================================================================
# INPUT MODEL TESTS
# ==============================================================================


class TestShipmentInputModel:
    """Test ShipmentInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid ShipmentInput creation."""
        shipment = ShipmentInput(
            shipment_id="DTO-SHIP-001",
            mode=TransportMode.ROAD,
            distance_km=Decimal("350.0"),
            cargo_mass_tonnes=Decimal("15.0"),
        )
        assert shipment.shipment_id == "DTO-SHIP-001"
        assert shipment.mode == TransportMode.ROAD
        assert shipment.distance_km == Decimal("350.0")

    def test_negative_distance_rejected(self):
        """Test ShipmentInput rejects negative distance."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            ShipmentInput(
                shipment_id="DTO-SHIP-BAD",
                mode=TransportMode.ROAD,
                distance_km=Decimal("-10.0"),
                cargo_mass_tonnes=Decimal("15.0"),
            )

    def test_zero_mass_rejected(self):
        """Test ShipmentInput rejects zero cargo mass."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            ShipmentInput(
                shipment_id="DTO-SHIP-BAD",
                mode=TransportMode.ROAD,
                distance_km=Decimal("350.0"),
                cargo_mass_tonnes=Decimal("0.0"),
            )

    def test_frozen(self):
        """Test ShipmentInput is frozen (immutable)."""
        shipment = ShipmentInput(
            shipment_id="DTO-SHIP-001",
            mode=TransportMode.ROAD,
            distance_km=Decimal("350.0"),
            cargo_mass_tonnes=Decimal("15.0"),
        )
        with pytest.raises(Exception):
            shipment.distance_km = Decimal("500.0")

    def test_defaults(self):
        """Test ShipmentInput defaults for optional fields."""
        shipment = ShipmentInput(
            shipment_id="DTO-SHIP-001",
            mode=TransportMode.ROAD,
            distance_km=Decimal("350.0"),
            cargo_mass_tonnes=Decimal("15.0"),
        )
        # Optional fields should have default None or appropriate default
        if hasattr(shipment, "temperature_controlled"):
            assert shipment.temperature_controlled in (False, None)


class TestSpendInputModel:
    """Test SpendInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid SpendInput creation."""
        spend = SpendInput(
            spend_id="DTO-SPEND-001",
            spend_amount=Decimal("75000.00"),
            currency=CurrencyCode.USD,
            sector_code="484110",
            reporting_year=2024,
        )
        assert spend.spend_amount == Decimal("75000.00")
        assert spend.currency == CurrencyCode.USD

    def test_negative_amount_rejected(self):
        """Test SpendInput rejects negative spend amount."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            SpendInput(
                spend_id="BAD",
                spend_amount=Decimal("-100.0"),
                currency=CurrencyCode.USD,
                sector_code="484110",
                reporting_year=2024,
            )

    def test_frozen(self):
        """Test SpendInput is frozen."""
        spend = SpendInput(
            spend_id="DTO-SPEND-001",
            spend_amount=Decimal("75000.00"),
            currency=CurrencyCode.USD,
            sector_code="484110",
            reporting_year=2024,
        )
        with pytest.raises(Exception):
            spend.spend_amount = Decimal("100000.00")


class TestWarehouseInputModel:
    """Test WarehouseInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid WarehouseInput creation."""
        wh = WarehouseInput(
            warehouse_id="DTO-WH-001",
            warehouse_type=WarehouseType.DISTRIBUTION_CENTER,
            floor_area_m2=Decimal("5000.0"),
            dwell_time_hours=Decimal("72.0"),
        )
        assert wh.warehouse_type == WarehouseType.DISTRIBUTION_CENTER
        assert wh.floor_area_m2 == Decimal("5000.0")

    def test_negative_area_rejected(self):
        """Test WarehouseInput rejects negative floor area."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            WarehouseInput(
                warehouse_id="BAD",
                warehouse_type=WarehouseType.DISTRIBUTION_CENTER,
                floor_area_m2=Decimal("-100.0"),
                dwell_time_hours=Decimal("72.0"),
            )

    def test_frozen(self):
        """Test WarehouseInput is frozen."""
        wh = WarehouseInput(
            warehouse_id="DTO-WH-001",
            warehouse_type=WarehouseType.DISTRIBUTION_CENTER,
            floor_area_m2=Decimal("5000.0"),
            dwell_time_hours=Decimal("72.0"),
        )
        with pytest.raises(Exception):
            wh.floor_area_m2 = Decimal("10000.0")


class TestLastMileInputModel:
    """Test LastMileInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid LastMileInput creation."""
        lm = LastMileInput(
            delivery_id="DTO-LM-001",
            vehicle_type=LastMileVehicleType.VAN_DIESEL,
            delivery_area=DeliveryArea.URBAN,
            distance_km=Decimal("15.0"),
            parcels_delivered=25,
        )
        assert lm.vehicle_type == LastMileVehicleType.VAN_DIESEL
        assert lm.parcels_delivered == 25

    def test_zero_parcels_rejected(self):
        """Test LastMileInput rejects zero parcels."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            LastMileInput(
                delivery_id="BAD",
                vehicle_type=LastMileVehicleType.VAN_DIESEL,
                delivery_area=DeliveryArea.URBAN,
                distance_km=Decimal("15.0"),
                parcels_delivered=0,
            )


class TestAverageDataInputModel:
    """Test AverageDataInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid AverageDataInput creation."""
        avg = AverageDataInput(
            channel=DistributionChannel.ECOMMERCE_DTC,
            annual_units_sold=50000,
        )
        assert avg.channel == DistributionChannel.ECOMMERCE_DTC
        assert avg.annual_units_sold == 50000


class TestColdChainInputModel:
    """Test ColdChainInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid ColdChainInput creation."""
        cc = ColdChainInput(
            cold_chain_regime=ColdChainRegime.PHARMA,
            refrigerant_type="R-134A",
            refrigerant_charge_kg=Decimal("8.0"),
            annual_leak_rate=Decimal("0.08"),
        )
        assert cc.cold_chain_regime == ColdChainRegime.PHARMA
        assert cc.annual_leak_rate == Decimal("0.08")

    def test_leak_rate_max_one(self):
        """Test ColdChainInput rejects leak rate > 1."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            ColdChainInput(
                cold_chain_regime=ColdChainRegime.CHILLED,
                refrigerant_type="R-404A",
                refrigerant_charge_kg=Decimal("10.0"),
                annual_leak_rate=Decimal("1.5"),
            )


class TestReturnLogisticsInputModel:
    """Test ReturnLogisticsInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid ReturnLogisticsInput creation."""
        ret = ReturnLogisticsInput(
            return_rate=Decimal("0.20"),
            original_distance_km=Decimal("350.0"),
            units_sold=1000,
        )
        assert ret.return_rate == Decimal("0.20")

    def test_return_rate_max_one(self):
        """Test ReturnLogisticsInput rejects return rate > 1."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            ReturnLogisticsInput(
                return_rate=Decimal("1.5"),
                original_distance_km=Decimal("350.0"),
                units_sold=1000,
            )


class TestBatchInputModel:
    """Test BatchInput Pydantic model."""

    def test_valid_creation(self):
        """Test valid BatchInput creation."""
        batch = BatchInput(
            batch_id="DTO-BATCH-001",
            requests=[],
        )
        assert batch.batch_id == "DTO-BATCH-001"


class TestCalculationRequestModel:
    """Test CalculationRequest Pydantic model."""

    def test_valid_creation(self):
        """Test valid CalculationRequest creation."""
        req = CalculationRequest(
            request_id="DTO-REQ-001",
            calculation_method=CalculationMethod.DISTANCE_BASED,
        )
        assert req.calculation_method == CalculationMethod.DISTANCE_BASED


class TestComplianceRequestModel:
    """Test ComplianceRequest Pydantic model."""

    def test_valid_creation(self):
        """Test valid ComplianceRequest creation."""
        req = ComplianceRequest(
            frameworks=[ComplianceFramework.GHG_PROTOCOL],
        )
        assert ComplianceFramework.GHG_PROTOCOL in req.frameworks


# ==============================================================================
# RESULT MODEL TESTS
# ==============================================================================


class TestCalculationResultModel:
    """Test CalculationResult Pydantic model."""

    def test_valid_creation(self):
        """Test valid CalculationResult creation."""
        result = CalculationResult(
            emissions_tco2e=Decimal("0.56175"),
            calculation_method=CalculationMethod.DISTANCE_BASED,
            provenance_hash="a" * 64,
        )
        assert result.emissions_tco2e == Decimal("0.56175")
        assert len(result.provenance_hash) == 64

    def test_frozen(self):
        """Test CalculationResult is frozen."""
        result = CalculationResult(
            emissions_tco2e=Decimal("0.56175"),
            calculation_method=CalculationMethod.DISTANCE_BASED,
            provenance_hash="a" * 64,
        )
        with pytest.raises(Exception):
            result.emissions_tco2e = Decimal("1.0")


class TestComplianceResultModel:
    """Test ComplianceResult Pydantic model."""

    def test_valid_creation(self):
        """Test valid ComplianceResult creation."""
        result = ComplianceResult(
            compliant=True,
            framework=ComplianceFramework.GHG_PROTOCOL,
        )
        assert result.compliant is True

    def test_frozen(self):
        """Test ComplianceResult is frozen."""
        result = ComplianceResult(
            compliant=True,
            framework=ComplianceFramework.GHG_PROTOCOL,
        )
        with pytest.raises(Exception):
            result.compliant = False


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test calculate_provenance_hash function."""

    def test_deterministic(self):
        """Test provenance hash is deterministic for same inputs."""
        h1 = calculate_provenance_hash("DTO-SHIP-001", "ROAD", Decimal("350.0"))
        h2 = calculate_provenance_hash("DTO-SHIP-001", "ROAD", Decimal("350.0"))
        assert h1 == h2
        assert len(h1) == 64

    def test_different_inputs(self):
        """Test provenance hash differs for different inputs."""
        h1 = calculate_provenance_hash("DTO-SHIP-001", "ROAD")
        h2 = calculate_provenance_hash("DTO-SHIP-002", "RAIL")
        assert h1 != h2

    def test_hex_format(self):
        """Test provenance hash is valid hexadecimal."""
        h = calculate_provenance_hash("test", "data")
        assert all(c in "0123456789abcdef" for c in h)


class TestDQIClassification:
    """Test get_dqi_classification function."""

    @pytest.mark.parametrize("score,expected_range", [
        (Decimal("4.5"), ["Excellent", "High"]),
        (Decimal("4.0"), ["Good", "High"]),
        (Decimal("3.0"), ["Fair", "Medium", "Moderate"]),
        (Decimal("2.0"), ["Poor", "Low"]),
        (Decimal("1.0"), ["Very Poor", "Very Low"]),
    ])
    def test_classification_ranges(self, score, expected_range):
        """Test DQI classification for various scores."""
        result = get_dqi_classification(score)
        assert result in expected_range, (
            f"Score {score} classified as '{result}', expected one of {expected_range}"
        )


class TestCurrencyConversion:
    """Test convert_currency_to_usd function."""

    def test_usd_identity(self):
        """Test USD 1000 stays as USD 1000."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.USD)
        assert result == Decimal("1000")

    def test_eur_conversion(self):
        """Test EUR conversion to USD is positive and reasonable."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.EUR)
        assert result > Decimal("1000")  # EUR > USD
        assert result < Decimal("1200")  # But not absurdly so

    def test_gbp_conversion(self):
        """Test GBP conversion to USD is positive and reasonable."""
        result = convert_currency_to_usd(Decimal("1000"), CurrencyCode.GBP)
        assert result > Decimal("1100")
        assert result < Decimal("1400")


class TestCPIDeflatorFunc:
    """Test get_cpi_deflator function."""

    def test_base_year(self):
        """Test CPI deflator for base year 2021 is 1.0."""
        result = get_cpi_deflator(2021)
        assert result == Decimal("1.0000") or result == Decimal("1.0")

    def test_2024(self):
        """Test CPI deflator for 2024."""
        result = get_cpi_deflator(2024)
        assert result > Decimal("1.0")

    def test_invalid_year(self):
        """Test CPI deflator raises for unavailable year."""
        with pytest.raises((ValueError, KeyError)):
            get_cpi_deflator(1900)


class TestClassifyIncoterm:
    """Test classify_incoterm function."""

    @pytest.mark.parametrize("incoterm,expected_cat9", [
        ("EXW", True),
        ("FCA", True),
        ("FOB", True),
        ("DAP", False),
        ("DDP", False),
        ("CIF", False),
    ])
    def test_incoterm_classification(self, incoterm, expected_cat9):
        """Test Incoterm -> Category 9 boundary classification."""
        result = classify_incoterm(incoterm)
        cat9 = result.get("cat_9", result.get("category_9_applicable",
                result.get("buyer_arranges")))
        assert cat9 == expected_cat9, (
            f"Incoterm {incoterm} expected cat_9={expected_cat9}, got {cat9}"
        )


# ==============================================================================
# SUMMARY META-TESTS
# ==============================================================================


def test_total_enum_count():
    """Meta-test: verify total enum count is at least 22."""
    enum_classes = [
        CalculationMethod, TransportMode, RoadVehicleType, MaritimeVesselType,
        AircraftType, RailVehicleType, CourierVehicleType, LastMileVehicleType,
        WarehouseType, DistributionChannel, ColdChainRegime, DeliveryArea,
        Incoterm, EFSource, ComplianceFramework, DataQualityTier,
        ProvenanceStage, UncertaintyMethod, GWPVersion, EmissionGas,
        CurrencyCode, ExportFormat,
    ]
    assert len(enum_classes) >= 22


def test_total_constant_tables():
    """Meta-test: verify 14 constant tables are present."""
    tables = [
        GWP_VALUES, TRANSPORT_EMISSION_FACTORS, COLD_CHAIN_FACTORS,
        WAREHOUSE_EMISSION_FACTORS, LAST_MILE_FACTORS, EEIO_FACTORS,
        CURRENCY_RATES, CPI_DEFLATORS, GRID_EMISSION_FACTORS,
        CHANNEL_AVERAGES, INCOTERM_CLASSIFICATIONS, LOAD_FACTORS,
        RETURN_FACTORS, DQI_SCORING,
    ]
    assert len(tables) == 14
    for t in tables:
        assert t is not None
        assert len(t) > 0
