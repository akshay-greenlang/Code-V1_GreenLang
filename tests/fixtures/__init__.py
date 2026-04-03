# -*- coding: utf-8 -*-
"""
GreenLang Test Fixtures Package.

Provides modular, importable test utilities organised into four sub-modules:

- **constants** -- shared IDs, mock credentials, emission factors, GWP values, tolerances
- **mocks** -- generic infrastructure mock factories (Redis, S3, HTTP, Prometheus, DB pool, agents)
- **generators** -- deterministic test-data generators (emissions, supply chain, facilities, compliance, etc.)
- **helpers** -- assertion helpers (provenance hash, Decimal close, agent response, PII, timestamps)

Quick start::

    # In a conftest.py
    from tests.fixtures import (
        TEST_TENANT_ID,
        MockRedisClient,
        EmissionDataGenerator,
        assert_valid_provenance_hash,
    )

    @pytest.fixture
    def redis():
        return MockRedisClient()
"""

# -- Constants ----------------------------------------------------------------
from tests.fixtures.constants import (
    CBAM_BENCHMARK_ALUMINUM,
    CBAM_BENCHMARK_CEMENT_CLINKER,
    CBAM_BENCHMARK_STEEL_HRC,
    COAL_EF_KG_CO2E_PER_KG,
    DEFAULT_DECIMAL_PLACES,
    DEFAULT_FLOAT_ABS_TOL,
    DEFAULT_FLOAT_REL_TOL,
    DIESEL_EF_KG_CO2E_PER_GALLON,
    DIESEL_EF_KG_CO2E_PER_LITRE,
    ELECTRICITY_EU_AVG_KG_CO2E_PER_KWH,
    ELECTRICITY_UK_KG_CO2E_PER_KWH,
    ELECTRICITY_US_AVG_KG_CO2E_PER_KWH,
    EMISSIONS_DECIMAL_PLACES,
    FINANCIAL_DECIMAL_PLACES,
    GWP_CH4,
    GWP_CH4_BIOGENIC,
    GWP_CO2,
    GWP_HFC134A,
    GWP_N2O,
    GWP_R410A,
    GWP_SF6,
    HIGH_DATA_QUALITY_SCORE,
    LPG_EF_KG_CO2E_PER_LITRE,
    MIN_DATA_QUALITY_SCORE,
    NATURAL_GAS_EF_KG_CO2E_PER_MJ,
    NATURAL_GAS_EF_KG_CO2E_PER_THERM,
    TEST_AGENT_ID,
    TEST_API_KEY,
    TEST_BASE_YEAR,
    TEST_CUTOFF_DATE,
    TEST_ERP_HOST,
    TEST_FACILITY_ID,
    TEST_JWT_TOKEN,
    TEST_ORGANIZATION_ID,
    TEST_REPORTING_YEAR,
    TEST_SESSION_ID,
    TEST_TENANT_ID,
    TEST_USER_ID,
    TEST_VAULT_TOKEN,
    VERIFICATION_CONFIDENCE_THRESHOLD,
)

# -- Mocks --------------------------------------------------------------------
from tests.fixtures.mocks import (
    MockAsyncRedisClient,
    MockDBConnection,
    MockDBPool,
    MockHTTPClient,
    MockHTTPResponse,
    MockPrometheusRegistry,
    MockRedisClient,
    MockRedisPipeline,
    MockS3Client,
    create_mock_agents,
    create_mock_config,
    create_mock_db_pool,
)

# -- Generators (existing) ----------------------------------------------------
from tests.fixtures.generators import (
    BaseTestDataGenerator,
    BuildingEnergyGenerator,
    CBAMShipmentGenerator,
    ComplianceDataGenerator,
    EdgeCaseGenerator,
    EmissionDataGenerator,
    EUDRCommodityGenerator,
    FacilityDataGenerator,
    IndustrialTestDataGenerator,
    SupplyChainDataGenerator,
    create_test_generators,
)

# -- Helpers -------------------------------------------------------------------
from tests.fixtures.helpers import (
    assert_decimal_close,
    assert_dict_has_keys,
    assert_emissions_result,
    assert_float_close,
    assert_list_of_dicts,
    assert_no_pii,
    assert_non_negative,
    assert_positive,
    assert_valid_agent_response,
    assert_valid_audit_entry,
    assert_valid_iso_timestamp,
    assert_valid_md5_hash,
    assert_valid_provenance_hash,
)


__all__ = [
    # --- Constants ---
    "CBAM_BENCHMARK_ALUMINUM",
    "CBAM_BENCHMARK_CEMENT_CLINKER",
    "CBAM_BENCHMARK_STEEL_HRC",
    "COAL_EF_KG_CO2E_PER_KG",
    "DEFAULT_DECIMAL_PLACES",
    "DEFAULT_FLOAT_ABS_TOL",
    "DEFAULT_FLOAT_REL_TOL",
    "DIESEL_EF_KG_CO2E_PER_GALLON",
    "DIESEL_EF_KG_CO2E_PER_LITRE",
    "ELECTRICITY_EU_AVG_KG_CO2E_PER_KWH",
    "ELECTRICITY_UK_KG_CO2E_PER_KWH",
    "ELECTRICITY_US_AVG_KG_CO2E_PER_KWH",
    "EMISSIONS_DECIMAL_PLACES",
    "FINANCIAL_DECIMAL_PLACES",
    "GWP_CH4",
    "GWP_CH4_BIOGENIC",
    "GWP_CO2",
    "GWP_HFC134A",
    "GWP_N2O",
    "GWP_R410A",
    "GWP_SF6",
    "HIGH_DATA_QUALITY_SCORE",
    "LPG_EF_KG_CO2E_PER_LITRE",
    "MIN_DATA_QUALITY_SCORE",
    "NATURAL_GAS_EF_KG_CO2E_PER_MJ",
    "NATURAL_GAS_EF_KG_CO2E_PER_THERM",
    "TEST_AGENT_ID",
    "TEST_API_KEY",
    "TEST_BASE_YEAR",
    "TEST_CUTOFF_DATE",
    "TEST_ERP_HOST",
    "TEST_FACILITY_ID",
    "TEST_JWT_TOKEN",
    "TEST_ORGANIZATION_ID",
    "TEST_REPORTING_YEAR",
    "TEST_SESSION_ID",
    "TEST_TENANT_ID",
    "TEST_USER_ID",
    "TEST_VAULT_TOKEN",
    "VERIFICATION_CONFIDENCE_THRESHOLD",
    # --- Mocks ---
    "MockAsyncRedisClient",
    "MockDBConnection",
    "MockDBPool",
    "MockHTTPClient",
    "MockHTTPResponse",
    "MockPrometheusRegistry",
    "MockRedisClient",
    "MockRedisPipeline",
    "MockS3Client",
    "create_mock_agents",
    "create_mock_config",
    "create_mock_db_pool",
    # --- Generators ---
    "BaseTestDataGenerator",
    "BuildingEnergyGenerator",
    "CBAMShipmentGenerator",
    "ComplianceDataGenerator",
    "EdgeCaseGenerator",
    "EmissionDataGenerator",
    "EUDRCommodityGenerator",
    "FacilityDataGenerator",
    "IndustrialTestDataGenerator",
    "SupplyChainDataGenerator",
    "create_test_generators",
    # --- Helpers ---
    "assert_decimal_close",
    "assert_dict_has_keys",
    "assert_emissions_result",
    "assert_float_close",
    "assert_list_of_dicts",
    "assert_no_pii",
    "assert_non_negative",
    "assert_positive",
    "assert_valid_agent_response",
    "assert_valid_audit_entry",
    "assert_valid_iso_timestamp",
    "assert_valid_md5_hash",
    "assert_valid_provenance_hash",
]
