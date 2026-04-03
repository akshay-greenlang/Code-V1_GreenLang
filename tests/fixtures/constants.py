# -*- coding: utf-8 -*-
"""
Shared test constants used across the GreenLang test suite.

These constants provide deterministic, well-known values for test
assertions. All numeric emission factors use ``Decimal`` for precision.
Mock credentials are clearly fake and safe to commit.

Usage::

    from tests.fixtures.constants import (
        TEST_TENANT_ID,
        DIESEL_EF_KG_CO2E_PER_GALLON,
        GWP_CH4,
    )
"""

from decimal import Decimal


# =============================================================================
# Test Identity Constants
# =============================================================================

TEST_TENANT_ID: str = "tenant-test-001"
"""Default tenant ID for isolated test runs."""

TEST_USER_ID: str = "user-test-001"
"""Default user ID for authentication stubs."""

TEST_FACILITY_ID: str = "facility-test-001"
"""Default facility ID for location-bound tests."""

TEST_ORGANIZATION_ID: str = "org-test-001"
"""Default organization ID for multi-tenant tests."""

TEST_AGENT_ID: str = "agent-test-001"
"""Default agent ID for pipeline tests."""

TEST_SESSION_ID: str = "session-test-001"
"""Default session ID for request-scoped tests."""


# =============================================================================
# Mock Credentials (clearly fake -- safe to commit)
# =============================================================================

TEST_API_KEY: str = "gl-test-key-00000000"
"""Fake API key for gateway/auth tests."""

TEST_JWT_TOKEN: str = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiJ1c2VyLXRlc3QtMDAxIiwidGVuYW50IjoidGVuYW50LXRlc3QtMDAxIn0"
    ".test_signature_not_real"
)
"""Fake JWT for auth middleware tests. Decodes to sub=user-test-001."""

TEST_ERP_HOST: str = "https://erp.test.greenlang.local"
"""Fake ERP endpoint for connector tests."""

TEST_VAULT_TOKEN: str = "hvs.test-vault-token-00000000"
"""Fake Vault token for secrets-manager tests."""


# =============================================================================
# Emission Factor Constants (authoritative test values)
# =============================================================================
# Sources: EPA 40 CFR 98, UK DEFRA 2023, eGRID 2023
# All values are Decimal for deterministic arithmetic.

DIESEL_EF_KG_CO2E_PER_GALLON: Decimal = Decimal("10.21")
"""EPA diesel combustion factor (kg CO2e / US gallon)."""

DIESEL_EF_KG_CO2E_PER_LITRE: Decimal = Decimal("2.68")
"""DEFRA diesel combustion factor (kg CO2e / litre)."""

NATURAL_GAS_EF_KG_CO2E_PER_THERM: Decimal = Decimal("5.30")
"""EPA natural gas combustion factor (kg CO2e / therm)."""

NATURAL_GAS_EF_KG_CO2E_PER_MJ: Decimal = Decimal("0.0561")
"""EPA natural gas combustion factor (kg CO2e / MJ)."""

ELECTRICITY_US_AVG_KG_CO2E_PER_KWH: Decimal = Decimal("0.385")
"""eGRID US national average grid factor (kg CO2e / kWh)."""

ELECTRICITY_UK_KG_CO2E_PER_KWH: Decimal = Decimal("0.207")
"""DEFRA UK grid factor (kg CO2e / kWh)."""

ELECTRICITY_EU_AVG_KG_CO2E_PER_KWH: Decimal = Decimal("0.256")
"""EU-27 average grid factor (kg CO2e / kWh)."""

LPG_EF_KG_CO2E_PER_LITRE: Decimal = Decimal("1.56")
"""DEFRA LPG combustion factor (kg CO2e / litre)."""

COAL_EF_KG_CO2E_PER_KG: Decimal = Decimal("2.42")
"""IPCC bituminous coal factor (kg CO2e / kg)."""


# =============================================================================
# Global Warming Potentials (AR6 -- 100 year horizon)
# =============================================================================
# Source: IPCC AR6 WG1 Table 7.15

GWP_CO2: int = 1
"""CO2 GWP (reference gas)."""

GWP_CH4: int = 28
"""CH4 GWP (fossil, AR6 100yr without climate-carbon feedback)."""

GWP_CH4_BIOGENIC: int = 27
"""CH4 GWP (biogenic, AR6 100yr)."""

GWP_N2O: int = 265
"""N2O GWP (AR6 100yr)."""

GWP_SF6: int = 23500
"""SF6 GWP (AR6 100yr)."""

GWP_HFC134A: int = 1300
"""HFC-134a GWP (AR6 100yr)."""

GWP_R410A: int = 1924
"""R-410A GWP (AR6 100yr, blend)."""


# =============================================================================
# CBAM Product Benchmarks (tCO2e / tonne product)
# =============================================================================
# Source: EU 2023/1773 Implementing Regulation

CBAM_BENCHMARK_STEEL_HRC: Decimal = Decimal("1.85")
"""CBAM benchmark for hot-rolled steel coil."""

CBAM_BENCHMARK_CEMENT_CLINKER: Decimal = Decimal("0.766")
"""CBAM benchmark for cement clinker."""

CBAM_BENCHMARK_ALUMINUM: Decimal = Decimal("8.60")
"""CBAM benchmark for unwrought aluminium."""


# =============================================================================
# Tolerance / Precision Defaults
# =============================================================================

DEFAULT_DECIMAL_PLACES: int = 6
"""Default number of decimal places for Decimal comparisons."""

DEFAULT_FLOAT_REL_TOL: float = 1e-9
"""Default relative tolerance for float comparisons."""

DEFAULT_FLOAT_ABS_TOL: float = 1e-9
"""Default absolute tolerance for float comparisons."""

EMISSIONS_DECIMAL_PLACES: int = 4
"""Decimal places for emission result assertions."""

FINANCIAL_DECIMAL_PLACES: int = 2
"""Decimal places for monetary value assertions."""


# =============================================================================
# Time Constants
# =============================================================================

TEST_REPORTING_YEAR: int = 2024
"""Default reporting year for period-bound tests."""

TEST_BASE_YEAR: int = 2019
"""Default base year for target-setting tests."""

TEST_CUTOFF_DATE: str = "2020-12-31"
"""EUDR deforestation-free cutoff date (ISO 8601)."""


# =============================================================================
# Data Quality Thresholds
# =============================================================================

MIN_DATA_QUALITY_SCORE: float = 0.7
"""Minimum acceptable data quality score for tests."""

HIGH_DATA_QUALITY_SCORE: float = 0.95
"""Score representing high-quality data."""

VERIFICATION_CONFIDENCE_THRESHOLD: float = 0.8
"""Minimum confidence for verification to pass."""
