# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-023: Processing of Sold Products Agent.

Provides comprehensive test fixtures for:
- Product inputs by category (ferrous, plastics, chemicals, food, electronics)
- Site-specific inputs (direct, energy, fuel methods)
- Average-data inputs (process EF, energy intensity, multi-step chain)
- Spend-based inputs (EEIO, multi-currency, CPI deflation)
- Calculation results, compliance results, provenance entries
- Mock database engine with EF lookups
- Configuration objects (18 frozen dataclass configs)
- 7 engine singletons with reset autouse fixture

Usage:
    def test_something(sample_product_ferrous, mock_database_engine):
        result = calculate(sample_product_ferrous, mock_database_engine)
        assert result.total_co2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import threading
import pytest


# ============================================================================
# GRACEFUL IMPORTS
# ============================================================================

try:
    from greenlang.agents.mrv.processing_sold_products.processing_database import (
        ProcessingDatabaseEngine,
    )
    PROCESSING_DB_AVAILABLE = True
except ImportError:
    PROCESSING_DB_AVAILABLE = False
    ProcessingDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.site_specific_calculator import (
        SiteSpecificCalculatorEngine,
    )
    SITE_SPECIFIC_AVAILABLE = True
except ImportError:
    SITE_SPECIFIC_AVAILABLE = False
    SiteSpecificCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    AVERAGE_DATA_AVAILABLE = True
except ImportError:
    AVERAGE_DATA_AVAILABLE = False
    AverageDataCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
    SPEND_BASED_AVAILABLE = True
except ImportError:
    SPEND_BASED_AVAILABLE = False
    SpendBasedCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridAggregatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.processing_pipeline import (
        ProcessingPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    ProcessingPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.processing_sold_products.config import get_config, reset_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from greenlang.agents.mrv.processing_sold_products.provenance import (
        ProvenanceTracker,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.processing_sold_products.setup import (
        ProcessingSoldProductsService,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False


# Engine availability flags
_ENGINE_CLASSES = [
    ("ProcessingDatabaseEngine", ProcessingDatabaseEngine, PROCESSING_DB_AVAILABLE),
    ("SiteSpecificCalculatorEngine", SiteSpecificCalculatorEngine, SITE_SPECIFIC_AVAILABLE),
    ("AverageDataCalculatorEngine", AverageDataCalculatorEngine, AVERAGE_DATA_AVAILABLE),
    ("SpendBasedCalculatorEngine", SpendBasedCalculatorEngine, SPEND_BASED_AVAILABLE),
    ("HybridAggregatorEngine", HybridAggregatorEngine, HYBRID_AVAILABLE),
    ("ComplianceCheckerEngine", ComplianceCheckerEngine, COMPLIANCE_AVAILABLE),
    ("ProcessingPipelineEngine", ProcessingPipelineEngine, PIPELINE_AVAILABLE),
]

_SKIP_DB = pytest.mark.skipif(
    not PROCESSING_DB_AVAILABLE,
    reason="ProcessingDatabaseEngine not available",
)
_SKIP_SITE = pytest.mark.skipif(
    not SITE_SPECIFIC_AVAILABLE,
    reason="SiteSpecificCalculatorEngine not available",
)
_SKIP_AVG = pytest.mark.skipif(
    not AVERAGE_DATA_AVAILABLE,
    reason="AverageDataCalculatorEngine not available",
)
_SKIP_SPEND = pytest.mark.skipif(
    not SPEND_BASED_AVAILABLE,
    reason="SpendBasedCalculatorEngine not available",
)
_SKIP_HYBRID = pytest.mark.skipif(
    not HYBRID_AVAILABLE,
    reason="HybridAggregatorEngine not available",
)
_SKIP_COMPLIANCE = pytest.mark.skipif(
    not COMPLIANCE_AVAILABLE,
    reason="ComplianceCheckerEngine not available",
)
_SKIP_PIPELINE = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="ProcessingPipelineEngine not available",
)
_SKIP_CONFIG = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="Config module not available",
)
_SKIP_PROVENANCE = pytest.mark.skipif(
    not PROVENANCE_AVAILABLE,
    reason="Provenance module not available",
)
_SKIP_SETUP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="Setup module not available",
)


# ============================================================================
# AUTOUSE SINGLETON RESET FIXTURE
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all 7 engine singletons before and after each test for isolation."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Attempt to call reset_instance or reset_singleton on each engine class."""
    for name, cls, available in _ENGINE_CLASSES:
        if not available or cls is None:
            continue
        for method_name in ("reset_instance", "reset_singleton", "_reset"):
            reset_fn = getattr(cls, method_name, None)
            if reset_fn is not None and callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
                break
    # Also reset config singleton
    if CONFIG_AVAILABLE:
        try:
            reset_config()
        except Exception:
            pass


# ============================================================================
# PRODUCT INPUT FIXTURES (by category)
# ============================================================================

@pytest.fixture
def sample_product_ferrous() -> Dict[str, Any]:
    """Hot-rolled steel coil sold to automotive stamper."""
    return {
        "product_id": "PRD-FER-001",
        "product_name": "Hot-rolled steel coil",
        "category": "metals_ferrous",
        "quantity_tonnes": Decimal("1000.0"),
        "unit": "tonnes",
        "customer_name": "AutoStamp Co",
        "customer_country": "US",
        "processing_type": "stamping",
        "processing_country": "US",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_plastics() -> Dict[str, Any]:
    """Polypropylene pellets sold to injection molder."""
    return {
        "product_id": "PRD-PLS-001",
        "product_name": "Polypropylene pellets",
        "category": "plastics_thermoplastic",
        "quantity_tonnes": Decimal("500.0"),
        "unit": "tonnes",
        "customer_name": "MoldTech Inc",
        "customer_country": "DE",
        "processing_type": "injection_molding",
        "processing_country": "DE",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_chemicals() -> Dict[str, Any]:
    """Ethylene oxide sold for chemical reaction processing."""
    return {
        "product_id": "PRD-CHM-001",
        "product_name": "Ethylene oxide",
        "category": "chemicals",
        "quantity_tonnes": Decimal("200.0"),
        "unit": "tonnes",
        "customer_name": "ChemProc Corp",
        "customer_country": "CN",
        "processing_type": "chemical_reaction",
        "processing_country": "CN",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_food() -> Dict[str, Any]:
    """Wheat flour sold for baking (drying/fermentation)."""
    return {
        "product_id": "PRD-FD-001",
        "product_name": "Wheat flour",
        "category": "food_ingredients",
        "quantity_tonnes": Decimal("300.0"),
        "unit": "tonnes",
        "customer_name": "BakeryGroup",
        "customer_country": "GB",
        "processing_type": "drying",
        "processing_country": "GB",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_electronics() -> Dict[str, Any]:
    """Semiconductor wafers sold for assembly processing."""
    return {
        "product_id": "PRD-ELC-001",
        "product_name": "Silicon wafers 300mm",
        "category": "electronics_components",
        "quantity_tonnes": Decimal("50.0"),
        "unit": "tonnes",
        "customer_name": "ChipAssembly Ltd",
        "customer_country": "TW",
        "processing_type": "assembly",
        "processing_country": "TW",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_textiles() -> Dict[str, Any]:
    """Cotton yarn sold for textile finishing."""
    return {
        "product_id": "PRD-TXT-001",
        "product_name": "Cotton yarn",
        "category": "textiles",
        "quantity_tonnes": Decimal("100.0"),
        "unit": "tonnes",
        "customer_name": "TextilFinish",
        "customer_country": "IN",
        "processing_type": "textile_finishing",
        "processing_country": "IN",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_glass() -> Dict[str, Any]:
    """Float glass sold for heat treatment."""
    return {
        "product_id": "PRD-GLS-001",
        "product_name": "Float glass 6mm",
        "category": "glass_ceramics",
        "quantity_tonnes": Decimal("400.0"),
        "unit": "tonnes",
        "customer_name": "GlassTreat Co",
        "customer_country": "JP",
        "processing_type": "heat_treatment",
        "processing_country": "JP",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_wood() -> Dict[str, Any]:
    """Wood pulp sold for milling."""
    return {
        "product_id": "PRD-WD-001",
        "product_name": "Wood pulp",
        "category": "wood_paper_pulp",
        "quantity_tonnes": Decimal("600.0"),
        "unit": "tonnes",
        "customer_name": "PaperMill Inc",
        "customer_country": "FI",
        "processing_type": "milling",
        "processing_country": "FI",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_minerals() -> Dict[str, Any]:
    """Iron ore concentrate sold for sintering."""
    return {
        "product_id": "PRD-MIN-001",
        "product_name": "Iron ore concentrate",
        "category": "minerals",
        "quantity_tonnes": Decimal("2000.0"),
        "unit": "tonnes",
        "customer_name": "SinterWorks",
        "customer_country": "BR",
        "processing_type": "sintering",
        "processing_country": "BR",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_product_agricultural() -> Dict[str, Any]:
    """Sugar cane juice sold for fermentation."""
    return {
        "product_id": "PRD-AG-001",
        "product_name": "Sugar cane juice",
        "category": "agricultural_commodities",
        "quantity_tonnes": Decimal("800.0"),
        "unit": "tonnes",
        "customer_name": "FermentCo",
        "customer_country": "BR",
        "processing_type": "fermentation",
        "processing_country": "BR",
        "reporting_year": 2024,
    }


# ============================================================================
# SITE-SPECIFIC INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_site_specific_direct() -> Dict[str, Any]:
    """Site-specific direct method: customer-reported processing emissions."""
    return {
        "method": "site_specific_direct",
        "product_id": "PRD-FER-001",
        "category": "metals_ferrous",
        "quantity_tonnes": Decimal("1000.0"),
        "processing_type": "stamping",
        "customer_reported_co2e_kg": Decimal("280000.0"),
        "customer_reporting_year": 2024,
        "customer_verification": "third_party",
        "region": "US",
    }


@pytest.fixture
def sample_site_specific_energy() -> Dict[str, Any]:
    """Site-specific energy method: energy consumption x grid EF."""
    return {
        "method": "site_specific_energy",
        "product_id": "PRD-PLS-001",
        "category": "plastics_thermoplastic",
        "quantity_tonnes": Decimal("500.0"),
        "processing_type": "injection_molding",
        "energy_consumption_kwh": Decimal("750000.0"),
        "grid_ef_kg_per_kwh": Decimal("0.42"),
        "region": "DE",
    }


@pytest.fixture
def sample_site_specific_fuel() -> Dict[str, Any]:
    """Site-specific fuel method: fuel consumption x combustion EF."""
    return {
        "method": "site_specific_fuel",
        "product_id": "PRD-GLS-001",
        "category": "glass_ceramics",
        "quantity_tonnes": Decimal("400.0"),
        "processing_type": "heat_treatment",
        "fuel_consumption_litres": Decimal("50000.0"),
        "fuel_type": "natural_gas",
        "fuel_ef_kg_per_litre": Decimal("2.02"),
        "region": "JP",
    }


@pytest.fixture
def sample_average_data_input() -> Dict[str, Any]:
    """Average-data method: process EF lookup."""
    return {
        "method": "average_data",
        "product_id": "PRD-FER-001",
        "category": "metals_ferrous",
        "quantity_tonnes": Decimal("1000.0"),
        "processing_type": "stamping",
        "region": "US",
        "reporting_year": 2024,
    }


@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """Spend-based EEIO method input."""
    return {
        "method": "spend_based",
        "product_id": "PRD-FER-001",
        "naics_code": "331110",
        "revenue_usd": Decimal("1000000.0"),
        "currency": "USD",
        "reporting_year": 2024,
        "margin_rate": Decimal("0.08"),
    }


# ============================================================================
# CALCULATION RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_calculation_result() -> Dict[str, Any]:
    """A fully populated calculation result for testing downstream engines."""
    return {
        "calculation_id": "calc-psp-001",
        "product_id": "PRD-FER-001",
        "product_name": "Hot-rolled steel coil",
        "category": "metals_ferrous",
        "processing_type": "stamping",
        "method": "site_specific_direct",
        "quantity_tonnes": Decimal("1000.0"),
        "total_co2e_kg": Decimal("280000.0"),
        "co2e_per_tonne": Decimal("280.0"),
        "dqi_score": Decimal("90.0"),
        "uncertainty_lower": Decimal("266000.0"),
        "uncertainty_upper": Decimal("294000.0"),
        "provenance_hash": "a" * 64,
        "region": "US",
        "reporting_year": 2024,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_compliance_result() -> Dict[str, Any]:
    """A fully compliant result for framework checks."""
    return {
        "total_co2e_kg": Decimal("280000.0"),
        "method": "site_specific_direct",
        "calculation_method": "site_specific_direct",
        "ef_sources": ["customer_reported"],
        "ef_source": "customer_reported",
        "product_category": "metals_ferrous",
        "processing_type": "stamping",
        "boundary": "intermediate_products_only",
        "exclusions": "None - all processing included",
        "dqi_score": Decimal("90.0"),
        "data_quality_score": Decimal("90.0"),
        "uncertainty_analysis": {"method": "propagation", "ci_95": [266000, 294000]},
        "uncertainty": {"method": "propagation"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 10, site-specific",
        "targets": "Reduce processing emissions 25% by 2030",
        "reduction_targets": "25% by 2030",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "reporting_period": "2024",
        "period": "2024",
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "ISO 14064"],
        "standards": ["GHG Protocol"],
        "total_scope3_co2e": Decimal("5000000.0"),
        "product_count": 10,
        "by_category": {
            "metals_ferrous": Decimal("280000.0"),
            "plastics_thermoplastic": Decimal("150000.0"),
        },
        "by_method": {
            "site_specific_direct": Decimal("280000.0"),
            "average_data": Decimal("150000.0"),
        },
        "completeness_score": Decimal("85.0"),
        "method_coverage": Decimal("90.0"),
    }


# ============================================================================
# EMISSION FACTOR DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def sample_process_ef_db() -> Dict[str, Any]:
    """Process emission factor database for average-data lookups.

    Keys: (category, processing_type, region) -> ef_kg_per_tonne.
    """
    return {
        ("metals_ferrous", "stamping", "US"): Decimal("280.0"),
        ("metals_ferrous", "machining", "US"): Decimal("195.0"),
        ("metals_ferrous", "welding", "US"): Decimal("320.0"),
        ("metals_ferrous", "heat_treatment", "US"): Decimal("410.0"),
        ("metals_ferrous", "casting", "US"): Decimal("550.0"),
        ("metals_ferrous", "forging", "US"): Decimal("380.0"),
        ("metals_non_ferrous", "machining", "US"): Decimal("240.0"),
        ("metals_non_ferrous", "casting", "US"): Decimal("620.0"),
        ("plastics_thermoplastic", "injection_molding", "DE"): Decimal("320.0"),
        ("plastics_thermoplastic", "extrusion", "DE"): Decimal("290.0"),
        ("plastics_thermoplastic", "blow_molding", "DE"): Decimal("310.0"),
        ("plastics_thermoset", "injection_molding", "US"): Decimal("350.0"),
        ("chemicals", "chemical_reaction", "CN"): Decimal("450.0"),
        ("chemicals", "refining", "US"): Decimal("380.0"),
        ("food_ingredients", "drying", "GB"): Decimal("120.0"),
        ("food_ingredients", "fermentation", "BR"): Decimal("95.0"),
        ("food_ingredients", "milling", "US"): Decimal("85.0"),
        ("textiles", "textile_finishing", "IN"): Decimal("210.0"),
        ("glass_ceramics", "heat_treatment", "JP"): Decimal("390.0"),
        ("glass_ceramics", "sintering", "JP"): Decimal("480.0"),
        ("wood_paper_pulp", "milling", "FI"): Decimal("110.0"),
        ("minerals", "sintering", "BR"): Decimal("520.0"),
        ("electronics_components", "assembly", "TW"): Decimal("680.0"),
        ("agricultural_commodities", "fermentation", "BR"): Decimal("140.0"),
        ("agricultural_commodities", "drying", "US"): Decimal("105.0"),
    }


@pytest.fixture
def sample_energy_intensity_db() -> Dict[str, Any]:
    """Energy intensity database for average-data energy method.

    Keys: processing_type -> kwh_per_tonne.
    """
    return {
        "stamping": Decimal("350.0"),
        "machining": Decimal("480.0"),
        "welding": Decimal("520.0"),
        "heat_treatment": Decimal("1200.0"),
        "casting": Decimal("900.0"),
        "forging": Decimal("650.0"),
        "injection_molding": Decimal("580.0"),
        "extrusion": Decimal("450.0"),
        "blow_molding": Decimal("500.0"),
        "coating": Decimal("380.0"),
        "assembly": Decimal("250.0"),
        "chemical_reaction": Decimal("1500.0"),
        "refining": Decimal("1800.0"),
        "milling": Decimal("200.0"),
        "drying": Decimal("600.0"),
        "sintering": Decimal("1100.0"),
        "fermentation": Decimal("300.0"),
        "textile_finishing": Decimal("700.0"),
    }


@pytest.fixture
def sample_grid_ef_db() -> Dict[str, Any]:
    """Grid emission factor database by region (kgCO2e/kWh)."""
    return {
        "US": Decimal("0.42"),
        "DE": Decimal("0.35"),
        "CN": Decimal("0.58"),
        "GB": Decimal("0.23"),
        "JP": Decimal("0.47"),
        "IN": Decimal("0.71"),
        "BR": Decimal("0.08"),
        "FI": Decimal("0.10"),
        "TW": Decimal("0.52"),
        "KR": Decimal("0.46"),
        "FR": Decimal("0.06"),
        "AU": Decimal("0.63"),
        "CA": Decimal("0.13"),
        "MX": Decimal("0.44"),
        "ZA": Decimal("0.92"),
        "GLOBAL": Decimal("0.44"),
    }


@pytest.fixture
def sample_eeio_factors() -> Dict[str, Any]:
    """EEIO sector emission factors for spend-based calculations.

    Keys: naics_code -> {name, ef_per_usd, margin_rate}.
    """
    return {
        "331110": {"name": "Iron and steel mills", "ef_per_usd": Decimal("0.82"), "margin_rate": Decimal("0.08")},
        "331200": {"name": "Steel product manufacturing", "ef_per_usd": Decimal("0.75"), "margin_rate": Decimal("0.09")},
        "326100": {"name": "Plastics product manufacturing", "ef_per_usd": Decimal("0.48"), "margin_rate": Decimal("0.12")},
        "325100": {"name": "Basic chemical manufacturing", "ef_per_usd": Decimal("0.95"), "margin_rate": Decimal("0.10")},
        "311200": {"name": "Grain and oilseed milling", "ef_per_usd": Decimal("0.35"), "margin_rate": Decimal("0.07")},
        "313300": {"name": "Textile finishing mills", "ef_per_usd": Decimal("0.42"), "margin_rate": Decimal("0.11")},
        "327200": {"name": "Glass and glass products", "ef_per_usd": Decimal("0.58"), "margin_rate": Decimal("0.09")},
        "322100": {"name": "Pulp, paper, and paperboard", "ef_per_usd": Decimal("0.52"), "margin_rate": Decimal("0.08")},
        "334400": {"name": "Semiconductor manufacturing", "ef_per_usd": Decimal("0.38"), "margin_rate": Decimal("0.22")},
        "327900": {"name": "Other nonmetallic minerals", "ef_per_usd": Decimal("0.65"), "margin_rate": Decimal("0.08")},
        "111000": {"name": "Crop production", "ef_per_usd": Decimal("0.45"), "margin_rate": Decimal("0.06")},
        "331300": {"name": "Alumina and aluminum", "ef_per_usd": Decimal("1.10"), "margin_rate": Decimal("0.07")},
    }


@pytest.fixture
def sample_currency_rates() -> Dict[str, Decimal]:
    """Currency to USD exchange rates."""
    return {
        "USD": Decimal("1.0000"),
        "EUR": Decimal("1.0850"),
        "GBP": Decimal("1.2650"),
        "JPY": Decimal("0.006667"),
        "CNY": Decimal("0.1380"),
        "INR": Decimal("0.01200"),
        "BRL": Decimal("0.2020"),
        "KRW": Decimal("0.000750"),
        "TWD": Decimal("0.03200"),
        "CAD": Decimal("0.7400"),
        "AUD": Decimal("0.6500"),
        "CHF": Decimal("1.1200"),
    }


@pytest.fixture
def sample_cpi_deflators() -> Dict[int, Decimal]:
    """CPI deflator table (base year 2021 = 1.0)."""
    return {
        2015: Decimal("0.8490"),
        2016: Decimal("0.8600"),
        2017: Decimal("0.8790"),
        2018: Decimal("0.9030"),
        2019: Decimal("0.9210"),
        2020: Decimal("0.9271"),
        2021: Decimal("1.0000"),
        2022: Decimal("1.0810"),
        2023: Decimal("1.1150"),
        2024: Decimal("1.1490"),
        2025: Decimal("1.1780"),
    }


@pytest.fixture
def sample_processing_chains() -> Dict[str, List[str]]:
    """Multi-step processing chain definitions.

    Keys: chain_name -> list of processing_type steps.
    """
    return {
        "steel_automotive": ["stamping", "welding", "coating", "assembly"],
        "steel_construction": ["machining", "welding", "coating"],
        "plastic_packaging": ["extrusion", "blow_molding", "coating"],
        "plastic_automotive": ["injection_molding", "assembly", "coating"],
        "chemical_pharma": ["chemical_reaction", "refining", "drying"],
        "food_bakery": ["milling", "drying", "fermentation"],
        "glass_automotive": ["heat_treatment", "coating", "assembly"],
        "electronics_pcb": ["assembly", "coating", "heat_treatment"],
    }


# ============================================================================
# MOCK DATABASE ENGINE FIXTURE
# ============================================================================

@pytest.fixture
def mock_database_engine(
    sample_process_ef_db,
    sample_energy_intensity_db,
    sample_grid_ef_db,
    sample_eeio_factors,
    sample_currency_rates,
    sample_cpi_deflators,
    sample_processing_chains,
) -> MagicMock:
    """Create a fully mocked ProcessingDatabaseEngine with EF lookups."""
    engine = MagicMock(spec=["get_process_ef", "get_energy_intensity",
                             "get_grid_ef", "get_eeio_factor",
                             "get_currency_rate", "get_cpi_deflator",
                             "get_processing_chain", "get_database_summary",
                             "get_lookup_count", "validate_category",
                             "validate_processing_type", "get_available_categories",
                             "get_available_processing_types",
                             "get_available_regions", "get_chain_steps",
                             "check_chain_compatibility",
                             "get_fuel_ef", "lookup_product_category"])

    def _get_process_ef(category, processing_type, region="GLOBAL"):
        key = (category, processing_type, region)
        if key in sample_process_ef_db:
            return {"ef_kg_per_tonne": sample_process_ef_db[key]}
        return None

    def _get_energy_intensity(processing_type):
        if processing_type in sample_energy_intensity_db:
            return {"kwh_per_tonne": sample_energy_intensity_db[processing_type]}
        return None

    def _get_grid_ef(region):
        if region in sample_grid_ef_db:
            return {"ef_kg_per_kwh": sample_grid_ef_db[region]}
        return None

    def _get_eeio_factor(naics_code):
        if naics_code in sample_eeio_factors:
            return sample_eeio_factors[naics_code]
        raise ValueError(f"EEIO factor not found for NAICS {naics_code}")

    def _get_currency_rate(currency):
        if currency in sample_currency_rates:
            return sample_currency_rates[currency]
        raise ValueError(f"Currency rate not found for {currency}")

    def _get_cpi_deflator(year):
        if year in sample_cpi_deflators:
            return sample_cpi_deflators[year]
        raise ValueError(f"CPI deflator not available for year {year}")

    def _get_processing_chain(chain_name):
        if chain_name in sample_processing_chains:
            return sample_processing_chains[chain_name]
        return None

    engine.get_process_ef.side_effect = _get_process_ef
    engine.get_energy_intensity.side_effect = _get_energy_intensity
    engine.get_grid_ef.side_effect = _get_grid_ef
    engine.get_eeio_factor.side_effect = _get_eeio_factor
    engine.get_currency_rate.side_effect = _get_currency_rate
    engine.get_cpi_deflator.side_effect = _get_cpi_deflator
    engine.get_processing_chain.side_effect = _get_processing_chain
    engine.get_lookup_count.return_value = 0
    engine.validate_category.return_value = True
    engine.validate_processing_type.return_value = True
    engine.get_available_categories.return_value = [
        "metals_ferrous", "metals_non_ferrous", "plastics_thermoplastic",
        "plastics_thermoset", "chemicals", "food_ingredients", "textiles",
        "electronics_components", "glass_ceramics", "wood_paper_pulp",
        "minerals", "agricultural_commodities",
    ]
    engine.get_available_processing_types.return_value = [
        "machining", "stamping", "welding", "heat_treatment",
        "injection_molding", "extrusion", "blow_molding",
        "casting", "forging", "coating", "assembly",
        "chemical_reaction", "refining", "milling",
        "drying", "sintering", "fermentation", "textile_finishing",
    ]
    engine.get_available_regions.return_value = [
        "US", "DE", "CN", "GB", "JP", "IN", "BR", "FI",
        "TW", "KR", "FR", "AU", "CA", "MX", "ZA", "GLOBAL",
    ]
    engine.get_chain_steps.side_effect = _get_processing_chain
    engine.check_chain_compatibility.return_value = True

    def _get_fuel_ef(fuel_type):
        fuel_efs = {
            "natural_gas": {"ef_kg_per_litre": Decimal("2.02")},
            "diesel": {"ef_kg_per_litre": Decimal("2.68")},
            "heavy_fuel_oil": {"ef_kg_per_litre": Decimal("3.12")},
            "lpg": {"ef_kg_per_litre": Decimal("1.55")},
        }
        if fuel_type in fuel_efs:
            return fuel_efs[fuel_type]
        raise ValueError(f"Fuel EF not found for {fuel_type}")

    engine.get_fuel_ef.side_effect = _get_fuel_ef
    engine.lookup_product_category.return_value = "metals_ferrous"

    engine.get_database_summary.return_value = {
        "categories": 12,
        "processing_types": 18,
        "regions": 16,
        "eeio_sectors": 12,
        "currencies": 12,
        "fuel_types": 4,
        "chains": 8,
    }

    return engine


# ============================================================================
# MOCK CONFIG FIXTURE
# ============================================================================

@pytest.fixture
def mock_config() -> MagicMock:
    """Create a fully mocked configuration object."""
    cfg = MagicMock()
    cfg.general.enabled = True
    cfg.general.debug = False
    cfg.general.log_level = "INFO"
    cfg.general.max_batch_size = 1000
    cfg.general.agent_id = "GL-MRV-S3-010"
    cfg.general.agent_component = "AGENT-MRV-023"
    cfg.general.version = "1.0.0"
    cfg.general.api_prefix = "/api/v1/processing-sold-products"
    cfg.general.default_gwp = "AR5"
    cfg.database.host = "localhost"
    cfg.database.port = 5432
    cfg.database.database = "greenlang"
    cfg.database.table_prefix = "gl_psp_"
    cfg.database.schema = "processing_sold_products_service"
    cfg.site_specific.enable_direct = True
    cfg.site_specific.enable_energy = True
    cfg.site_specific.enable_fuel = True
    cfg.average_data.default_region = "GLOBAL"
    cfg.average_data.enable_chain_calculations = True
    cfg.spend.enable_cpi_deflation = True
    cfg.spend.enable_margin_removal = True
    cfg.spend.default_margin_rate = Decimal("0.08")
    cfg.spend.base_year = 2021
    cfg.spend.default_currency = "USD"
    cfg.hybrid.method_priority = [
        "site_specific_direct", "site_specific_energy", "site_specific_fuel",
        "average_data", "spend_based",
    ]
    cfg.hybrid.enable_gap_filling = True
    cfg.hybrid.allocation_method = "mass"
    cfg.compliance.get_frameworks.return_value = [
        "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
        "CDP", "SBTI", "SB_253", "GRI",
    ]
    cfg.compliance.strict_mode = False
    cfg.compliance.materiality_threshold = Decimal("0.01")
    cfg.provenance.enable_provenance = True
    cfg.provenance.hash_algorithm = "sha256"
    cfg.cache.enable_cache = True
    cfg.cache.ttl_seconds = 3600
    cfg.metrics.enable_metrics = True
    cfg.metrics.prefix = "gl_psp_"
    cfg.uncertainty.default_method = "propagation"
    cfg.uncertainty.confidence_level = Decimal("0.95")
    cfg.api.enable_api = True
    cfg.api.prefix = "/api/v1/processing-sold-products"
    return cfg


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_full_compliance_result(**overrides) -> Dict[str, Any]:
    """Build a fully compliant result dict with all required fields.

    Pass keyword overrides to modify or nullify specific fields.
    """
    base = {
        "total_co2e_kg": Decimal("280000.0"),
        "method": "site_specific_direct",
        "calculation_method": "site_specific_direct",
        "ef_sources": ["customer_reported"],
        "ef_source": "customer_reported",
        "product_category": "metals_ferrous",
        "processing_type": "stamping",
        "boundary": "intermediate_products_only",
        "exclusions": "None - all processing included",
        "dqi_score": Decimal("90.0"),
        "data_quality_score": Decimal("90.0"),
        "uncertainty_analysis": {"method": "propagation", "ci_95": [266000, 294000]},
        "uncertainty": {"method": "propagation"},
        "base_year": 2019,
        "methodology": "GHG Protocol Scope 3 Category 10, site-specific",
        "targets": "Reduce processing emissions 25% by 2030",
        "reduction_targets": "25% by 2030",
        "actions": "Supplier engagement for energy efficiency",
        "reduction_actions": "Energy efficiency programs",
        "verification_status": "limited_assurance",
        "verified": True,
        "assurance": "limited",
        "assurance_opinion": "limited_assurance",
        "reporting_period": "2024",
        "period": "2024",
        "gases_included": ["CO2", "CH4", "N2O"],
        "emission_gases": ["CO2", "CH4", "N2O"],
        "standards_used": ["GHG Protocol", "ISO 14064"],
        "standards": ["GHG Protocol"],
        "total_scope3_co2e": Decimal("5000000.0"),
        "product_count": 10,
        "by_category": {"metals_ferrous": Decimal("280000.0")},
        "by_method": {"site_specific_direct": Decimal("280000.0")},
        "completeness_score": Decimal("85.0"),
        "method_coverage": Decimal("90.0"),
        "intermediate_only": True,
        "end_product_excluded": True,
        "customer_data_provided": True,
        "processing_steps_documented": True,
        "allocation_method": "mass",
        "progress_tracking": {"2023": 300000, "2024": 280000},
        "year_over_year_change": Decimal("-6.7"),
        "target_coverage": "75%",
        "sbti_coverage": "75%",
    }
    base.update(overrides)
    return base
