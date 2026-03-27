# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-025: End-of-Life Treatment of Sold Products Agent.

Provides comprehensive test fixtures for:
- Product inputs by type (electronics, appliance, packaging, clothing, furniture,
  battery, tire, food product, building material, mixed product)
- Material treatment emission factors (15 materials x 7 treatments)
- Product compositions (20 product categories)
- Regional treatment mixes (12 regions)
- Landfill FOD parameters (DOC, DOCf, MCF, k, F, OX by material and climate)
- Incineration parameters (fossil carbon, combustion efficiency, energy recovery)
- Recycling factors (recovery rate, quality factor, avoided EFs)
- Composting and AD factors (CH4, N2O process emissions)
- Mock engines for all 7 engine singletons
- Compliance result builder helper
- Configuration objects (18 config sections)
- Provenance, metrics, and batch fixtures

Usage:
    def test_something(sample_electronics_product, mock_database_engine):
        result = calculate(sample_electronics_product, mock_database_engine)
        assert result.gross_emissions_tco2e > 0
        assert result.avoided_emissions_tco2e is not None  # always separate

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
    from greenlang.agents.mrv.end_of_life_treatment.eol_product_database import (
        EOLProductDatabaseEngine,
    )
    EOL_DB_AVAILABLE = True
except ImportError:
    EOL_DB_AVAILABLE = False
    EOLProductDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.waste_type_specific_calculator import (
        WasteTypeSpecificCalculatorEngine,
    )
    WASTE_TYPE_AVAILABLE = True
except ImportError:
    WASTE_TYPE_AVAILABLE = False
    WasteTypeSpecificCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    AVERAGE_DATA_AVAILABLE = True
except ImportError:
    AVERAGE_DATA_AVAILABLE = False
    AverageDataCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.producer_specific_calculator import (
        ProducerSpecificCalculatorEngine,
    )
    PRODUCER_SPECIFIC_AVAILABLE = True
except ImportError:
    PRODUCER_SPECIFIC_AVAILABLE = False
    ProducerSpecificCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridAggregatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.end_of_life_treatment_pipeline import (
        EndOfLifeTreatmentPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    EndOfLifeTreatmentPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.config import get_config, reset_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from greenlang.agents.mrv.end_of_life_treatment.provenance import (
        ProvenanceChainBuilder,
    )
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.end_of_life_treatment.setup import (
        EndOfLifeTreatmentService,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False


# Engine availability flags
_ENGINE_CLASSES = [
    ("EOLProductDatabaseEngine", EOLProductDatabaseEngine, EOL_DB_AVAILABLE),
    ("WasteTypeSpecificCalculatorEngine", WasteTypeSpecificCalculatorEngine, WASTE_TYPE_AVAILABLE),
    ("AverageDataCalculatorEngine", AverageDataCalculatorEngine, AVERAGE_DATA_AVAILABLE),
    ("ProducerSpecificCalculatorEngine", ProducerSpecificCalculatorEngine, PRODUCER_SPECIFIC_AVAILABLE),
    ("HybridAggregatorEngine", HybridAggregatorEngine, HYBRID_AVAILABLE),
    ("ComplianceCheckerEngine", ComplianceCheckerEngine, COMPLIANCE_AVAILABLE),
    ("EndOfLifeTreatmentPipelineEngine", EndOfLifeTreatmentPipelineEngine, PIPELINE_AVAILABLE),
]

_SKIP_DB = pytest.mark.skipif(
    not EOL_DB_AVAILABLE,
    reason="EOLProductDatabaseEngine not available",
)
_SKIP_WASTE_TYPE = pytest.mark.skipif(
    not WASTE_TYPE_AVAILABLE,
    reason="WasteTypeSpecificCalculatorEngine not available",
)
_SKIP_AVG = pytest.mark.skipif(
    not AVERAGE_DATA_AVAILABLE,
    reason="AverageDataCalculatorEngine not available",
)
_SKIP_PRODUCER = pytest.mark.skipif(
    not PRODUCER_SPECIFIC_AVAILABLE,
    reason="ProducerSpecificCalculatorEngine not available",
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
    reason="EndOfLifeTreatmentPipelineEngine not available",
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
        for method_name in ("reset_instance", "reset_singleton", "_reset", "reset"):
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
# PRODUCT INPUT FIXTURES (10 products)
# ============================================================================

@pytest.fixture
def sample_electronics_product() -> Dict[str, Any]:
    """Consumer electronics product (smartphone), 2.5 kg, plastic/metal/glass.

    A typical consumer electronics device with mixed material composition.
    Expected end-of-life: 40% recycling, 30% landfill, 25% incineration, 5% other.
    """
    return {
        "product_id": "PRD-ELEC-001",
        "product_name": "Smartphone Model X",
        "product_category": "consumer_electronics",
        "total_mass_kg": Decimal("0.180"),
        "units_sold": 100000,
        "total_mass_tonnes": Decimal("18.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "plastic_abs", "mass_fraction": Decimal("0.35"), "mass_kg": Decimal("0.063")},
            {"material": "aluminum", "mass_fraction": Decimal("0.25"), "mass_kg": Decimal("0.045")},
            {"material": "glass", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("0.036")},
            {"material": "copper", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.018")},
            {"material": "lithium_battery", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.018")},
        ],
        "expected_lifetime_years": 3,
        "country_of_sale": "US",
    }


@pytest.fixture
def sample_appliance_product() -> Dict[str, Any]:
    """Large home appliance (washing machine), 35 kg, metal/plastic.

    Expected end-of-life: 70% recycling (metal recovery), 20% landfill, 10% incineration.
    """
    return {
        "product_id": "PRD-APPL-001",
        "product_name": "Washing Machine EcoWash 3000",
        "product_category": "large_appliances",
        "total_mass_kg": Decimal("35.0"),
        "units_sold": 50000,
        "total_mass_tonnes": Decimal("1750.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "steel", "mass_fraction": Decimal("0.55"), "mass_kg": Decimal("19.25")},
            {"material": "plastic_pp", "mass_fraction": Decimal("0.25"), "mass_kg": Decimal("8.75")},
            {"material": "copper", "mass_fraction": Decimal("0.08"), "mass_kg": Decimal("2.80")},
            {"material": "rubber", "mass_fraction": Decimal("0.07"), "mass_kg": Decimal("2.45")},
            {"material": "glass", "mass_fraction": Decimal("0.05"), "mass_kg": Decimal("1.75")},
        ],
        "expected_lifetime_years": 12,
        "country_of_sale": "DE",
    }


@pytest.fixture
def sample_packaging_product() -> Dict[str, Any]:
    """Flexible packaging (food wrapper), 0.5 kg per unit, paper/plastic.

    Expected end-of-life: 50% landfill, 30% incineration, 15% recycling, 5% litter.
    """
    return {
        "product_id": "PRD-PKG-001",
        "product_name": "Food Packaging Multi-Layer",
        "product_category": "packaging",
        "total_mass_kg": Decimal("0.005"),
        "units_sold": 10000000,
        "total_mass_tonnes": Decimal("50.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.60"), "mass_kg": Decimal("0.003")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.35"), "mass_kg": Decimal("0.00175")},
            {"material": "aluminum_foil", "mass_fraction": Decimal("0.05"), "mass_kg": Decimal("0.00025")},
        ],
        "expected_lifetime_years": 0,
        "country_of_sale": "GB",
    }


@pytest.fixture
def sample_clothing_product() -> Dict[str, Any]:
    """Textile clothing item (t-shirt), 0.3 kg, cotton/polyester.

    Expected end-of-life: 60% landfill, 20% incineration, 15% reuse/recycling, 5% other.
    """
    return {
        "product_id": "PRD-CLT-001",
        "product_name": "Cotton-Blend T-Shirt",
        "product_category": "clothing",
        "total_mass_kg": Decimal("0.300"),
        "units_sold": 500000,
        "total_mass_tonnes": Decimal("150.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "cotton", "mass_fraction": Decimal("0.65"), "mass_kg": Decimal("0.195")},
            {"material": "polyester", "mass_fraction": Decimal("0.35"), "mass_kg": Decimal("0.105")},
        ],
        "expected_lifetime_years": 3,
        "country_of_sale": "US",
    }


@pytest.fixture
def sample_furniture_product() -> Dict[str, Any]:
    """Office furniture (desk), 25 kg, wood/metal/textile.

    Expected end-of-life: 40% landfill, 35% recycling, 15% incineration, 10% reuse.
    """
    return {
        "product_id": "PRD-FRN-001",
        "product_name": "Adjustable Office Desk",
        "product_category": "furniture",
        "total_mass_kg": Decimal("25.0"),
        "units_sold": 20000,
        "total_mass_tonnes": Decimal("500.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "wood_mdf", "mass_fraction": Decimal("0.50"), "mass_kg": Decimal("12.5")},
            {"material": "steel", "mass_fraction": Decimal("0.35"), "mass_kg": Decimal("8.75")},
            {"material": "textile_polyester", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("2.5")},
            {"material": "plastic_abs", "mass_fraction": Decimal("0.05"), "mass_kg": Decimal("1.25")},
        ],
        "expected_lifetime_years": 15,
        "country_of_sale": "US",
    }


@pytest.fixture
def sample_battery_product() -> Dict[str, Any]:
    """Lithium-ion battery pack, 0.8 kg.

    Expected end-of-life: 60% recycling (regulated), 30% landfill, 10% other.
    """
    return {
        "product_id": "PRD-BAT-001",
        "product_name": "Li-ion Battery Pack 18650",
        "product_category": "batteries",
        "total_mass_kg": Decimal("0.800"),
        "units_sold": 200000,
        "total_mass_tonnes": Decimal("160.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "lithium_cobalt_oxide", "mass_fraction": Decimal("0.30"), "mass_kg": Decimal("0.240")},
            {"material": "aluminum", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("0.160")},
            {"material": "copper", "mass_fraction": Decimal("0.15"), "mass_kg": Decimal("0.120")},
            {"material": "electrolyte_organic", "mass_fraction": Decimal("0.15"), "mass_kg": Decimal("0.120")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.080")},
            {"material": "steel", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.080")},
        ],
        "expected_lifetime_years": 5,
        "country_of_sale": "JP",
    }


@pytest.fixture
def sample_tire_product() -> Dict[str, Any]:
    """Automotive tire, 10 kg, rubber/steel/textile.

    Expected end-of-life: 45% incineration (TDF), 30% recycling, 15% landfill, 10% retreading.
    """
    return {
        "product_id": "PRD-TIR-001",
        "product_name": "All-Season Radial Tire 205/55R16",
        "product_category": "tires",
        "total_mass_kg": Decimal("10.0"),
        "units_sold": 100000,
        "total_mass_tonnes": Decimal("1000.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.45"), "mass_kg": Decimal("4.5")},
            {"material": "rubber_natural", "mass_fraction": Decimal("0.15"), "mass_kg": Decimal("1.5")},
            {"material": "steel_cord", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("2.0")},
            {"material": "textile_nylon", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("1.0")},
            {"material": "carbon_black", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("1.0")},
        ],
        "expected_lifetime_years": 5,
        "country_of_sale": "US",
    }


@pytest.fixture
def sample_food_product() -> Dict[str, Any]:
    """Packaged food product, 1 kg total, organic material.

    Expected end-of-life: 40% landfill, 25% composting, 20% incineration, 15% AD.
    """
    return {
        "product_id": "PRD-FD-001",
        "product_name": "Organic Cereal Box 500g",
        "product_category": "food_products",
        "total_mass_kg": Decimal("0.600"),
        "units_sold": 2000000,
        "total_mass_tonnes": Decimal("1200.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "food_organic", "mass_fraction": Decimal("0.50"), "mass_kg": Decimal("0.300")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.40"), "mass_kg": Decimal("0.240")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.060")},
        ],
        "expected_lifetime_years": 0,
        "country_of_sale": "DE",
    }


@pytest.fixture
def sample_building_material() -> Dict[str, Any]:
    """Building material (concrete/steel panel), 50 kg, concrete/steel.

    Expected end-of-life: 60% landfill (inert), 30% recycling, 10% reuse.
    """
    return {
        "product_id": "PRD-BLD-001",
        "product_name": "Precast Concrete Panel",
        "product_category": "building_materials",
        "total_mass_kg": Decimal("50.0"),
        "units_sold": 100000,
        "total_mass_tonnes": Decimal("5000.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "concrete", "mass_fraction": Decimal("0.75"), "mass_kg": Decimal("37.5")},
            {"material": "steel_rebar", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("10.0")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.05"), "mass_kg": Decimal("2.5")},
        ],
        "expected_lifetime_years": 50,
        "country_of_sale": "US",
    }


@pytest.fixture
def sample_mixed_product() -> Dict[str, Any]:
    """Mixed material product for testing multi-pathway decomposition.

    Contains 6 distinct materials across different treatment pathways.
    """
    return {
        "product_id": "PRD-MIX-001",
        "product_name": "Multi-Material Assembly",
        "product_category": "mixed",
        "total_mass_kg": Decimal("5.0"),
        "units_sold": 50000,
        "total_mass_tonnes": Decimal("250.0"),
        "reporting_year": 2024,
        "composition": [
            {"material": "steel", "mass_fraction": Decimal("0.25"), "mass_kg": Decimal("1.25")},
            {"material": "plastic_abs", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("1.00")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.20"), "mass_kg": Decimal("1.00")},
            {"material": "glass", "mass_fraction": Decimal("0.15"), "mass_kg": Decimal("0.75")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.50")},
            {"material": "food_organic", "mass_fraction": Decimal("0.10"), "mass_kg": Decimal("0.50")},
        ],
        "expected_lifetime_years": 5,
        "country_of_sale": "US",
    }


# ============================================================================
# EMISSION FACTOR DATABASE FIXTURES
# ============================================================================

@pytest.fixture
def material_treatment_efs() -> Dict[str, Dict[str, Decimal]]:
    """Material x treatment emission factors (kg CO2e per kg material).

    15 materials across 7 treatment pathways. These are gross emission factors
    (excluding avoided emissions, which are always tracked separately).
    """
    return {
        "steel": {
            "landfill": Decimal("0.004"),
            "incineration": Decimal("0.020"),
            "recycling": Decimal("0.050"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.030"),
            "wastewater": Decimal("0.0"),
        },
        "aluminum": {
            "landfill": Decimal("0.004"),
            "incineration": Decimal("0.020"),
            "recycling": Decimal("0.080"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.030"),
            "wastewater": Decimal("0.0"),
        },
        "copper": {
            "landfill": Decimal("0.004"),
            "incineration": Decimal("0.015"),
            "recycling": Decimal("0.100"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.025"),
            "wastewater": Decimal("0.0"),
        },
        "plastic_abs": {
            "landfill": Decimal("0.042"),
            "incineration": Decimal("2.380"),
            "recycling": Decimal("0.310"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("3.100"),
            "wastewater": Decimal("0.0"),
        },
        "plastic_pe": {
            "landfill": Decimal("0.042"),
            "incineration": Decimal("2.850"),
            "recycling": Decimal("0.290"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("3.200"),
            "wastewater": Decimal("0.0"),
        },
        "plastic_pp": {
            "landfill": Decimal("0.042"),
            "incineration": Decimal("2.700"),
            "recycling": Decimal("0.280"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("3.050"),
            "wastewater": Decimal("0.0"),
        },
        "glass": {
            "landfill": Decimal("0.004"),
            "incineration": Decimal("0.010"),
            "recycling": Decimal("0.180"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.010"),
            "wastewater": Decimal("0.0"),
        },
        "paper_cardboard": {
            "landfill": Decimal("1.420"),
            "incineration": Decimal("0.850"),
            "recycling": Decimal("0.150"),
            "composting": Decimal("0.180"),
            "anaerobic_digestion": Decimal("0.120"),
            "open_burning": Decimal("1.100"),
            "wastewater": Decimal("0.0"),
        },
        "wood_mdf": {
            "landfill": Decimal("1.060"),
            "incineration": Decimal("0.120"),
            "recycling": Decimal("0.080"),
            "composting": Decimal("0.150"),
            "anaerobic_digestion": Decimal("0.100"),
            "open_burning": Decimal("0.180"),
            "wastewater": Decimal("0.0"),
        },
        "cotton": {
            "landfill": Decimal("1.200"),
            "incineration": Decimal("0.750"),
            "recycling": Decimal("0.200"),
            "composting": Decimal("0.250"),
            "anaerobic_digestion": Decimal("0.180"),
            "open_burning": Decimal("0.950"),
            "wastewater": Decimal("0.0"),
        },
        "polyester": {
            "landfill": Decimal("0.042"),
            "incineration": Decimal("2.500"),
            "recycling": Decimal("0.350"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("3.000"),
            "wastewater": Decimal("0.0"),
        },
        "rubber_synthetic": {
            "landfill": Decimal("0.042"),
            "incineration": Decimal("2.200"),
            "recycling": Decimal("0.250"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("2.800"),
            "wastewater": Decimal("0.0"),
        },
        "food_organic": {
            "landfill": Decimal("0.580"),
            "incineration": Decimal("0.150"),
            "recycling": Decimal("0.0"),
            "composting": Decimal("0.080"),
            "anaerobic_digestion": Decimal("0.055"),
            "open_burning": Decimal("0.250"),
            "wastewater": Decimal("0.0"),
        },
        "concrete": {
            "landfill": Decimal("0.002"),
            "incineration": Decimal("0.0"),
            "recycling": Decimal("0.010"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.0"),
            "wastewater": Decimal("0.0"),
        },
        "lithium_battery": {
            "landfill": Decimal("0.150"),
            "incineration": Decimal("0.350"),
            "recycling": Decimal("0.450"),
            "composting": Decimal("0.0"),
            "anaerobic_digestion": Decimal("0.0"),
            "open_burning": Decimal("0.500"),
            "wastewater": Decimal("0.0"),
        },
    }


@pytest.fixture
def product_compositions() -> Dict[str, List[Dict[str, Any]]]:
    """Default product compositions by product category (20 categories).

    Used for average-data method when specific composition is unknown.
    """
    return {
        "consumer_electronics": [
            {"material": "plastic_abs", "mass_fraction": Decimal("0.35")},
            {"material": "aluminum", "mass_fraction": Decimal("0.25")},
            {"material": "glass", "mass_fraction": Decimal("0.20")},
            {"material": "copper", "mass_fraction": Decimal("0.10")},
            {"material": "lithium_battery", "mass_fraction": Decimal("0.10")},
        ],
        "large_appliances": [
            {"material": "steel", "mass_fraction": Decimal("0.55")},
            {"material": "plastic_pp", "mass_fraction": Decimal("0.25")},
            {"material": "copper", "mass_fraction": Decimal("0.08")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.07")},
            {"material": "glass", "mass_fraction": Decimal("0.05")},
        ],
        "packaging": [
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.60")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.35")},
            {"material": "aluminum", "mass_fraction": Decimal("0.05")},
        ],
        "clothing": [
            {"material": "cotton", "mass_fraction": Decimal("0.65")},
            {"material": "polyester", "mass_fraction": Decimal("0.35")},
        ],
        "furniture": [
            {"material": "wood_mdf", "mass_fraction": Decimal("0.50")},
            {"material": "steel", "mass_fraction": Decimal("0.35")},
            {"material": "polyester", "mass_fraction": Decimal("0.10")},
            {"material": "plastic_abs", "mass_fraction": Decimal("0.05")},
        ],
        "batteries": [
            {"material": "lithium_battery", "mass_fraction": Decimal("0.50")},
            {"material": "aluminum", "mass_fraction": Decimal("0.20")},
            {"material": "copper", "mass_fraction": Decimal("0.15")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.10")},
            {"material": "steel", "mass_fraction": Decimal("0.05")},
        ],
        "tires": [
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.60")},
            {"material": "steel", "mass_fraction": Decimal("0.20")},
            {"material": "polyester", "mass_fraction": Decimal("0.10")},
            {"material": "carbon_black", "mass_fraction": Decimal("0.10")},
        ],
        "food_products": [
            {"material": "food_organic", "mass_fraction": Decimal("0.50")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.40")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.10")},
        ],
        "building_materials": [
            {"material": "concrete", "mass_fraction": Decimal("0.75")},
            {"material": "steel", "mass_fraction": Decimal("0.20")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.05")},
        ],
        "automotive_parts": [
            {"material": "steel", "mass_fraction": Decimal("0.60")},
            {"material": "aluminum", "mass_fraction": Decimal("0.15")},
            {"material": "plastic_pp", "mass_fraction": Decimal("0.15")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.10")},
        ],
        "medical_devices": [
            {"material": "plastic_pp", "mass_fraction": Decimal("0.40")},
            {"material": "steel", "mass_fraction": Decimal("0.30")},
            {"material": "glass", "mass_fraction": Decimal("0.20")},
            {"material": "copper", "mass_fraction": Decimal("0.10")},
        ],
        "toys": [
            {"material": "plastic_abs", "mass_fraction": Decimal("0.60")},
            {"material": "steel", "mass_fraction": Decimal("0.15")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.15")},
            {"material": "cotton", "mass_fraction": Decimal("0.10")},
        ],
        "sporting_goods": [
            {"material": "plastic_abs", "mass_fraction": Decimal("0.30")},
            {"material": "aluminum", "mass_fraction": Decimal("0.25")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.20")},
            {"material": "polyester", "mass_fraction": Decimal("0.15")},
            {"material": "steel", "mass_fraction": Decimal("0.10")},
        ],
        "cosmetics": [
            {"material": "glass", "mass_fraction": Decimal("0.40")},
            {"material": "plastic_pe", "mass_fraction": Decimal("0.35")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.15")},
            {"material": "aluminum", "mass_fraction": Decimal("0.10")},
        ],
        "office_equipment": [
            {"material": "plastic_abs", "mass_fraction": Decimal("0.40")},
            {"material": "steel", "mass_fraction": Decimal("0.30")},
            {"material": "copper", "mass_fraction": Decimal("0.15")},
            {"material": "glass", "mass_fraction": Decimal("0.15")},
        ],
        "small_appliances": [
            {"material": "plastic_pp", "mass_fraction": Decimal("0.40")},
            {"material": "steel", "mass_fraction": Decimal("0.30")},
            {"material": "copper", "mass_fraction": Decimal("0.15")},
            {"material": "aluminum", "mass_fraction": Decimal("0.10")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.05")},
        ],
        "garden_tools": [
            {"material": "steel", "mass_fraction": Decimal("0.50")},
            {"material": "wood_mdf", "mass_fraction": Decimal("0.25")},
            {"material": "plastic_pp", "mass_fraction": Decimal("0.15")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.10")},
        ],
        "pet_products": [
            {"material": "plastic_pe", "mass_fraction": Decimal("0.45")},
            {"material": "cotton", "mass_fraction": Decimal("0.25")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.15")},
            {"material": "steel", "mass_fraction": Decimal("0.15")},
        ],
        "lighting": [
            {"material": "glass", "mass_fraction": Decimal("0.45")},
            {"material": "aluminum", "mass_fraction": Decimal("0.25")},
            {"material": "plastic_abs", "mass_fraction": Decimal("0.20")},
            {"material": "copper", "mass_fraction": Decimal("0.10")},
        ],
        "mixed": [
            {"material": "steel", "mass_fraction": Decimal("0.25")},
            {"material": "plastic_abs", "mass_fraction": Decimal("0.20")},
            {"material": "paper_cardboard", "mass_fraction": Decimal("0.20")},
            {"material": "glass", "mass_fraction": Decimal("0.15")},
            {"material": "rubber_synthetic", "mass_fraction": Decimal("0.10")},
            {"material": "food_organic", "mass_fraction": Decimal("0.10")},
        ],
    }


@pytest.fixture
def regional_treatment_mixes() -> Dict[str, Dict[str, Decimal]]:
    """Regional treatment mixes (12 regions) - fraction to each pathway.

    Each region sums to 1.0 across all treatment pathways.
    """
    return {
        "US": {
            "landfill": Decimal("0.52"),
            "incineration": Decimal("0.13"),
            "recycling": Decimal("0.25"),
            "composting": Decimal("0.06"),
            "anaerobic_digestion": Decimal("0.02"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "DE": {
            "landfill": Decimal("0.01"),
            "incineration": Decimal("0.32"),
            "recycling": Decimal("0.47"),
            "composting": Decimal("0.12"),
            "anaerobic_digestion": Decimal("0.06"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "GB": {
            "landfill": Decimal("0.23"),
            "incineration": Decimal("0.28"),
            "recycling": Decimal("0.32"),
            "composting": Decimal("0.10"),
            "anaerobic_digestion": Decimal("0.05"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "JP": {
            "landfill": Decimal("0.02"),
            "incineration": Decimal("0.72"),
            "recycling": Decimal("0.20"),
            "composting": Decimal("0.03"),
            "anaerobic_digestion": Decimal("0.02"),
            "open_burning": Decimal("0.00"),
            "wastewater": Decimal("0.01"),
        },
        "FR": {
            "landfill": Decimal("0.22"),
            "incineration": Decimal("0.34"),
            "recycling": Decimal("0.26"),
            "composting": Decimal("0.10"),
            "anaerobic_digestion": Decimal("0.06"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "CN": {
            "landfill": Decimal("0.55"),
            "incineration": Decimal("0.25"),
            "recycling": Decimal("0.12"),
            "composting": Decimal("0.04"),
            "anaerobic_digestion": Decimal("0.02"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "IN": {
            "landfill": Decimal("0.60"),
            "incineration": Decimal("0.05"),
            "recycling": Decimal("0.10"),
            "composting": Decimal("0.05"),
            "anaerobic_digestion": Decimal("0.02"),
            "open_burning": Decimal("0.17"),
            "wastewater": Decimal("0.01"),
        },
        "BR": {
            "landfill": Decimal("0.58"),
            "incineration": Decimal("0.02"),
            "recycling": Decimal("0.15"),
            "composting": Decimal("0.05"),
            "anaerobic_digestion": Decimal("0.02"),
            "open_burning": Decimal("0.17"),
            "wastewater": Decimal("0.01"),
        },
        "AU": {
            "landfill": Decimal("0.40"),
            "incineration": Decimal("0.05"),
            "recycling": Decimal("0.37"),
            "composting": Decimal("0.10"),
            "anaerobic_digestion": Decimal("0.04"),
            "open_burning": Decimal("0.03"),
            "wastewater": Decimal("0.01"),
        },
        "KR": {
            "landfill": Decimal("0.08"),
            "incineration": Decimal("0.25"),
            "recycling": Decimal("0.55"),
            "composting": Decimal("0.07"),
            "anaerobic_digestion": Decimal("0.03"),
            "open_burning": Decimal("0.01"),
            "wastewater": Decimal("0.01"),
        },
        "CA": {
            "landfill": Decimal("0.45"),
            "incineration": Decimal("0.05"),
            "recycling": Decimal("0.30"),
            "composting": Decimal("0.12"),
            "anaerobic_digestion": Decimal("0.05"),
            "open_burning": Decimal("0.02"),
            "wastewater": Decimal("0.01"),
        },
        "GLOBAL": {
            "landfill": Decimal("0.40"),
            "incineration": Decimal("0.16"),
            "recycling": Decimal("0.20"),
            "composting": Decimal("0.08"),
            "anaerobic_digestion": Decimal("0.04"),
            "open_burning": Decimal("0.11"),
            "wastewater": Decimal("0.01"),
        },
    }


@pytest.fixture
def landfill_fod_params() -> Dict[str, Dict[str, Decimal]]:
    """Landfill first-order decay parameters by material and climate zone.

    DOC: degradable organic carbon fraction
    DOCf: fraction of DOC that decomposes (default 0.5)
    MCF: methane correction factor (1.0 for managed anaerobic)
    k: first-order decay rate (yr^-1)
    F: fraction of CH4 in landfill gas (default 0.5)
    OX: oxidation factor (default 0.1 for managed)
    """
    return {
        "paper_cardboard_temperate_wet": {
            "doc": Decimal("0.40"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.06"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "paper_cardboard_tropical_wet": {
            "doc": Decimal("0.40"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.07"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "food_organic_temperate_wet": {
            "doc": Decimal("0.15"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.18"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "food_organic_tropical_wet": {
            "doc": Decimal("0.15"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.40"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "wood_mdf_temperate_wet": {
            "doc": Decimal("0.43"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.03"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "cotton_temperate_wet": {
            "doc": Decimal("0.24"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.06"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
        "plastic_abs_temperate_wet": {
            "doc": Decimal("0.0"), "docf": Decimal("0.50"),
            "mcf": Decimal("1.0"), "k": Decimal("0.0"),
            "f": Decimal("0.50"), "ox": Decimal("0.10"),
        },
    }


@pytest.fixture
def incineration_params() -> Dict[str, Dict[str, Decimal]]:
    """Incineration parameters by material type.

    fossil_carbon_fraction: fraction of carbon that is fossil-origin
    carbon_content: total carbon content (dry weight basis)
    combustion_efficiency: fraction of waste completely combusted
    n2o_ef: N2O emission factor (kg N2O per tonne waste)
    ch4_ef: CH4 emission factor (kg CH4 per tonne waste)
    energy_recovery_ef: grid displacement factor (kgCO2e per MWh)
    """
    return {
        "plastic_abs": {
            "fossil_carbon_fraction": Decimal("1.00"),
            "carbon_content": Decimal("0.676"),
            "combustion_efficiency": Decimal("0.995"),
            "n2o_ef": Decimal("0.040"),
            "ch4_ef": Decimal("0.010"),
            "energy_recovery_ef": Decimal("420.0"),
        },
        "plastic_pe": {
            "fossil_carbon_fraction": Decimal("1.00"),
            "carbon_content": Decimal("0.857"),
            "combustion_efficiency": Decimal("0.995"),
            "n2o_ef": Decimal("0.040"),
            "ch4_ef": Decimal("0.010"),
            "energy_recovery_ef": Decimal("420.0"),
        },
        "paper_cardboard": {
            "fossil_carbon_fraction": Decimal("0.01"),
            "carbon_content": Decimal("0.460"),
            "combustion_efficiency": Decimal("0.995"),
            "n2o_ef": Decimal("0.060"),
            "ch4_ef": Decimal("0.030"),
            "energy_recovery_ef": Decimal("420.0"),
        },
        "food_organic": {
            "fossil_carbon_fraction": Decimal("0.0"),
            "carbon_content": Decimal("0.380"),
            "combustion_efficiency": Decimal("0.995"),
            "n2o_ef": Decimal("0.050"),
            "ch4_ef": Decimal("0.020"),
            "energy_recovery_ef": Decimal("420.0"),
        },
        "rubber_synthetic": {
            "fossil_carbon_fraction": Decimal("0.80"),
            "carbon_content": Decimal("0.700"),
            "combustion_efficiency": Decimal("0.990"),
            "n2o_ef": Decimal("0.040"),
            "ch4_ef": Decimal("0.020"),
            "energy_recovery_ef": Decimal("420.0"),
        },
        "cotton": {
            "fossil_carbon_fraction": Decimal("0.0"),
            "carbon_content": Decimal("0.444"),
            "combustion_efficiency": Decimal("0.995"),
            "n2o_ef": Decimal("0.060"),
            "ch4_ef": Decimal("0.020"),
            "energy_recovery_ef": Decimal("420.0"),
        },
    }


@pytest.fixture
def recycling_factors() -> Dict[str, Dict[str, Decimal]]:
    """Recycling process factors by material.

    recovery_rate: fraction of material recovered
    quality_factor: downcycling quality factor (1.0 = no loss)
    process_ef: recycling process emissions (kg CO2e per kg)
    avoided_ef: virgin production emissions displaced (kg CO2e per kg)
    transport_ef: transport to recycling facility (kg CO2e per kg)
    """
    return {
        "steel": {
            "recovery_rate": Decimal("0.90"),
            "quality_factor": Decimal("0.95"),
            "process_ef": Decimal("0.050"),
            "avoided_ef": Decimal("1.800"),
            "transport_ef": Decimal("0.010"),
        },
        "aluminum": {
            "recovery_rate": Decimal("0.85"),
            "quality_factor": Decimal("0.90"),
            "process_ef": Decimal("0.080"),
            "avoided_ef": Decimal("8.500"),
            "transport_ef": Decimal("0.012"),
        },
        "copper": {
            "recovery_rate": Decimal("0.92"),
            "quality_factor": Decimal("0.95"),
            "process_ef": Decimal("0.100"),
            "avoided_ef": Decimal("3.500"),
            "transport_ef": Decimal("0.015"),
        },
        "glass": {
            "recovery_rate": Decimal("0.80"),
            "quality_factor": Decimal("0.85"),
            "process_ef": Decimal("0.180"),
            "avoided_ef": Decimal("0.550"),
            "transport_ef": Decimal("0.020"),
        },
        "paper_cardboard": {
            "recovery_rate": Decimal("0.78"),
            "quality_factor": Decimal("0.80"),
            "process_ef": Decimal("0.150"),
            "avoided_ef": Decimal("0.920"),
            "transport_ef": Decimal("0.015"),
        },
        "plastic_pe": {
            "recovery_rate": Decimal("0.70"),
            "quality_factor": Decimal("0.75"),
            "process_ef": Decimal("0.290"),
            "avoided_ef": Decimal("1.800"),
            "transport_ef": Decimal("0.012"),
        },
        "plastic_abs": {
            "recovery_rate": Decimal("0.65"),
            "quality_factor": Decimal("0.70"),
            "process_ef": Decimal("0.310"),
            "avoided_ef": Decimal("2.100"),
            "transport_ef": Decimal("0.014"),
        },
        "rubber_synthetic": {
            "recovery_rate": Decimal("0.60"),
            "quality_factor": Decimal("0.65"),
            "process_ef": Decimal("0.250"),
            "avoided_ef": Decimal("2.500"),
            "transport_ef": Decimal("0.018"),
        },
    }


@pytest.fixture
def product_weights() -> Dict[str, Decimal]:
    """Default product weight estimates by category (kg per unit)."""
    return {
        "consumer_electronics": Decimal("0.200"),
        "large_appliances": Decimal("35.0"),
        "small_appliances": Decimal("2.5"),
        "packaging": Decimal("0.010"),
        "clothing": Decimal("0.300"),
        "furniture": Decimal("25.0"),
        "batteries": Decimal("0.500"),
        "tires": Decimal("10.0"),
        "food_products": Decimal("0.500"),
        "building_materials": Decimal("50.0"),
        "automotive_parts": Decimal("15.0"),
        "medical_devices": Decimal("1.0"),
        "toys": Decimal("0.500"),
        "sporting_goods": Decimal("2.0"),
        "cosmetics": Decimal("0.200"),
        "office_equipment": Decimal("5.0"),
        "garden_tools": Decimal("3.0"),
        "pet_products": Decimal("1.0"),
        "lighting": Decimal("0.500"),
        "mixed": Decimal("5.0"),
    }


@pytest.fixture
def composting_ad_factors() -> Dict[str, Dict[str, Decimal]]:
    """Composting and anaerobic digestion emission factors by material.

    ch4_ef: CH4 from composting (kg CH4/tonne wet waste)
    n2o_ef: N2O from composting (kg N2O/tonne wet waste)
    ad_ch4_fugitive: fugitive CH4 from AD (fraction of biogas)
    ad_n2o_ef: N2O from digestate (kg N2O/tonne)
    """
    return {
        "food_organic": {
            "ch4_ef": Decimal("4.0"),
            "n2o_ef": Decimal("0.30"),
            "ad_ch4_fugitive": Decimal("0.02"),
            "ad_n2o_ef": Decimal("0.10"),
        },
        "paper_cardboard": {
            "ch4_ef": Decimal("2.0"),
            "n2o_ef": Decimal("0.15"),
            "ad_ch4_fugitive": Decimal("0.02"),
            "ad_n2o_ef": Decimal("0.05"),
        },
        "wood_mdf": {
            "ch4_ef": Decimal("1.5"),
            "n2o_ef": Decimal("0.10"),
            "ad_ch4_fugitive": Decimal("0.02"),
            "ad_n2o_ef": Decimal("0.04"),
        },
        "cotton": {
            "ch4_ef": Decimal("3.0"),
            "n2o_ef": Decimal("0.20"),
            "ad_ch4_fugitive": Decimal("0.02"),
            "ad_n2o_ef": Decimal("0.08"),
        },
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_database_engine(
    material_treatment_efs,
    product_compositions,
    regional_treatment_mixes,
    landfill_fod_params,
    incineration_params,
    recycling_factors,
    product_weights,
    composting_ad_factors,
) -> MagicMock:
    """Create a fully mocked EOLProductDatabaseEngine with EF lookups."""
    engine = MagicMock(spec=[
        "get_material_treatment_ef", "get_product_composition",
        "get_regional_treatment_mix", "get_landfill_fod_params",
        "get_incineration_params", "get_recycling_factors",
        "get_product_weight", "get_composting_ad_factors",
        "get_database_summary", "get_lookup_count",
        "get_available_materials", "get_available_treatments",
        "get_available_regions", "get_available_categories",
        "validate_material", "validate_treatment",
    ])

    def _get_material_treatment_ef(material, treatment):
        mat_efs = material_treatment_efs.get(material)
        if mat_efs is not None:
            ef = mat_efs.get(treatment)
            if ef is not None:
                return {"ef_kg_co2e_per_kg": ef, "material": material, "treatment": treatment}
        return None

    def _get_product_composition(category):
        comp = product_compositions.get(category)
        if comp is not None:
            return comp
        return None

    def _get_regional_treatment_mix(region):
        mix = regional_treatment_mixes.get(region)
        if mix is not None:
            return mix
        return regional_treatment_mixes.get("GLOBAL")

    def _get_landfill_fod_params(material, climate="temperate_wet"):
        key = f"{material}_{climate}"
        return landfill_fod_params.get(key)

    def _get_incineration_params(material):
        return incineration_params.get(material)

    def _get_recycling_factors(material):
        return recycling_factors.get(material)

    def _get_product_weight(category):
        return product_weights.get(category)

    def _get_composting_ad_factors(material):
        return composting_ad_factors.get(material)

    engine.get_material_treatment_ef.side_effect = _get_material_treatment_ef
    engine.get_product_composition.side_effect = _get_product_composition
    engine.get_regional_treatment_mix.side_effect = _get_regional_treatment_mix
    engine.get_landfill_fod_params.side_effect = _get_landfill_fod_params
    engine.get_incineration_params.side_effect = _get_incineration_params
    engine.get_recycling_factors.side_effect = _get_recycling_factors
    engine.get_product_weight.side_effect = _get_product_weight
    engine.get_composting_ad_factors.side_effect = _get_composting_ad_factors
    engine.get_lookup_count.return_value = 0
    engine.validate_material.return_value = True
    engine.validate_treatment.return_value = True
    engine.get_available_materials.return_value = list(material_treatment_efs.keys())
    engine.get_available_treatments.return_value = [
        "landfill", "incineration", "recycling", "composting",
        "anaerobic_digestion", "open_burning", "wastewater",
    ]
    engine.get_available_regions.return_value = list(regional_treatment_mixes.keys())
    engine.get_available_categories.return_value = list(product_compositions.keys())
    engine.get_database_summary.return_value = {
        "materials": 15,
        "treatments": 7,
        "regions": 12,
        "categories": 20,
        "landfill_fod_entries": 7,
        "incineration_entries": 6,
        "recycling_entries": 8,
    }

    return engine


@pytest.fixture
def mock_waste_type_engine() -> MagicMock:
    """Mock WasteTypeSpecificCalculatorEngine."""
    engine = MagicMock()
    engine.calculate.return_value = {
        "gross_emissions_kgco2e": Decimal("1250.0"),
        "avoided_emissions_kgco2e": Decimal("450.0"),
        "biogenic_co2e_kgco2e": Decimal("120.0"),
        "by_treatment": {
            "landfill": Decimal("380.0"),
            "incineration": Decimal("520.0"),
            "recycling": Decimal("150.0"),
            "composting": Decimal("80.0"),
            "anaerobic_digestion": Decimal("40.0"),
            "open_burning": Decimal("80.0"),
        },
        "avoided_by_treatment": {
            "recycling": Decimal("350.0"),
            "anaerobic_digestion": Decimal("100.0"),
        },
        "method": "waste_type_specific",
        "dqi_score": Decimal("75.0"),
    }
    return engine


@pytest.fixture
def mock_average_data_engine() -> MagicMock:
    """Mock AverageDataCalculatorEngine."""
    engine = MagicMock()
    engine.calculate.return_value = {
        "gross_emissions_kgco2e": Decimal("1400.0"),
        "avoided_emissions_kgco2e": Decimal("0.0"),
        "method": "average_data",
        "dqi_score": Decimal("45.0"),
        "uncertainty_pct": Decimal("40.0"),
    }
    return engine


@pytest.fixture
def mock_producer_specific_engine() -> MagicMock:
    """Mock ProducerSpecificCalculatorEngine."""
    engine = MagicMock()
    engine.calculate.return_value = {
        "gross_emissions_kgco2e": Decimal("1100.0"),
        "avoided_emissions_kgco2e": Decimal("520.0"),
        "method": "producer_specific",
        "dqi_score": Decimal("90.0"),
        "epd_id": "EPD-2024-001",
        "verification_status": "third_party_verified",
    }
    return engine


@pytest.fixture
def mock_hybrid_engine() -> MagicMock:
    """Mock HybridAggregatorEngine."""
    engine = MagicMock()
    engine.aggregate.return_value = {
        "gross_emissions_kgco2e": Decimal("1200.0"),
        "avoided_emissions_kgco2e": Decimal("480.0"),
        "method": "hybrid",
        "circularity_index": Decimal("0.35"),
        "recycling_rate": Decimal("0.28"),
        "diversion_rate": Decimal("0.48"),
    }
    return engine


@pytest.fixture
def mock_compliance_engine() -> MagicMock:
    """Mock ComplianceCheckerEngine."""
    engine = MagicMock()
    engine.check_compliance.return_value = {
        "compliant": True,
        "framework": "GHG_PROTOCOL_SCOPE3",
        "issues": [],
        "warnings": [],
        "dc_checks_passed": 8,
        "dc_checks_total": 8,
        "completeness_score": Decimal("85.0"),
    }
    return engine


@pytest.fixture
def mock_pipeline_engine() -> MagicMock:
    """Mock EndOfLifeTreatmentPipelineEngine."""
    engine = MagicMock()
    engine.execute.return_value = {
        "calculation_id": "CALC-EOL-001",
        "gross_emissions_tco2e": Decimal("1.25"),
        "avoided_emissions_tco2e": Decimal("0.48"),
        "biogenic_co2e_tco2e": Decimal("0.12"),
        "by_treatment": {
            "landfill": Decimal("0.38"),
            "incineration": Decimal("0.52"),
            "recycling": Decimal("0.15"),
            "composting": Decimal("0.08"),
            "anaerobic_digestion": Decimal("0.04"),
            "open_burning": Decimal("0.08"),
        },
        "circularity_metrics": {
            "recycling_rate": Decimal("0.28"),
            "diversion_rate": Decimal("0.48"),
            "circularity_index": Decimal("0.35"),
        },
        "compliant": True,
        "dqi_score": Decimal("75.0"),
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
    cfg.general.agent_id = "GL-MRV-S3-012"
    cfg.general.agent_component = "AGENT-MRV-025"
    cfg.general.version = "1.0.0"
    cfg.general.api_prefix = "/api/v1/end-of-life-treatment"
    cfg.general.default_gwp = "AR5"
    cfg.database.host = "localhost"
    cfg.database.port = 5432
    cfg.database.database = "greenlang"
    cfg.database.table_prefix = "gl_eol_"
    cfg.database.schema = "end_of_life_treatment_service"
    cfg.waste_type.enable_fod_model = True
    cfg.waste_type.default_climate = "temperate_wet"
    cfg.waste_type.projection_years = 50
    cfg.waste_type.default_docf = Decimal("0.50")
    cfg.waste_type.default_oxidation = Decimal("0.10")
    cfg.average_data.default_region = "GLOBAL"
    cfg.average_data.enable_weight_estimation = True
    cfg.producer_specific.require_epd = False
    cfg.producer_specific.min_verification_level = "self_declared"
    cfg.hybrid.method_priority = [
        "producer_specific", "waste_type_specific", "average_data",
    ]
    cfg.hybrid.enable_gap_filling = True
    cfg.hybrid.separate_avoided_emissions = True
    cfg.circularity.enable_circularity_metrics = True
    cfg.circularity.track_waste_hierarchy = True
    cfg.compliance.get_frameworks.return_value = [
        "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
        "CSRD_ESRS_E5", "CDP", "SBTI", "GRI",
    ]
    cfg.compliance.strict_mode = False
    cfg.compliance.materiality_threshold = Decimal("0.01")
    cfg.provenance.enable_provenance = True
    cfg.provenance.hash_algorithm = "sha256"
    cfg.cache.enable_cache = True
    cfg.cache.ttl_seconds = 3600
    cfg.metrics.enable_metrics = True
    cfg.metrics.prefix = "gl_eol_"
    cfg.uncertainty.default_method = "propagation"
    cfg.uncertainty.confidence_level = Decimal("0.95")
    cfg.api.enable_api = True
    cfg.api.prefix = "/api/v1/end-of-life-treatment"
    return cfg


# ============================================================================
# MOCK SERVICE AND METRICS FIXTURES
# ============================================================================

@pytest.fixture
def mock_service() -> Mock:
    """Mock end-of-life treatment service facade."""
    mock = Mock()
    mock.calculate_eol_emissions = AsyncMock(return_value={
        "calculation_id": "CALC-EOL-001",
        "gross_emissions_tco2e": Decimal("1.25"),
        "avoided_emissions_tco2e": Decimal("0.48"),
        "status": "success",
    })
    return mock


@pytest.fixture
def mock_metrics() -> Mock:
    """Mock Prometheus metrics recorder."""
    mock = Mock()
    mock.record_calculation = Mock()
    mock.record_emissions = Mock()
    mock.record_product_mass = Mock()
    mock.record_treatment_pathway = Mock()
    mock.record_error = Mock()
    return mock


@pytest.fixture
def mock_provenance() -> Mock:
    """Mock provenance tracker."""
    mock = Mock()
    mock.generate_hash = Mock(return_value="a" * 64)
    mock.record_calculation = Mock()
    return mock


# ============================================================================
# BATCH AND COMPLIANCE FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch_inputs(
    sample_electronics_product,
    sample_packaging_product,
    sample_clothing_product,
    sample_food_product,
    sample_building_material,
) -> List[Dict[str, Any]]:
    """Batch of 5 diverse product inputs for pipeline testing."""
    return [
        sample_electronics_product,
        sample_packaging_product,
        sample_clothing_product,
        sample_food_product,
        sample_building_material,
    ]


@pytest.fixture
def sample_compliance_input() -> Dict[str, Any]:
    """Sample compliance check input for GHG Protocol framework."""
    return {
        "framework": "GHG_PROTOCOL_SCOPE3",
        "calculation_id": "CALC-EOL-001",
        "tenant_id": "tenant-001",
        "check_double_counting": True,
        "check_boundary": True,
        "check_avoided_emissions_separate": True,
        "check_data_quality": True,
        "min_dqi_score": Decimal("3.0"),
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_full_compliance_result(**overrides) -> Dict[str, Any]:
    """Build a fully compliant result dict with all required fields.

    CRITICAL: avoided_emissions_tco2e is ALWAYS separate from
    gross_emissions_tco2e and is NEVER netted against it.

    Pass keyword overrides to modify or nullify specific fields.
    """
    base = {
        "gross_emissions_kgco2e": Decimal("1250.0"),
        "avoided_emissions_kgco2e": Decimal("450.0"),
        "biogenic_co2e_kgco2e": Decimal("120.0"),
        "method": "waste_type_specific",
        "calculation_method": "waste_type_specific",
        "ef_sources": ["epa_warm", "defra", "ipcc"],
        "ef_source": "epa_warm",
        "product_category": "consumer_electronics",
        "boundary": "sold_products_eol_only",
        "exclusions": "None - all treatment pathways included",
        "dqi_score": Decimal("75.0"),
        "data_quality_score": Decimal("75.0"),
        "uncertainty_analysis": {"method": "propagation", "ci_95": [1000, 1500]},
        "uncertainty": {"method": "propagation"},
        "base_year": 2020,
        "methodology": "GHG Protocol Scope 3 Category 12, waste-type-specific",
        "targets": "Reduce EOL emissions 30% by 2030",
        "reduction_targets": "30% by 2030",
        "actions": "Design for recyclability",
        "reduction_actions": "Material substitution and design for disassembly",
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
        "by_treatment": {
            "landfill": Decimal("380.0"),
            "incineration": Decimal("520.0"),
            "recycling": Decimal("150.0"),
            "composting": Decimal("80.0"),
            "anaerobic_digestion": Decimal("40.0"),
            "open_burning": Decimal("80.0"),
        },
        "by_method": {"waste_type_specific": Decimal("1250.0")},
        "completeness_score": Decimal("85.0"),
        "method_coverage": Decimal("90.0"),
        "avoided_emissions_separate": True,
        "dc_cat5_excluded": True,
        "dc_cat10_excluded": True,
        "dc_cat11_excluded": True,
        "dc_cat13_excluded": True,
        "dc_scope1_excluded": True,
        "dc_scope2_excluded": True,
        "circularity_index": Decimal("0.35"),
        "recycling_rate": Decimal("0.28"),
        "diversion_rate": Decimal("0.48"),
        "waste_hierarchy_compliance": Decimal("70.0"),
        "progress_tracking": {"2023": 1400, "2024": 1250},
        "year_over_year_change": Decimal("-10.7"),
        "target_coverage": "80%",
        "sbti_coverage": "80%",
        "esrs_e5_compliant": True,
    }
    base.update(overrides)
    return base
