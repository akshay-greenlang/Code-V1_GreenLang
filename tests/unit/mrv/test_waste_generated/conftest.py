# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-018: Waste Generated in Operations Agent.

Provides comprehensive test fixtures for:
- Waste stream inputs (food waste, plastics, paper, metals, etc.)
- Landfill emissions (managed anaerobic, semi-aerobic, unmanaged)
- Incineration emissions (continuous stoker, batch, with/without energy recovery)
- Recycling and composting (open loop, closed loop, quality factors)
- Anaerobic digestion (biowaste plant, gastight storage)
- Wastewater treatment (COD/BOD basis, aerobic/anaerobic systems)
- Waste composition data (multi-component analysis)
- Configuration objects (landfill, incineration, recycling, wastewater)
- Mock engines (database, landfill, incineration, recycling, compliance)
- Emission factors, classification results, and batch inputs

Usage:
    def test_something(sample_waste_stream_input, mock_database_engine):
        result = calculate(sample_waste_stream_input, mock_database_engine)
        assert result.emissions_tco2e > 0
"""

from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest

# Note: Adjust imports when actual models are implemented
# from greenlang.agents.mrv.waste_generated.models import (
#     CalculationMethod, WasteTreatmentMethod, WasteCategory, WasteStream,
#     LandfillType, ClimateZone, IncineratorType, RecyclingType,
#     WastewaterSystem, GasCollectionSystem, EFSource, ComplianceFramework,
#     WasteStreamInput, LandfillInput, IncinerationInput, RecyclingInput,
#     CompostingInput, AnaerobicDigestionInput, WastewaterInput,
#     WasteComposition, CalculationRequest, CalculationResult
# )
# from greenlang.agents.mrv.waste_generated.config import WasteGeneratedConfig


# ============================================================================
# WASTE STREAM INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_waste_stream_input() -> Dict[str, Any]:
    """
    Sample waste stream input - Food waste to landfill.

    Represents a typical commercial food waste disposal scenario:
    - Waste Category: Food waste (high DOC content)
    - Treatment: Managed anaerobic landfill with gas capture
    - Mass: 100 tonnes per year
    - Climate: Temperate wet (moderate decay rate)
    - Gas collection: Active with geomembrane cap (90% capture efficiency)
    """
    return {
        "stream_id": "WS-2026-001",
        "facility_id": "FAC-001",
        "tenant_id": "tenant-001",
        "waste_category": "FOOD_WASTE",
        "treatment_method": "LANDFILL_WITH_GAS_CAPTURE",
        "waste_stream": "COMMERCIAL_INDUSTRIAL",
        "mass_tonnes": Decimal("100.0"),
        "reporting_year": 2026,
        "reporting_period_start": date(2026, 1, 1),
        "reporting_period_end": date(2026, 12, 31),
        "supplier_name": "WasteCo Inc.",
        "facility_location": "Munich, DE",
        "description": "Commercial food waste to managed landfill"
    }


@pytest.fixture
def sample_landfill_input() -> Dict[str, Any]:
    """
    Sample landfill emissions input - Managed anaerobic landfill.

    Represents a typical modern sanitary landfill:
    - Type: Managed anaerobic (MCF = 1.0)
    - Climate: Temperate wet (k = 0.09 for food waste)
    - Gas collection: Active with geomembrane cap (90% efficiency)
    - Oxidation: 10% of uncaptured CH4 oxidized in soil cover
    - Energy recovery: None (gas flared only)
    """
    return {
        "landfill_type": "MANAGED_ANAEROBIC",
        "climate_zone": "TEMPERATE_WET",
        "gas_collection": "ACTIVE_GEOMEMBRANE",
        "oxidation_factor": Decimal("0.10"),  # 10% oxidation
        "energy_recovery": False,
        "flare_efficiency": Decimal("0.98"),  # 98% CH4 destroyed in flare
        "doc_value": None,  # Use default for waste type
        "docf_value": None,  # Use default
        "mcf_override": None,  # Use default for landfill type
        "description": "Modern managed landfill with active gas collection"
    }


@pytest.fixture
def sample_incineration_input() -> Dict[str, Any]:
    """
    Sample incineration input - Continuous stoker with energy recovery.

    Represents a modern waste-to-energy facility:
    - Incinerator: Continuous stoker (mass burn)
    - Waste: Mixed plastics (high fossil carbon content)
    - Energy recovery: Combined heat and power (CHP)
    - Combustion efficiency: 99.5%
    - Air pollution controls: SNCR for NOx reduction
    """
    return {
        "incinerator_type": "CONTINUOUS_STOKER",
        "energy_recovery": True,
        "combustion_efficiency": Decimal("0.995"),
        "fossil_carbon_fraction": Decimal("0.85"),  # Plastics are mostly fossil
        "dry_matter_fraction": Decimal("0.90"),  # 90% dry matter
        "carbon_content": None,  # Use default for waste type
        "n2o_ef_kg_per_tonne": None,  # Use default
        "ch4_ef_kg_per_tonne": None,  # Use default
        "electricity_generated_mwh": Decimal("500.0"),  # Energy recovered
        "heat_generated_mwh": Decimal("200.0"),
        "description": "Waste-to-energy facility with CHP"
    }


@pytest.fixture
def sample_recycling_input() -> Dict[str, Any]:
    """
    Sample recycling input - Open loop paper recycling.

    Represents a typical paper recycling operation:
    - Type: Open loop (paper to cardboard)
    - Quality factor: 0.9 (10% quality degradation)
    - Material recovery rate: 85%
    - Contamination rate: 5%
    - Substitution approach: Avoided emissions from virgin production
    """
    return {
        "recycling_type": "OPEN_LOOP",
        "quality_factor": Decimal("0.9"),
        "material_recovery_rate": Decimal("0.85"),
        "contamination_rate": Decimal("0.05"),
        "substitution_approach": True,  # Credit avoided emissions
        "virgin_production_ef_kg_co2e_per_kg": None,  # Use default
        "recycling_process_ef_kg_co2e_per_kg": None,  # Use default
        "transport_distance_km": Decimal("50.0"),  # To recycling facility
        "description": "Open loop paper recycling to cardboard"
    }


@pytest.fixture
def sample_composting_input() -> Dict[str, Any]:
    """
    Sample composting input - Windrow composting of food waste.

    Represents an aerobic composting facility:
    - Method: Windrow (turned piles)
    - Waste basis: Wet weight (as received)
    - CH4 emission factor: 4 kg CH4/tonne wet waste (IPCC default)
    - N2O emission factor: 0.3 kg N2O/tonne wet waste
    - Moisture content: 70%
    """
    return {
        "composting_method": "WINDROW",
        "wet_weight_basis": True,
        "moisture_content": Decimal("0.70"),
        "ch4_ef_kg_per_tonne": Decimal("4.0"),  # IPCC default
        "n2o_ef_kg_per_tonne": Decimal("0.3"),  # IPCC default
        "duration_days": 90,  # Composting duration
        "description": "Windrow composting of food waste"
    }


@pytest.fixture
def sample_anaerobic_digestion_input() -> Dict[str, Any]:
    """
    Sample anaerobic digestion input - Biowaste plant with gastight storage.

    Represents a modern anaerobic digestion facility:
    - System: Biowaste AD plant (wet, mesophilic)
    - Biogas storage: Gastight (no fugitive CH4)
    - Digestate storage: Covered (no fugitive CH4)
    - Biogas use: CHP for electricity and heat
    - Methane yield: 100 m³ CH4/tonne VS (volatile solids)
    """
    return {
        "ad_system": "BIOWASTE_PLANT_WET",
        "biogas_storage_gastight": True,
        "digestate_storage_covered": True,
        "biogas_to_energy": True,
        "methane_yield_m3_per_tonne_vs": Decimal("100.0"),
        "volatile_solids_fraction": Decimal("0.80"),  # 80% VS in dry matter
        "biogas_ch4_fraction": Decimal("0.60"),  # 60% CH4 in biogas
        "ch4_leakage_rate": Decimal("0.01"),  # 1% fugitive CH4
        "electricity_generated_mwh": Decimal("150.0"),
        "heat_generated_mwh": Decimal("100.0"),
        "description": "Anaerobic digestion with CHP"
    }


@pytest.fixture
def sample_wastewater_input() -> Dict[str, Any]:
    """
    Sample wastewater treatment input - COD basis, centralized aerobic.

    Represents a well-managed municipal wastewater treatment plant:
    - System: Centralized aerobic (activated sludge)
    - Organic load basis: COD (Chemical Oxygen Demand)
    - COD load: 1000 kg COD
    - MCF: 0.00 (well-managed aerobic, no CH4)
    - N2O emissions: From nitrification/denitrification
    """
    return {
        "wastewater_system": "CENTRALIZED_AEROBIC_GOOD",
        "organic_load_basis": "COD",  # Chemical Oxygen Demand
        "cod_load_kg": Decimal("1000.0"),
        "bod_load_kg": None,  # COD basis used instead
        "total_nitrogen_kg": Decimal("50.0"),
        "mcf_override": None,  # Use default (0.00 for aerobic)
        "n2o_ef_kg_n2o_per_kg_n": Decimal("0.005"),  # IPCC default
        "sludge_removed_kg": Decimal("200.0"),
        "sludge_treatment": "ANAEROBIC_DIGESTION",
        "description": "Municipal activated sludge WWTP"
    }


@pytest.fixture
def sample_waste_composition() -> Dict[str, Any]:
    """
    Sample waste composition - Mixed commercial waste stream.

    Represents a typical commercial waste composition:
    - Paper/cardboard: 40%
    - Food waste: 35%
    - Plastics (mixed): 25%
    - Total: 100 tonnes
    """
    return {
        "composition_id": "COMP-2026-001",
        "tenant_id": "tenant-001",
        "description": "Commercial waste composition",
        "total_mass_tonnes": Decimal("100.0"),
        "components": [
            {
                "waste_category": "PAPER_CARDBOARD",
                "mass_fraction": Decimal("0.40"),
                "mass_tonnes": Decimal("40.0"),
                "doc_fraction": Decimal("0.40"),  # Degradable organic carbon
                "fossil_carbon_fraction": Decimal("0.0"),
                "moisture_content": Decimal("0.10")
            },
            {
                "waste_category": "FOOD_WASTE",
                "mass_fraction": Decimal("0.35"),
                "mass_tonnes": Decimal("35.0"),
                "doc_fraction": Decimal("0.15"),
                "fossil_carbon_fraction": Decimal("0.0"),
                "moisture_content": Decimal("0.70")
            },
            {
                "waste_category": "PLASTICS_MIXED",
                "mass_fraction": Decimal("0.25"),
                "mass_tonnes": Decimal("25.0"),
                "doc_fraction": Decimal("0.0"),  # Plastics don't degrade
                "fossil_carbon_fraction": Decimal("0.85"),
                "moisture_content": Decimal("0.02")
            }
        ]
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Sample waste generated configuration.

    Represents a production-ready configuration with defaults for:
    - GWP source: IPCC AR5 (100-year)
    - Landfill FOD model: Enabled
    - Double counting check: Enabled
    - Data quality validation: Enabled
    - Compliance frameworks: GHG Protocol + CSRD
    """
    return {
        "general": {
            "enabled": True,
            "debug": False,
            "log_level": "INFO",
            "agent_id": "GL-MRV-S3-005",
            "version": "1.0.0"
        },
        "calculation": {
            "default_gwp_source": "AR5",
            "gwp_timeframe": 100,  # 100-year GWP
            "enable_biogenic_accounting": True,
            "biogenic_separate": True,
            "default_method": "WASTE_TYPE_SPECIFIC"
        },
        "landfill": {
            "enable_fod_model": True,
            "projection_years": 50,
            "default_docf": Decimal("0.5"),
            "default_f": Decimal("0.5"),
            "default_oxidation": Decimal("0.10")
        },
        "incineration": {
            "default_combustion_efficiency": Decimal("0.995"),
            "default_energy_recovery": False,
            "separate_fossil_biogenic": True
        },
        "recycling": {
            "default_quality_factor": Decimal("1.0"),
            "default_substitution": True,
            "allocation_method": "CUT_OFF"
        },
        "wastewater": {
            "default_organic_basis": "COD",
            "default_mcf_aerobic": Decimal("0.00"),
            "default_mcf_anaerobic": Decimal("0.80")
        },
        "compliance": {
            "enabled_frameworks": ["GHG_PROTOCOL", "CSRD"],
            "double_counting_check": True,
            "boundary_enforcement": True
        },
        "data_quality": {
            "min_score_threshold": Decimal("3.0"),
            "validation_enabled": True
        }
    }


@pytest.fixture
def sample_compliance_input() -> Dict[str, Any]:
    """
    Sample compliance check input - GHG Protocol framework.

    Validates waste generated emissions calculation against:
    - GHG Protocol Scope 3 Category 5 guidance
    - Double counting exclusion (no owned facilities)
    - Biogenic/fossil separation requirements
    - Data quality requirements
    """
    return {
        "framework": "GHG_PROTOCOL",
        "calculation_id": "CALC-2026-001",
        "tenant_id": "tenant-001",
        "check_double_counting": True,
        "check_boundary": True,
        "check_data_quality": True,
        "min_dqi_score": Decimal("3.0"),
        "description": "GHG Protocol Scope 3 compliance check"
    }


# ============================================================================
# BATCH INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch_inputs() -> List[Dict[str, Any]]:
    """
    Sample batch of 5 diverse waste stream inputs.

    Represents a typical organization's waste profile:
    1. Food waste to landfill (100 tonnes)
    2. Plastics to incineration with energy recovery (50 tonnes)
    3. Paper/cardboard to recycling (75 tonnes)
    4. Garden waste to composting (30 tonnes)
    5. Industrial wastewater treatment (500 kg COD)
    """
    return [
        {
            "stream_id": "WS-2026-001",
            "facility_id": "FAC-001",
            "tenant_id": "tenant-001",
            "waste_category": "FOOD_WASTE",
            "treatment_method": "LANDFILL_WITH_GAS_CAPTURE",
            "mass_tonnes": Decimal("100.0"),
            "reporting_year": 2026
        },
        {
            "stream_id": "WS-2026-002",
            "facility_id": "FAC-001",
            "tenant_id": "tenant-001",
            "waste_category": "PLASTICS_MIXED",
            "treatment_method": "INCINERATION_WITH_ENERGY_RECOVERY",
            "mass_tonnes": Decimal("50.0"),
            "reporting_year": 2026
        },
        {
            "stream_id": "WS-2026-003",
            "facility_id": "FAC-001",
            "tenant_id": "tenant-001",
            "waste_category": "PAPER_CARDBOARD",
            "treatment_method": "RECYCLING_OPEN_LOOP",
            "mass_tonnes": Decimal("75.0"),
            "reporting_year": 2026
        },
        {
            "stream_id": "WS-2026-004",
            "facility_id": "FAC-002",
            "tenant_id": "tenant-001",
            "waste_category": "GARDEN_WASTE",
            "treatment_method": "COMPOSTING",
            "mass_tonnes": Decimal("30.0"),
            "reporting_year": 2026
        },
        {
            "stream_id": "WS-2026-005",
            "facility_id": "FAC-002",
            "tenant_id": "tenant-001",
            "waste_category": "WASTEWATER",
            "treatment_method": "WASTEWATER_TREATMENT",
            "mass_tonnes": Decimal("0.5"),  # 500 kg COD
            "reporting_year": 2026
        }
    ]


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_database_engine() -> Mock:
    """
    Mock waste classification database engine.

    Provides emission factors, DOC values, MCF values, and decay rates
    for all waste categories and treatment methods.
    """
    mock = Mock()

    # Mock emission factor lookup
    mock.get_emission_factor = Mock(return_value={
        "ef_value": Decimal("0.45"),  # kg CO2e/kg waste
        "ef_source": "EPA_WARM",
        "waste_category": "FOOD_WASTE",
        "treatment_method": "LANDFILL",
        "uncertainty": Decimal("0.15")
    })

    # Mock DOC lookup
    mock.get_doc_value = Mock(return_value=Decimal("0.15"))  # 15% DOC for food

    # Mock MCF lookup
    mock.get_mcf_value = Mock(return_value=Decimal("1.0"))  # Anaerobic landfill

    # Mock decay rate
    mock.get_decay_rate = Mock(return_value=Decimal("0.09"))  # k for food, temperate wet

    return mock


@pytest.fixture
def mock_landfill_engine() -> Mock:
    """
    Mock landfill emissions calculation engine.

    Simulates IPCC First Order Decay (FOD) model calculations for:
    - CH4 generation from degradable organic carbon
    - Multi-year projections (50-year default)
    - Gas collection efficiency and oxidation
    """
    mock = Mock()

    # Mock FOD calculation
    mock.calculate_fod_emissions = Mock(return_value={
        "ch4_generated_kg": Decimal("2500.0"),
        "ch4_captured_kg": Decimal("2250.0"),  # 90% capture
        "ch4_oxidized_kg": Decimal("25.0"),  # 10% of fugitive
        "ch4_emitted_kg": Decimal("225.0"),
        "co2e_tonnes": Decimal("5.625"),  # CH4 GWP100 = 25
        "n2o_kg": Decimal("0.0"),  # Negligible N2O from landfill
        "projection_years": 50
    })

    return mock


@pytest.fixture
def mock_incineration_engine() -> Mock:
    """
    Mock incineration emissions calculation engine.

    Simulates combustion emissions calculations for:
    - Fossil CO2 from plastics/synthetic materials
    - Biogenic CO2 from organic materials (reported separately)
    - N2O emissions from nitrogen combustion
    - CH4 from incomplete combustion
    - Energy recovery credits
    """
    mock = Mock()

    # Mock incineration calculation
    mock.calculate_incineration = Mock(return_value={
        "fossil_co2_kg": Decimal("1200.0"),
        "biogenic_co2_kg": Decimal("300.0"),
        "n2o_kg": Decimal("0.5"),
        "ch4_kg": Decimal("0.1"),
        "total_co2e_tonnes": Decimal("1.35"),  # Fossil only
        "biogenic_co2e_tonnes": Decimal("0.30"),
        "energy_recovered_mwh": Decimal("0.8"),
        "avoided_emissions_co2e_tonnes": Decimal("0.4")
    })

    return mock


@pytest.fixture
def mock_recycling_engine() -> Mock:
    """
    Mock recycling and composting emissions engine.

    Simulates recycling/recovery process emissions and credits:
    - Process emissions from reprocessing
    - Avoided emissions from virgin production substitution
    - Quality degradation factors (downcycling)
    - Material recovery rates
    """
    mock = Mock()

    # Mock recycling calculation
    mock.calculate_recycling = Mock(return_value={
        "process_emissions_co2e_tonnes": Decimal("0.05"),
        "avoided_emissions_co2e_tonnes": Decimal("1.20"),
        "net_emissions_co2e_tonnes": Decimal("-1.15"),  # Net benefit
        "material_recovered_tonnes": Decimal("63.75"),  # 85% recovery
        "contamination_tonnes": Decimal("3.75")
    })

    # Mock composting calculation
    mock.calculate_composting = Mock(return_value={
        "ch4_kg": Decimal("120.0"),  # 4 kg/tonne * 30 tonnes
        "n2o_kg": Decimal("9.0"),  # 0.3 kg/tonne * 30 tonnes
        "co2e_tonnes": Decimal("5.67"),
        "compost_produced_tonnes": Decimal("9.0")  # 30% yield
    })

    return mock


@pytest.fixture
def mock_wastewater_engine() -> Mock:
    """
    Mock wastewater treatment emissions engine.

    Simulates wastewater emissions calculations:
    - CH4 from anaerobic decomposition (MCF-based)
    - N2O from nitrification/denitrification
    - COD/BOD conversion
    """
    mock = Mock()

    # Mock wastewater calculation
    mock.calculate_wastewater = Mock(return_value={
        "ch4_kg": Decimal("0.0"),  # Aerobic, MCF = 0
        "n2o_kg": Decimal("0.25"),  # From nitrogen
        "co2e_tonnes": Decimal("0.074"),
        "sludge_emissions_co2e_tonnes": Decimal("0.05")
    })

    return mock


@pytest.fixture
def mock_compliance_engine() -> Mock:
    """
    Mock compliance checker engine.

    Validates calculations against regulatory frameworks:
    - GHG Protocol Scope 3 Category 5
    - ISO 14064-1:2018
    - CSRD ESRS E5
    - Double counting checks
    - Boundary validation
    """
    mock = Mock()

    # Mock compliance check
    mock.check_compliance = Mock(return_value={
        "compliant": True,
        "framework": "GHG_PROTOCOL",
        "issues": [],
        "warnings": [],
        "timestamp": datetime.now()
    })

    return mock


@pytest.fixture
def mock_pipeline_engine() -> Mock:
    """
    Mock waste generated pipeline orchestrator.

    Coordinates all calculation engines and manages workflow:
    - Classification and routing
    - Multi-pathway calculations
    - Aggregation
    - Compliance checks
    """
    mock = Mock()

    # Mock pipeline execution
    mock.execute = Mock(return_value={
        "calculation_id": "CALC-2026-001",
        "total_emissions_co2e_tonnes": Decimal("10.5"),
        "biogenic_co2e_tonnes": Decimal("0.3"),
        "avoided_emissions_co2e_tonnes": Decimal("1.2"),
        "net_emissions_co2e_tonnes": Decimal("9.6"),
        "pathway_breakdown": {
            "landfill": Decimal("5.625"),
            "incineration": Decimal("1.35"),
            "recycling": Decimal("-1.15"),
            "composting": Decimal("5.67"),
            "wastewater": Decimal("0.074")
        },
        "compliant": True,
        "data_quality_score": Decimal("4.2")
    })

    return mock


@pytest.fixture
def mock_service() -> Mock:
    """Mock waste generated service facade."""
    mock = Mock()
    mock.calculate_waste_emissions = AsyncMock(return_value={
        "calculation_id": "CALC-2026-001",
        "emissions_tco2e": Decimal("10.5"),
        "status": "success"
    })
    return mock


# ============================================================================
# METRICS AND PROVENANCE FIXTURES
# ============================================================================

@pytest.fixture
def mock_metrics() -> Mock:
    """Mock Prometheus metrics recorder."""
    mock = Mock()
    mock.record_calculation = Mock()
    mock.record_emissions = Mock()
    mock.record_waste_mass = Mock()
    mock.record_error = Mock()
    return mock


@pytest.fixture
def mock_provenance() -> Mock:
    """Mock provenance tracker."""
    mock = Mock()
    mock.generate_hash = Mock(return_value="a" * 64)  # SHA-256 hash
    mock.record_calculation = Mock()
    return mock


# ============================================================================
# EMISSION FACTOR FIXTURES
# ============================================================================

@pytest.fixture
def sample_emission_factor() -> Dict[str, Any]:
    """
    Sample emission factor record.

    EPA WARM v16 emission factor for:
    - Material: Food waste
    - Disposal: Landfill
    - Value: 0.45 kg CO2e/kg waste
    - Uncertainty: ±15%
    """
    return {
        "ef_id": "EF-WARM-FOOD-LANDFILL",
        "waste_category": "FOOD_WASTE",
        "treatment_method": "LANDFILL",
        "ef_value_kg_co2e_per_kg": Decimal("0.45"),
        "ef_source": "EPA_WARM",
        "ef_version": "v16",
        "uncertainty_percent": Decimal("15.0"),
        "valid_from": date(2024, 1, 1),
        "valid_to": None,
        "description": "EPA WARM food waste landfill emission factor"
    }


@pytest.fixture
def sample_classification_result() -> Dict[str, Any]:
    """
    Sample waste classification result.

    Classifies waste according to:
    - European Waste Catalogue (EWC) code
    - Basel Convention hazard class
    - Treatment recommendation
    - Regulatory requirements
    """
    return {
        "stream_id": "WS-2026-001",
        "waste_category": "FOOD_WASTE",
        "waste_stream": "COMMERCIAL_INDUSTRIAL",
        "ewc_code": "20 01 08",  # Biodegradable kitchen waste
        "basel_hazard": None,  # Non-hazardous
        "recommended_treatment": "ANAEROBIC_DIGESTION",
        "regulatory_requirements": ["CSRD_ESRS_E5"],
        "hazardous": False
    }
