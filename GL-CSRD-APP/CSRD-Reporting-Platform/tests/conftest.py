"""
GL-CSRD Shared Test Fixtures (conftest.py)

Provides shared fixtures for all 975 tests across 14 test files.
Includes fixtures for 12 ESRS standards and common test utilities.

This file is automatically loaded by pytest for all test files.

Author: GreenLang CSRD Team
Version: 1.0.0
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml


# ============================================================================
# PATH FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def base_path() -> Path:
    """Get base path for project root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_path(base_path: Path) -> Path:
    """Get data directory path."""
    return base_path / "data"


@pytest.fixture(scope="session")
def tests_path(base_path: Path) -> Path:
    """Get tests directory path."""
    return base_path / "tests"


@pytest.fixture(scope="session")
def examples_path(base_path: Path) -> Path:
    """Get examples directory path."""
    return base_path / "examples"


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# ESRS DATA FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def esrs_formulas_path(data_path: Path) -> Path:
    """Path to ESRS formulas YAML."""
    return data_path / "esrs_formulas.yaml"


@pytest.fixture(scope="session")
def emission_factors_path(data_path: Path) -> Path:
    """Path to GHG emission factors JSON."""
    return data_path / "emission_factors.json"


@pytest.fixture(scope="session")
def esrs_data_points_path(data_path: Path) -> Path:
    """Path to ESRS data points catalog (1,082 data points)."""
    return data_path / "esrs_data_points.json"


@pytest.fixture(scope="session")
def esrs_formulas(esrs_formulas_path: Path) -> Dict[str, Any]:
    """Load ESRS formulas database (520+ formulas)."""
    if not esrs_formulas_path.exists():
        # Return mock formulas if file doesn't exist
        return {
            "formulas": {
                "total_ghg_emissions": {
                    "formula": "scope1 + scope2 + scope3",
                    "unit": "tonnes_co2e",
                    "esrs_standard": "E1"
                }
            }
        }

    with open(esrs_formulas_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def emission_factors(emission_factors_path: Path) -> Dict[str, Any]:
    """Load GHG emission factors database."""
    if not emission_factors_path.exists():
        # Return mock emission factors
        return {
            "electricity": {
                "grid": {"factor": 0.5, "unit": "kg_co2e_per_kwh"}
            },
            "natural_gas": {
                "combustion": {"factor": 2.0, "unit": "kg_co2e_per_m3"}
            }
        }

    with open(emission_factors_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove metadata if present
    if isinstance(data, dict) and "metadata" in data:
        del data["metadata"]

    return data


@pytest.fixture(scope="session")
def esrs_data_points(esrs_data_points_path: Path) -> Dict[str, Any]:
    """Load ESRS data points catalog (1,082 data points)."""
    if not esrs_data_points_path.exists():
        # Return mock data points
        return {
            "data_points": [
                {
                    "code": "E1-1",
                    "name": "Scope 1 GHG emissions",
                    "standard": "ESRS E1",
                    "unit": "tonnes_co2e"
                }
            ]
        }

    with open(esrs_data_points_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# ESRS STANDARD FIXTURES (12 Standards)
# ============================================================================


@pytest.fixture
def esrs1_data() -> Dict[str, Any]:
    """ESRS 1 - General Requirements test data."""
    return {
        "standard": "ESRS 1",
        "name": "General Requirements",
        "reporting_entity": "Test Company Ltd",
        "reporting_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        },
        "materiality_assessment": {
            "double_materiality": True,
            "impact_materiality": True,
            "financial_materiality": True
        }
    }


@pytest.fixture
def esrs2_data() -> Dict[str, Any]:
    """ESRS 2 - General Disclosures test data."""
    return {
        "standard": "ESRS 2",
        "name": "General Disclosures",
        "governance": {
            "board_oversight": True,
            "management_responsibility": True
        },
        "strategy": {
            "sustainability_strategy": "Net Zero by 2050",
            "business_model": "Manufacturing"
        },
        "impacts_risks_opportunities": {
            "identified": True,
            "assessed": True,
            "managed": True
        }
    }


@pytest.fixture
def esrs_e1_data() -> Dict[str, Any]:
    """ESRS E1 - Climate Change test data."""
    return {
        "standard": "ESRS E1",
        "name": "Climate Change",
        "ghg_emissions": {
            "scope1": 10000.0,  # tonnes CO2e
            "scope2": 5000.0,
            "scope3": 25000.0,
            "total": 40000.0
        },
        "climate_targets": {
            "net_zero_target": "2050",
            "interim_targets": ["2030", "2040"]
        },
        "transition_plan": {
            "exists": True,
            "aligned_with_1_5C": True
        }
    }


@pytest.fixture
def esrs_e2_data() -> Dict[str, Any]:
    """ESRS E2 - Pollution test data."""
    return {
        "standard": "ESRS E2",
        "name": "Pollution",
        "air_pollution": {
            "pollutants": ["NOx", "SOx", "PM"],
            "total_emissions": 100.0  # tonnes
        },
        "water_pollution": {
            "pollutants": ["COD", "BOD"],
            "total_discharge": 50.0  # tonnes
        },
        "soil_contamination": {
            "sites_affected": 0
        }
    }


@pytest.fixture
def esrs_e3_data() -> Dict[str, Any]:
    """ESRS E3 - Water and Marine Resources test data."""
    return {
        "standard": "ESRS E3",
        "name": "Water and Marine Resources",
        "water_consumption": {
            "total": 100000.0,  # m³
            "by_source": {
                "municipal": 60000.0,
                "groundwater": 40000.0
            }
        },
        "water_discharge": {
            "total": 80000.0,  # m³
            "treatment_level": "tertiary"
        },
        "marine_impacts": {
            "operations_in_marine_areas": False
        }
    }


@pytest.fixture
def esrs_e4_data() -> Dict[str, Any]:
    """ESRS E4 - Biodiversity and Ecosystems test data."""
    return {
        "standard": "ESRS E4",
        "name": "Biodiversity and Ecosystems",
        "biodiversity_impacts": {
            "operations_in_protected_areas": False,
            "biodiversity_sensitive_areas": 0
        },
        "ecosystem_services": {
            "dependence": "low",
            "impact": "low"
        },
        "deforestation": {
            "commodities_linked_to_deforestation": []
        }
    }


@pytest.fixture
def esrs_e5_data() -> Dict[str, Any]:
    """ESRS E5 - Resource Use and Circular Economy test data."""
    return {
        "standard": "ESRS E5",
        "name": "Resource Use and Circular Economy",
        "resource_inflows": {
            "materials": 50000.0,  # tonnes
            "renewable_percentage": 30.0
        },
        "resource_outflows": {
            "products": 45000.0,  # tonnes
            "waste": 5000.0
        },
        "circular_economy": {
            "recycled_content": 25.0,  # %
            "recyclability": 80.0  # %
        }
    }


@pytest.fixture
def esrs_s1_data() -> Dict[str, Any]:
    """ESRS S1 - Own Workforce test data."""
    return {
        "standard": "ESRS S1",
        "name": "Own Workforce",
        "workforce_composition": {
            "total_employees": 1000,
            "permanent": 900,
            "temporary": 100,
            "full_time": 950,
            "part_time": 50
        },
        "working_conditions": {
            "health_safety_incidents": 5,
            "training_hours": 20000.0
        },
        "equal_treatment": {
            "gender_pay_gap": 5.0,  # %
            "diversity_metrics": {"women_in_management": 35.0}
        }
    }


@pytest.fixture
def esrs_s2_data() -> Dict[str, Any]:
    """ESRS S2 - Workers in Value Chain test data."""
    return {
        "standard": "ESRS S2",
        "name": "Workers in Value Chain",
        "supply_chain": {
            "suppliers": 150,
            "suppliers_audited": 100
        },
        "working_conditions": {
            "forced_labor_risk": "low",
            "child_labor_risk": "low"
        },
        "labor_rights": {
            "freedom_of_association": True,
            "collective_bargaining": True
        }
    }


@pytest.fixture
def esrs_s3_data() -> Dict[str, Any]:
    """ESRS S3 - Affected Communities test data."""
    return {
        "standard": "ESRS S3",
        "name": "Affected Communities",
        "community_engagement": {
            "stakeholder_consultations": 12,  # per year
            "grievance_mechanisms": True
        },
        "impacts": {
            "positive_impacts": ["employment", "local_procurement"],
            "negative_impacts": []
        },
        "land_rights": {
            "indigenous_peoples_affected": False
        }
    }


@pytest.fixture
def esrs_s4_data() -> Dict[str, Any]:
    """ESRS S4 - Consumers and End-users test data."""
    return {
        "standard": "ESRS S4",
        "name": "Consumers and End-users",
        "product_safety": {
            "recalls": 0,
            "safety_incidents": 0
        },
        "consumer_rights": {
            "data_privacy_compliant": True,
            "complaints": 50
        },
        "accessibility": {
            "products_accessible": True,
            "information_accessible": True
        }
    }


@pytest.fixture
def esrs_g1_data() -> Dict[str, Any]:
    """ESRS G1 - Business Conduct test data."""
    return {
        "standard": "ESRS G1",
        "name": "Business Conduct",
        "corporate_culture": {
            "code_of_conduct": True,
            "whistleblower_protection": True
        },
        "anti_corruption": {
            "policy_exists": True,
            "training_provided": True,
            "incidents": 0
        },
        "political_influence": {
            "lobbying_activities": True,
            "political_contributions": 0.0
        }
    }


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_esg_data() -> pd.DataFrame:
    """Sample ESG data for testing (100 rows)."""
    return pd.DataFrame({
        "metric_name": [f"metric_{i}" for i in range(100)],
        "value": [float(i * 10) for i in range(100)],
        "unit": ["tonnes_co2e"] * 50 + ["m3"] * 30 + ["kwh"] * 20,
        "reporting_period": ["2024"] * 100,
        "data_quality": ["high"] * 80 + ["medium"] * 15 + ["low"] * 5
    })


@pytest.fixture
def sample_ghg_data() -> Dict[str, float]:
    """Sample GHG emissions data."""
    return {
        "scope1_emissions": 10000.0,
        "scope2_emissions": 5000.0,
        "scope3_emissions": 25000.0,
        "electricity_consumed": 1000000.0,  # kWh
        "natural_gas_consumed": 50000.0,  # m³
        "fleet_fuel_consumed": 100000.0  # liters
    }


@pytest.fixture
def sample_company_info() -> Dict[str, Any]:
    """Sample company information."""
    return {
        "company_name": "Test Company Ltd",
        "company_id": "TEST123",
        "lei": "TEST1234567890123456",
        "reporting_period": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        "industry": "Manufacturing",
        "employees": 1000,
        "revenue": 100000000.0  # EUR
    }


# ============================================================================
# FRAMEWORK INTEGRATION FIXTURES
# ============================================================================


@pytest.fixture
def tcfd_metrics() -> List[Dict[str, Any]]:
    """Sample TCFD metrics for testing framework integration."""
    return [
        {
            "framework": "TCFD",
            "pillar": "Governance",
            "metric": "Board oversight of climate risks",
            "value": True,
            "esrs_mapping": ["ESRS 2", "ESRS E1"]
        },
        {
            "framework": "TCFD",
            "pillar": "Metrics",
            "metric": "Scope 1 GHG emissions",
            "value": 10000.0,
            "esrs_mapping": ["ESRS E1"]
        }
    ]


@pytest.fixture
def gri_metrics() -> List[Dict[str, Any]]:
    """Sample GRI metrics for testing framework integration."""
    return [
        {
            "framework": "GRI",
            "standard": "GRI 305",
            "metric": "Direct GHG emissions (Scope 1)",
            "value": 10000.0,
            "esrs_mapping": ["ESRS E1"]
        }
    ]


@pytest.fixture
def sasb_metrics() -> List[Dict[str, Any]]:
    """Sample SASB metrics for testing framework integration."""
    return [
        {
            "framework": "SASB",
            "industry": "Manufacturing",
            "metric": "Energy Management",
            "value": 1000000.0,
            "esrs_mapping": ["ESRS E1"]
        }
    ]


# ============================================================================
# MOCK AGENT FIXTURES
# ============================================================================


@pytest.fixture
def mock_calculator_agent():
    """Mock CalculatorAgent for testing."""
    agent = Mock()
    agent.calculate.return_value = {"result": 100.0, "unit": "tonnes_co2e"}
    agent.validate.return_value = True
    return agent


@pytest.fixture
def mock_intake_agent():
    """Mock IntakeAgent for testing."""
    agent = Mock()
    agent.ingest.return_value = pd.DataFrame({"metric": ["test"], "value": [100]})
    agent.validate_schema.return_value = True
    return agent


@pytest.fixture
def mock_reporting_agent():
    """Mock ReportingAgent for testing."""
    agent = Mock()
    agent.generate_report.return_value = {"report_id": "test123", "status": "success"}
    agent.generate_xbrl.return_value = b"<xbrl>test</xbrl>"
    return agent


# ============================================================================
# TIMESTAMP FIXTURES
# ============================================================================


@pytest.fixture
def current_timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


@pytest.fixture
def reporting_year() -> int:
    """Current reporting year."""
    return datetime.now().year


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers (already in pytest.ini, but can add programmatically)
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "critical: marks tests as critical")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark critical tests
        if "calculator" in item.nodeid.lower() or "zero_hallucination" in item.nodeid.lower():
            item.add_marker(pytest.mark.critical)

        # Auto-mark slow tests
        if "e2e" in item.nodeid.lower() or "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)


# ============================================================================
# HELPER FIXTURES
# ============================================================================


@pytest.fixture
def assert_valid_esrs_code():
    """Helper to validate ESRS code format."""
    def _assert(code: str) -> bool:
        """Validate ESRS code format (e.g., E1-1, S1-9, G1-1)."""
        import re
        pattern = r"^(E[1-5]|S[1-4]|G1|ESRS[12])-[0-9]+$"
        return bool(re.match(pattern, code))

    return _assert


@pytest.fixture
def assert_valid_ghg_data():
    """Helper to validate GHG emissions data."""
    def _assert(data: Dict[str, float]) -> bool:
        """Validate GHG data has required fields and valid values."""
        required = ["scope1_emissions", "scope2_emissions", "scope3_emissions"]
        if not all(k in data for k in required):
            return False
        return all(isinstance(v, (int, float)) and v >= 0 for v in data.values())

    return _assert


# ============================================================================
# SESSION FIXTURES
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def test_session_info(request):
    """Print test session information."""
    print("\n" + "=" * 70)
    print("GL-CSRD Test Suite")
    print("=" * 70)
    print(f"Total Tests: 975")
    print(f"Test Files: 14")
    print(f"ESRS Standards: 12")
    print(f"Session Start: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    def finalizer():
        print("\n" + "=" * 70)
        print(f"Session End: {datetime.now().isoformat()}")
        print("=" * 70 + "\n")

    request.addfinalizer(finalizer)
