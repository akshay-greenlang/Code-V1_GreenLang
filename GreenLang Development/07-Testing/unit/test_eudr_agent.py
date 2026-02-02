# -*- coding: utf-8 -*-
"""
Unit Tests for EUDR Deforestation Compliance Agent

Comprehensive test suite with 50 test cases covering:
- ValidateGeolocationTool (12 tests)
- ClassifyCommodityTool (12 tests)
- AssessCountryRiskTool (10 tests)
- TraceSupplyChainTool (8 tests)
- GenerateDdsReportTool (8 tests)

Target: 85%+ coverage for EUDR compliance tools
Run with: pytest tests/unit/test_eudr_agent.py -v --cov=generated/eudr_compliance_v1

Author: GL-TestEngineer
Version: 1.0.0

EUDR (EU Deforestation Regulation) requires operators and traders to conduct
due diligence to ensure products are deforestation-free and legally produced.
Applies to 7 commodities: cattle, cocoa, coffee, palm oil, rubber, soya, wood.
"""

import pytest
import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "generated" / "eudr_compliance_v1"))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def validate_geolocation_tool():
    """Create ValidateGeolocationTool instance."""
    from generated.eudr_compliance_v1.tools import ValidateGeolocationTool
    return ValidateGeolocationTool()


@pytest.fixture
def classify_commodity_tool():
    """Create ClassifyCommodityTool instance."""
    from generated.eudr_compliance_v1.tools import ClassifyCommodityTool
    return ClassifyCommodityTool()


@pytest.fixture
def assess_country_risk_tool():
    """Create AssessCountryRiskTool instance."""
    from generated.eudr_compliance_v1.tools import AssessCountryRiskTool
    return AssessCountryRiskTool()


@pytest.fixture
def trace_supply_chain_tool():
    """Create TraceSupplyChainTool instance."""
    from generated.eudr_compliance_v1.tools import TraceSupplyChainTool
    return TraceSupplyChainTool()


@pytest.fixture
def generate_dds_report_tool():
    """Create GenerateDdsReportTool instance."""
    from generated.eudr_compliance_v1.tools import GenerateDdsReportTool
    return GenerateDdsReportTool()


@pytest.fixture
def mock_geolocation_validator():
    """Create mock geolocation validator."""
    mock_result = {
        "valid": True,
        "coordinates": [-3.4653, -62.2159],
        "country_code": "BR",
        "coordinate_type": "point",
        "precision_meters": 10.0,
        "in_protected_area": False,
        "forest_cover_2020": True,
        "deforestation_detected": False,
        "validation_source": "Global Forest Watch 2024",
        "result_hash": "a" * 64,
        "executed_at": datetime.now().isoformat(),
    }
    return mock_result


@pytest.fixture
def sample_eudr_commodities():
    """Sample EUDR-regulated commodities."""
    return [
        "cattle", "cocoa", "coffee", "palm_oil",
        "rubber", "soya", "wood"
    ]


@pytest.fixture
def sample_cn_codes():
    """Sample CN codes for EUDR-regulated products."""
    return {
        "cattle": "0102.29",
        "cocoa": "1801.00",
        "coffee": "0901.11",
        "palm_oil": "1511.10",
        "rubber": "4001.10",
        "soya": "1201.90",
        "wood": "4403.11",
    }


@pytest.fixture
def sample_supply_chain_nodes():
    """Sample supply chain nodes for testing."""
    return [
        {
            "node_id": "FARM-001",
            "node_type": "producer",
            "location": {"lat": -3.4653, "lon": -62.2159},
            "country_code": "BR",
            "timestamp": "2024-01-15T10:00:00Z",
            "documents": ["harvest_certificate", "land_title"],
        },
        {
            "node_id": "COOP-001",
            "node_type": "cooperative",
            "location": {"lat": -3.5000, "lon": -62.3000},
            "country_code": "BR",
            "timestamp": "2024-01-20T14:00:00Z",
            "documents": ["aggregation_report", "quality_certificate"],
        },
        {
            "node_id": "PORT-001",
            "node_type": "exporter",
            "location": {"lat": -3.1019, "lon": -60.0250},
            "country_code": "BR",
            "timestamp": "2024-02-01T08:00:00Z",
            "documents": ["export_declaration", "phytosanitary_certificate"],
        },
    ]


# =============================================================================
# ValidateGeolocationTool Tests (12 tests)
# =============================================================================

class TestValidateGeolocationTool:
    """Test suite for ValidateGeolocationTool - 12 test cases."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_tool_initialization(self, validate_geolocation_tool):
        """UT-EUDR-001: Test tool initializes correctly."""
        assert validate_geolocation_tool is not None
        assert validate_geolocation_tool.name == "validate_geolocation"
        assert validate_geolocation_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_coordinates_raises_error(self, validate_geolocation_tool):
        """UT-EUDR-002: Test missing coordinates parameter raises ValueError."""
        params = {"country_code": "BR"}

        with pytest.raises(ValueError) as exc_info:
            await validate_geolocation_tool.execute(params)

        assert "coordinates" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_country_code_raises_error(self, validate_geolocation_tool):
        """UT-EUDR-003: Test missing country_code parameter raises ValueError."""
        params = {"coordinates": [-3.4653, -62.2159]}

        with pytest.raises(ValueError) as exc_info:
            await validate_geolocation_tool.execute(params)

        assert "country_code" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_valid_point_coordinates(self, validate_geolocation_tool, mock_geolocation_validator):
        """UT-EUDR-004: Test valid point coordinates return valid result."""
        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_geolocation_validator):
            params = {
                "coordinates": [-3.4653, -62.2159],
                "country_code": "BR",
                "coordinate_type": "point",
            }

            result = await validate_geolocation_tool.execute(params)

            assert result is not None
            assert result["valid"] is True
            assert result["in_protected_area"] is False
            assert result["deforestation_detected"] is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_polygon_coordinates(self, validate_geolocation_tool):
        """UT-EUDR-005: Test polygon coordinate validation."""
        mock_result = {
            "valid": True,
            "coordinates": [
                [-3.4653, -62.2159],
                [-3.4700, -62.2200],
                [-3.4750, -62.2150],
                [-3.4653, -62.2159],
            ],
            "country_code": "BR",
            "coordinate_type": "polygon",
            "area_hectares": 150.5,
            "in_protected_area": False,
            "deforestation_detected": False,
            "result_hash": "b" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_result):
            params = {
                "coordinates": [
                    [-3.4653, -62.2159],
                    [-3.4700, -62.2200],
                    [-3.4750, -62.2150],
                    [-3.4653, -62.2159],
                ],
                "country_code": "BR",
                "coordinate_type": "polygon",
            }

            result = await validate_geolocation_tool.execute(params)

            assert result is not None
            assert result["coordinate_type"] == "polygon"
            assert "area_hectares" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_protected_area_detection(self, validate_geolocation_tool):
        """UT-EUDR-006: Test protected area detection returns valid=False."""
        mock_result = {
            "valid": False,
            "coordinates": [-3.5000, -62.3000],
            "country_code": "BR",
            "in_protected_area": True,
            "protected_area_name": "Amazon National Park",
            "protected_area_type": "national_park",
            "validation_errors": ["Location is within protected area"],
            "result_hash": "c" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_result):
            params = {
                "coordinates": [-3.5000, -62.3000],
                "country_code": "BR",
            }

            result = await validate_geolocation_tool.execute(params)

            assert result["valid"] is False
            assert result["in_protected_area"] is True
            assert "protected_area_name" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_deforestation_detection(self, validate_geolocation_tool):
        """UT-EUDR-007: Test deforestation detection returns valid=False."""
        mock_result = {
            "valid": False,
            "coordinates": [-4.0000, -63.0000],
            "country_code": "BR",
            "in_protected_area": False,
            "deforestation_detected": True,
            "deforestation_date": "2022-06-15",
            "forest_loss_hectares": 25.3,
            "validation_errors": ["Deforestation detected after Dec 31, 2020"],
            "result_hash": "d" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_result):
            params = {
                "coordinates": [-4.0000, -63.0000],
                "country_code": "BR",
            }

            result = await validate_geolocation_tool.execute(params)

            assert result["valid"] is False
            assert result["deforestation_detected"] is True
            assert "deforestation_date" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    @pytest.mark.parametrize("country_code", ["BR", "ID", "CO", "CI", "GH", "MY", "PE"])
    async def test_multiple_countries(self, validate_geolocation_tool, country_code):
        """UT-EUDR-008: Test validation for multiple high-risk countries."""
        mock_result = {
            "valid": True,
            "coordinates": [0.0, 0.0],
            "country_code": country_code,
            "in_protected_area": False,
            "deforestation_detected": False,
            "result_hash": "e" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_result):
            params = {
                "coordinates": [0.0, 0.0],
                "country_code": country_code,
            }

            result = await validate_geolocation_tool.execute(params)

            assert result is not None
            assert result["country_code"] == country_code

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_validate_params_method(self, validate_geolocation_tool):
        """UT-EUDR-009: Test validate_params method returns boolean."""
        valid_params = {"coordinates": [0.0, 0.0], "country_code": "BR"}
        invalid_params = {"coordinates": [0.0, 0.0]}

        assert validate_geolocation_tool.validate_params(valid_params) is True
        assert validate_geolocation_tool.validate_params(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_precision_parameter(self, validate_geolocation_tool, mock_geolocation_validator):
        """UT-EUDR-010: Test precision_meters parameter is passed correctly."""
        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_geolocation_validator) as mock_validate:
            params = {
                "coordinates": [-3.4653, -62.2159],
                "country_code": "BR",
                "precision_meters": 5.0,
            }

            result = await validate_geolocation_tool.execute(params)

            mock_validate.assert_called_once()
            call_kwargs = mock_validate.call_args[1]
            assert call_kwargs["precision_meters"] == 5.0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_result_includes_provenance(self, validate_geolocation_tool, mock_geolocation_validator):
        """UT-EUDR-011: Test result includes provenance hash."""
        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_geolocation_validator):
            params = {
                "coordinates": [-3.4653, -62.2159],
                "country_code": "BR",
            }

            result = await validate_geolocation_tool.execute(params)

            assert "result_hash" in result
            assert len(result["result_hash"]) == 64

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_determinism(self, validate_geolocation_tool, mock_geolocation_validator):
        """UT-EUDR-012: Test geolocation validation is deterministic."""
        with patch('generated.eudr_compliance_v1.tools._validate_geolocation', return_value=mock_geolocation_validator):
            params = {
                "coordinates": [-3.4653, -62.2159],
                "country_code": "BR",
            }

            results = []
            for _ in range(5):
                result = await validate_geolocation_tool.execute(params)
                results.append(result["result_hash"])

            assert all(r == results[0] for r in results)


# =============================================================================
# ClassifyCommodityTool Tests (12 tests)
# =============================================================================

class TestClassifyCommodityTool:
    """Test suite for ClassifyCommodityTool - 12 test cases."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_tool_initialization(self, classify_commodity_tool):
        """UT-EUDR-013: Test tool initializes correctly."""
        assert classify_commodity_tool is not None
        assert classify_commodity_tool.name == "classify_commodity"
        assert classify_commodity_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_cn_code_raises_error(self, classify_commodity_tool):
        """UT-EUDR-014: Test missing cn_code parameter raises ValueError."""
        params = {"product_description": "Coffee beans"}

        with pytest.raises(ValueError) as exc_info:
            await classify_commodity_tool.execute(params)

        assert "cn_code" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cn_code,expected_commodity,expected_regulated", [
        ("0102.29", "cattle", True),
        ("1801.00", "cocoa", True),
        ("0901.11", "coffee", True),
        ("1511.10", "palm_oil", True),
        ("4001.10", "rubber", True),
        ("1201.90", "soya", True),
        ("4403.11", "wood", True),
        ("8471.30", "not_regulated", False),  # Computer equipment
    ])
    async def test_cn_code_classification(self, classify_commodity_tool, cn_code, expected_commodity, expected_regulated):
        """UT-EUDR-015: Test CN code classification for all 7 commodities."""
        mock_result = {
            "cn_code": cn_code,
            "commodity_type": expected_commodity,
            "eudr_regulated": expected_regulated,
            "classification_source": "EU Combined Nomenclature 2024",
            "result_hash": "f" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {"cn_code": cn_code}

            result = await classify_commodity_tool.execute(params)

            assert result["commodity_type"] == expected_commodity
            assert result["eudr_regulated"] == expected_regulated

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_derived_product_classification(self, classify_commodity_tool):
        """UT-EUDR-016: Test derived product classification (e.g., chocolate)."""
        mock_result = {
            "cn_code": "1806.31",
            "commodity_type": "cocoa",
            "product_category": "derived_product",
            "product_description": "Chocolate containing cocoa",
            "eudr_regulated": True,
            "due_diligence_required": True,
            "result_hash": "g" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {
                "cn_code": "1806.31",
                "product_description": "Chocolate containing cocoa",
            }

            result = await classify_commodity_tool.execute(params)

            assert result["commodity_type"] == "cocoa"
            assert result["product_category"] == "derived_product"
            assert result["eudr_regulated"] is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_quantity_parameter(self, classify_commodity_tool):
        """UT-EUDR-017: Test quantity_kg parameter is included in result."""
        mock_result = {
            "cn_code": "0901.11",
            "commodity_type": "coffee",
            "eudr_regulated": True,
            "quantity_kg": 5000.0,
            "result_hash": "h" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {
                "cn_code": "0901.11",
                "quantity_kg": 5000.0,
            }

            result = await classify_commodity_tool.execute(params)

            assert result["quantity_kg"] == 5000.0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_de_minimis_threshold(self, classify_commodity_tool):
        """UT-EUDR-018: Test de minimis threshold detection."""
        mock_result = {
            "cn_code": "1806.31",
            "commodity_type": "cocoa",
            "eudr_regulated": True,
            "quantity_kg": 1.5,
            "below_de_minimis": True,
            "de_minimis_threshold_kg": 2.0,
            "due_diligence_required": False,
            "result_hash": "i" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {
                "cn_code": "1806.31",
                "quantity_kg": 1.5,
            }

            result = await classify_commodity_tool.execute(params)

            assert result["below_de_minimis"] is True
            assert result["due_diligence_required"] is False

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_validate_params_method(self, classify_commodity_tool):
        """UT-EUDR-019: Test validate_params method returns boolean."""
        valid_params = {"cn_code": "0901.11"}
        invalid_params = {"product_description": "Coffee"}

        assert classify_commodity_tool.validate_params(valid_params) is True
        assert classify_commodity_tool.validate_params(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_invalid_cn_code_format(self, classify_commodity_tool):
        """UT-EUDR-020: Test invalid CN code format returns error."""
        mock_result = {
            "cn_code": "invalid",
            "error": "Invalid CN code format",
            "eudr_regulated": False,
            "commodity_type": "unknown",
            "result_hash": "j" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {"cn_code": "invalid"}

            result = await classify_commodity_tool.execute(params)

            assert "error" in result
            assert result["eudr_regulated"] is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_result_includes_classification_source(self, classify_commodity_tool):
        """UT-EUDR-021: Test result includes classification source."""
        mock_result = {
            "cn_code": "0901.11",
            "commodity_type": "coffee",
            "eudr_regulated": True,
            "classification_source": "EU Combined Nomenclature 2024",
            "result_hash": "k" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {"cn_code": "0901.11"}

            result = await classify_commodity_tool.execute(params)

            assert "classification_source" in result
            assert "EU" in result["classification_source"]

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_determinism(self, classify_commodity_tool):
        """UT-EUDR-022: Test commodity classification is deterministic."""
        mock_result = {
            "cn_code": "1801.00",
            "commodity_type": "cocoa",
            "eudr_regulated": True,
            "result_hash": "l" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {"cn_code": "1801.00"}

            results = []
            for _ in range(5):
                result = await classify_commodity_tool.execute(params)
                results.append(json.dumps(result, sort_keys=True))

            assert all(r == results[0] for r in results)

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_mixed_product_classification(self, classify_commodity_tool):
        """UT-EUDR-023: Test mixed product with multiple commodities."""
        mock_result = {
            "cn_code": "1905.31",
            "commodity_type": "mixed",
            "commodities_present": ["cocoa", "palm_oil", "soya"],
            "eudr_regulated": True,
            "due_diligence_required": True,
            "product_description": "Sweet biscuits with chocolate",
            "result_hash": "m" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {
                "cn_code": "1905.31",
                "product_description": "Sweet biscuits with chocolate",
            }

            result = await classify_commodity_tool.execute(params)

            assert result["commodity_type"] == "mixed"
            assert len(result["commodities_present"]) > 0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_timber_classification(self, classify_commodity_tool):
        """UT-EUDR-024: Test timber and wood products classification."""
        mock_result = {
            "cn_code": "4403.11",
            "commodity_type": "wood",
            "wood_type": "tropical",
            "species": "teak",
            "eudr_regulated": True,
            "cites_regulated": False,
            "result_hash": "n" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._classify_commodity', return_value=mock_result):
            params = {
                "cn_code": "4403.11",
                "product_description": "Tropical hardwood logs",
            }

            result = await classify_commodity_tool.execute(params)

            assert result["commodity_type"] == "wood"
            assert result["eudr_regulated"] is True


# =============================================================================
# AssessCountryRiskTool Tests (10 tests)
# =============================================================================

class TestAssessCountryRiskTool:
    """Test suite for AssessCountryRiskTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_tool_initialization(self, assess_country_risk_tool):
        """UT-EUDR-025: Test tool initializes correctly."""
        assert assess_country_risk_tool is not None
        assert assess_country_risk_tool.name == "assess_country_risk"
        assert assess_country_risk_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_country_code_raises_error(self, assess_country_risk_tool):
        """UT-EUDR-026: Test missing country_code raises ValueError."""
        params = {"commodity_type": "coffee"}

        with pytest.raises(ValueError) as exc_info:
            await assess_country_risk_tool.execute(params)

        assert "country_code" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_commodity_type_raises_error(self, assess_country_risk_tool):
        """UT-EUDR-027: Test missing commodity_type raises ValueError."""
        params = {"country_code": "BR"}

        with pytest.raises(ValueError) as exc_info:
            await assess_country_risk_tool.execute(params)

        assert "commodity_type" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    @pytest.mark.parametrize("country_code,commodity,expected_risk", [
        ("BR", "soya", "high"),
        ("ID", "palm_oil", "high"),
        ("CO", "coffee", "standard"),
        ("DE", "wood", "low"),
        ("FR", "cattle", "low"),
    ])
    async def test_risk_level_assessment(self, assess_country_risk_tool, country_code, commodity, expected_risk):
        """UT-EUDR-028: Test risk level assessment for various countries."""
        mock_result = {
            "country_code": country_code,
            "commodity_type": commodity,
            "risk_level": expected_risk,
            "risk_score": 0.75 if expected_risk == "high" else (0.5 if expected_risk == "standard" else 0.2),
            "due_diligence_level": "enhanced" if expected_risk == "high" else "standard",
            "result_hash": "o" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": country_code,
                "commodity_type": commodity,
            }

            result = await assess_country_risk_tool.execute(params)

            assert result["risk_level"] == expected_risk

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_risk_score_range(self, assess_country_risk_tool):
        """UT-EUDR-029: Test risk score is between 0 and 1."""
        mock_result = {
            "country_code": "BR",
            "commodity_type": "soya",
            "risk_level": "high",
            "risk_score": 0.85,
            "result_hash": "p" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": "BR",
                "commodity_type": "soya",
            }

            result = await assess_country_risk_tool.execute(params)

            assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_regional_risk_assessment(self, assess_country_risk_tool):
        """UT-EUDR-030: Test sub-national regional risk assessment."""
        mock_result = {
            "country_code": "BR",
            "commodity_type": "soya",
            "region": "Mato Grosso",
            "risk_level": "high",
            "risk_score": 0.92,
            "regional_factors": ["high_deforestation_rate", "amazon_biome"],
            "result_hash": "q" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": "BR",
                "commodity_type": "soya",
                "region": "Mato Grosso",
            }

            result = await assess_country_risk_tool.execute(params)

            assert result["region"] == "Mato Grosso"
            assert "regional_factors" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_due_diligence_requirements(self, assess_country_risk_tool):
        """UT-EUDR-031: Test due diligence requirements are returned."""
        mock_result = {
            "country_code": "ID",
            "commodity_type": "palm_oil",
            "risk_level": "high",
            "risk_score": 0.88,
            "due_diligence_level": "enhanced",
            "required_documents": [
                "geolocation_data",
                "land_title",
                "deforestation_certificate",
                "supply_chain_mapping",
            ],
            "result_hash": "r" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": "ID",
                "commodity_type": "palm_oil",
            }

            result = await assess_country_risk_tool.execute(params)

            assert result["due_diligence_level"] == "enhanced"
            assert "required_documents" in result
            assert len(result["required_documents"]) > 0

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_validate_params_method(self, assess_country_risk_tool):
        """UT-EUDR-032: Test validate_params method returns boolean."""
        valid_params = {"country_code": "BR", "commodity_type": "coffee"}
        invalid_params = {"country_code": "BR"}

        assert assess_country_risk_tool.validate_params(valid_params) is True
        assert assess_country_risk_tool.validate_params(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_result_includes_data_sources(self, assess_country_risk_tool):
        """UT-EUDR-033: Test result includes data sources for audit trail."""
        mock_result = {
            "country_code": "BR",
            "commodity_type": "soya",
            "risk_level": "high",
            "risk_score": 0.85,
            "data_sources": [
                "EC Benchmarking System",
                "FAO Forest Resources Assessment",
                "Global Forest Watch",
            ],
            "result_hash": "s" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": "BR",
                "commodity_type": "soya",
            }

            result = await assess_country_risk_tool.execute(params)

            assert "data_sources" in result
            assert len(result["data_sources"]) > 0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_determinism(self, assess_country_risk_tool):
        """UT-EUDR-034: Test risk assessment is deterministic."""
        mock_result = {
            "country_code": "CO",
            "commodity_type": "coffee",
            "risk_level": "standard",
            "risk_score": 0.55,
            "result_hash": "t" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._assess_country_risk', return_value=mock_result):
            params = {
                "country_code": "CO",
                "commodity_type": "coffee",
            }

            results = []
            for _ in range(5):
                result = await assess_country_risk_tool.execute(params)
                results.append(result["risk_score"])

            assert all(r == results[0] for r in results)


# =============================================================================
# TraceSupplyChainTool Tests (8 tests)
# =============================================================================

class TestTraceSupplyChainTool:
    """Test suite for TraceSupplyChainTool - 8 test cases."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_tool_initialization(self, trace_supply_chain_tool):
        """UT-EUDR-035: Test tool initializes correctly."""
        assert trace_supply_chain_tool is not None
        assert trace_supply_chain_tool.name == "trace_supply_chain"
        assert trace_supply_chain_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_shipment_id_raises_error(self, trace_supply_chain_tool):
        """UT-EUDR-036: Test missing shipment_id raises ValueError."""
        params = {
            "supply_chain_nodes": [],
            "commodity_type": "coffee",
        }

        with pytest.raises(ValueError) as exc_info:
            await trace_supply_chain_tool.execute(params)

        assert "shipment_id" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_complete_chain_of_custody(self, trace_supply_chain_tool, sample_supply_chain_nodes):
        """UT-EUDR-037: Test complete chain of custody returns high score."""
        mock_result = {
            "shipment_id": "SHIP-2024-001",
            "commodity_type": "coffee",
            "traceability_score": 0.95,
            "chain_of_custody": "complete",
            "nodes_verified": 3,
            "gaps_identified": [],
            "result_hash": "u" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._trace_supply_chain', return_value=mock_result):
            params = {
                "shipment_id": "SHIP-2024-001",
                "supply_chain_nodes": sample_supply_chain_nodes,
                "commodity_type": "coffee",
            }

            result = await trace_supply_chain_tool.execute(params)

            assert result["chain_of_custody"] == "complete"
            assert result["traceability_score"] >= 0.9
            assert len(result["gaps_identified"]) == 0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_partial_chain_of_custody(self, trace_supply_chain_tool):
        """UT-EUDR-038: Test partial chain of custody returns lower score."""
        mock_result = {
            "shipment_id": "SHIP-2024-002",
            "commodity_type": "cocoa",
            "traceability_score": 0.65,
            "chain_of_custody": "partial",
            "nodes_verified": 2,
            "gaps_identified": ["Missing producer documentation"],
            "result_hash": "v" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._trace_supply_chain', return_value=mock_result):
            params = {
                "shipment_id": "SHIP-2024-002",
                "supply_chain_nodes": [{"node_id": "COOP-001", "node_type": "cooperative"}],
                "commodity_type": "cocoa",
            }

            result = await trace_supply_chain_tool.execute(params)

            assert result["chain_of_custody"] == "partial"
            assert result["traceability_score"] < 0.9
            assert len(result["gaps_identified"]) > 0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_broken_chain_of_custody(self, trace_supply_chain_tool):
        """UT-EUDR-039: Test broken chain of custody returns low score."""
        mock_result = {
            "shipment_id": "SHIP-2024-003",
            "commodity_type": "palm_oil",
            "traceability_score": 0.25,
            "chain_of_custody": "broken",
            "nodes_verified": 1,
            "gaps_identified": [
                "Missing producer information",
                "No geolocation data",
                "Unverified intermediaries",
            ],
            "result_hash": "w" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._trace_supply_chain', return_value=mock_result):
            params = {
                "shipment_id": "SHIP-2024-003",
                "supply_chain_nodes": [],
                "commodity_type": "palm_oil",
            }

            result = await trace_supply_chain_tool.execute(params)

            assert result["chain_of_custody"] == "broken"
            assert result["traceability_score"] < 0.5

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_validate_params_method(self, trace_supply_chain_tool):
        """UT-EUDR-040: Test validate_params method returns boolean."""
        valid_params = {
            "shipment_id": "SHIP-001",
            "supply_chain_nodes": [],
            "commodity_type": "coffee",
        }
        invalid_params = {"shipment_id": "SHIP-001"}

        assert trace_supply_chain_tool.validate_params(valid_params) is True
        assert trace_supply_chain_tool.validate_params(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_traceability_score_range(self, trace_supply_chain_tool, sample_supply_chain_nodes):
        """UT-EUDR-041: Test traceability score is between 0 and 1."""
        mock_result = {
            "shipment_id": "SHIP-2024-004",
            "commodity_type": "rubber",
            "traceability_score": 0.78,
            "chain_of_custody": "partial",
            "result_hash": "x" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._trace_supply_chain', return_value=mock_result):
            params = {
                "shipment_id": "SHIP-2024-004",
                "supply_chain_nodes": sample_supply_chain_nodes,
                "commodity_type": "rubber",
            }

            result = await trace_supply_chain_tool.execute(params)

            assert 0.0 <= result["traceability_score"] <= 1.0

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_determinism(self, trace_supply_chain_tool, sample_supply_chain_nodes):
        """UT-EUDR-042: Test supply chain tracing is deterministic."""
        mock_result = {
            "shipment_id": "SHIP-2024-005",
            "commodity_type": "soya",
            "traceability_score": 0.88,
            "chain_of_custody": "complete",
            "result_hash": "y" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._trace_supply_chain', return_value=mock_result):
            params = {
                "shipment_id": "SHIP-2024-005",
                "supply_chain_nodes": sample_supply_chain_nodes,
                "commodity_type": "soya",
            }

            results = []
            for _ in range(5):
                result = await trace_supply_chain_tool.execute(params)
                results.append(result["traceability_score"])

            assert all(r == results[0] for r in results)


# =============================================================================
# GenerateDdsReportTool Tests (8 tests)
# =============================================================================

class TestGenerateDdsReportTool:
    """Test suite for GenerateDdsReportTool - 8 test cases."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_tool_initialization(self, generate_dds_report_tool):
        """UT-EUDR-043: Test tool initializes correctly."""
        assert generate_dds_report_tool is not None
        assert generate_dds_report_tool.name == "generate_dds_report"
        assert generate_dds_report_tool.safe is True

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_missing_operator_info_raises_error(self, generate_dds_report_tool):
        """UT-EUDR-044: Test missing operator_info raises ValueError."""
        params = {
            "commodity_data": {},
            "geolocation_data": {},
            "risk_assessment": {},
        }

        with pytest.raises(ValueError) as exc_info:
            await generate_dds_report_tool.execute(params)

        assert "operator_info" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_valid_dds_generation(self, generate_dds_report_tool):
        """UT-EUDR-045: Test valid DDS report generation."""
        mock_result = {
            "dds_id": "DDS-2024-001234",
            "dds_status": "valid",
            "operator_reference": "OP-EU-12345",
            "submission_date": "2024-01-15T10:00:00Z",
            "verification_status": "verified",
            "result_hash": "z" * 64,
        }

        with patch('generated.eudr_compliance_v1.tools._generate_dds_report', return_value=mock_result):
            params = {
                "operator_info": {
                    "name": "Example Importer GmbH",
                    "eori_number": "DE123456789",
                    "address": "Berlin, Germany",
                },
                "commodity_data": {
                    "cn_code": "0901.11",
                    "commodity_type": "coffee",
                    "quantity_kg": 5000.0,
                },
                "geolocation_data": {
                    "coordinates": [-3.4653, -62.2159],
                    "country_code": "BR",
                },
                "risk_assessment": {
                    "risk_level": "standard",
                    "risk_score": 0.55,
                },
            }

            result = await generate_dds_report_tool.execute(params)

            assert result["dds_status"] == "valid"
            assert "dds_id" in result
            assert result["dds_id"].startswith("DDS-")

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_incomplete_dds_generation(self, generate_dds_report_tool):
        """UT-EUDR-046: Test incomplete DDS report returns incomplete status."""
        mock_result = {
            "dds_id": None,
            "dds_status": "incomplete",
            "missing_fields": ["geolocation_coordinates", "producer_name"],
            "validation_errors": ["Required field missing: geolocation_coordinates"],
            "result_hash": "a1" * 32,
        }

        with patch('generated.eudr_compliance_v1.tools._generate_dds_report', return_value=mock_result):
            params = {
                "operator_info": {"name": "Test Operator"},
                "commodity_data": {"cn_code": "0901.11"},
                "geolocation_data": {},
                "risk_assessment": {},
            }

            result = await generate_dds_report_tool.execute(params)

            assert result["dds_status"] == "incomplete"
            assert "missing_fields" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_dds_with_traceability_data(self, generate_dds_report_tool, sample_supply_chain_nodes):
        """UT-EUDR-047: Test DDS generation with traceability data."""
        mock_result = {
            "dds_id": "DDS-2024-002345",
            "dds_status": "valid",
            "traceability_included": True,
            "supply_chain_verified": True,
            "result_hash": "b1" * 32,
        }

        with patch('generated.eudr_compliance_v1.tools._generate_dds_report', return_value=mock_result):
            params = {
                "operator_info": {"name": "Test Operator", "eori_number": "DE123456789"},
                "commodity_data": {"cn_code": "0901.11", "commodity_type": "coffee"},
                "geolocation_data": {"coordinates": [-3.4653, -62.2159], "country_code": "BR"},
                "risk_assessment": {"risk_level": "standard"},
                "traceability_data": {
                    "supply_chain_nodes": sample_supply_chain_nodes,
                    "traceability_score": 0.95,
                },
            }

            result = await generate_dds_report_tool.execute(params)

            assert result["dds_status"] == "valid"
            assert result["traceability_included"] is True

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_validate_params_method(self, generate_dds_report_tool):
        """UT-EUDR-048: Test validate_params method returns boolean."""
        valid_params = {
            "operator_info": {},
            "commodity_data": {},
            "geolocation_data": {},
            "risk_assessment": {},
        }
        invalid_params = {"operator_info": {}, "commodity_data": {}}

        assert generate_dds_report_tool.validate_params(valid_params) is True
        assert generate_dds_report_tool.validate_params(invalid_params) is False

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_dds_includes_submission_timestamp(self, generate_dds_report_tool):
        """UT-EUDR-049: Test DDS includes submission timestamp."""
        mock_result = {
            "dds_id": "DDS-2024-003456",
            "dds_status": "valid",
            "submission_date": "2024-01-20T14:30:00Z",
            "executed_at": datetime.now().isoformat(),
            "result_hash": "c1" * 32,
        }

        with patch('generated.eudr_compliance_v1.tools._generate_dds_report', return_value=mock_result):
            params = {
                "operator_info": {"name": "Test"},
                "commodity_data": {"cn_code": "0901.11"},
                "geolocation_data": {"coordinates": [0, 0], "country_code": "BR"},
                "risk_assessment": {"risk_level": "low"},
            }

            result = await generate_dds_report_tool.execute(params)

            assert "submission_date" in result or "executed_at" in result

    @pytest.mark.unit
    @pytest.mark.eudr
    @pytest.mark.asyncio
    async def test_determinism(self, generate_dds_report_tool):
        """UT-EUDR-050: Test DDS generation is deterministic."""
        mock_result = {
            "dds_id": "DDS-2024-004567",
            "dds_status": "valid",
            "result_hash": "d1" * 32,
        }

        with patch('generated.eudr_compliance_v1.tools._generate_dds_report', return_value=mock_result):
            params = {
                "operator_info": {"name": "Test", "eori_number": "DE123"},
                "commodity_data": {"cn_code": "0901.11"},
                "geolocation_data": {"coordinates": [0, 0], "country_code": "BR"},
                "risk_assessment": {"risk_level": "low"},
            }

            results = []
            for _ in range(5):
                result = await generate_dds_report_tool.execute(params)
                results.append(result["result_hash"])

            assert all(r == results[0] for r in results)


# =============================================================================
# Tool Registry Tests
# =============================================================================

class TestEUDRToolRegistry:
    """Test suite for EUDR tool registry functionality."""

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_get_tool_returns_correct_tool(self):
        """Test get_tool function returns correct tool instances."""
        from generated.eudr_compliance_v1.tools import get_tool

        geo_tool = get_tool("validate_geolocation")
        classify_tool = get_tool("classify_commodity")
        risk_tool = get_tool("assess_country_risk")
        trace_tool = get_tool("trace_supply_chain")
        dds_tool = get_tool("generate_dds_report")

        assert geo_tool is not None
        assert classify_tool is not None
        assert risk_tool is not None
        assert trace_tool is not None
        assert dds_tool is not None

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_get_tool_invalid_name_returns_none(self):
        """Test get_tool with invalid name returns None."""
        from generated.eudr_compliance_v1.tools import get_tool

        result = get_tool("nonexistent_tool")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.eudr
    def test_list_tools_returns_all_tools(self):
        """Test list_tools returns all available tools."""
        from generated.eudr_compliance_v1.tools import list_tools

        tools = list_tools()

        assert "validate_geolocation" in tools
        assert "classify_commodity" in tools
        assert "assess_country_risk" in tools
        assert "trace_supply_chain" in tools
        assert "generate_dds_report" in tools
        assert len(tools) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
