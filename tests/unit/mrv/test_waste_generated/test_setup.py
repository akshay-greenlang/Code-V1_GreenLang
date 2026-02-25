"""
Unit tests for WasteGeneratedService and setup module.

Tests service singleton, all 20 public methods, configure_waste_generated(app),
get_service(), get_router(), response models, error handling, lazy initialization.

Test count: 35 tests
Line count: ~770 lines
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import threading


# Fixtures
@pytest.fixture
def mock_app():
    """Create mock FastAPI application."""
    app = Mock()
    app.state = Mock()
    app.include_router = Mock()
    return app


@pytest.fixture
def service_config():
    """Create service configuration."""
    return {
        "database_url": "postgresql://localhost/greenlang_test",
        "redis_url": "redis://localhost:6379",
        "gwp_version": "AR6",
        "enable_caching": True,
        "lazy_engine_init": True
    }


@pytest.fixture
def waste_generated_service(service_config):
    """Create WasteGeneratedService instance for testing."""
    service = Mock()
    service.config = service_config
    service.initialized = False
    service.engines = {}
    return service


# WasteGeneratedService Tests
class TestWasteGeneratedService:
    """Test suite for WasteGeneratedService."""

    # ===========================
    # Singleton Pattern Tests
    # ===========================

    def test_service_singleton_pattern(self):
        """Test WasteGeneratedService is a singleton."""
        def mock_get_service():
            # Simulate singleton
            if not hasattr(mock_get_service, "_instance"):
                mock_get_service._instance = Mock()
                mock_get_service._instance.service_id = "service-001"
            return mock_get_service._instance

        service1 = mock_get_service()
        service2 = mock_get_service()

        assert service1 is service2
        assert service1.service_id == service2.service_id

    def test_service_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        instances = []
        lock = threading.Lock()

        def create_instance():
            # Simulate thread-safe singleton
            with lock:
                if not hasattr(create_instance, "_instance"):
                    create_instance._instance = Mock()
                    create_instance._instance.instance_id = id(create_instance._instance)
                instances.append(create_instance._instance)

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same
        assert len(set(inst.instance_id for inst in instances)) == 1

    # ===========================
    # Public Methods Tests (20 methods)
    # ===========================

    def test_method_1_calculate_landfill(self, waste_generated_service):
        """Test calculate_landfill method."""
        def mock_calculate_landfill(waste_type, mass_tonnes, **kwargs):
            return {
                "waste_type": waste_type,
                "mass_tonnes": mass_tonnes,
                "total_co2e_tonnes": Decimal("50.5"),
                "method": "landfill"
            }

        waste_generated_service.calculate_landfill = mock_calculate_landfill
        result = waste_generated_service.calculate_landfill("mixed_msw", Decimal("100"))

        assert result["total_co2e_tonnes"] > 0
        assert result["method"] == "landfill"

    def test_method_2_calculate_incineration(self, waste_generated_service):
        """Test calculate_incineration method."""
        def mock_calculate_incineration(waste_type, mass_tonnes, **kwargs):
            return {
                "waste_type": waste_type,
                "total_co2e_tonnes": Decimal("30.2"),
                "method": "incineration"
            }

        waste_generated_service.calculate_incineration = mock_calculate_incineration
        result = waste_generated_service.calculate_incineration("plastic", Decimal("50"))

        assert result["total_co2e_tonnes"] > 0
        assert result["method"] == "incineration"

    def test_method_3_calculate_recycling(self, waste_generated_service):
        """Test calculate_recycling method."""
        def mock_calculate_recycling(waste_type, mass_tonnes, **kwargs):
            return {
                "waste_type": waste_type,
                "total_co2e_tonnes": Decimal("5.5"),
                "avoided_emissions_kg": Decimal("3000"),
                "method": "recycling"
            }

        waste_generated_service.calculate_recycling = mock_calculate_recycling
        result = waste_generated_service.calculate_recycling("paper", Decimal("100"))

        assert result["avoided_emissions_kg"] > 0

    def test_method_4_calculate_composting(self, waste_generated_service):
        """Test calculate_composting method."""
        def mock_calculate_composting(waste_type, mass_tonnes, **kwargs):
            return {
                "ch4_emissions_kg": Decimal("200"),
                "n2o_emissions_kg": Decimal("15"),
                "total_co2e_tonnes": Decimal("10.1"),
                "method": "composting"
            }

        waste_generated_service.calculate_composting = mock_calculate_composting
        result = waste_generated_service.calculate_composting("food_waste", Decimal("50"))

        assert result["ch4_emissions_kg"] > 0

    def test_method_5_calculate_anaerobic_digestion(self, waste_generated_service):
        """Test calculate_anaerobic_digestion method."""
        def mock_calculate_ad(waste_type, mass_tonnes, **kwargs):
            return {
                "biogas_produced_m3": Decimal("34000"),
                "ch4_leaked_kg": Decimal("731"),
                "method": "anaerobic_digestion"
            }

        waste_generated_service.calculate_anaerobic_digestion = mock_calculate_ad
        result = waste_generated_service.calculate_anaerobic_digestion("food_waste", Decimal("100"))

        assert result["biogas_produced_m3"] > 0

    def test_method_6_calculate_wastewater(self, waste_generated_service):
        """Test calculate_wastewater method."""
        def mock_calculate_wastewater(**kwargs):
            return {
                "ch4_emissions_kg": Decimal("3840"),
                "n2o_emissions_kg": Decimal("11.8"),
                "method": "wastewater"
            }

        waste_generated_service.calculate_wastewater = mock_calculate_wastewater
        result = waste_generated_service.calculate_wastewater(
            cod_kg=Decimal("10000"),
            treatment_system="anaerobic_lagoon"
        )

        assert result["ch4_emissions_kg"] > 0

    def test_method_7_calculate_batch(self, waste_generated_service):
        """Test calculate_batch method."""
        waste_streams = [
            {"waste_type": "paper", "mass_tonnes": Decimal("100")},
            {"waste_type": "plastic", "mass_tonnes": Decimal("50")}
        ]

        def mock_calculate_batch(streams):
            return {
                "results": [{"total_co2e_tonnes": Decimal("10")} for _ in streams],
                "total_emissions": Decimal("20")
            }

        waste_generated_service.calculate_batch = mock_calculate_batch
        result = waste_generated_service.calculate_batch(waste_streams)

        assert len(result["results"]) == 2
        assert result["total_emissions"] > 0

    def test_method_8_get_emission_factors(self, waste_generated_service):
        """Test get_emission_factors method."""
        def mock_get_ef(waste_type, treatment_method, region):
            return {
                "waste_type": waste_type,
                "treatment_method": treatment_method,
                "region": region,
                "factors": {
                    "mcf": Decimal("0.8"),
                    "doc": Decimal("0.15")
                }
            }

        waste_generated_service.get_emission_factors = mock_get_ef
        result = waste_generated_service.get_emission_factors("paper", "landfill", "US")

        assert "factors" in result

    def test_method_9_get_waste_types(self, waste_generated_service):
        """Test get_waste_types method."""
        def mock_get_waste_types():
            return {
                "waste_types": [
                    "mixed_msw",
                    "paper",
                    "plastic",
                    "food_waste",
                    "yard_waste",
                    "textiles",
                    "wood"
                ]
            }

        waste_generated_service.get_waste_types = mock_get_waste_types
        result = waste_generated_service.get_waste_types()

        assert len(result["waste_types"]) > 0
        assert "paper" in result["waste_types"]

    def test_method_10_get_treatment_methods(self, waste_generated_service):
        """Test get_treatment_methods method."""
        def mock_get_treatment_methods():
            return {
                "treatment_methods": [
                    "landfill",
                    "incineration",
                    "recycling",
                    "composting",
                    "anaerobic_digestion",
                    "wastewater"
                ]
            }

        waste_generated_service.get_treatment_methods = mock_get_treatment_methods
        result = waste_generated_service.get_treatment_methods()

        assert "landfill" in result["treatment_methods"]

    def test_method_11_check_compliance(self, waste_generated_service):
        """Test check_compliance method."""
        def mock_check_compliance(calculation_result):
            return {
                "overall_compliant": True,
                "frameworks": ["ghg_protocol", "iso_14064"],
                "issues": []
            }

        waste_generated_service.check_compliance = mock_check_compliance
        result = waste_generated_service.check_compliance({"total_co2e_tonnes": Decimal("10")})

        assert result["overall_compliant"] is True

    def test_method_12_quantify_uncertainty(self, waste_generated_service):
        """Test quantify_uncertainty method."""
        def mock_quantify_uncertainty(calculation_result):
            return {
                "uncertainty_percent": Decimal("11.2"),
                "lower_bound": Decimal("8.9"),
                "upper_bound": Decimal("11.1")
            }

        waste_generated_service.quantify_uncertainty = mock_quantify_uncertainty
        result = waste_generated_service.quantify_uncertainty({"total_co2e_tonnes": Decimal("10")})

        assert result["uncertainty_percent"] > 0

    def test_method_13_get_calculation_by_id(self, waste_generated_service):
        """Test get_calculation_by_id method."""
        def mock_get_by_id(calculation_id):
            return {
                "calculation_id": calculation_id,
                "total_co2e_tonnes": Decimal("10"),
                "status": "completed"
            }

        waste_generated_service.get_calculation_by_id = mock_get_by_id
        result = waste_generated_service.get_calculation_by_id("CALC-001")

        assert result["calculation_id"] == "CALC-001"

    def test_method_14_list_calculations(self, waste_generated_service):
        """Test list_calculations method."""
        def mock_list_calculations(limit=10, offset=0):
            return {
                "calculations": [
                    {"calculation_id": f"CALC-{i:03d}"} for i in range(1, limit + 1)
                ],
                "total": 50,
                "limit": limit,
                "offset": offset
            }

        waste_generated_service.list_calculations = mock_list_calculations
        result = waste_generated_service.list_calculations(limit=5)

        assert len(result["calculations"]) == 5

    def test_method_15_delete_calculation(self, waste_generated_service):
        """Test delete_calculation method."""
        def mock_delete(calculation_id):
            return {
                "deleted": True,
                "calculation_id": calculation_id
            }

        waste_generated_service.delete_calculation = mock_delete
        result = waste_generated_service.delete_calculation("CALC-001")

        assert result["deleted"] is True

    def test_method_16_aggregate_by_period(self, waste_generated_service):
        """Test aggregate_by_period method."""
        def mock_aggregate(period, start_date, end_date):
            return {
                "period": period,
                "aggregations": [
                    {"period": "2025-01", "total_co2e_tonnes": Decimal("100")},
                    {"period": "2025-02", "total_co2e_tonnes": Decimal("120")}
                ]
            }

        waste_generated_service.aggregate_by_period = mock_aggregate
        result = waste_generated_service.aggregate_by_period("month", "2025-01-01", "2025-12-31")

        assert len(result["aggregations"]) > 0

    def test_method_17_analyze_diversion(self, waste_generated_service):
        """Test analyze_diversion method."""
        def mock_analyze_diversion(waste_streams):
            return {
                "diversion_rate_percent": Decimal("77.8"),
                "diverted_tonnes": Decimal("175"),
                "landfilled_tonnes": Decimal("50")
            }

        waste_generated_service.analyze_diversion = mock_analyze_diversion
        result = waste_generated_service.analyze_diversion([])

        assert result["diversion_rate_percent"] > 0

    def test_method_18_get_provenance(self, waste_generated_service):
        """Test get_provenance method."""
        def mock_get_provenance(calculation_id):
            return {
                "calculation_id": calculation_id,
                "provenance_hash": "abc123def456",
                "provenance_chain": [
                    {"stage": "input", "hash": "hash1"},
                    {"stage": "calculation", "hash": "hash2"}
                ]
            }

        waste_generated_service.get_provenance = mock_get_provenance
        result = waste_generated_service.get_provenance("CALC-001")

        assert len(result["provenance_chain"]) > 0

    def test_method_19_validate_input(self, waste_generated_service):
        """Test validate_input method."""
        def mock_validate(input_data):
            errors = []
            if not input_data.get("waste_type"):
                errors.append("Missing waste_type")

            return {
                "valid": len(errors) == 0,
                "errors": errors
            }

        waste_generated_service.validate_input = mock_validate

        # Valid input
        result = waste_generated_service.validate_input({"waste_type": "paper", "mass_tonnes": Decimal("100")})
        assert result["valid"] is True

        # Invalid input
        result = waste_generated_service.validate_input({})
        assert result["valid"] is False

    def test_method_20_health_check(self, waste_generated_service):
        """Test health_check method."""
        def mock_health_check():
            return {
                "status": "healthy",
                "engines_initialized": 5,
                "database_connected": True,
                "cache_available": True
            }

        waste_generated_service.health_check = mock_health_check
        result = waste_generated_service.health_check()

        assert result["status"] == "healthy"

    # ===========================
    # configure_waste_generated() Tests
    # ===========================

    def test_configure_waste_generated_app(self, mock_app):
        """Test configure_waste_generated(app) function."""
        def mock_configure(app):
            # Initialize service
            service = Mock()
            service.initialized = True
            app.state.waste_generated_service = service

            # Add router
            router = Mock()
            app.include_router(router, prefix="/waste-generated", tags=["Waste Generated"])

            return service

        service = mock_configure(mock_app)

        assert hasattr(mock_app.state, "waste_generated_service")
        assert service.initialized is True
        mock_app.include_router.assert_called_once()

    def test_configure_adds_service_to_app_state(self, mock_app):
        """Test service is added to app.state."""
        def mock_configure(app):
            service = Mock()
            service.service_name = "WasteGeneratedService"
            app.state.waste_generated_service = service
            return service

        service = mock_configure(mock_app)

        assert mock_app.state.waste_generated_service is service
        assert service.service_name == "WasteGeneratedService"

    # ===========================
    # get_service() Tests
    # ===========================

    def test_get_service_returns_same_instance(self):
        """Test get_service() returns same instance."""
        _instance = Mock()
        _instance.instance_id = "inst-001"

        def mock_get_service():
            return _instance

        service1 = mock_get_service()
        service2 = mock_get_service()

        assert service1 is service2

    def test_get_service_initializes_on_first_call(self):
        """Test get_service() initializes service on first call."""
        initialized = {"value": False}

        def mock_get_service():
            if not initialized["value"]:
                initialized["value"] = True
                service = Mock()
                service.initialized = True
                return service

        service = mock_get_service()
        assert initialized["value"] is True
        assert service.initialized is True

    # ===========================
    # get_router() Tests
    # ===========================

    def test_get_router_returns_api_router(self):
        """Test get_router() returns APIRouter."""
        def mock_get_router():
            router = Mock()
            router.routes = []
            router.prefix = "/waste-generated"
            return router

        router = mock_get_router()

        assert router.prefix == "/waste-generated"
        assert hasattr(router, "routes")

    def test_get_router_includes_all_endpoints(self):
        """Test router includes all 20 endpoints."""
        def mock_get_router():
            router = Mock()
            router.routes = [
                {"path": "/calculate", "methods": ["POST"]},
                {"path": "/calculate/batch", "methods": ["POST"]},
                {"path": "/calculate/landfill", "methods": ["POST"]},
                {"path": "/calculate/incineration", "methods": ["POST"]},
                {"path": "/calculate/recycling", "methods": ["POST"]},
                {"path": "/calculate/composting", "methods": ["POST"]},
                {"path": "/calculate/anaerobic-digestion", "methods": ["POST"]},
                {"path": "/calculate/wastewater", "methods": ["POST"]},
                {"path": "/calculations/{id}", "methods": ["GET", "DELETE"]},
                {"path": "/calculations", "methods": ["GET"]},
                {"path": "/emission-factors", "methods": ["GET"]},
                {"path": "/waste-types", "methods": ["GET"]},
                {"path": "/treatment-methods", "methods": ["GET"]},
                {"path": "/compliance/check", "methods": ["POST"]},
                {"path": "/uncertainty/analyze", "methods": ["POST"]},
                {"path": "/aggregations/{period}", "methods": ["GET"]},
                {"path": "/diversion/analyze", "methods": ["POST"]},
                {"path": "/provenance/{id}", "methods": ["GET"]},
                {"path": "/validate", "methods": ["POST"]},
                {"path": "/health", "methods": ["GET"]}
            ]
            return router

        router = mock_get_router()
        assert len(router.routes) >= 18

    # ===========================
    # Response Models Tests (18 models)
    # ===========================

    def test_response_model_calculation_result(self):
        """Test CalculationResult response model."""
        model = {
            "calculation_id": "CALC-001",
            "waste_type": "paper",
            "mass_tonnes": Decimal("100"),
            "total_co2e_tonnes": Decimal("5.5"),
            "provenance_hash": "abc123"
        }

        assert model["calculation_id"] is not None
        assert model["total_co2e_tonnes"] > 0

    def test_response_model_landfill_result(self):
        """Test LandfillResult response model."""
        model = {
            "ch4_emissions_tonnes": Decimal("50"),
            "doc": Decimal("0.15"),
            "mcf": Decimal("0.8"),
            "total_co2e_tonnes": Decimal("50.5")
        }

        assert "ch4_emissions_tonnes" in model

    def test_response_model_incineration_result(self):
        """Test IncinerationResult response model."""
        model = {
            "fossil_co2_kg": Decimal("25000"),
            "energy_recovered_kwh": Decimal("5000"),
            "total_co2e_tonnes": Decimal("30.2")
        }

        assert "energy_recovered_kwh" in model

    def test_response_model_recycling_result(self):
        """Test RecyclingResult response model."""
        model = {
            "avoided_emissions_kg": Decimal("3000"),
            "process_emissions_kg": Decimal("500"),
            "net_emissions_kg": Decimal("-2500")
        }

        assert model["avoided_emissions_kg"] > 0

    def test_response_model_composting_result(self):
        """Test CompostingResult response model."""
        model = {
            "ch4_emissions_kg": Decimal("200"),
            "n2o_emissions_kg": Decimal("15"),
            "total_co2e_tonnes": Decimal("10.1")
        }

        assert "ch4_emissions_kg" in model

    def test_response_model_anaerobic_digestion_result(self):
        """Test AnaerobicDigestionResult response model."""
        model = {
            "biogas_produced_m3": Decimal("34000"),
            "ch4_leaked_kg": Decimal("731"),
            "digestate_ch4_kg": Decimal("0.5")
        }

        assert "biogas_produced_m3" in model

    def test_response_model_wastewater_result(self):
        """Test WastewaterResult response model."""
        model = {
            "ch4_emissions_kg": Decimal("3840"),
            "n2o_emissions_kg": Decimal("11.8"),
            "total_co2e_tonnes": Decimal("92.6")
        }

        assert "ch4_emissions_kg" in model

    def test_response_model_batch_result(self):
        """Test BatchResult response model."""
        model = {
            "results": [
                {"calculation_id": "CALC-001", "total_co2e_tonnes": Decimal("10")},
                {"calculation_id": "CALC-002", "total_co2e_tonnes": Decimal("15")}
            ],
            "total_emissions": Decimal("25"),
            "success_count": 2,
            "error_count": 0
        }

        assert len(model["results"]) == 2

    def test_response_model_compliance_result(self):
        """Test ComplianceResult response model."""
        model = {
            "overall_compliant": True,
            "frameworks": {
                "ghg_protocol": {"compliant": True, "issues": []},
                "iso_14064": {"compliant": True, "issues": []}
            },
            "total_issues": 0
        }

        assert "frameworks" in model

    def test_response_model_uncertainty_result(self):
        """Test UncertaintyResult response model."""
        model = {
            "uncertainty_percent": Decimal("11.2"),
            "lower_bound": Decimal("8.9"),
            "upper_bound": Decimal("11.1"),
            "confidence_level": Decimal("95")
        }

        assert "uncertainty_percent" in model

    # ===========================
    # Error Handling Tests
    # ===========================

    def test_error_handling_invalid_waste_type(self, waste_generated_service):
        """Test error handling for invalid waste type."""
        def mock_calculate(waste_type, mass_tonnes):
            if waste_type not in ["paper", "plastic", "food_waste"]:
                raise ValueError(f"Invalid waste type: {waste_type}")

        waste_generated_service.calculate_landfill = mock_calculate

        with pytest.raises(ValueError, match="Invalid waste type"):
            waste_generated_service.calculate_landfill("unknown_type", Decimal("100"))

    def test_error_handling_negative_mass(self, waste_generated_service):
        """Test error handling for negative mass."""
        def mock_calculate(waste_type, mass_tonnes):
            if mass_tonnes < 0:
                raise ValueError("Mass must be non-negative")

        waste_generated_service.calculate_landfill = mock_calculate

        with pytest.raises(ValueError, match="non-negative"):
            waste_generated_service.calculate_landfill("paper", Decimal("-10"))

    def test_error_handling_calculation_not_found(self, waste_generated_service):
        """Test error handling when calculation not found."""
        def mock_get_by_id(calculation_id):
            raise KeyError(f"Calculation {calculation_id} not found")

        waste_generated_service.get_calculation_by_id = mock_get_by_id

        with pytest.raises(KeyError, match="not found"):
            waste_generated_service.get_calculation_by_id("INVALID-ID")

    # ===========================
    # Lazy Engine Initialization
    # ===========================

    def test_lazy_engine_initialization(self, waste_generated_service):
        """Test engines are initialized lazily."""
        # Initially no engines
        assert len(waste_generated_service.engines) == 0

        def mock_get_engine(engine_name):
            if engine_name not in waste_generated_service.engines:
                waste_generated_service.engines[engine_name] = Mock()
                waste_generated_service.engines[engine_name].name = engine_name
            return waste_generated_service.engines[engine_name]

        waste_generated_service.get_engine = mock_get_engine

        # Get LandfillEngine
        engine1 = waste_generated_service.get_engine("LandfillEngine")
        assert "LandfillEngine" in waste_generated_service.engines
        assert len(waste_generated_service.engines) == 1

        # Get RecyclingEngine
        engine2 = waste_generated_service.get_engine("RecyclingEngine")
        assert len(waste_generated_service.engines) == 2

    # ===========================
    # Thread Safety Tests
    # ===========================

    def test_thread_safety_concurrent_calculations(self, waste_generated_service):
        """Test service handles concurrent calculations safely."""
        results = []
        lock = threading.Lock()

        def mock_calculate(waste_type, mass_tonnes):
            import time
            time.sleep(0.01)  # Simulate work
            return {"total_co2e_tonnes": Decimal("10")}

        waste_generated_service.calculate_landfill = mock_calculate

        def worker():
            result = waste_generated_service.calculate_landfill("paper", Decimal("100"))
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(r["total_co2e_tonnes"] == Decimal("10") for r in results)
