"""
GL-015 INSULSCAN - Test Suite

Comprehensive test suite for the Insulation Analysis Agent achieving 85%+ coverage.
Tests all components including heat loss calculations, economic thickness optimization,
surface temperature compliance, condensation prevention, and IR thermography integration.

Test Categories:
    - Unit Tests: Individual component testing
    - Integration Tests: Multi-component workflows
    - Performance Tests: Throughput and latency validation
    - Compliance Tests: Regulatory validation (OSHA, ASTM, NAIMA)

Coverage Target: 85%+
"""

from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_config import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_schemas import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_materials import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_heat_loss import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_economic_thickness import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_surface_temperature import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_condensation import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_ir_survey import *
from greenlang.agents.process_heat.gl_015_insulation_analysis.tests.test_analyzer import *

__all__ = [
    "TestEconomicConfig",
    "TestSafetyConfig",
    "TestInsulationAnalysisConfig",
    "TestSchemas",
    "TestInsulationMaterial",
    "TestInsulationMaterialDatabase",
    "TestHeatLossCalculator",
    "TestEconomicThicknessOptimizer",
    "TestSurfaceTemperatureCalculator",
    "TestCondensationAnalyzer",
    "TestIRThermographySurvey",
    "TestInsulationAnalysisAgent",
]
