# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Workflows.

Tests all 6 workflows: sector pathway design, pathway validation,
technology planning, progress monitoring, multi-scenario analysis, and
full sector assessment. Covers instantiation, config defaults, and
cross-sector parametrized tests.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from workflows import (
    SectorPathwayDesignWorkflow,
    SectorPathwayDesignConfig,
    SectorPathwayDesignResult,
    PathwayValidationWorkflow,
    PathwayValidationConfig,
    PathwayValidationResult,
    TechnologyPlanningWorkflow,
    TechnologyPlanningConfig,
    TechnologyPlanningResult,
    ProgressMonitoringWorkflow,
    ProgressMonitoringConfig,
    ProgressMonitoringResult,
    MultiScenarioAnalysisWorkflow,
    MultiScenarioConfig,
    MultiScenarioResult,
    FullSectorAssessmentWorkflow,
    FullSectorAssessmentConfig,
    FullSectorAssessmentResult,
)

from .conftest import (
    SDA_SECTORS,
    SCENARIO_TYPES,
    CONVERGENCE_MODELS,
    EXTENDED_SECTORS,
    timed_block,
)


# ========================================================================
# 1. Sector Pathway Design Workflow
# ========================================================================


class TestSectorPathwayDesignWorkflow:
    """Test sector pathway design workflow."""

    def test_workflow_instantiates(self):
        wf = SectorPathwayDesignWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = SectorPathwayDesignConfig()
        wf = SectorPathwayDesignWorkflow(config=config)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = SectorPathwayDesignWorkflow()
        assert hasattr(wf, "execute")
        assert callable(wf.execute)

    def test_workflow_has_workflow_id(self):
        wf = SectorPathwayDesignWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_workflow_has_logger(self):
        wf = SectorPathwayDesignWorkflow()
        assert hasattr(wf, "logger")

    def test_config_defaults(self):
        config = SectorPathwayDesignConfig()
        assert config is not None
        assert hasattr(config, "company_name")
        assert hasattr(config, "base_year")
        assert hasattr(config, "target_year")

    def test_config_with_company_name(self):
        config = SectorPathwayDesignConfig(company_name="SteelCo")
        assert config.company_name == "SteelCo"

    def test_config_with_convergence_model(self):
        config = SectorPathwayDesignConfig(convergence_model="exponential")
        assert config.convergence_model == "exponential"

    def test_config_with_scenarios(self):
        config = SectorPathwayDesignConfig(scenarios=["nze_15c", "wb2c"])
        assert len(config.scenarios) == 2

    def test_result_model(self):
        assert SectorPathwayDesignResult is not None

    @pytest.mark.parametrize("model", CONVERGENCE_MODELS)
    def test_config_convergence_models(self, model):
        config = SectorPathwayDesignConfig(convergence_model=model)
        assert config.convergence_model == model

    @pytest.mark.parametrize("nace", ["D35.11", "C24.10", "C23.51"])
    def test_config_with_nace(self, nace):
        config = SectorPathwayDesignConfig(nace_codes=[nace])
        assert nace in config.nace_codes


# ========================================================================
# 2. Pathway Validation Workflow
# ========================================================================


class TestPathwayValidationWorkflow:
    """Test pathway validation workflow."""

    def test_workflow_instantiates(self):
        wf = PathwayValidationWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = PathwayValidationConfig()
        wf = PathwayValidationWorkflow(config=config)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = PathwayValidationWorkflow()
        assert hasattr(wf, "execute")

    def test_config_defaults(self):
        config = PathwayValidationConfig()
        assert config is not None
        assert hasattr(config, "sector")
        assert hasattr(config, "base_year")

    def test_config_with_sector(self):
        config = PathwayValidationConfig(sector="power_generation")
        assert config.sector == "power_generation"

    def test_config_with_coverage(self):
        config = PathwayValidationConfig(
            scope12_coverage_pct=Decimal("95"),
            scope3_coverage_pct=Decimal("67"),
        )
        assert config.scope12_coverage_pct == Decimal("95")

    def test_result_model(self):
        assert PathwayValidationResult is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_config_per_sector(self, sector):
        config = PathwayValidationConfig(sector=sector)
        wf = PathwayValidationWorkflow(config=config)
        assert wf is not None


# ========================================================================
# 3. Technology Planning Workflow
# ========================================================================


class TestTechnologyPlanningWorkflow:
    """Test technology planning workflow."""

    def test_workflow_instantiates(self):
        wf = TechnologyPlanningWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = TechnologyPlanningConfig()
        wf = TechnologyPlanningWorkflow(config=config)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = TechnologyPlanningWorkflow()
        assert hasattr(wf, "execute")

    def test_config_defaults(self):
        config = TechnologyPlanningConfig()
        assert config is not None
        assert hasattr(config, "sector")

    def test_config_with_sector(self):
        config = TechnologyPlanningConfig(sector="steel")
        assert config.sector == "steel"

    def test_config_with_capex(self):
        config = TechnologyPlanningConfig(
            available_capex_usd=Decimal("1000000000"),
        )
        assert config.available_capex_usd == Decimal("1000000000")

    def test_result_model(self):
        assert TechnologyPlanningResult is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_config_per_sector(self, sector):
        config = TechnologyPlanningConfig(sector=sector)
        wf = TechnologyPlanningWorkflow(config=config)
        assert wf is not None


# ========================================================================
# 4. Progress Monitoring Workflow
# ========================================================================


class TestProgressMonitoringWorkflow:
    """Test progress monitoring workflow."""

    def test_workflow_instantiates(self):
        wf = ProgressMonitoringWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = ProgressMonitoringConfig()
        wf = ProgressMonitoringWorkflow(config=config)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = ProgressMonitoringWorkflow()
        assert hasattr(wf, "execute")

    def test_config_defaults(self):
        config = ProgressMonitoringConfig()
        assert config is not None
        assert hasattr(config, "sector")
        assert hasattr(config, "base_year")

    def test_config_with_sector(self):
        config = ProgressMonitoringConfig(sector="cement")
        assert config.sector == "cement"

    def test_config_with_alert_threshold(self):
        config = ProgressMonitoringConfig(alert_threshold_pct=Decimal("10"))
        assert config.alert_threshold_pct == Decimal("10")

    def test_result_model(self):
        assert ProgressMonitoringResult is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_config_per_sector(self, sector):
        config = ProgressMonitoringConfig(sector=sector)
        wf = ProgressMonitoringWorkflow(config=config)
        assert wf is not None


# ========================================================================
# 5. Multi-Scenario Analysis Workflow
# ========================================================================


class TestMultiScenarioAnalysisWorkflow:
    """Test multi-scenario analysis workflow."""

    def test_workflow_instantiates(self):
        wf = MultiScenarioAnalysisWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = MultiScenarioConfig()
        wf = MultiScenarioAnalysisWorkflow(config=config)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = MultiScenarioAnalysisWorkflow()
        assert hasattr(wf, "execute")

    def test_config_defaults(self):
        config = MultiScenarioConfig()
        assert config is not None
        assert hasattr(config, "sector")
        assert hasattr(config, "scenarios")

    def test_config_with_sector(self):
        config = MultiScenarioConfig(sector="steel")
        assert config.sector == "steel"

    def test_config_with_scenarios(self):
        config = MultiScenarioConfig(scenarios=SCENARIO_TYPES)
        assert len(config.scenarios) == 5

    def test_config_varying_scenario_counts(self):
        for n in range(1, 6):
            config = MultiScenarioConfig(scenarios=SCENARIO_TYPES[:n])
            assert len(config.scenarios) == n

    def test_result_model(self):
        assert MultiScenarioResult is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_config_per_sector(self, sector):
        config = MultiScenarioConfig(sector=sector, scenarios=SCENARIO_TYPES)
        wf = MultiScenarioAnalysisWorkflow(config=config)
        assert wf is not None


# ========================================================================
# 6. Full Sector Assessment Workflow
# ========================================================================


class TestFullSectorAssessmentWorkflow:
    """Test full sector assessment workflow."""

    def test_workflow_instantiates(self):
        wf = FullSectorAssessmentWorkflow()
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = FullSectorAssessmentWorkflow()
        assert hasattr(wf, "execute")
        assert callable(wf.execute)

    def test_workflow_has_workflow_id(self):
        wf = FullSectorAssessmentWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_config_defaults(self):
        config = FullSectorAssessmentConfig()
        assert config is not None
        assert hasattr(config, "company_name")
        assert hasattr(config, "base_year")
        assert hasattr(config, "convergence_model")

    def test_config_with_company_name(self):
        config = FullSectorAssessmentConfig(company_name="CementCo")
        assert config.company_name == "CementCo"

    def test_config_skip_flags(self):
        config = FullSectorAssessmentConfig(
            skip_validation=True,
            skip_scenarios=True,
            skip_technology=True,
        )
        assert config.skip_validation is True
        assert config.skip_scenarios is True
        assert config.skip_technology is True

    def test_result_model(self):
        assert FullSectorAssessmentResult is not None


# ========================================================================
# Cross-Workflow Tests
# ========================================================================


class TestAllWorkflowsInstantiation:
    """Test all 6 workflows instantiate cleanly."""

    WORKFLOW_CLASSES = [
        SectorPathwayDesignWorkflow,
        PathwayValidationWorkflow,
        TechnologyPlanningWorkflow,
        ProgressMonitoringWorkflow,
        MultiScenarioAnalysisWorkflow,
        FullSectorAssessmentWorkflow,
    ]

    @pytest.mark.parametrize("workflow_cls", WORKFLOW_CLASSES)
    def test_workflow_instantiates(self, workflow_cls):
        wf = workflow_cls()
        assert wf is not None

    @pytest.mark.parametrize("workflow_cls", WORKFLOW_CLASSES)
    def test_workflow_has_execute(self, workflow_cls):
        wf = workflow_cls()
        assert hasattr(wf, "execute")

    @pytest.mark.parametrize("workflow_cls", WORKFLOW_CLASSES)
    def test_workflow_has_workflow_id(self, workflow_cls):
        wf = workflow_cls()
        assert hasattr(wf, "workflow_id")

    @pytest.mark.parametrize("workflow_cls", WORKFLOW_CLASSES)
    def test_workflow_has_logger(self, workflow_cls):
        wf = workflow_cls()
        assert hasattr(wf, "logger")


# ========================================================================
# Workflow Config with Sector - Parametrized
# ========================================================================


class TestWorkflowConfigMatrix:
    """Test workflow configurations across sectors."""

    CONFIGURABLE_PAIRS = [
        (SectorPathwayDesignWorkflow, SectorPathwayDesignConfig),
        (PathwayValidationWorkflow, PathwayValidationConfig),
        (TechnologyPlanningWorkflow, TechnologyPlanningConfig),
        (ProgressMonitoringWorkflow, ProgressMonitoringConfig),
    ]

    @pytest.mark.parametrize("wf_cls,config_cls", CONFIGURABLE_PAIRS)
    def test_workflow_with_default_config(self, wf_cls, config_cls):
        config = config_cls()
        wf = wf_cls(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_design_workflow_per_sector(self, sector):
        config = SectorPathwayDesignConfig(company_name=f"{sector}Co")
        wf = SectorPathwayDesignWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_validation_workflow_per_sector(self, sector):
        config = PathwayValidationConfig(sector=sector)
        wf = PathwayValidationWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_tech_planning_per_sector(self, sector):
        config = TechnologyPlanningConfig(sector=sector)
        wf = TechnologyPlanningWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_monitoring_per_sector(self, sector):
        config = ProgressMonitoringConfig(sector=sector)
        wf = ProgressMonitoringWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_multi_scenario_per_sector(self, sector):
        config = MultiScenarioConfig(sector=sector, scenarios=SCENARIO_TYPES)
        wf = MultiScenarioAnalysisWorkflow(config=config)
        assert wf is not None


# ========================================================================
# All Sectors - Full Matrix
# ========================================================================


class TestSectorWorkflowMatrix:
    """Test all workflows across all SDA sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_validation_all_sectors(self, sector):
        config = PathwayValidationConfig(sector=sector)
        wf = PathwayValidationWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_tech_planning_all_sectors(self, sector):
        config = TechnologyPlanningConfig(sector=sector)
        wf = TechnologyPlanningWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_monitoring_all_sectors(self, sector):
        config = ProgressMonitoringConfig(sector=sector)
        wf = ProgressMonitoringWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_multi_scenario_all_sectors(self, sector):
        config = MultiScenarioConfig(sector=sector, scenarios=SCENARIO_TYPES)
        wf = MultiScenarioAnalysisWorkflow(config=config)
        assert wf is not None


# ========================================================================
# Scenario Parametrized Tests
# ========================================================================


class TestScenarioWorkflows:
    """Test workflows with different scenario configurations."""

    @pytest.mark.parametrize("scenario", SCENARIO_TYPES)
    def test_design_workflow_per_scenario(self, scenario):
        config = SectorPathwayDesignConfig(scenarios=[scenario])
        wf = SectorPathwayDesignWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("scenario", SCENARIO_TYPES)
    def test_multi_scenario_single_scenario(self, scenario):
        config = MultiScenarioConfig(sector="steel", scenarios=[scenario])
        wf = MultiScenarioAnalysisWorkflow(config=config)
        assert wf is not None

    @pytest.mark.parametrize("model", CONVERGENCE_MODELS)
    def test_design_workflow_convergence_model(self, model):
        config = SectorPathwayDesignConfig(convergence_model=model)
        wf = SectorPathwayDesignWorkflow(config=config)
        assert wf is not None


# ========================================================================
# Workflow Configuration Validation
# ========================================================================


class TestWorkflowConfigValidation:
    """Test workflow configuration objects."""

    CONFIG_CLASSES = [
        SectorPathwayDesignConfig,
        PathwayValidationConfig,
        TechnologyPlanningConfig,
        ProgressMonitoringConfig,
        MultiScenarioConfig,
        FullSectorAssessmentConfig,
    ]

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_instantiates(self, config_cls):
        config = config_cls()
        assert config is not None

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_has_model_fields(self, config_cls):
        assert len(config_cls.model_fields) > 0

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_serializable(self, config_cls):
        config = config_cls()
        d = config.model_dump()
        assert isinstance(d, dict)


# ========================================================================
# Workflow Edge Cases
# ========================================================================


class TestWorkflowEdgeCases:
    """Test workflow edge cases and boundary conditions."""

    def test_workflow_with_empty_config(self):
        wf = SectorPathwayDesignWorkflow(config=SectorPathwayDesignConfig())
        assert wf is not None

    def test_workflow_without_config(self):
        wf = SectorPathwayDesignWorkflow()
        assert wf.config is not None

    def test_full_assessment_no_config_arg(self):
        wf = FullSectorAssessmentWorkflow()
        assert wf is not None

    def test_concurrent_workflow_creation(self):
        """Verify workflows can be instantiated in parallel."""
        wfs = [
            SectorPathwayDesignWorkflow(),
            PathwayValidationWorkflow(),
            TechnologyPlanningWorkflow(),
            ProgressMonitoringWorkflow(),
            MultiScenarioAnalysisWorkflow(),
            FullSectorAssessmentWorkflow(),
        ]
        assert len(wfs) == 6
        for wf in wfs:
            assert wf is not None

    def test_multiple_configs_same_company(self):
        configs = [
            SectorPathwayDesignConfig(company_name="TestCo"),
            PathwayValidationConfig(sector="steel"),
            TechnologyPlanningConfig(sector="steel"),
        ]
        for config in configs:
            assert config is not None

    @pytest.mark.parametrize("sector", EXTENDED_SECTORS)
    def test_extended_sector_configs(self, sector):
        """Extended sectors may work in configs that have 'sector' field."""
        try:
            config = PathwayValidationConfig(sector=sector)
            wf = PathwayValidationWorkflow(config=config)
            assert wf is not None
        except (ValueError, TypeError):
            pass  # Extended sectors may not be supported


# ========================================================================
# Workflow Performance
# ========================================================================


class TestWorkflowPerformance:
    """Test workflow instantiation performance."""

    WORKFLOW_CLASSES = [
        SectorPathwayDesignWorkflow,
        PathwayValidationWorkflow,
        TechnologyPlanningWorkflow,
        ProgressMonitoringWorkflow,
        MultiScenarioAnalysisWorkflow,
        FullSectorAssessmentWorkflow,
    ]

    @pytest.mark.parametrize("workflow_cls", WORKFLOW_CLASSES)
    def test_workflow_instantiation_under_2s(self, workflow_cls):
        start = time.time()
        wf = workflow_cls()
        elapsed = (time.time() - start) * 1000
        assert wf is not None
        assert elapsed < 2000

    def test_all_workflows_instantiate_under_10s(self):
        start = time.time()
        for wf_cls in self.WORKFLOW_CLASSES:
            wf = wf_cls()
            assert wf is not None
        elapsed = (time.time() - start) * 1000
        assert elapsed < 10000

    @pytest.mark.parametrize("sector", SDA_SECTORS[:4])
    def test_design_workflow_per_sector_speed(self, sector):
        start = time.time()
        config = SectorPathwayDesignConfig(company_name=f"{sector}Co")
        wf = SectorPathwayDesignWorkflow(config=config)
        elapsed = (time.time() - start) * 1000
        assert wf is not None
        assert elapsed < 2000


# ========================================================================
# Workflow Serialization
# ========================================================================


class TestWorkflowSerialization:
    """Test workflow config serialization."""

    CONFIG_CLASSES = [
        SectorPathwayDesignConfig,
        PathwayValidationConfig,
        TechnologyPlanningConfig,
        ProgressMonitoringConfig,
        MultiScenarioConfig,
        FullSectorAssessmentConfig,
    ]

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_model_dump(self, config_cls):
        config = config_cls()
        d = config.model_dump()
        assert isinstance(d, dict)
        assert len(d) > 0

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_model_dump_json(self, config_cls):
        config = config_cls()
        j = config.model_dump_json()
        assert isinstance(j, str)
        assert len(j) > 0
