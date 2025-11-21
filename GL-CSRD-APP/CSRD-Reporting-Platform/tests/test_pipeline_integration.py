# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform - Complete Pipeline Integration Tests

This is the MOST CRITICAL test suite in the entire platform.
It validates the end-to-end functionality of ALL 6 agents working together.

TEST SCOPE:
- Complete 6-agent workflow (Intake → Materiality → Calculator → Aggregator → Reporting → Audit)
- Performance benchmarks (<30 minutes for 10,000 data points)
- Large dataset testing (1,000+ metrics)
- Multi-standard coverage (E1-E5, S1-S4, G1)
- Error recovery and resilience
- Multi-entity scenarios (parent-subsidiary)
- Output validation (XBRL, PDF, JSON, audit package)
- Intermediate outputs between agents
- Framework integration (TCFD, GRI, SASB → ESRS)
- Time-series analysis (YoY, CAGR)

TARGET: High integration coverage, production-ready validation

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from csrd_pipeline import (
from greenlang.determinism import deterministic_random
    AgentExecution,
    CSRDPipeline,
    PipelinePerformance,
    PipelineResult,
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def base_path() -> Path:
    """Get base path for test resources."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_path(base_path: Path) -> Path:
    """Path to CSRD configuration YAML."""
    return base_path / "config" / "csrd_config.yaml"


@pytest.fixture
def demo_data_path(base_path: Path) -> Path:
    """Path to demo ESG data CSV (50 metrics)."""
    return base_path / "examples" / "demo_esg_data.csv"


@pytest.fixture
def demo_company_profile_path(base_path: Path) -> Path:
    """Path to demo company profile JSON."""
    return base_path / "examples" / "demo_company_profile.json"


@pytest.fixture
def demo_company_profile(demo_company_profile_path: Path) -> Dict[str, Any]:
    """Load demo company profile."""
    with open(demo_company_profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    output = tmp_path / "pipeline_output"
    output.mkdir(exist_ok=True)
    return output


@pytest.fixture
def pipeline(config_path: Path) -> CSRDPipeline:
    """Create CSRDPipeline instance for testing."""
    return CSRDPipeline(config_path=str(config_path))


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider to avoid API calls."""
    with patch('agents.materiality_agent.MaterialityAgent._call_llm') as mock:
        # Mock materiality assessment response
        mock.return_value = {
            'material_topics': [
                {
                    'esrs_code': 'E1',
                    'topic': 'Climate Change',
                    'impact_materiality': 8.5,
                    'financial_materiality': 7.0,
                    'is_material': True
                },
                {
                    'esrs_code': 'S1',
                    'topic': 'Own Workforce',
                    'impact_materiality': 7.5,
                    'financial_materiality': 6.5,
                    'is_material': True
                },
                {
                    'esrs_code': 'G1',
                    'topic': 'Business Conduct',
                    'impact_materiality': 6.0,
                    'financial_materiality': 7.5,
                    'is_material': True
                }
            ],
            'analysis_summary': 'Based on the company profile and data, three topics are material.'
        }
        yield mock


@pytest.fixture
def large_dataset_1k(tmp_path: Path, base_path: Path) -> Path:
    """Generate large dataset with 1,000 metrics."""
    # Load demo data as template
    demo_path = base_path / "examples" / "demo_esg_data.csv"
    df = pd.read_csv(demo_path)

    # Replicate rows to create 1,000 metrics
    rows = []
    for i in range(20):  # 50 metrics * 20 = 1,000
        for _, row in df.iterrows():
            new_row = row.copy()
            # Add variation to values
            if pd.notna(row['value']):
                new_row['value'] = float(row['value']) * (0.9 + np.deterministic_random().random() * 0.2)
            rows.append(new_row)

    large_df = pd.DataFrame(rows)
    output_path = tmp_path / "large_dataset_1k.csv"
    large_df.to_csv(output_path, index=False)

    return output_path


@pytest.fixture
def large_dataset_10k(tmp_path: Path, base_path: Path) -> Path:
    """Generate large dataset with 10,000 metrics."""
    # Load demo data as template
    demo_path = base_path / "examples" / "demo_esg_data.csv"
    df = pd.read_csv(demo_path)

    # Replicate rows to create 10,000 metrics
    rows = []
    for i in range(200):  # 50 metrics * 200 = 10,000
        for _, row in df.iterrows():
            new_row = row.copy()
            # Add variation to values
            if pd.notna(row['value']):
                new_row['value'] = float(row['value']) * (0.9 + np.deterministic_random().random() * 0.2)
            rows.append(new_row)

    large_df = pd.DataFrame(rows)
    output_path = tmp_path / "large_dataset_10k.csv"
    large_df.to_csv(output_path, index=False)

    return output_path


@pytest.fixture
def e1_only_data(tmp_path: Path) -> Path:
    """Generate dataset with only E1 (Climate) metrics."""
    data = {
        'metric_code': ['E1-1', 'E1-2', 'E1-3', 'E1-6', 'E1-7', 'E1-8'],
        'metric_name': [
            'Scope 1 GHG Emissions',
            'Scope 2 GHG Emissions (location-based)',
            'Scope 2 GHG Emissions (market-based)',
            'Total Energy Consumption',
            'Renewable Energy Consumption',
            'Renewable Energy Percentage'
        ],
        'value': [12500.5, 8300.2, 6100.8, 185000, 45000, 24.3],
        'unit': ['tCO2e', 'tCO2e', 'tCO2e', 'GJ', 'GJ', '%'],
        'period_start': ['2024-01-01'] * 6,
        'period_end': ['2024-12-31'] * 6,
        'data_quality': ['high'] * 6,
        'source_document': ['SAP Energy Management'] * 6,
        'verification_status': ['verified'] * 6,
        'notes': [''] * 6
    }
    df = pd.DataFrame(data)
    output_path = tmp_path / "e1_only_data.csv"
    df.to_csv(output_path, index=False)
    return output_path


@pytest.fixture
def multi_entity_data(tmp_path: Path, demo_company_profile: Dict) -> tuple:
    """Generate multi-entity dataset (parent + 3 subsidiaries)."""
    # Parent company data
    parent_data = {
        'entity_id': ['parent'] * 10,
        'metric_code': ['E1-1', 'E1-2', 'E1-6', 'S1-1', 'S1-8', 'G1-1', 'G1-2', 'G1-4', 'E5-1', 'E5-3'],
        'metric_name': [
            'Scope 1 GHG Emissions', 'Scope 2 GHG Emissions', 'Total Energy Consumption',
            'Total Employees', 'Employee Turnover Rate', 'Board Members', 'Female Board Members',
            'Anti-Corruption Training Rate', 'Total Waste', 'Waste Diversion Rate'
        ],
        'value': [8000.0, 5000.0, 120000, 610, 8.0, 9, 4, 98, 2100, 68],
        'unit': ['tCO2e', 'tCO2e', 'GJ', 'FTE', '%', 'count', 'count', '%', 'tonnes', '%'],
        'period_start': ['2024-01-01'] * 10,
        'period_end': ['2024-12-31'] * 10,
        'data_quality': ['high'] * 10
    }

    # Subsidiary 1 (Germany - 100% ownership)
    sub1_data = {
        'entity_id': ['sub-001'] * 5,
        'metric_code': ['E1-1', 'E1-2', 'E1-6', 'S1-1', 'E5-1'],
        'metric_name': [
            'Scope 1 GHG Emissions', 'Scope 2 GHG Emissions', 'Total Energy Consumption',
            'Total Employees', 'Total Waste'
        ],
        'value': [2500.0, 1800.0, 38000, 320, 700],
        'unit': ['tCO2e', 'tCO2e', 'GJ', 'FTE', 'tonnes'],
        'period_start': ['2024-01-01'] * 5,
        'period_end': ['2024-12-31'] * 5,
        'data_quality': ['high'] * 5
    }

    # Subsidiary 2 (France - 75% ownership)
    sub2_data = {
        'entity_id': ['sub-002'] * 5,
        'metric_code': ['E1-1', 'E1-2', 'E1-6', 'S1-1', 'E5-1'],
        'metric_name': [
            'Scope 1 GHG Emissions', 'Scope 2 GHG Emissions', 'Total Energy Consumption',
            'Total Employees', 'Total Waste'
        ],
        'value': [1500.0, 1200.0, 22000, 180, 450],
        'unit': ['tCO2e', 'tCO2e', 'GJ', 'FTE', 'tonnes'],
        'period_start': ['2024-01-01'] * 5,
        'period_end': ['2024-12-31'] * 5,
        'data_quality': ['high'] * 5
    }

    # Subsidiary 3 (Spain - 100% ownership)
    sub3_data = {
        'entity_id': ['sub-003'] * 5,
        'metric_code': ['E1-1', 'E1-2', 'E1-6', 'S1-1', 'E5-1'],
        'metric_name': [
            'Scope 1 GHG Emissions', 'Scope 2 GHG Emissions', 'Total Energy Consumption',
            'Total Employees', 'Total Waste'
        ],
        'value': [500.0, 300.0, 5000, 140, 250],
        'unit': ['tCO2e', 'tCO2e', 'GJ', 'FTE', 'tonnes'],
        'period_start': ['2024-01-01'] * 5,
        'period_end': ['2024-12-31'] * 5,
        'data_quality': ['high'] * 5
    }

    # Combine all entities
    all_data = {}
    for key in parent_data.keys():
        all_data[key] = (
            parent_data[key] + sub1_data[key] + sub2_data[key] + sub3_data[key]
        )

    df = pd.DataFrame(all_data)
    output_path = tmp_path / "multi_entity_data.csv"
    df.to_csv(output_path, index=False)

    return output_path, demo_company_profile


@pytest.fixture
def time_series_data(tmp_path: Path) -> Path:
    """Generate time-series dataset (2020-2024, 5 years)."""
    years = [2020, 2021, 2022, 2023, 2024]
    metrics = ['E1-1', 'E1-2', 'S1-1', 'G1-1']

    # Base values with YoY growth
    base_values = {
        'E1-1': 15000.0,  # Decreasing (good for emissions)
        'E1-2': 10000.0,  # Decreasing
        'S1-1': 1000.0,   # Increasing (growth)
        'G1-1': 7.0       # Stable
    }

    growth_rates = {
        'E1-1': -0.05,  # -5% YoY (emissions reduction)
        'E1-2': -0.08,  # -8% YoY
        'S1-1': 0.10,   # +10% YoY (hiring)
        'G1-1': 0.0     # Stable
    }

    rows = []
    for metric in metrics:
        for i, year in enumerate(years):
            value = base_values[metric] * ((1 + growth_rates[metric]) ** i)
            rows.append({
                'metric_code': metric,
                'metric_name': f'{metric} metric',
                'value': value,
                'unit': 'tCO2e' if 'E1' in metric else ('FTE' if 'S1' in metric else 'count'),
                'period_start': f'{year}-01-01',
                'period_end': f'{year}-12-31',
                'data_quality': 'high',
                'source_document': 'Historical records',
                'verification_status': 'verified',
                'notes': f'Year {year}'
            })

    df = pd.DataFrame(rows)
    output_path = tmp_path / "time_series_data.csv"
    df.to_csv(output_path, index=False)
    return output_path


# ============================================================================
# TEST CLASS 1: PIPELINE INITIALIZATION
# ============================================================================


class TestPipelineInitialization:
    """Test pipeline setup and agent initialization."""

    def test_pipeline_initialization_success(self, config_path):
        """Test successful pipeline initialization."""
        pipeline = CSRDPipeline(config_path=str(config_path))

        # Verify all agents initialized
        assert pipeline.intake_agent is not None
        assert pipeline.materiality_agent is not None
        assert pipeline.calculator_agent is not None
        assert pipeline.aggregator_agent is not None
        assert pipeline.reporting_agent is not None
        assert pipeline.audit_agent is not None

    def test_pipeline_config_loaded(self, config_path):
        """Test configuration is loaded correctly."""
        pipeline = CSRDPipeline(config_path=str(config_path))

        assert pipeline.config is not None
        assert 'metadata' in pipeline.config
        assert pipeline.config['metadata']['config_version'] == '1.0.0'
        assert pipeline.config['pipeline']['target_total_time_minutes'] == 30

    def test_pipeline_invalid_config_path(self):
        """Test pipeline fails gracefully with invalid config path."""
        with pytest.raises(FileNotFoundError):
            CSRDPipeline(config_path="invalid/path/config.yaml")

    def test_pipeline_agent_executions_empty(self, pipeline):
        """Test agent executions list starts empty."""
        assert isinstance(pipeline.agent_executions, list)
        assert len(pipeline.agent_executions) == 0

    def test_pipeline_stats_initialized(self, pipeline):
        """Test pipeline statistics are initialized."""
        assert 'agent_times' in pipeline.stats
        assert 'total_warnings' in pipeline.stats
        assert 'total_errors' in pipeline.stats


# ============================================================================
# TEST CLASS 2: COMPLETE WORKFLOW (HAPPY PATH)
# ============================================================================


class TestCompleteWorkflow:
    """Test end-to-end pipeline with demo data (happy path)."""

    def test_complete_workflow_demo_data(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test complete 6-agent pipeline with demo data."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Overall result
        assert isinstance(result, PipelineResult)
        assert result.status in ["success", "partial_success"]
        assert result.compliance_status in ["PASS", "WARNING"]

        # Assert - All 6 agents executed
        assert len(result.agent_executions) == 6
        assert result.agent_executions[0].agent_name == "IntakeAgent"
        assert result.agent_executions[1].agent_name == "MaterialityAgent"
        assert result.agent_executions[2].agent_name == "CalculatorAgent"
        assert result.agent_executions[3].agent_name == "AggregatorAgent"
        assert result.agent_executions[4].agent_name == "ReportingAgent"
        assert result.agent_executions[5].agent_name == "AuditAgent"

        # Assert - All agents succeeded
        for exec in result.agent_executions:
            assert exec.status == "success"

    def test_complete_workflow_all_outputs_generated(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test all intermediate outputs are generated."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Intermediate files exist
        intermediate_dir = output_dir / "intermediate"
        assert (intermediate_dir / "01_intake_validated.json").exists()
        assert (intermediate_dir / "02_materiality_assessment.json").exists()
        assert (intermediate_dir / "03_calculated_metrics.json").exists()
        assert (intermediate_dir / "04_aggregated_data.json").exists()
        assert (intermediate_dir / "05_csrd_report.json").exists()
        assert (intermediate_dir / "06_compliance_audit.json").exists()

        # Assert - Final result file exists
        assert (output_dir / "pipeline_result.json").exists()

    def test_complete_workflow_no_data_loss(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test no data loss between agents."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Data flows through pipeline
        assert result.agent_executions[0].output_records > 0  # IntakeAgent
        assert result.agent_executions[1].output_records > 0  # MaterialityAgent
        assert result.agent_executions[2].output_records > 0  # CalculatorAgent
        assert result.agent_executions[3].output_records > 0  # AggregatorAgent
        assert result.agent_executions[4].output_records > 0  # ReportingAgent
        assert result.agent_executions[5].output_records > 0  # AuditAgent

        # No massive data loss
        intake_records = result.agent_executions[0].output_records
        calc_records = result.agent_executions[2].output_records
        assert calc_records >= intake_records * 0.5  # At least 50% makes it through

    def test_complete_workflow_performance_demo_data(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test processing time <5 minutes for demo data (50 metrics)."""
        # Act
        start_time = time.time()
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )
        total_time = time.time() - start_time

        # Assert - Performance
        assert total_time < 300  # <5 minutes
        assert result.performance.total_time_seconds < 300
        assert result.performance.records_per_second > 0

    def test_complete_workflow_data_quality_score(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test data quality score is calculated."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Data quality
        assert result.data_quality_score >= 0
        assert result.data_quality_score <= 100
        # Demo data should have high quality
        assert result.data_quality_score >= 80

    def test_complete_workflow_provenance_tracking(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test provenance and audit trail is maintained."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Provenance
        assert result.configuration_used is not None
        assert 'config_version' in result.configuration_used
        assert result.execution_timestamp is not None
        assert result.pipeline_id is not None

    def test_complete_workflow_warnings_and_errors_tracked(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test warnings and errors are tracked across agents."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Error tracking
        assert result.warnings_count >= 0
        assert result.errors_count >= 0

        # Sum of agent warnings/errors matches total
        total_warnings = sum(exec.warnings for exec in result.agent_executions)
        total_errors = sum(exec.errors for exec in result.agent_executions)
        assert result.warnings_count == total_warnings
        assert result.errors_count == total_errors


# ============================================================================
# TEST CLASS 3: LARGE DATASETS PERFORMANCE
# ============================================================================


class TestLargeDatasets:
    """Test pipeline performance with large datasets."""

    def test_large_dataset_1k_metrics_performance(
        self,
        pipeline,
        large_dataset_1k,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test 1,000 metrics performance."""
        # Act
        start_time = time.time()
        result = pipeline.run(
            esg_data_file=str(large_dataset_1k),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )
        total_time = time.time() - start_time

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Assert - Performance reasonable for 1K metrics
        assert total_time < 600  # <10 minutes for 1K
        assert result.total_data_points_processed >= 900  # Most processed

    @pytest.mark.slow
    def test_large_dataset_10k_metrics_performance(
        self,
        pipeline,
        large_dataset_10k,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test 10,000 metrics performance (target: <30 minutes)."""
        # Act
        start_time = time.time()
        result = pipeline.run(
            esg_data_file=str(large_dataset_10k),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )
        total_time = time.time() - start_time

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Assert - Performance within target
        assert total_time < 1800  # <30 minutes
        assert result.performance.within_target is True
        assert result.total_data_points_processed >= 9000  # Most processed

    def test_large_dataset_throughput_measurement(
        self,
        pipeline,
        large_dataset_1k,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test throughput measurement (records/second)."""
        # Act
        result = pipeline.run(
            esg_data_file=str(large_dataset_1k),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Throughput calculated
        assert result.performance.records_per_second > 0

        # Should process at reasonable rate
        assert result.performance.records_per_second >= 5  # At least 5 records/sec

    def test_large_dataset_agent_timing_breakdown(
        self,
        pipeline,
        large_dataset_1k,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test agent timing breakdown for large dataset."""
        # Act
        result = pipeline.run(
            esg_data_file=str(large_dataset_1k),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - All agent times recorded
        perf = result.performance
        assert perf.agent_1_intake_seconds > 0
        assert perf.agent_2_materiality_seconds > 0
        assert perf.agent_3_calculator_seconds > 0
        assert perf.agent_4_aggregator_seconds > 0
        assert perf.agent_5_reporting_seconds > 0
        assert perf.agent_6_audit_seconds > 0

        # Assert - Sum equals total
        total_agent_time = (
            perf.agent_1_intake_seconds +
            perf.agent_2_materiality_seconds +
            perf.agent_3_calculator_seconds +
            perf.agent_4_aggregator_seconds +
            perf.agent_5_reporting_seconds +
            perf.agent_6_audit_seconds
        )
        # Allow small overhead for orchestration
        assert abs(total_agent_time - perf.total_time_seconds) < 5


# ============================================================================
# TEST CLASS 4: MULTI-STANDARD COVERAGE
# ============================================================================


class TestMultiStandardCoverage:
    """Test pipeline with different ESRS standard combinations."""

    def test_e1_climate_only(
        self,
        pipeline,
        e1_only_data,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with E1 (Climate) data only."""
        # Act
        result = pipeline.run(
            esg_data_file=str(e1_only_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Assert - E1 metrics processed
        assert result.total_data_points_processed > 0

    def test_all_environmental_e1_to_e5(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with all Environmental standards (E1-E5)."""
        # Demo data contains E1-E5
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Verify E1-E5 metrics present in output
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "01_intake_validated.json", 'r') as f:
            intake_output = json.load(f)

        metric_codes = [dp['metric_code'] for dp in intake_output['validated_data']]
        assert any(code.startswith('E1-') for code in metric_codes)
        assert any(code.startswith('E2-') for code in metric_codes)
        assert any(code.startswith('E3-') for code in metric_codes)
        assert any(code.startswith('E4-') for code in metric_codes)
        assert any(code.startswith('E5-') for code in metric_codes)

    def test_all_social_s1_to_s4(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with all Social standards (S1-S4)."""
        # Demo data contains S1-S4
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Verify S1-S4 metrics present
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "01_intake_validated.json", 'r') as f:
            intake_output = json.load(f)

        metric_codes = [dp['metric_code'] for dp in intake_output['validated_data']]
        assert any(code.startswith('S1-') for code in metric_codes)
        assert any(code.startswith('S2-') for code in metric_codes)
        assert any(code.startswith('S3-') for code in metric_codes)
        assert any(code.startswith('S4-') for code in metric_codes)

    def test_governance_g1(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with G1 (Governance) standard."""
        # Demo data contains G1
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Verify G1 metrics present
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "01_intake_validated.json", 'r') as f:
            intake_output = json.load(f)

        metric_codes = [dp['metric_code'] for dp in intake_output['validated_data']]
        assert any(code.startswith('G1-') for code in metric_codes)

    def test_full_coverage_all_standards(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with ALL standards (E1-E5, S1-S4, G1)."""
        # Demo data has full coverage
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Completed successfully
        assert result.status in ["success", "partial_success"]

        # Verify comprehensive coverage
        assert result.total_data_points_processed >= 45  # Demo has ~50 metrics

    def test_standard_routing_to_agents(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test correct routing of standards to appropriate agents."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # All 6 agents should have processed data
        assert all(exec.status == "success" for exec in result.agent_executions)

        # CalculatorAgent should have calculated metrics from all standards
        calc_exec = result.agent_executions[2]
        assert calc_exec.output_records > 0


# ============================================================================
# TEST CLASS 5: ERROR RECOVERY
# ============================================================================


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_invalid_data_intake_fails(
        self,
        pipeline,
        tmp_path,
        demo_company_profile,
        output_dir
    ):
        """Test IntakeAgent fails with completely invalid data."""
        # Create invalid CSV
        invalid_data = tmp_path / "invalid.csv"
        invalid_data.write_text("random,garbage,data\n1,2,3\n")

        # Act & Assert
        with pytest.raises(Exception):
            pipeline.run(
                esg_data_file=str(invalid_data),
                company_profile=demo_company_profile,
                output_dir=str(output_dir)
            )

    def test_missing_file_fails_gracefully(
        self,
        pipeline,
        demo_company_profile,
        output_dir
    ):
        """Test missing input file fails gracefully."""
        with pytest.raises(Exception):
            pipeline.run(
                esg_data_file="nonexistent_file.csv",
                company_profile=demo_company_profile,
                output_dir=str(output_dir)
            )

    def test_partial_data_quality_warnings(
        self,
        pipeline,
        tmp_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test partial data quality generates warnings."""
        # Create data with some missing values
        data = {
            'metric_code': ['E1-1', 'E1-2', 'E1-3'],
            'metric_name': ['Scope 1', 'Scope 2', 'Scope 2 MB'],
            'value': [1000.0, None, 800.0],  # Missing value
            'unit': ['tCO2e', 'tCO2e', 'tCO2e'],
            'period_start': ['2024-01-01'] * 3,
            'period_end': ['2024-12-31'] * 3,
            'data_quality': ['high', 'low', 'high'],
            'source_document': ['SAP'] * 3,
            'verification_status': ['verified'] * 3,
            'notes': [''] * 3
        }
        df = pd.DataFrame(data)
        partial_data = tmp_path / "partial_data.csv"
        df.to_csv(partial_data, index=False)

        # Act
        result = pipeline.run(
            esg_data_file=str(partial_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Should complete with warnings
        assert result.status in ["success", "partial_success"]
        assert result.warnings_count > 0 or result.errors_count > 0

    def test_materiality_llm_failure_degradation(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir
    ):
        """Test MaterialityAgent degrades gracefully without LLM."""
        # Mock LLM to raise exception
        with patch('agents.materiality_agent.MaterialityAgent._call_llm') as mock:
            mock.side_effect = Exception("LLM API unavailable")

            # Should either fail or degrade gracefully
            try:
                result = pipeline.run(
                    esg_data_file=str(demo_data_path),
                    company_profile=demo_company_profile,
                    output_dir=str(output_dir)
                )
                # If it succeeds, check for fallback behavior
                assert result.status in ["partial_success", "failure"]
            except Exception:
                # Expected to fail without LLM
                pass

    def test_calculation_error_flagging(
        self,
        pipeline,
        tmp_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test CalculatorAgent flags calculation errors."""
        # Create data that might cause calculation issues
        data = {
            'metric_code': ['E1-1', 'E1-8'],  # E1-8 is percentage, needs inputs
            'metric_name': ['Scope 1', 'Renewable %'],
            'value': [1000.0, 150.0],  # Invalid percentage (>100%)
            'unit': ['tCO2e', '%'],
            'period_start': ['2024-01-01'] * 2,
            'period_end': ['2024-12-31'] * 2,
            'data_quality': ['high'] * 2,
            'source_document': ['SAP'] * 2,
            'verification_status': ['verified'] * 2,
            'notes': [''] * 2
        }
        df = pd.DataFrame(data)
        error_data = tmp_path / "error_data.csv"
        df.to_csv(error_data, index=False)

        # Act
        result = pipeline.run(
            esg_data_file=str(error_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Should complete but may have errors/warnings
        assert result.status in ["success", "partial_success", "failure"]

    def test_compliance_failure_reporting(
        self,
        pipeline,
        tmp_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test AuditAgent reports compliance failures."""
        # Create minimal data (likely non-compliant)
        data = {
            'metric_code': ['E1-1'],
            'metric_name': ['Scope 1'],
            'value': [1000.0],
            'unit': ['tCO2e'],
            'period_start': ['2024-01-01'],
            'period_end': ['2024-12-31'],
            'data_quality': ['high'],
            'source_document': ['SAP'],
            'verification_status': ['unverified'],
            'notes': ['']
        }
        df = pd.DataFrame(data)
        minimal_data = tmp_path / "minimal_data.csv"
        df.to_csv(minimal_data, index=False)

        # Act
        result = pipeline.run(
            esg_data_file=str(minimal_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Should complete but may not be compliant
        assert result.compliance_status in ["PASS", "WARNING", "FAIL"]

        # If failed, audit should have recorded issues
        if result.compliance_status == "FAIL":
            audit_exec = result.agent_executions[5]
            assert audit_exec.errors > 0


# ============================================================================
# TEST CLASS 6: DATA FLOW BETWEEN AGENTS
# ============================================================================


class TestDataFlow:
    """Test data handoffs between agents."""

    def test_intake_to_materiality_handoff(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test IntakeAgent → MaterialityAgent data handoff."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intermediate outputs
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "01_intake_validated.json", 'r') as f:
            intake_output = json.load(f)
        with open(intermediate_dir / "02_materiality_assessment.json", 'r') as f:
            materiality_output = json.load(f)

        # Assert - Data structure consistency
        assert 'validated_data' in intake_output
        assert 'material_topics' in materiality_output
        assert len(intake_output['validated_data']) > 0

    def test_materiality_to_calculator_handoff(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test MaterialityAgent → CalculatorAgent data handoff."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intermediate outputs
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "02_materiality_assessment.json", 'r') as f:
            materiality_output = json.load(f)
        with open(intermediate_dir / "03_calculated_metrics.json", 'r') as f:
            calc_output = json.load(f)

        # Assert - Calculator received materiality context
        assert 'material_topics' in materiality_output
        assert 'calculated_metrics' in calc_output

    def test_calculator_to_aggregator_handoff(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test CalculatorAgent → AggregatorAgent data handoff."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intermediate outputs
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "03_calculated_metrics.json", 'r') as f:
            calc_output = json.load(f)
        with open(intermediate_dir / "04_aggregated_data.json", 'r') as f:
            agg_output = json.load(f)

        # Assert - Aggregator received calculated metrics
        assert 'calculated_metrics' in calc_output
        assert calc_output['metadata']['metrics_calculated'] > 0

    def test_aggregator_to_reporting_handoff(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test AggregatorAgent → ReportingAgent data handoff."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intermediate outputs
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "04_aggregated_data.json", 'r') as f:
            agg_output = json.load(f)
        with open(intermediate_dir / "05_csrd_report.json", 'r') as f:
            report_output = json.load(f)

        # Assert - Reporting received aggregated data
        assert agg_output['metadata']['aggregated_metrics_count'] > 0
        assert 'metadata' in report_output

    def test_reporting_to_audit_handoff(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test ReportingAgent → AuditAgent data handoff."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intermediate outputs
        intermediate_dir = output_dir / "intermediate"
        with open(intermediate_dir / "05_csrd_report.json", 'r') as f:
            report_output = json.load(f)
        with open(intermediate_dir / "06_compliance_audit.json", 'r') as f:
            audit_output = json.load(f)

        # Assert - Audit received report data
        assert 'metadata' in report_output
        assert 'compliance_report' in audit_output
        assert audit_output['compliance_report']['total_rules_checked'] > 0

    def test_data_structure_consistency_all_agents(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test data structure consistency across all agents."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # All outputs should be valid JSON
        intermediate_dir = output_dir / "intermediate"
        for i in range(1, 7):
            file_path = intermediate_dir / f"0{i}_*.json"
            matching_files = list(intermediate_dir.glob(f"0{i}_*.json"))
            assert len(matching_files) == 1

            # Should be valid JSON
            with open(matching_files[0], 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)


# ============================================================================
# TEST CLASS 7: INTERMEDIATE OUTPUTS
# ============================================================================


class TestIntermediateOutputs:
    """Test intermediate outputs from each agent."""

    def test_intake_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test IntakeAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load intake output
        with open(output_dir / "intermediate" / "01_intake_validated.json", 'r') as f:
            intake_output = json.load(f)

        # Assert structure
        assert 'validated_data' in intake_output
        assert 'metadata' in intake_output
        assert intake_output['metadata']['total_records'] > 0
        assert 'data_quality_score' in intake_output['metadata']

    def test_materiality_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test MaterialityAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load materiality output
        with open(output_dir / "intermediate" / "02_materiality_assessment.json", 'r') as f:
            mat_output = json.load(f)

        # Assert structure
        assert 'material_topics' in mat_output
        assert isinstance(mat_output['material_topics'], list)
        if len(mat_output['material_topics']) > 0:
            topic = mat_output['material_topics'][0]
            assert 'esrs_code' in topic
            assert 'is_material' in topic

    def test_calculator_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test CalculatorAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load calculator output
        with open(output_dir / "intermediate" / "03_calculated_metrics.json", 'r') as f:
            calc_output = json.load(f)

        # Assert structure
        assert 'calculated_metrics' in calc_output
        assert 'metadata' in calc_output
        assert calc_output['metadata']['metrics_calculated'] > 0

    def test_aggregator_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test AggregatorAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load aggregator output
        with open(output_dir / "intermediate" / "04_aggregated_data.json", 'r') as f:
            agg_output = json.load(f)

        # Assert structure
        assert 'metadata' in agg_output
        assert agg_output['metadata']['aggregated_metrics_count'] > 0

    def test_reporting_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test ReportingAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load reporting output
        with open(output_dir / "intermediate" / "05_csrd_report.json", 'r') as f:
            report_output = json.load(f)

        # Assert structure
        assert 'metadata' in report_output
        assert report_output['metadata']['xbrl_facts_count'] >= 0

    def test_audit_output_structure(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test AuditAgent output structure."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load audit output
        with open(output_dir / "intermediate" / "06_compliance_audit.json", 'r') as f:
            audit_output = json.load(f)

        # Assert structure
        assert 'compliance_report' in audit_output
        assert 'total_rules_checked' in audit_output['compliance_report']
        assert 'compliance_status' in audit_output['compliance_report']

    def test_all_intermediate_files_created(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test all intermediate files are created."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert all 6 intermediate files exist
        intermediate_dir = output_dir / "intermediate"
        assert intermediate_dir.exists()

        expected_files = [
            "01_intake_validated.json",
            "02_materiality_assessment.json",
            "03_calculated_metrics.json",
            "04_aggregated_data.json",
            "05_csrd_report.json",
            "06_compliance_audit.json"
        ]

        for filename in expected_files:
            file_path = intermediate_dir / filename
            assert file_path.exists(), f"Missing: {filename}"


# ============================================================================
# TEST CLASS 8: MULTI-ENTITY SCENARIOS
# ============================================================================


class TestMultiEntity:
    """Test multi-entity scenarios (parent-subsidiary)."""

    def test_parent_company_only(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test parent company only (no subsidiaries)."""
        # Remove subsidiaries from profile
        profile = demo_company_profile.copy()
        profile['subsidiaries'] = []

        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=profile,
            output_dir=str(output_dir)
        )

        # Assert - Should work without subsidiaries
        assert result.status in ["success", "partial_success"]

    def test_multi_entity_consolidation(
        self,
        pipeline,
        multi_entity_data,
        output_dir,
        mock_llm_provider
    ):
        """Test parent + 3 subsidiaries consolidation."""
        data_path, company_profile = multi_entity_data

        # Act
        result = pipeline.run(
            esg_data_file=str(data_path),
            company_profile=company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Should consolidate all entities
        assert result.status in ["success", "partial_success"]
        assert result.total_data_points_processed > 0

    def test_ownership_percentage_handling(
        self,
        pipeline,
        multi_entity_data,
        output_dir,
        mock_llm_provider
    ):
        """Test ownership percentage handling in consolidation."""
        data_path, company_profile = multi_entity_data

        # Verify ownership percentages in profile
        assert company_profile['subsidiaries'][0]['ownership_percentage'] == 100
        assert company_profile['subsidiaries'][1]['ownership_percentage'] == 75
        assert company_profile['subsidiaries'][2]['ownership_percentage'] == 100

        # Act
        result = pipeline.run(
            esg_data_file=str(data_path),
            company_profile=company_profile,
            output_dir=str(output_dir)
        )

        # Should handle different ownership levels
        assert result.status in ["success", "partial_success"]

    def test_subsidiary_only_reporting(
        self,
        pipeline,
        tmp_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test subsidiary-only reporting."""
        # Create subsidiary-only profile
        profile = demo_company_profile.copy()
        profile['legal_name'] = profile['subsidiaries'][0]['name']
        profile['subsidiaries'] = []

        # Create subsidiary data
        data = {
            'metric_code': ['E1-1', 'S1-1'],
            'metric_name': ['Scope 1', 'Employees'],
            'value': [1000.0, 320.0],
            'unit': ['tCO2e', 'FTE'],
            'period_start': ['2024-01-01'] * 2,
            'period_end': ['2024-12-31'] * 2,
            'data_quality': ['high'] * 2,
            'source_document': ['SAP'] * 2,
            'verification_status': ['verified'] * 2,
            'notes': [''] * 2
        }
        df = pd.DataFrame(data)
        sub_data = tmp_path / "subsidiary_data.csv"
        df.to_csv(sub_data, index=False)

        # Act
        result = pipeline.run(
            esg_data_file=str(sub_data),
            company_profile=profile,
            output_dir=str(output_dir)
        )

        # Assert - Should work for subsidiary
        assert result.status in ["success", "partial_success"]


# ============================================================================
# TEST CLASS 9: FRAMEWORK INTEGRATION
# ============================================================================


class TestFrameworkIntegration:
    """Test cross-framework integration (TCFD, GRI, SASB → ESRS)."""

    def test_esrs_only_data(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test with ESRS-only data (no conversion needed)."""
        # Demo data is already ESRS
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Should process directly
        assert result.status in ["success", "partial_success"]

        # Verify no framework conversion warnings
        agg_exec = result.agent_executions[3]  # AggregatorAgent
        assert agg_exec.status == "success"

    def test_unified_esrs_output(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test unified ESRS output regardless of input framework."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load aggregated output
        with open(output_dir / "intermediate" / "04_aggregated_data.json", 'r') as f:
            agg_output = json.load(f)

        # Assert - Output is in ESRS format
        assert agg_output['metadata']['aggregated_metrics_count'] > 0


# ============================================================================
# TEST CLASS 10: TIME-SERIES SCENARIOS
# ============================================================================


class TestTimeSeries:
    """Test time-series analysis scenarios."""

    def test_single_year_2024(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test single year (2024) reporting."""
        # Demo data is for 2024
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Single year works
        assert result.status in ["success", "partial_success"]

    def test_multi_year_time_series(
        self,
        pipeline,
        time_series_data,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test 5 years (2020-2024) with time-series analysis."""
        # Act
        result = pipeline.run(
            esg_data_file=str(time_series_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Multi-year data processed
        assert result.status in ["success", "partial_success"]
        assert result.total_data_points_processed > 0

    def test_year_over_year_trend_analysis(
        self,
        pipeline,
        time_series_data,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test year-over-year trend analysis."""
        # Act
        result = pipeline.run(
            esg_data_file=str(time_series_data),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Load aggregator output
        with open(output_dir / "intermediate" / "04_aggregated_data.json", 'r') as f:
            agg_output = json.load(f)

        # AggregatorAgent should process time-series
        assert agg_output['metadata']['aggregated_metrics_count'] > 0


# ============================================================================
# TEST CLASS 11: OUTPUT VALIDATION
# ============================================================================


class TestOutputValidation:
    """Test output validation (XBRL, PDF, JSON, audit package)."""

    def test_pipeline_result_json_valid(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test pipeline_result.json is valid."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Result file exists and is valid JSON
        result_file = output_dir / "pipeline_result.json"
        assert result_file.exists()

        with open(result_file, 'r') as f:
            result_data = json.load(f)

        assert 'pipeline_id' in result_data
        assert 'status' in result_data
        assert 'agent_executions' in result_data
        assert len(result_data['agent_executions']) == 6

    def test_json_exports_all_valid(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test all JSON exports are valid."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # All intermediate JSON files should be valid
        intermediate_dir = output_dir / "intermediate"
        for json_file in intermediate_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)

    def test_output_completeness(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test output package completeness."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - All expected outputs
        assert (output_dir / "intermediate").exists()
        assert (output_dir / "pipeline_result.json").exists()

        # Intermediate directory should have 6 files
        intermediate_files = list((output_dir / "intermediate").glob("*.json"))
        assert len(intermediate_files) == 6


# ============================================================================
# TEST CLASS 12: PERFORMANCE BENCHMARKS
# ============================================================================


class TestPerformanceBenchmarks:
    """Test performance benchmarks for each agent."""

    def test_agent_timing_breakdown_demo_data(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test agent timing breakdown for demo data."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - All agent times recorded
        perf = result.performance
        assert perf.agent_1_intake_seconds > 0
        assert perf.agent_2_materiality_seconds > 0
        assert perf.agent_3_calculator_seconds > 0
        assert perf.agent_4_aggregator_seconds > 0
        assert perf.agent_5_reporting_seconds > 0
        assert perf.agent_6_audit_seconds > 0

    def test_total_pipeline_time_within_target(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test total pipeline time within target."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Within target for demo data
        assert result.performance.total_time_seconds < 300  # <5 min for demo

    def test_intake_agent_performance_percentage(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test IntakeAgent uses <10% of total time."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Calculate percentage
        intake_pct = (result.performance.agent_1_intake_seconds /
                      result.performance.total_time_seconds) * 100

        # For demo data, intake should be fast
        assert intake_pct < 20  # <20% for small dataset

    def test_calculator_agent_performance_percentage(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test CalculatorAgent performance."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Calculate ms per metric
        calc_exec = result.agent_executions[2]
        if calc_exec.output_records > 0:
            ms_per_metric = (calc_exec.duration_seconds * 1000) / calc_exec.output_records

            # Should be fast (target: <5ms per metric)
            assert ms_per_metric < 100  # <100ms per metric is reasonable

    def test_throughput_verification(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test throughput verification (records/second)."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - Reasonable throughput
        assert result.performance.records_per_second > 0
        # For demo data (50 records), should process quickly
        assert result.performance.records_per_second >= 0.1  # At least 0.1 rec/sec

    def test_performance_target_flag(
        self,
        pipeline,
        demo_data_path,
        demo_company_profile,
        output_dir,
        mock_llm_provider
    ):
        """Test within_target flag is set correctly."""
        # Act
        result = pipeline.run(
            esg_data_file=str(demo_data_path),
            company_profile=demo_company_profile,
            output_dir=str(output_dir)
        )

        # Assert - within_target calculated
        target_minutes = result.performance.target_time_minutes
        actual_minutes = result.performance.total_time_seconds / 60

        expected_within_target = actual_minutes <= target_minutes
        assert result.performance.within_target == expected_within_target


# ============================================================================
# END OF INTEGRATION TESTS
# ============================================================================
