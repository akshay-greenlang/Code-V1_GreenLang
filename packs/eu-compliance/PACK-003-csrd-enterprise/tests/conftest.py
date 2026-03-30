# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Shared Test Fixtures
=========================================================

Provides reusable pytest fixtures for all PACK-003 test modules including
multi-tenant isolation, SSO/SAML, white-label branding, predictive analytics,
narrative generation, workflow builder, IoT streaming, carbon credits,
supply chain ESG, regulatory filing, API management, marketplace,
and enterprise workflow testing.

All fixtures are self-contained with no external dependencies.
Every external service is mocked via stub classes in this module.

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

from greenlang.schemas import utcnow

# ---------------------------------------------------------------------------
# Paths & sys.path setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent

_engines_dir = str(PACK_ROOT / "engines")
if _engines_dir not in sys.path:
    sys.path.insert(0, _engines_dir)

_pack_root_str = str(PACK_ROOT)
if _pack_root_str not in sys.path:
    sys.path.insert(0, _pack_root_str)

_config_dir = str(PACK_ROOT / "config")
if _config_dir not in sys.path:
    sys.path.insert(0, _config_dir)

PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
ENGINES_DIR = PACK_ROOT / "engines"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    elif hasattr(data, "model_dump"):
        raw = json.dumps(data.model_dump(mode="json"), sort_keys=True, default=str)
    else:
        raw = json.dumps(str(data), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _new_uuid() -> str:
    """Generate a new UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Pack YAML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML_PATH

@pytest.fixture(scope="session")
def pack_yaml_raw(pack_yaml_path) -> str:
    """Return the raw text content of pack.yaml."""
    return pack_yaml_path.read_text(encoding="utf-8")

@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary."""
    return yaml.safe_load(pack_yaml_raw)

# ---------------------------------------------------------------------------
# Preset / Sector YAML loading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def preset_files() -> Dict[str, Path]:
    """Return mapping of preset ID to file path."""
    result: Dict[str, Path] = {}
    if PRESETS_DIR.exists():
        for f in PRESETS_DIR.glob("*.yaml"):
            result[f.stem] = f
    return result

@pytest.fixture(scope="session")
def sector_files() -> Dict[str, Path]:
    """Return mapping of sector ID to file path."""
    result: Dict[str, Path] = {}
    if SECTORS_DIR.exists():
        for f in SECTORS_DIR.glob("*.yaml"):
            result[f.stem] = f
    return result

@pytest.fixture(scope="session")
def demo_config_path() -> Path:
    """Return path to demo configuration."""
    return DEMO_DIR / "demo_config.yaml"

@pytest.fixture(scope="session")
def demo_config(demo_config_path) -> Dict[str, Any]:
    """Return parsed demo configuration."""
    if demo_config_path.exists():
        return yaml.safe_load(demo_config_path.read_text(encoding="utf-8"))
    return {}

@pytest.fixture(scope="session")
def demo_tenant_profiles_path() -> Path:
    """Return path to demo tenant profiles."""
    return DEMO_DIR / "demo_tenant_profiles.json"

@pytest.fixture(scope="session")
def demo_tenant_profiles(demo_tenant_profiles_path) -> Any:
    """Return parsed demo tenant profiles."""
    if demo_tenant_profiles_path.exists():
        return json.loads(demo_tenant_profiles_path.read_text(encoding="utf-8"))
    return []

@pytest.fixture(scope="session")
def demo_iot_stream_path() -> Path:
    """Return path to demo IoT stream CSV."""
    return DEMO_DIR / "demo_iot_stream.csv"

# ---------------------------------------------------------------------------
# Enterprise PackConfig fixture (dict-based, no external deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_enterprise_config() -> Dict[str, Any]:
    """Create a full EnterprisePackConfig dict with all sub-configs.

    This is a comprehensive configuration that enables all enterprise
    features, suitable for unit testing of all PACK-003 functionality.
    """
    return {
        "metadata": {
            "name": "csrd-enterprise",
            "version": "1.0.0",
            "display_name": "CSRD Enterprise Pack",
            "description": "Enterprise-grade CSRD reporting with multi-tenant isolation",
            "category": "eu-compliance",
            "tier": "enterprise",
            "author": "GreenLang Platform Team",
            "license": "Proprietary",
            "min_platform_version": "2.0.0",
            "release_date": "2026-03-14",
            "support_tier": "enterprise-premium",
            "tags": [
                "csrd", "esrs", "multi-tenant", "predictive-analytics",
                "iot", "carbon-credits", "supply-chain-esg",
            ],
            "compliance_references": [
                {
                    "id": "CSRD",
                    "name": "Corporate Sustainability Reporting Directive",
                    "regulation": "Directive (EU) 2022/2464",
                    "effective_date": "2024-01-01",
                    "description": "EU sustainability reporting mandate",
                },
                {
                    "id": "ESRS",
                    "name": "European Sustainability Reporting Standards",
                    "regulation": "Delegated Regulation (EU) 2023/2772",
                    "effective_date": "2024-01-01",
                    "description": "ESRS Set 1",
                },
            ],
        },
        "enterprise": {
            "tenant": {
                "enabled": True,
                "isolation_level": "NAMESPACE",
                "max_tenants": 100,
                "resource_quotas": {
                    "max_agents": 200,
                    "max_storage_gb": 500,
                    "max_api_calls_per_day": 1000000,
                    "max_users": 500,
                },
                "cross_tenant_benchmarking": True,
                "tenant_provisioning_timeout_seconds": 300,
                "data_residency_enforcement": True,
            },
            "sso": {
                "saml_enabled": True,
                "oauth_enabled": True,
                "scim_enabled": True,
                "idp_metadata_url": "https://idp.example.com/metadata",
                "default_role": "viewer",
                "jit_provisioning": True,
                "allowed_domains": ["example.com", "corp.example.com"],
                "session_timeout_minutes": 480,
                "mfa_required": True,
            },
            "white_label": {
                "enabled": True,
                "logo_url": "https://assets.example.com/logo.png",
                "primary_color": "#1B5E20",
                "secondary_color": "#388E3C",
                "accent_color": "#4CAF50",
                "font_family": "Inter, sans-serif",
                "custom_domain": "sustainability.example.com",
                "powered_by_visible": False,
                "email_branding": True,
                "favicon_url": "https://assets.example.com/favicon.ico",
                "report_header_logo": "https://assets.example.com/report-logo.png",
            },
            "predictive": {
                "models_enabled": [
                    "emission_forecast", "anomaly_detection", "drift_monitor",
                ],
                "forecast_horizon_months": 12,
                "confidence_level": 0.95,
                "retrain_interval_days": 30,
                "anomaly_sensitivity": 0.85,
                "explainability_enabled": True,
                "feature_importance_tracking": True,
                "model_versioning": True,
                "auto_retrain_on_drift": True,
                "drift_psi_threshold": 0.2,
            },
            "narrative": {
                "languages": ["en", "de", "fr", "es", "it", "nl"],
                "tone": "regulatory",
                "fact_checking_enabled": True,
                "max_draft_tokens": 8000,
                "revision_tracking": True,
                "source_citation_required": True,
                "template_library_enabled": True,
                "human_review_required": True,
            },
            "workflow_builder": {
                "max_steps": 50,
                "allowed_step_types": [
                    "agent", "approval", "condition", "timer",
                    "notification", "data_transform", "quality_gate", "external_api",
                ],
                "template_sharing": True,
                "conditional_logic": True,
                "parallel_execution": True,
                "timer_steps": True,
                "human_in_loop": True,
                "max_custom_workflows": 100,
                "version_control": True,
            },
            "iot": {
                "enabled": True,
                "protocols": ["MQTT", "HTTP", "OPCUA", "MODBUS"],
                "aggregation_window_minutes": 15,
                "max_devices": 1000,
                "buffer_size_mb": 512,
                "anomaly_alerting": True,
                "data_retention_days": 365,
                "downsampling_enabled": True,
                "downsampling_after_days": 90,
            },
            "carbon_credit": {
                "enabled": True,
                "registries_enabled": ["VCS", "GoldStandard", "ACR", "CAR", "CDM", "Article6"],
                "auto_retirement": False,
                "vintage_tracking": True,
                "price_tracking": True,
                "net_zero_accounting": True,
                "additionality_verification": True,
                "buffer_pool_percent": 10.0,
            },
            "supply_chain": {
                "enabled": True,
                "max_tiers": 4,
                "questionnaire_frequency_months": 6,
                "scoring_weights": {
                    "environmental": 0.40,
                    "social": 0.35,
                    "governance": 0.25,
                },
                "risk_threshold": 0.6,
                "auto_dispatch": True,
                "critical_supplier_monitoring": True,
                "deforestation_screening": True,
            },
            "filing": {
                "enabled": True,
                "targets": ["ESAP", "national_registries"],
                "auto_submit": False,
                "validation_strictness": "strict",
                "deadline_buffer_days": 14,
                "amendment_tracking": True,
                "submission_receipt_archival": True,
            },
            "api_management": {
                "rate_limit_per_minute": 600,
                "rate_limit_per_day": 1000000,
                "api_key_rotation_days": 90,
                "graphql_enabled": True,
                "webhook_max_retries": 5,
                "burst_limit": 100,
                "api_versioning": True,
                "cors_enabled": True,
            },
            "marketplace": {
                "plugins_enabled": True,
                "max_plugins": 50,
                "auto_update": False,
                "sandbox_mode": True,
                "allowed_categories": [
                    "data_connector", "report_template", "calculation_engine",
                    "visualization", "notification", "integration",
                ],
                "plugin_review_required": True,
            },
            "consolidation": {
                "enabled": True,
                "max_subsidiaries": 500,
                "default_approach": "operational_control",
            },
            "approval": {
                "enabled": True,
            },
            "quality_gates": {
                "enabled": True,
                "overall_pass_threshold": 90.0,
            },
            "cross_framework": {
                "enabled": True,
                "enabled_frameworks": {
                    "cdp": True, "tcfd": True, "sbti": True, "taxonomy": True,
                    "gri": True, "sasb": True, "issb": True, "tnfd": True,
                },
            },
            "scenarios": {
                "enabled": True,
                "monte_carlo_enabled": True,
                "monte_carlo_iterations": 50000,
            },
            "benchmarking": {
                "enabled": True,
                "cross_tenant_benchmarking": True,
            },
            "stakeholder": {
                "enabled": True,
            },
            "regulatory": {
                "enabled": True,
            },
            "data_governance": {
                "enabled": True,
                "data_retention_years": 10,
            },
            "webhooks": {
                "enabled": True,
            },
            "assurance": {
                "enabled": True,
                "assurance_level": "reasonable",
            },
            "intensity_metrics": {
                "enabled": True,
            },
        },
    }

# ---------------------------------------------------------------------------
# Tenant Profile fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tenant_profile() -> Dict[str, Any]:
    """Create a sample tenant profile for testing."""
    return {
        "tenant_id": "tn-001-test",
        "tenant_name": "Acme Sustainability Corp",
        "tier": "enterprise",
        "isolation_level": "namespace",
        "admin_email": "admin@acme-sustain.com",
        "region": "eu-west-1",
        "features_enabled": [
            "basic_csrd", "multi_entity", "predictive_analytics",
            "iot_streaming", "carbon_credits", "white_label",
        ],
        "resource_quotas": {
            "max_agents": 200,
            "max_storage_gb": 500,
            "max_api_calls_per_day": 1000000,
            "max_users": 500,
            "max_subsidiaries": 250,
        },
        "status": "active",
        "health_score": 95.0,
        "created_at": "2026-01-15T00:00:00Z",
    }

# ---------------------------------------------------------------------------
# White Label / Brand Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_brand_config() -> Dict[str, Any]:
    """Create a WhiteLabelConfig with test colors and logo."""
    return {
        "enabled": True,
        "logo_url": "https://cdn.example.com/acme-logo.svg",
        "primary_color": "#003366",
        "secondary_color": "#0066CC",
        "accent_color": "#FF9900",
        "font_family": "Roboto, Arial, sans-serif",
        "custom_domain": "sustainability.acme.com",
        "custom_css": "",
        "powered_by_visible": False,
        "email_branding": True,
        "favicon_url": "https://cdn.example.com/acme-favicon.ico",
        "report_header_logo": "https://cdn.example.com/acme-report-logo.png",
        "dark_mode": {
            "primary_color": "#1A4D80",
            "secondary_color": "#3388DD",
            "accent_color": "#FFAA33",
            "background_color": "#121212",
            "text_color": "#E0E0E0",
        },
    }

# ---------------------------------------------------------------------------
# Forecast / Predictive Analytics fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_forecast_data() -> List[Dict[str, Any]]:
    """Create 5 years of monthly emission data for predictive testing."""
    data = []
    base_date = datetime(2021, 1, 1, tzinfo=timezone.utc)
    base_value = 1000.0
    for i in range(60):  # 5 years * 12 months
        month_date = base_date + timedelta(days=30 * i)
        seasonal_factor = 1.0 + 0.15 * ((i % 12) / 12.0)
        trend_factor = 1.0 - 0.02 * (i / 12.0)
        value = round(base_value * seasonal_factor * trend_factor, 2)
        data.append({
            "date": month_date.strftime("%Y-%m-%d"),
            "value": value,
            "unit": "tCO2e",
            "scope": "scope_1",
            "source": "stationary_combustion",
            "confidence": 0.95,
        })
    return data

# ---------------------------------------------------------------------------
# Narrative fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_narrative_data() -> Dict[str, Any]:
    """Create ESRS data points for narrative generation testing."""
    return {
        "company_name": "GlobalTech Industries AG",
        "reporting_year": 2025,
        "esrs_standard": "E1",
        "disclosure_id": "E1-6",
        "disclosure_name": "Gross GHG Emissions",
        "data_points": {
            "scope_1_total_tco2e": 45230.5,
            "scope_2_location_tco2e": 28100.0,
            "scope_2_market_tco2e": 15400.0,
            "scope_3_total_tco2e": 312000.0,
            "base_year": 2020,
            "base_year_scope_1": 52000.0,
            "reduction_pct_from_base": 13.0,
            "target_2030_reduction_pct": 42.0,
            "target_2050": "net_zero",
        },
        "methodology": "GHG Protocol Corporate Standard",
        "verification_status": "limited_assurance",
        "sources": [
            {"id": "EF-001", "name": "ecoinvent 3.10", "type": "emission_factor"},
            {"id": "GRID-DE", "name": "IEA 2024 Grid Factor - Germany", "type": "grid_factor"},
        ],
    }

# ---------------------------------------------------------------------------
# Workflow Definition fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow_definition() -> Dict[str, Any]:
    """Create a custom workflow with 10 steps for testing."""
    return {
        "workflow_id": "wf-custom-001",
        "name": "Custom ESG Data Pipeline",
        "description": "Custom workflow for quarterly ESG data processing",
        "version": "1.0.0",
        "created_by": "admin@example.com",
        "steps": [
            {
                "step_id": "step-1",
                "type": "agent",
                "agent_id": "AGENT-DATA-002",
                "name": "Import Excel Data",
                "inputs": {"file_path": "/data/quarterly_data.xlsx"},
                "outputs": ["normalized_data"],
                "next_steps": ["step-2"],
            },
            {
                "step_id": "step-2",
                "type": "quality_gate",
                "name": "Data Quality Check",
                "config": {"threshold": 90.0},
                "inputs": ["normalized_data"],
                "outputs": ["quality_report"],
                "next_steps": ["step-3"],
            },
            {
                "step_id": "step-3",
                "type": "condition",
                "name": "Quality Gate Pass?",
                "config": {
                    "condition": "quality_report.score >= 90",
                    "true_branch": "step-4",
                    "false_branch": "step-10",
                },
                "inputs": ["quality_report"],
                "outputs": [],
                "next_steps": ["step-4", "step-10"],
            },
            {
                "step_id": "step-4",
                "type": "agent",
                "agent_id": "AGENT-MRV-001",
                "name": "Calculate Scope 1 Emissions",
                "inputs": ["normalized_data"],
                "outputs": ["scope1_result"],
                "next_steps": ["step-5"],
            },
            {
                "step_id": "step-5",
                "type": "agent",
                "agent_id": "AGENT-MRV-009",
                "name": "Calculate Scope 2 Emissions",
                "inputs": ["normalized_data"],
                "outputs": ["scope2_result"],
                "next_steps": ["step-6"],
            },
            {
                "step_id": "step-6",
                "type": "data_transform",
                "name": "Consolidate Results",
                "config": {"operation": "merge"},
                "inputs": ["scope1_result", "scope2_result"],
                "outputs": ["consolidated_result"],
                "next_steps": ["step-7"],
            },
            {
                "step_id": "step-7",
                "type": "approval",
                "name": "Manager Approval",
                "config": {"approver_role": "reviewer", "timeout_hours": 48},
                "inputs": ["consolidated_result"],
                "outputs": ["approval_status"],
                "next_steps": ["step-8"],
            },
            {
                "step_id": "step-8",
                "type": "timer",
                "name": "Wait for Approval Period",
                "config": {"delay_seconds": 0, "description": "Cooling period"},
                "inputs": [],
                "outputs": [],
                "next_steps": ["step-9"],
            },
            {
                "step_id": "step-9",
                "type": "notification",
                "name": "Notify Stakeholders",
                "config": {
                    "channel": "email",
                    "template": "quarterly_update",
                    "recipients": ["finance@example.com"],
                },
                "inputs": ["consolidated_result"],
                "outputs": ["notification_status"],
                "next_steps": [],
            },
            {
                "step_id": "step-10",
                "type": "notification",
                "name": "Alert Data Team",
                "config": {
                    "channel": "slack",
                    "template": "data_quality_failure",
                    "recipients": ["#data-quality"],
                },
                "inputs": ["quality_report"],
                "outputs": ["alert_status"],
                "next_steps": [],
            },
        ],
        "max_execution_time_seconds": 3600,
        "retry_policy": {"max_retries": 2, "backoff_seconds": 30},
    }

# ---------------------------------------------------------------------------
# IoT Sensor Reading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_iot_readings() -> List[Dict[str, Any]]:
    """Create 100 IoT sensor readings for testing."""
    readings = []
    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    device_types = [
        ("energy_meter", "kWh", 100.0, 500.0),
        ("gas_flow", "m3/h", 5.0, 50.0),
        ("temperature", "celsius", 15.0, 35.0),
        ("water_meter", "m3", 0.5, 10.0),
    ]
    for i in range(100):
        dtype, unit, min_val, max_val = device_types[i % len(device_types)]
        device_id = f"device-{dtype}-{(i // len(device_types)) + 1:03d}"
        value = round(min_val + (max_val - min_val) * ((i * 7 + 3) % 100) / 100.0, 3)
        quality = "good" if (i % 10) != 7 else "suspect"
        readings.append({
            "reading_id": f"rdg-{i + 1:04d}",
            "device_id": device_id,
            "device_type": dtype,
            "timestamp": (base_time + timedelta(minutes=15 * i)).isoformat(),
            "value": value,
            "unit": unit,
            "quality_flag": quality,
            "facility_id": f"facility-{(i % 5) + 1:02d}",
            "protocol": "MQTT" if i % 2 == 0 else "HTTP",
            "calibration_status": "valid",
            "battery_pct": 85.0 + (i % 15),
        })
    return readings

# ---------------------------------------------------------------------------
# Carbon Credit fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_carbon_credits() -> List[Dict[str, Any]]:
    """Create a portfolio of 20 carbon credits across registries."""
    registries = ["VCS", "GoldStandard", "ACR", "CAR", "CDM", "Article6"]
    project_types = [
        "reforestation", "solar", "wind", "cookstove",
        "methane_capture", "REDD+", "blue_carbon", "DAC",
    ]
    credits = []
    for i in range(20):
        registry = registries[i % len(registries)]
        project = project_types[i % len(project_types)]
        vintage = 2022 + (i % 4)
        quantity = (i + 1) * 500
        price_per_tonne = round(5.0 + i * 1.5, 2)
        credits.append({
            "credit_id": f"CC-{registry}-{i + 1:04d}",
            "registry": registry,
            "project_type": project,
            "project_name": f"{project.replace('_', ' ').title()} Project {i + 1}",
            "project_country": ["BR", "IN", "KE", "ID", "CO", "MX"][i % 6],
            "vintage_year": vintage,
            "quantity_tco2e": quantity,
            "price_per_tonne_usd": price_per_tonne,
            "total_value_usd": round(quantity * price_per_tonne, 2),
            "status": "active" if i < 16 else "retired",
            "retirement_date": None if i < 16 else "2025-12-31",
            "additionality_score": round(0.6 + (i % 10) * 0.04, 2),
            "permanence_risk": "low" if i % 3 == 0 else "medium",
            "verification_standard": "VCS v4.0" if registry == "VCS" else f"{registry} Standard",
            "serial_numbers": f"{registry}-{vintage}-{i * 1000 + 1:08d}-{i * 1000 + quantity:08d}",
            "provenance_hash": _compute_hash({"id": f"CC-{registry}-{i + 1:04d}", "vintage": vintage}),
        })
    return credits

# ---------------------------------------------------------------------------
# Supplier Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_supplier_data() -> List[Dict[str, Any]]:
    """Create 15 suppliers with ESG scores for testing."""
    suppliers = []
    sectors = [
        "manufacturing", "logistics", "chemicals", "electronics",
        "agriculture", "textiles", "mining", "energy",
    ]
    countries = ["DE", "CN", "IN", "US", "BR", "JP", "VN", "TR"]
    for i in range(15):
        env_score = round(0.3 + (i * 13 % 70) / 100.0, 2)
        soc_score = round(0.4 + (i * 17 % 60) / 100.0, 2)
        gov_score = round(0.5 + (i * 11 % 50) / 100.0, 2)
        composite = round(env_score * 0.4 + soc_score * 0.35 + gov_score * 0.25, 4)
        risk_tier = "high" if composite < 0.5 else ("medium" if composite < 0.7 else "low")
        suppliers.append({
            "supplier_id": f"SUP-{i + 1:04d}",
            "name": f"Supplier {chr(65 + i)} Corp",
            "country": countries[i % len(countries)],
            "sector": sectors[i % len(sectors)],
            "tier": (i % 3) + 1,
            "annual_spend_eur": round((i + 1) * 500000, 2),
            "environmental_score": env_score,
            "social_score": soc_score,
            "governance_score": gov_score,
            "composite_esg_score": composite,
            "risk_tier": risk_tier,
            "last_assessment_date": "2025-06-15",
            "questionnaire_status": "completed" if i < 12 else "pending",
            "scope3_contribution_pct": round(2.0 + i * 0.8, 1),
            "deforestation_risk": "low" if i % 4 != 0 else "medium",
        })
    return suppliers

# ---------------------------------------------------------------------------
# Filing Package fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_filing_package() -> Dict[str, Any]:
    """Create ESEF/iXBRL filing data for testing."""
    return {
        "filing_id": "FIL-2025-001",
        "company_lei": "529900DEMO9876543210",
        "company_name": "GlobalTech Industries AG",
        "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "filing_target": "ESAP",
        "format": "inline_xbrl",
        "taxonomy_version": "ESRS_2023",
        "language": "en",
        "status": "prepared",
        "deadline": "2026-04-30",
        "validation_results": {
            "total_checks": 150,
            "passed": 148,
            "warnings": 2,
            "errors": 0,
            "validation_score": 98.67,
        },
        "content_sections": [
            {"section": "E1", "disclosure_count": 12, "status": "complete"},
            {"section": "E2", "disclosure_count": 8, "status": "complete"},
            {"section": "E3", "disclosure_count": 6, "status": "complete"},
            {"section": "S1", "disclosure_count": 15, "status": "complete"},
            {"section": "G1", "disclosure_count": 10, "status": "complete"},
        ],
        "xbrl_tags_count": 1250,
        "file_size_mb": 4.2,
        "submission_receipt": None,
        "amendment_history": [],
        "provenance_hash": _compute_hash({"filing_id": "FIL-2025-001"}),
    }

# ---------------------------------------------------------------------------
# API Key fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_api_keys() -> List[Dict[str, Any]]:
    """Create API key fixtures for testing."""
    return [
        {
            "key_id": "ak-001",
            "key_prefix": "gl_ent_",
            "tenant_id": "tn-001-test",
            "name": "Production API Key",
            "scopes": ["read:emissions", "write:emissions", "read:reports"],
            "rate_limit_per_minute": 600,
            "rate_limit_per_day": 1000000,
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-04-01T00:00:00Z",
            "last_used_at": "2026-03-10T12:00:00Z",
            "status": "active",
            "usage_count_today": 1523,
        },
        {
            "key_id": "ak-002",
            "key_prefix": "gl_ent_",
            "tenant_id": "tn-001-test",
            "name": "Read-Only API Key",
            "scopes": ["read:emissions", "read:reports"],
            "rate_limit_per_minute": 300,
            "rate_limit_per_day": 500000,
            "created_at": "2026-02-01T00:00:00Z",
            "expires_at": "2026-05-01T00:00:00Z",
            "last_used_at": "2026-03-12T08:30:00Z",
            "status": "active",
            "usage_count_today": 423,
        },
        {
            "key_id": "ak-003",
            "key_prefix": "gl_ent_",
            "tenant_id": "tn-002-test",
            "name": "Expired Key",
            "scopes": ["read:emissions"],
            "rate_limit_per_minute": 100,
            "rate_limit_per_day": 100000,
            "created_at": "2025-01-01T00:00:00Z",
            "expires_at": "2025-12-31T00:00:00Z",
            "last_used_at": "2025-11-20T15:00:00Z",
            "status": "expired",
            "usage_count_today": 0,
        },
    ]

# ---------------------------------------------------------------------------
# Audit Engagement fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_audit_engagement() -> Dict[str, Any]:
    """Create audit engagement data for testing."""
    return {
        "engagement_id": "AUD-2025-001",
        "auditor_firm": "KPMG Sustainability Assurance",
        "lead_auditor": "Dr. Anna Mueller",
        "engagement_type": "limited_assurance",
        "standard": "ISAE 3000 (Revised)",
        "scope": [
            "scope_1_emissions", "scope_2_emissions",
            "scope_3_material_categories", "esrs_e1_disclosures",
        ],
        "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "status": "in_progress",
        "phases": [
            {
                "phase": "planning",
                "status": "completed",
                "start_date": "2026-01-15",
                "end_date": "2026-01-31",
            },
            {
                "phase": "fieldwork",
                "status": "in_progress",
                "start_date": "2026-02-01",
                "end_date": "2026-03-15",
            },
            {
                "phase": "reporting",
                "status": "pending",
                "start_date": "2026-03-16",
                "end_date": "2026-04-15",
            },
        ],
        "evidence_packages": [
            {"package_id": "EP-001", "category": "scope_1", "document_count": 45, "status": "submitted"},
            {"package_id": "EP-002", "category": "scope_2", "document_count": 32, "status": "submitted"},
            {"package_id": "EP-003", "category": "scope_3", "document_count": 78, "status": "pending"},
        ],
        "findings": [
            {
                "finding_id": "F-001",
                "severity": "observation",
                "description": "Emission factor source documentation could be strengthened",
                "status": "open",
            },
        ],
        "portal_access_url": "https://auditor.greenlang.io/engagement/AUD-2025-001",
        "provenance_hash": _compute_hash({"engagement_id": "AUD-2025-001"}),
    }

# ---------------------------------------------------------------------------
# Stub classes for external dependencies
# ---------------------------------------------------------------------------

class StubTenantManager:
    """Stub for TenantManager that operates in-memory."""

    def __init__(self):
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def create_tenant(self, config: Dict[str, Any]) -> Dict[str, Any]:
        tenant_id = config.get("tenant_id", _new_uuid())
        tenant = {
            "tenant_id": tenant_id,
            "name": config.get("tenant_name", "Test Tenant"),
            "tier": config.get("tier", "starter"),
            "status": "active",
            "created_at": utcnow().isoformat(),
            "provenance_hash": _compute_hash(config),
        }
        self.tenants[tenant_id] = tenant
        self.audit_log.append({"event": "created", "tenant_id": tenant_id})
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        return self.tenants.get(tenant_id)

    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if tenant_id not in self.tenants:
            raise KeyError(f"Tenant {tenant_id} not found")
        self.tenants[tenant_id].update(updates)
        self.audit_log.append({"event": "updated", "tenant_id": tenant_id})
        return self.tenants[tenant_id]

    def list_tenants(self) -> List[Dict[str, Any]]:
        return list(self.tenants.values())

    def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            self.audit_log.append({"event": "deleted", "tenant_id": tenant_id})
            return True
        return False

class StubSAMLProvider:
    """Stub for SAML identity provider."""

    def __init__(self):
        self.configured = False
        self.metadata_url = ""
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def configure(self, metadata_url: str, allowed_domains: List[str] = None) -> Dict[str, Any]:
        self.configured = True
        self.metadata_url = metadata_url
        return {
            "status": "configured",
            "idp_entity_id": "https://idp.example.com",
            "sso_url": "https://idp.example.com/sso",
            "slo_url": "https://idp.example.com/slo",
            "certificate": "MOCK_CERT_DATA",
        }

    def authenticate(self, saml_response: str) -> Dict[str, Any]:
        return {
            "authenticated": True,
            "user_id": "user-001",
            "email": "user@example.com",
            "roles": ["viewer"],
            "session_id": _new_uuid(),
            "attributes": {"department": "sustainability", "location": "DE"},
        }

    def provision_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "user_id": user_data.get("user_id", _new_uuid()),
            "email": user_data.get("email", "new@example.com"),
            "role": user_data.get("role", "viewer"),
            "provisioned": True,
            "jit": True,
        }

class StubGraphQLSchema:
    """Stub for GraphQL schema manager."""

    def __init__(self):
        self.types: Dict[str, Dict[str, Any]] = {}
        self.queries: Dict[str, Any] = {}

    def register_type(self, type_name: str, fields: Dict[str, str]) -> Dict[str, Any]:
        self.types[type_name] = fields
        return {"type": type_name, "fields": list(fields.keys()), "registered": True}

    def resolve_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "data": {
                "emissions": {
                    "scope1_total": 45230.5,
                    "scope2_total": 28100.0,
                    "unit": "tCO2e",
                    "reporting_year": 2025,
                }
            },
            "errors": None,
            "extensions": {"query_complexity": 12, "execution_ms": 45},
        }

    def check_field_auth(self, field_path: str, user_roles: List[str]) -> bool:
        protected_prefixes = {"financialData", "internalNotes", "auditFindings"}
        parts = field_path.split(".")
        for part in parts:
            if part in protected_prefixes:
                return "admin" in user_roles or "auditor" in user_roles
        return True

class StubMLModel:
    """Stub for ML predictive models."""

    def __init__(self, model_type: str = "emission_forecast"):
        self.model_type = model_type
        self.version = "1.0.0"
        self.trained = True
        self.drift_score = 0.05

    def predict(self, data: List[Dict[str, Any]], horizon: int = 12) -> Dict[str, Any]:
        predictions = []
        base_value = data[-1]["value"] if data else 100.0
        for i in range(horizon):
            trend = 1.0 - 0.005 * i
            predictions.append({
                "month": i + 1,
                "predicted_value": round(base_value * trend, 2),
                "lower_bound": round(base_value * trend * 0.9, 2),
                "upper_bound": round(base_value * trend * 1.1, 2),
                "confidence": 0.95,
            })
        return {
            "model_type": self.model_type,
            "model_version": self.version,
            "horizon_months": horizon,
            "predictions": predictions,
            "r_squared": 0.92,
            "mae": 12.5,
            "rmse": 18.3,
            "provenance_hash": _compute_hash({"model": self.model_type, "horizon": horizon}),
        }

    def detect_anomalies(self, data: List[Dict[str, Any]], sensitivity: float = 0.85) -> List[Dict[str, Any]]:
        anomalies = []
        if len(data) < 5:
            return anomalies
        values = [d.get("value", 0) for d in data]
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        threshold = mean_val + 2 * std_val * (1.0 / sensitivity)
        for i, d in enumerate(data):
            if d.get("value", 0) > threshold:
                anomalies.append({
                    "index": i,
                    "value": d["value"],
                    "expected_range": [round(mean_val - 2 * std_val, 2), round(threshold, 2)],
                    "anomaly_score": round((d["value"] - threshold) / std_val, 4),
                    "type": "spike",
                })
        return anomalies

    def check_drift(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "drift_detected": self.drift_score > 0.2,
            "psi_score": self.drift_score,
            "threshold": 0.2,
            "recommendation": "no_action" if self.drift_score <= 0.2 else "retrain",
        }

    def explain(self, prediction_idx: int = 0) -> Dict[str, Any]:
        return {
            "prediction_index": prediction_idx,
            "feature_importance": {
                "historical_trend": 0.35,
                "seasonal_pattern": 0.25,
                "production_volume": 0.20,
                "weather_data": 0.10,
                "energy_price": 0.10,
            },
            "method": "SHAP",
        }

class StubAuditorPortal:
    """Stub for auditor collaboration portal."""

    def __init__(self):
        self.engagements: Dict[str, Dict[str, Any]] = {}
        self.evidence_packages: Dict[str, List[Dict[str, Any]]] = {}
        self.findings: List[Dict[str, Any]] = []

    def create_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        eng_id = data.get("engagement_id", f"AUD-{_new_uuid()[:8]}")
        # Copy data but override status to 'active' for new engagements
        engagement_data = dict(data)
        engagement_data.pop("status", None)
        engagement = {
            "engagement_id": eng_id,
            "status": "active",
            "created_at": utcnow().isoformat(),
            "portal_url": f"https://auditor.greenlang.io/engagement/{eng_id}",
            **engagement_data,
        }
        self.engagements[eng_id] = engagement
        return engagement

    def package_evidence(self, engagement_id: str, category: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        package = {
            "package_id": f"EP-{_new_uuid()[:8]}",
            "engagement_id": engagement_id,
            "category": category,
            "document_count": len(documents),
            "status": "submitted",
            "provenance_hash": _compute_hash({"engagement_id": engagement_id, "category": category}),
        }
        self.evidence_packages.setdefault(engagement_id, []).append(package)
        return package

    def submit_finding(self, engagement_id: str, finding: Dict[str, Any]) -> Dict[str, Any]:
        finding_record = {
            "finding_id": f"F-{len(self.findings) + 1:03d}",
            "engagement_id": engagement_id,
            "status": "open",
            **finding,
        }
        self.findings.append(finding_record)
        return finding_record

    def get_opinion(self, engagement_id: str) -> Dict[str, Any]:
        return {
            "engagement_id": engagement_id,
            "opinion_type": "limited_assurance",
            "conclusion": "unmodified",
            "scope_coverage_pct": 95.0,
            "material_findings": 0,
            "observations": 2,
            "issued_date": utcnow().strftime("%Y-%m-%d"),
        }

class StubMarketplace:
    """Stub for plugin marketplace."""

    def __init__(self):
        self.plugins = [
            {
                "plugin_id": "plg-001",
                "name": "SAP S/4HANA Connector",
                "category": "data_connector",
                "version": "2.1.0",
                "author": "GreenLang Partners",
                "installed": False,
                "compatible": True,
            },
            {
                "plugin_id": "plg-002",
                "name": "Custom Dashboard Widgets",
                "category": "visualization",
                "version": "1.3.0",
                "author": "Community",
                "installed": True,
                "compatible": True,
            },
            {
                "plugin_id": "plg-003",
                "name": "TNFD Report Template",
                "category": "report_template",
                "version": "1.0.0",
                "author": "GreenLang",
                "installed": False,
                "compatible": True,
            },
        ]
        self.installed: List[str] = ["plg-002"]

    def discover(self, category: str = None) -> List[Dict[str, Any]]:
        if category:
            return [p for p in self.plugins if p["category"] == category]
        return list(self.plugins)

    def install(self, plugin_id: str) -> Dict[str, Any]:
        for p in self.plugins:
            if p["plugin_id"] == plugin_id:
                p["installed"] = True
                self.installed.append(plugin_id)
                return {"status": "installed", "plugin_id": plugin_id}
        raise KeyError(f"Plugin {plugin_id} not found")

    def check_compatibility(self, plugin_id: str) -> Dict[str, Any]:
        for p in self.plugins:
            if p["plugin_id"] == plugin_id:
                return {"compatible": p["compatible"], "plugin_id": plugin_id}
        return {"compatible": False, "plugin_id": plugin_id}

    def get_quotas(self, tenant_id: str) -> Dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "max_plugins": 50,
            "installed_count": len(self.installed),
            "remaining": 50 - len(self.installed),
        }

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tenant_manager() -> StubTenantManager:
    """Return a StubTenantManager instance."""
    return StubTenantManager()

@pytest.fixture
def mock_saml_provider() -> StubSAMLProvider:
    """Return a StubSAMLProvider instance."""
    return StubSAMLProvider()

@pytest.fixture
def mock_graphql_schema() -> StubGraphQLSchema:
    """Return a StubGraphQLSchema instance."""
    return StubGraphQLSchema()

@pytest.fixture
def mock_ml_models() -> Dict[str, StubMLModel]:
    """Return mock ML models for each model type."""
    return {
        "emission_forecast": StubMLModel("emission_forecast"),
        "anomaly_detection": StubMLModel("anomaly_detection"),
        "drift_monitor": StubMLModel("drift_monitor"),
    }

@pytest.fixture
def mock_auditor_portal() -> StubAuditorPortal:
    """Return a StubAuditorPortal instance."""
    return StubAuditorPortal()

@pytest.fixture
def mock_marketplace() -> StubMarketplace:
    """Return a StubMarketplace instance."""
    return StubMarketplace()

# ---------------------------------------------------------------------------
# MultiTenantEngine fixture (loaded from engines directory)
# ---------------------------------------------------------------------------

def _load_multi_tenant_module():
    """Load the multi_tenant_engine module and rebuild Pydantic models.

    Uses sys.modules registration so Pydantic's type resolution can find
    the enum types (TenantTier, IsolationLevel, etc.) in the module namespace.
    """
    module_name = "multi_tenant_engine"
    engine_path = str(ENGINES_DIR / "multi_tenant_engine.py")
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        # Register in sys.modules so Pydantic forward refs resolve
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        # Rebuild Pydantic models with module globals as namespace
        module_ns = vars(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if (
                isinstance(attr, type)
                and hasattr(attr, "model_rebuild")
                and hasattr(attr, "model_fields")
            ):
                try:
                    attr.model_rebuild(_types_namespace=module_ns)
                except Exception:
                    pass
        return mod
    return None

@pytest.fixture
def multi_tenant_engine():
    """Create a MultiTenantEngine instance for testing.

    Dynamically imports from the engines directory to avoid hyphenated
    package path issues.
    """
    mod = _load_multi_tenant_module()
    if mod is None:
        pytest.skip("multi_tenant_engine.py not found")
    return mod.MultiTenantEngine()

@pytest.fixture
def multi_tenant_module():
    """Return the multi_tenant_engine module for model access."""
    mod = _load_multi_tenant_module()
    if mod is None:
        pytest.skip("multi_tenant_engine.py not found")
    return mod

# ---------------------------------------------------------------------------
# PackConfig fixture (loaded from config directory)
# ---------------------------------------------------------------------------

@pytest.fixture
def pack_config_module():
    """Return the pack_config module for model access."""
    spec = importlib.util.spec_from_file_location(
        "pack_config",
        str(CONFIG_DIR / "pack_config.py"),
    )
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    pytest.skip("pack_config.py not found")

# ---------------------------------------------------------------------------
# Template render helper
# ---------------------------------------------------------------------------

def render_template_stub(
    template_id: str,
    data: Dict[str, Any],
    output_format: str = "markdown",
) -> Dict[str, Any]:
    """Stub template renderer for testing.

    Generates minimal output in the requested format without
    requiring Jinja2 templates on disk.
    """
    title = template_id.replace("_", " ").title()
    provenance_hash = _compute_hash({"template_id": template_id, "data": data})

    if output_format == "markdown":
        content = f"# {title}\n\n"
        for key, val in data.items():
            content += f"- **{key}**: {val}\n"
        content += f"\n---\nProvenance: {provenance_hash}\n"
    elif output_format == "html":
        content = f"<html><head><title>{title}</title></head><body>"
        content += f"<h1>{title}</h1><dl>"
        for key, val in data.items():
            content += f"<dt>{key}</dt><dd>{val}</dd>"
        content += f"</dl><footer>Provenance: {provenance_hash}</footer></body></html>"
    elif output_format == "json":
        content = json.dumps({
            "template_id": template_id,
            "title": title,
            "data": data,
            "provenance_hash": provenance_hash,
            "generated_at": utcnow().isoformat(),
        }, indent=2, default=str)
    else:
        content = f"{title}: {json.dumps(data, default=str)}"

    return {
        "template_id": template_id,
        "format": output_format,
        "content": content,
        "provenance_hash": provenance_hash,
        "generated_at": utcnow().isoformat(),
    }

@pytest.fixture
def template_renderer():
    """Return the template render stub function."""
    return render_template_stub

# ---------------------------------------------------------------------------
# ENTERPRISE TEMPLATE IDS (the 9 PACK-003 templates)
# ---------------------------------------------------------------------------

ENTERPRISE_TEMPLATE_IDS = [
    "emission_forecast_report",
    "iot_monitoring_dashboard",
    "carbon_credit_portfolio",
    "supply_chain_esg_scorecard",
    "regulatory_filing_package",
    "multi_language_narrative",
    "tenant_analytics_report",
    "enterprise_audit_package",
    "white_label_report",
]

@pytest.fixture
def enterprise_template_ids() -> List[str]:
    """Return the 9 enterprise template IDs."""
    return list(ENTERPRISE_TEMPLATE_IDS)

# ---------------------------------------------------------------------------
# ENTERPRISE WORKFLOW IDS (the 8 PACK-003 workflows)
# ---------------------------------------------------------------------------

ENTERPRISE_WORKFLOW_IDS = [
    "predictive_forecasting",
    "iot_continuous_monitoring",
    "carbon_credit_management",
    "supply_chain_esg_assessment",
    "regulatory_filing",
    "narrative_generation",
    "custom_workflow_management",
    "tenant_onboarding",
]

@pytest.fixture
def enterprise_workflow_ids() -> List[str]:
    """Return the 8 enterprise workflow IDs."""
    return list(ENTERPRISE_WORKFLOW_IDS)
