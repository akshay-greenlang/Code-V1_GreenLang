# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Shared Test Fixtures (conftest.py)
=======================================================================

Provides pytest fixtures for the PACK-018 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample data fixtures for all 8 engines
  - Demo configuration loading

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Engine module fixtures
  4. Workflow module fixtures
  5. Template module fixtures
  6. Integration module fixtures
  7. Domain-specific sample data

Engine Coverage (8 engines):
  - Claim Substantiation Engine
  - Evidence Chain Engine
  - Lifecycle Assessment Engine
  - Label Compliance Engine
  - Comparative Claims Engine
  - Greenwashing Detection Engine
  - Trader Obligation Engine
  - Green Claims Benchmark Engine

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import importlib
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"

# ---------------------------------------------------------------------------
# Engine file mapping: logical name -> file name on disk (8 engines)
# ---------------------------------------------------------------------------
ENGINE_FILES = {
    "claim_substantiation": "claim_substantiation_engine.py",
    "evidence_chain": "evidence_chain_engine.py",
    "lifecycle_assessment": "lifecycle_assessment_engine.py",
    "label_compliance": "label_compliance_engine.py",
    "comparative_claims": "comparative_claims_engine.py",
    "greenwashing_detection": "greenwashing_detection_engine.py",
    "trader_obligation": "trader_obligation_engine.py",
    "green_claims_benchmark": "green_claims_benchmark_engine.py",
}

# Engine class names expected in each module
ENGINE_CLASSES = {
    "claim_substantiation": "ClaimSubstantiationEngine",
    "evidence_chain": "EvidenceChainEngine",
    "lifecycle_assessment": "LifecycleAssessmentEngine",
    "label_compliance": "LabelComplianceEngine",
    "comparative_claims": "ComparativeClaimsEngine",
    "greenwashing_detection": "GreenwashingDetectionEngine",
    "trader_obligation": "TraderObligationEngine",
    "green_claims_benchmark": "GreenClaimsBenchmarkEngine",
}

# ---------------------------------------------------------------------------
# Workflow file mapping (8 workflows)
# ---------------------------------------------------------------------------
WORKFLOW_FILES = {
    "claim_assessment": "claim_assessment_workflow.py",
    "evidence_collection": "evidence_collection_workflow.py",
    "lifecycle_verification": "lifecycle_verification_workflow.py",
    "label_audit": "label_audit_workflow.py",
    "greenwashing_screening": "greenwashing_screening_workflow.py",
    "compliance_gap": "compliance_gap_workflow.py",
    "remediation_planning": "remediation_planning_workflow.py",
    "regulatory_submission": "regulatory_submission_workflow.py",
}

WORKFLOW_CLASSES = {
    "claim_assessment": "ClaimAssessmentWorkflow",
    "evidence_collection": "EvidenceCollectionWorkflow",
    "lifecycle_verification": "LifecycleVerificationWorkflow",
    "label_audit": "LabelAuditWorkflow",
    "greenwashing_screening": "GreenwashingScreeningWorkflow",
    "compliance_gap": "ComplianceGapWorkflow",
    "remediation_planning": "RemediationPlanningWorkflow",
    "regulatory_submission": "RegulatorySubmissionWorkflow",
}

# ---------------------------------------------------------------------------
# Template file mapping (8 templates)
# ---------------------------------------------------------------------------
TEMPLATE_FILES = {
    "claim_assessment_report": "claim_assessment_report.py",
    "evidence_dossier_report": "evidence_dossier_report.py",
    "lifecycle_summary_report": "lifecycle_summary_report.py",
    "label_compliance_report": "label_compliance_report.py",
    "greenwashing_risk_report": "greenwashing_risk_report.py",
    "compliance_gap_report": "compliance_gap_report.py",
    "green_claims_scorecard": "green_claims_scorecard.py",
    "regulatory_submission_report": "regulatory_submission_report.py",
}

TEMPLATE_CLASSES = {
    "claim_assessment_report": "ClaimAssessmentReportTemplate",
    "evidence_dossier_report": "EvidenceDossierReportTemplate",
    "lifecycle_summary_report": "LifecycleSummaryReportTemplate",
    "label_compliance_report": "LabelComplianceReportTemplate",
    "greenwashing_risk_report": "GreenwashingRiskReportTemplate",
    "compliance_gap_report": "ComplianceGapReportTemplate",
    "green_claims_scorecard": "GreenClaimsScorecardTemplate",
    "regulatory_submission_report": "RegulatorySubmissionReportTemplate",
}

# ---------------------------------------------------------------------------
# Integration file mapping (10 integrations)
# ---------------------------------------------------------------------------
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "csrd_bridge": "csrd_pack_bridge.py",
    "mrv_claims_bridge": "mrv_claims_bridge.py",
    "mrv_bridge": "mrv_claims_bridge.py",
    "data_claims_bridge": "data_claims_bridge.py",
    "data_bridge": "data_claims_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "pef_bridge": "pef_bridge.py",
    "dpp_bridge": "dpp_bridge.py",
    "ecgt_bridge": "ecgt_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "GreenClaimsOrchestrator",
    "csrd_pack_bridge": "CSRDPackBridge",
    "csrd_bridge": "CSRDPackBridge",
    "mrv_claims_bridge": "MRVClaimsBridge",
    "mrv_bridge": "MRVClaimsBridge",
    "data_claims_bridge": "DataClaimsBridge",
    "data_bridge": "DataClaimsBridge",
    "taxonomy_bridge": "TaxonomyBridge",
    "pef_bridge": "PEFBridge",
    "dpp_bridge": "DPPBridge",
    "ecgt_bridge": "ECGTBridge",
    "health_check": "GreenClaimsHealthCheck",
    "setup_wizard": "GreenClaimsSetupWizard",
}

# Preset names
PRESET_NAMES = [
    "manufacturing",
    "retail",
    "financial_services",
    "energy",
    "technology",
    "sme",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-018 as a Python package.  The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module.
        subdir: Subdirectory under PACK_ROOT.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the module file does not exist.
        ImportError: If the module cannot be loaded.
    """
    subdir_map = {
        "engines": ENGINES_DIR,
        "workflows": WORKFLOWS_DIR,
        "templates": TEMPLATES_DIR,
        "integrations": INTEGRATIONS_DIR,
        "config": CONFIG_DIR,
    }
    base_dir = subdir_map.get(subdir, PACK_ROOT / subdir)
    file_path = base_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Module file not found: {file_path}. "
            f"Ensure PACK-018 source files are present."
        )

    full_module_name = f"pack018_test.{subdir}.{module_name}"

    if full_module_name in sys.modules:
        return sys.modules[full_module_name]

    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(full_module_name, None)
        raise ImportError(
            f"Failed to load module {full_module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_workflow(workflow_key: str):
    """Load a workflow module by its logical key."""
    file_name = WORKFLOW_FILES[workflow_key]
    return _load_module(workflow_key, file_name, "workflows")


def _load_template(template_key: str):
    """Load a template module by its logical key."""
    file_name = TEMPLATE_FILES[template_key]
    return _load_module(template_key, file_name, "templates")


def _load_integration(integration_key: str):
    """Load an integration module by its logical key."""
    file_name = INTEGRATION_FILES[integration_key]
    return _load_module(integration_key, file_name, "integrations")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-018 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    if yaml is None:
        pytest.skip("pyyaml not installed")
    if not pack_yaml_path.exists():
        pytest.skip("pack.yaml not found")
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed to None"
    return data


@pytest.fixture(scope="session")
def demo_yaml_path() -> Path:
    """Return the absolute path to the demo configuration YAML."""
    return DEMO_DIR / "demo_config.yaml"


@pytest.fixture(scope="session")
def demo_yaml_data(demo_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the demo_config.yaml as a dictionary."""
    if yaml is None:
        pytest.skip("pyyaml not installed")
    if not demo_yaml_path.exists():
        pytest.skip(f"demo_config.yaml not found at {demo_yaml_path}")
    with open(demo_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "demo_config.yaml parsed to None"
    return data


# =============================================================================
# 2. Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def config_module():
    """Load and return the pack_config module."""
    return _load_config_module()


# =============================================================================
# 3. Engine Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def claim_substantiation_module():
    """Load the Claim Substantiation engine module."""
    return _load_engine("claim_substantiation")


@pytest.fixture(scope="session")
def evidence_chain_module():
    """Load the Evidence Chain engine module."""
    return _load_engine("evidence_chain")


@pytest.fixture(scope="session")
def lifecycle_assessment_module():
    """Load the Lifecycle Assessment engine module."""
    return _load_engine("lifecycle_assessment")


@pytest.fixture(scope="session")
def label_compliance_module():
    """Load the Label Compliance engine module."""
    return _load_engine("label_compliance")


@pytest.fixture(scope="session")
def comparative_claims_module():
    """Load the Comparative Claims engine module."""
    return _load_engine("comparative_claims")


@pytest.fixture(scope="session")
def greenwashing_detection_module():
    """Load the Greenwashing Detection engine module."""
    return _load_engine("greenwashing_detection")


@pytest.fixture(scope="session")
def trader_obligation_module():
    """Load the Trader Obligation engine module."""
    return _load_engine("trader_obligation")


@pytest.fixture(scope="session")
def green_claims_benchmark_module():
    """Load the Green Claims Benchmark engine module."""
    return _load_engine("green_claims_benchmark")


# =============================================================================
# 4. Workflow Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def claim_assessment_workflow_module():
    """Load the Claim Assessment workflow module."""
    return _load_workflow("claim_assessment")


@pytest.fixture(scope="session")
def evidence_collection_workflow_module():
    """Load the Evidence Collection workflow module."""
    return _load_workflow("evidence_collection")


# =============================================================================
# 5. Template Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def claim_assessment_report_module():
    """Load the Claim Assessment report template module."""
    return _load_template("claim_assessment_report")


@pytest.fixture(scope="session")
def greenwashing_risk_report_module():
    """Load the Greenwashing Risk report template module."""
    return _load_template("greenwashing_risk_report")


# =============================================================================
# 6. Domain-Specific Sample Data Fixtures
# =============================================================================


# ---------------------------------------------------------------------------
# 6.1  Environmental Claims Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_claims() -> List[Dict[str, Any]]:
    """Sample environmental claims for substantiation testing.

    Covers a range of claim types from low-risk to critical per
    EU Green Claims Directive Article 3.
    """
    return [
        {
            "claim_id": "CLM-001",
            "claim_text": "Our packaging is 100% recyclable",
            "claim_type": "RECYCLABLE",
            "product_or_org": "EcoClean All-Purpose Cleaner",
            "scope_description": "Product packaging only",
            "lifecycle_stages_covered": ["RAW_MATERIALS", "END_OF_LIFE"],
        },
        {
            "claim_id": "CLM-002",
            "claim_text": "Carbon neutral delivery service",
            "claim_type": "CARBON_NEUTRAL",
            "product_or_org": "Delivery operations",
            "scope_description": "Last-mile delivery Scope 1+2+3",
            "lifecycle_stages_covered": [
                "TRANSPORTATION",
                "DISTRIBUTION",
            ],
        },
        {
            "claim_id": "CLM-003",
            "claim_text": "Eco-friendly product",
            "claim_type": "ECO_FRIENDLY",
            "product_or_org": "GreenWash Detergent",
            "scope_description": "Overall product",
            "lifecycle_stages_covered": ["MANUFACTURING"],
        },
        {
            "claim_id": "CLM-004",
            "claim_text": "Made with 60% recycled plastic",
            "claim_type": "RECYCLABLE",
            "product_or_org": "ReBottle Water Bottle",
            "scope_description": "Product material composition",
            "lifecycle_stages_covered": ["RAW_MATERIALS", "MANUFACTURING"],
        },
        {
            "claim_id": "CLM-005",
            "claim_text": "We are a sustainable company",
            "claim_type": "SUSTAINABLE",
            "product_or_org": "Corporate level",
            "scope_description": "Entire organization",
            "lifecycle_stages_covered": [],
        },
    ]


@pytest.fixture
def sample_evidence() -> List[Dict[str, Any]]:
    """Sample evidence documents for claim substantiation.

    Includes certifications, LCA studies, test reports, and
    third-party verifications per Article 10.
    """
    return [
        {
            "evidence_id": "EV-001",
            "evidence_type": "CERTIFICATION",
            "source": "TUV Rheinland",
            "description": "Material recyclability certification ISO 18604",
            "is_third_party": True,
            "valid_from": "2024-01-01",
            "valid_to": "2026-12-31",
            "claim_ids": ["CLM-001"],
        },
        {
            "evidence_id": "EV-002",
            "evidence_type": "LCA_STUDY",
            "source": "Fraunhofer Institute",
            "description": "Full LCA per ISO 14044 for EcoClean product line",
            "is_third_party": True,
            "valid_from": "2024-06-01",
            "valid_to": "2027-05-31",
            "claim_ids": ["CLM-001", "CLM-004"],
        },
        {
            "evidence_id": "EV-003",
            "evidence_type": "AUDIT_REPORT",
            "source": "SGS",
            "description": "Carbon footprint verification for delivery operations",
            "is_third_party": True,
            "valid_from": "2025-01-01",
            "valid_to": "2025-12-31",
            "claim_ids": ["CLM-002"],
        },
        {
            "evidence_id": "EV-004",
            "evidence_type": "OFFSET_REGISTRY",
            "source": "Gold Standard",
            "description": "Carbon offset credits for residual emissions",
            "is_third_party": True,
            "valid_from": "2025-01-01",
            "valid_to": "2025-12-31",
            "claim_ids": ["CLM-002"],
        },
        {
            "evidence_id": "EV-005",
            "evidence_type": "MEASUREMENT",
            "source": "Internal lab",
            "description": "Recycled content measurement per ISO 14021",
            "is_third_party": False,
            "valid_from": "2025-03-01",
            "valid_to": "2026-02-28",
            "claim_ids": ["CLM-004"],
        },
    ]


# ---------------------------------------------------------------------------
# 6.2  Label Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_labels() -> List[Dict[str, Any]]:
    """Sample environmental labels for label compliance testing.

    Covers recognized EU labels, private schemes, and self-declared
    labels per Articles 6-9.
    """
    return [
        {
            "label_id": "LBL-001",
            "label_name": "EU Ecolabel",
            "label_type": "TYPE_I_ECOLABEL",
            "scheme_owner": "European Commission",
            "accredited": True,
            "scientific_basis": True,
            "third_party_verification": True,
            "complaint_mechanism": True,
            "periodic_review": True,
        },
        {
            "label_id": "LBL-002",
            "label_name": "Company Green Seal",
            "label_type": "COMPANY_OWN",
            "scheme_owner": "EcoProducts GmbH",
            "accredited": False,
            "scientific_basis": False,
            "third_party_verification": False,
            "complaint_mechanism": False,
            "periodic_review": False,
        },
        {
            "label_id": "LBL-003",
            "label_name": "Blue Angel",
            "label_type": "TYPE_I_ECOLABEL",
            "scheme_owner": "German Federal Environment Agency",
            "accredited": True,
            "scientific_basis": True,
            "third_party_verification": True,
            "complaint_mechanism": True,
            "periodic_review": True,
        },
        {
            "label_id": "LBL-004",
            "label_name": "GreenChoice Private Label",
            "label_type": "PRIVATE_SCHEME",
            "scheme_owner": "GreenChoice Foundation",
            "accredited": False,
            "scientific_basis": True,
            "third_party_verification": True,
            "complaint_mechanism": True,
            "periodic_review": False,
        },
    ]


# ---------------------------------------------------------------------------
# 6.3  Lifecycle Impact Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_lifecycle_impacts() -> List[Dict[str, Any]]:
    """Sample lifecycle impact data for PEF/LCA testing.

    16 PEF impact categories across 6 lifecycle phases for a
    sample consumer product.
    """
    return [
        {
            "phase": "RAW_MATERIALS",
            "category": "CLIMATE_CHANGE",
            "value": Decimal("12.5"),
            "unit": "kg CO2 eq",
            "data_quality_rating": 3,
        },
        {
            "phase": "MANUFACTURING",
            "category": "CLIMATE_CHANGE",
            "value": Decimal("8.3"),
            "unit": "kg CO2 eq",
            "data_quality_rating": 2,
        },
        {
            "phase": "TRANSPORTATION",
            "category": "CLIMATE_CHANGE",
            "value": Decimal("3.1"),
            "unit": "kg CO2 eq",
            "data_quality_rating": 3,
        },
        {
            "phase": "USE",
            "category": "CLIMATE_CHANGE",
            "value": Decimal("15.8"),
            "unit": "kg CO2 eq",
            "data_quality_rating": 4,
        },
        {
            "phase": "END_OF_LIFE",
            "category": "CLIMATE_CHANGE",
            "value": Decimal("2.1"),
            "unit": "kg CO2 eq",
            "data_quality_rating": 3,
        },
        {
            "phase": "RAW_MATERIALS",
            "category": "WATER_USE",
            "value": Decimal("450.0"),
            "unit": "m3 eq",
            "data_quality_rating": 3,
        },
        {
            "phase": "MANUFACTURING",
            "category": "WATER_USE",
            "value": Decimal("120.0"),
            "unit": "m3 eq",
            "data_quality_rating": 2,
        },
        {
            "phase": "RAW_MATERIALS",
            "category": "LAND_USE",
            "value": Decimal("25.0"),
            "unit": "pt",
            "data_quality_rating": 4,
        },
    ]


# ---------------------------------------------------------------------------
# 6.4  Trader Profile Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trader_profile() -> Dict[str, Any]:
    """Sample trader profile for obligation tracking.

    Represents a medium-sized manufacturer with EU market presence
    per Article 8.
    """
    return {
        "entity_id": "TRADER-001",
        "entity_name": "EcoProducts GmbH",
        "entity_type": "MANUFACTURER",
        "entity_size": "MEDIUM",
        "annual_turnover_eur": Decimal("25000000"),
        "employee_count": 180,
        "eu_market_presence": True,
        "claims_count": 12,
    }


@pytest.fixture
def sample_sme_profile() -> Dict[str, Any]:
    """Sample micro-enterprise profile for SME exemption testing.

    Per Article 12, micro-enterprises have simplified requirements.
    """
    return {
        "entity_id": "TRADER-002",
        "entity_name": "BioClean Startup",
        "entity_type": "MANUFACTURER",
        "entity_size": "MICRO",
        "annual_turnover_eur": Decimal("800000"),
        "employee_count": 8,
        "eu_market_presence": True,
        "claims_count": 3,
    }


# ---------------------------------------------------------------------------
# 6.5  Comparative Claims Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_comparative_claims() -> List[Dict[str, Any]]:
    """Sample comparative and improvement claims for validation.

    Per Articles 3(4) and 5, comparative claims require baseline,
    methodology, and independent monitoring.
    """
    return [
        {
            "claim_id": "CMP-001",
            "claim_text": "30% less CO2 than our 2020 product",
            "comparison_type": "YEAR_OVER_YEAR",
            "baseline_value": Decimal("100"),
            "current_value": Decimal("70"),
            "baseline_year": 2020,
            "current_year": 2025,
            "unit": "kg CO2 eq",
            "methodology": "ISO 14067",
            "has_binding_target": True,
            "independent_monitoring": True,
        },
        {
            "claim_id": "CMP-002",
            "claim_text": "We will be carbon neutral by 2030",
            "comparison_type": "IMPROVEMENT_OVER_TIME",
            "baseline_value": Decimal("5000"),
            "current_value": Decimal("3500"),
            "baseline_year": 2020,
            "current_year": 2025,
            "unit": "tonnes CO2 eq",
            "methodology": "GHG Protocol",
            "has_binding_target": False,
            "independent_monitoring": False,
        },
    ]


# ---------------------------------------------------------------------------
# 6.6  Greenwashing Screening Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_marketing_claims() -> List[Dict[str, Any]]:
    """Sample marketing claims for greenwashing screening.

    Includes vague, misleading, and compliant claims for
    TerraChoice Seven Sins detection.
    """
    return [
        {
            "text": "Our product is eco-friendly and good for the planet",
            "claim_type": "ECO_FRIENDLY",
            "evidence_count": 0,
            "lifecycle_coverage": [],
        },
        {
            "text": "100% natural ingredients",
            "claim_type": "SUSTAINABLE",
            "evidence_count": 1,
            "lifecycle_coverage": ["RAW_MATERIALS"],
        },
        {
            "text": "Carbon neutral certified by SGS (Scope 1+2+3)",
            "claim_type": "CARBON_NEUTRAL",
            "evidence_count": 3,
            "lifecycle_coverage": [
                "RAW_MATERIALS",
                "MANUFACTURING",
                "TRANSPORTATION",
                "USE",
                "END_OF_LIFE",
            ],
        },
        {
            "text": "CFC-free product",
            "claim_type": "ENVIRONMENTALLY_FRIENDLY",
            "evidence_count": 0,
            "lifecycle_coverage": [],
        },
        {
            "text": "Green product - save the earth!",
            "claim_type": "GREEN",
            "evidence_count": 0,
            "lifecycle_coverage": [],
        },
    ]


# ---------------------------------------------------------------------------
# 6.7  Benchmark Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_portfolio_results() -> List[Dict[str, Any]]:
    """Sample portfolio assessment results for benchmarking.

    Represents aggregated results across multiple claims.
    """
    return [
        {
            "claim_id": "CLM-001",
            "substantiation_score": Decimal("75"),
            "greenwashing_risk": Decimal("15"),
            "evidence_complete": True,
            "verification_ready": True,
        },
        {
            "claim_id": "CLM-002",
            "substantiation_score": Decimal("82"),
            "greenwashing_risk": Decimal("10"),
            "evidence_complete": True,
            "verification_ready": True,
        },
        {
            "claim_id": "CLM-003",
            "substantiation_score": Decimal("25"),
            "greenwashing_risk": Decimal("80"),
            "evidence_complete": False,
            "verification_ready": False,
        },
        {
            "claim_id": "CLM-004",
            "substantiation_score": Decimal("68"),
            "greenwashing_risk": Decimal("20"),
            "evidence_complete": True,
            "verification_ready": False,
        },
        {
            "claim_id": "CLM-005",
            "substantiation_score": Decimal("10"),
            "greenwashing_risk": Decimal("95"),
            "evidence_complete": False,
            "verification_ready": False,
        },
    ]


# ---------------------------------------------------------------------------
# 6.8  Political Activity / Supplier Data (for G1 cross-reference)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for evidence chain testing.

    Represents various evidence document types with validity periods.
    """
    return [
        {
            "doc_id": "DOC-001",
            "title": "Material Recyclability Certificate",
            "evidence_type": "CERTIFICATION",
            "source_url": "https://tuv.com/cert/12345",
            "issue_date": "2024-01-15",
            "expiry_date": "2026-12-31",
            "issuing_body": "TUV Rheinland",
            "accreditation_reference": "DAkkS D-ZE-12345-01",
        },
        {
            "doc_id": "DOC-002",
            "title": "Full Product LCA Study",
            "evidence_type": "LCA_STUDY",
            "source_url": "https://fraunhofer.de/lca/67890",
            "issue_date": "2024-06-01",
            "expiry_date": "2027-05-31",
            "issuing_body": "Fraunhofer IVV",
            "accreditation_reference": "ISO 14044:2006 compliant",
        },
        {
            "doc_id": "DOC-003",
            "title": "GHG Verification Statement",
            "evidence_type": "THIRD_PARTY_VERIFICATION",
            "source_url": "https://sgs.com/verify/11111",
            "issue_date": "2025-01-01",
            "expiry_date": "2025-12-31",
            "issuing_body": "SGS",
            "accreditation_reference": "UKAS 0120",
        },
        {
            "doc_id": "DOC-004",
            "title": "Expired Test Report",
            "evidence_type": "TEST_REPORT",
            "source_url": "https://lab.de/test/old",
            "issue_date": "2020-01-01",
            "expiry_date": "2022-12-31",
            "issuing_body": "Internal Lab",
            "accreditation_reference": "",
        },
    ]
