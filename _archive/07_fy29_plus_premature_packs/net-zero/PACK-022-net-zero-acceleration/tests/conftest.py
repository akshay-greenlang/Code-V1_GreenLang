# -*- coding: utf-8 -*-
"""
PACK-022 Net Zero Acceleration Pack - Test Configuration
==========================================================

Shared test infrastructure for all PACK-022 test modules.
Adds the pack root directory to sys.path so that imports like
``from engines.scenario_modeling_engine import ...`` resolve correctly.

Author:  GL-TestEngineer
Pack:    PACK-022 Net Zero Acceleration
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACK_ROOT.parents[2]

# Add pack root so `from engines.X import ...` works
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# Add repo root so cross-pack references work if needed
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Path Constants (available to test modules via conftest import)
# ---------------------------------------------------------------------------

ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"

# ---------------------------------------------------------------------------
# Engine and Workflow File Registries
# ---------------------------------------------------------------------------

ENGINE_FILES = {
    "scenario_modeling": "engines/scenario_modeling_engine.py",
    "temperature_scoring": "engines/temperature_scoring_engine.py",
    "sda_pathway": "engines/sda_pathway_engine.py",
    "variance_decomposition": "engines/variance_decomposition_engine.py",
    "multi_entity": "engines/multi_entity_engine.py",
    "supplier_engagement": "engines/supplier_engagement_engine.py",
    "scope3_activity": "engines/scope3_activity_engine.py",
    "vcmi_validation": "engines/vcmi_validation_engine.py",
    "climate_finance": "engines/climate_finance_engine.py",
    "assurance_workpaper": "engines/assurance_workpaper_engine.py",
}

WORKFLOW_FILES = {
    "scenario_analysis": "workflows/scenario_analysis_workflow.py",
    "sda_target": "workflows/sda_target_workflow.py",
    "supplier_program": "workflows/supplier_program_workflow.py",
    "transition_finance": "workflows/transition_finance_workflow.py",
    "advanced_progress": "workflows/advanced_progress_workflow.py",
    "temperature_alignment": "workflows/temperature_alignment_workflow.py",
    "vcmi_certification": "workflows/vcmi_certification_workflow.py",
    "full_acceleration": "workflows/full_acceleration_workflow.py",
}
