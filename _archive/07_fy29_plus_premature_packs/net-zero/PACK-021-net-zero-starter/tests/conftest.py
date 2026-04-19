# -*- coding: utf-8 -*-
"""
PACK-021 Net Zero Starter Pack - Test Configuration
=====================================================

Shared test infrastructure for all PACK-021 test modules.
Adds the pack root directory to sys.path so that imports like
``from engines.residual_emissions_engine import ...`` resolve correctly.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
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
    "net_zero_baseline": "engines/net_zero_baseline_engine.py",
    "net_zero_target": "engines/net_zero_target_engine.py",
    "net_zero_gap": "engines/net_zero_gap_engine.py",
    "reduction_pathway": "engines/reduction_pathway_engine.py",
    "residual_emissions": "engines/residual_emissions_engine.py",
    "offset_portfolio": "engines/offset_portfolio_engine.py",
    "net_zero_scorecard": "engines/net_zero_scorecard_engine.py",
    "net_zero_benchmark": "engines/net_zero_benchmark_engine.py",
}

WORKFLOW_FILES = {
    "net_zero_onboarding": "workflows/net_zero_onboarding_workflow.py",
    "target_setting": "workflows/target_setting_workflow.py",
    "reduction_planning": "workflows/reduction_planning_workflow.py",
    "offset_strategy": "workflows/offset_strategy_workflow.py",
    "progress_review": "workflows/progress_review_workflow.py",
    "full_net_zero_assessment": "workflows/full_net_zero_assessment_workflow.py",
}
