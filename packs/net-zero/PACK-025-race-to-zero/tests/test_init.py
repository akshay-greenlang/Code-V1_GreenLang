# -*- coding: utf-8 -*-
"""
Tests for PACK-025 Race to Zero Pack package __init__ modules.

Validates that each sub-package (__init__.py) correctly exposes its
public API, version metadata, and all expected classes/enums.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ========================================================================
# Engines __init__
# ========================================================================


class TestEnginesInit:
    """Tests for engines/__init__.py exports."""

    def test_imports_all_10_engines(self):
        from engines import __all__
        engine_classes = [e for e in __all__ if e.endswith("Engine")]
        assert len(engine_classes) == 10

    def test_version_exported(self):
        from engines import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from engines import __pack_id__
        assert __pack_id__ == "PACK-025"

    def test_pack_name_exported(self):
        from engines import __pack_name__
        assert __pack_name__ == "Race to Zero Pack"

    def test_all_list_is_list(self):
        from engines import __all__
        assert isinstance(__all__, list)
        assert len(__all__) >= 13  # 10 engines + version + pack_id + pack_name


# ========================================================================
# Workflows __init__
# ========================================================================


class TestWorkflowsInit:
    """Tests for workflows/__init__.py exports."""

    def test_imports_all_8_workflows(self):
        from workflows import __all__
        workflow_classes = [w for w in __all__ if w.endswith("Workflow")]
        assert len(workflow_classes) == 8

    def test_version_exported(self):
        from workflows import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from workflows import __pack_id__
        assert __pack_id__ == "PACK-025"

    def test_pack_name_exported(self):
        from workflows import __pack_name__
        assert __pack_name__ == "Race to Zero Pack"

    def test_all_list_has_many_exports(self):
        from workflows import __all__
        assert isinstance(__all__, list)
        # 8 workflows + configs + results + enums + models
        assert len(__all__) >= 50

    def test_pledge_onboarding_exports(self):
        from workflows import (
            PledgeOnboardingWorkflow,
            PledgeOnboardingConfig,
            PledgeOnboardingResult,
            OnboardingPhase,
        )
        assert all(
            c is not None
            for c in [
                PledgeOnboardingWorkflow,
                PledgeOnboardingConfig,
                PledgeOnboardingResult,
                OnboardingPhase,
            ]
        )

    def test_full_r2z_exports(self):
        from workflows import (
            FullRaceToZeroWorkflow,
            FullR2ZConfig,
            FullR2ZResult,
            R2ZPhase,
        )
        assert all(
            c is not None
            for c in [
                FullRaceToZeroWorkflow,
                FullR2ZConfig,
                FullR2ZResult,
                R2ZPhase,
            ]
        )


# ========================================================================
# Templates __init__
# ========================================================================


class TestTemplatesInit:
    """Tests for templates/__init__.py exports."""

    def test_imports_all_10_templates(self):
        from templates import __all__
        template_classes = [t for t in __all__ if t.endswith("Template")]
        assert len(template_classes) == 10

    def test_version_exported(self):
        from templates import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from templates import __pack_id__
        assert __pack_id__ == "PACK-025"


# ========================================================================
# Integrations __init__
# ========================================================================


class TestIntegrationsInit:
    """Tests for integrations/__init__.py exports."""

    def test_version_exported(self):
        from integrations import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from integrations import __pack_id__
        assert __pack_id__ == "PACK-025"

    def test_all_list_has_many_exports(self):
        from integrations import __all__
        assert isinstance(__all__, list)
        # 12 integration classes + configs + models + enums
        assert len(__all__) >= 80

    def test_orchestrator_exported(self):
        from integrations import RaceToZeroOrchestrator
        assert RaceToZeroOrchestrator is not None

    def test_setup_wizard_exported(self):
        from integrations import RaceToZeroSetupWizard
        assert RaceToZeroSetupWizard is not None

    def test_health_check_exported(self):
        from integrations import RaceToZeroHealthCheck
        assert RaceToZeroHealthCheck is not None


# ========================================================================
# Presets __init__
# ========================================================================


class TestPresetsInit:
    """Tests for config/presets/__init__.py exports."""

    def test_version_exported(self):
        from config.presets import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from config.presets import __pack_id__
        assert __pack_id__ == "PACK-025"

    def test_available_presets_exported(self):
        from config.presets import AVAILABLE_PRESETS
        assert isinstance(AVAILABLE_PRESETS, dict)
        assert len(AVAILABLE_PRESETS) == 8

    def test_actor_type_map_exported(self):
        from config.presets import ACTOR_TYPE_PRESET_MAP
        assert isinstance(ACTOR_TYPE_PRESET_MAP, dict)
        assert len(ACTOR_TYPE_PRESET_MAP) == 8

    def test_utility_functions_exported(self):
        from config.presets import get_preset_path, get_preset_for_actor_type
        assert callable(get_preset_path)
        assert callable(get_preset_for_actor_type)
