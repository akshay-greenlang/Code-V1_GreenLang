# -*- coding: utf-8 -*-
"""
GreenLang Models Package

This module provides core data models for emission factors and calculations.
"""

from .emission_factor import (
    DataQualityTier,
    Scope,
    GeographyLevel,
    EmissionFactor,
    EmissionResult,
    SourceProvenance,
)

# Backward-compatible re-export from greenlang.data.models module
# NOTE: There's a naming conflict between models.py and models/ directory
# This ensures imports from greenlang.data.models work for both
import sys
import importlib.util
from pathlib import Path

# Import BuildingInput from the models.py file (not this package)
_models_file = Path(__file__).parent.parent / "models.py"
if _models_file.exists():
    spec = importlib.util.spec_from_file_location("greenlang.data._models_compat", _models_file)
    if spec and spec.loader:
        _models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_models_module)
        BuildingInput = _models_module.BuildingInput

__all__ = [
    "DataQualityTier",
    "Scope",
    "GeographyLevel",
    "EmissionFactor",
    "EmissionResult",
    "SourceProvenance",
    "BuildingInput",
]
