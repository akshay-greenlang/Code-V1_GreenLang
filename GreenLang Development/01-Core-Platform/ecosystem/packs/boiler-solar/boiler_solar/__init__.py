# -*- coding: utf-8 -*-
"""
Boiler Solar Pack
=================

Solar thermal analysis pack for industrial boilers using GreenLang.

This pack provides comprehensive analysis of solar thermal integration
for industrial boilers, including energy modeling, carbon calculations,
and economic analysis.
"""

__version__ = "0.2.0"
__author__ = "GreenLang Team"
__license__ = "Commercial"

from pathlib import Path

# Pack directory
PACK_DIR = Path(__file__).parent.parent

# Export key components
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "PACK_DIR",
    "get_manifest_path",
]

def get_manifest_path() -> str:
    """
    Get the path to the pack manifest file.
    
    This function is used as the entry point for pip-installed packs,
    allowing GreenLang to discover this pack when installed.
    
    Returns:
        str: Absolute path to the pack.yaml file
    """
    return str(PACK_DIR / "pack.yaml")