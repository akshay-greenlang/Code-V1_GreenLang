# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-024 Carbon Neutral Pack.

Adds the pack root to sys.path so that ``from engines.X import Y`` works
in every test module without requiring an installed package.
"""

import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
