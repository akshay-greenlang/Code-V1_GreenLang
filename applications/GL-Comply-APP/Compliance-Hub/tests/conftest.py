# -*- coding: utf-8 -*-
"""Shared fixtures + path wiring for Comply-Hub tests.

Tests run with PYTHONPATH=applications/GL-Comply-APP/Compliance-Hub so that
`from schemas.models import ...` resolves.
"""

import sys
from pathlib import Path

# Make the Comply-Hub root importable without PYTHONPATH manipulation
_HUB_ROOT = Path(__file__).resolve().parent.parent
if str(_HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_HUB_ROOT))
