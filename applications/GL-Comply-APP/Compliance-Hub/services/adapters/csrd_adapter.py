# -*- coding: utf-8 -*-
"""CSRD adapter — delegates to Scope Engine with ESRS E1 projection.

Future: integrate with applications/GL-CSRD-APP/CSRD-Reporting-Platform/csrd_pipeline.py
for full ESRS E1-E5 + S1-S4 + G1 disclosure assembly (narrative, materiality,
XBRL tagging). This adapter currently delivers the E1-6 quantitative core.
"""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class CSRDAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.CSRD
