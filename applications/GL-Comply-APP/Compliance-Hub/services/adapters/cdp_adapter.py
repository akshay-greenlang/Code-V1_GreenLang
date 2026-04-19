# -*- coding: utf-8 -*-
"""CDP Climate Change questionnaire adapter.

CDP C6 (Emissions Data) section maps to GHG Protocol. This adapter surfaces
the quantitative data used for CDP questionnaire autofill; free-text narrative
responses live in applications/GL-CDP-APP.
"""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class CDPAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.CDP
