# -*- coding: utf-8 -*-
"""CBAM adapter — delegates to Scope Engine with CBAM embedded-emissions view.

Future: integrate with applications/GL-CBAM-APP for goods-CN-code matching,
quarterly report XML generation, and customs declaration integration.
"""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class CBAMAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.CBAM
