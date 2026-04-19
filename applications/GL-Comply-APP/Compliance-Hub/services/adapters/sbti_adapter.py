# -*- coding: utf-8 -*-
"""SBTi target baseline adapter."""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class SBTiAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.SBTI
