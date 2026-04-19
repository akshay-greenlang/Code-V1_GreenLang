# -*- coding: utf-8 -*-
"""GHG Protocol Corporate Standard adapter."""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class GHGProtocolAdapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.GHG_PROTOCOL
