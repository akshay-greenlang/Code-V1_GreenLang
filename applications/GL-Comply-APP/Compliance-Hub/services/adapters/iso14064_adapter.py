# -*- coding: utf-8 -*-
"""ISO 14064-1:2018 adapter."""

from __future__ import annotations

from schemas.models import FrameworkEnum
from services.adapters.base import ScopeEngineAdapterBase


class ISO14064Adapter(ScopeEngineAdapterBase):
    framework = FrameworkEnum.ISO_14064
