# -*- coding: utf-8 -*-
"""
Public re-export module for the Factors FastAPI factory.

The actual implementation lives in :mod:`greenlang.factors.factors_app`;
this module exists so callers (tests, OEM wrappers, the conftest) can
write::

    from greenlang.factors.api import create_factors_app

without having to know the implementation file name.
"""

from __future__ import annotations

from greenlang.factors.factors_app import app, create_factors_app

__all__ = ["app", "create_factors_app"]
