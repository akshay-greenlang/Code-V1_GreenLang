# -*- coding: utf-8 -*-
"""
Factor Broker Data Sources
GL-VCCI Scope 3 Platform

Emission factor data source integrations.

Version: 1.0.0
"""

from .base import FactorSource
from .ecoinvent import EcoinventSource
from .desnz import DESNZSource
from .epa import EPASource
from .proxy import ProxySource

__all__ = [
    "FactorSource",
    "EcoinventSource",
    "DESNZSource",
    "EPASource",
    "ProxySource",
]
