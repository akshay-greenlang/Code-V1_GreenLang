# -*- coding: utf-8 -*-
"""
Scenario Modeling Framework
GL-VCCI Scope 3 Platform

Framework for emission reduction scenario modeling.
Full implementation planned for Week 27+.

Version: 1.0.0 (Framework + Stubs)
"""

from .scenario_engine import ScenarioEngine
from .supplier_switching import SupplierSwitchingModule
from .modal_shift import ModalShiftModule
from .product_substitution import ProductSubstitutionModule

__all__ = [
    "ScenarioEngine",
    "SupplierSwitchingModule",
    "ModalShiftModule",
    "ProductSubstitutionModule",
]
