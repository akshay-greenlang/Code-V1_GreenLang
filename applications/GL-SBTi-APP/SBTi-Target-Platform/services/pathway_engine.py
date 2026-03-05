"""
Pathway Engine -- Facade for PathwayCalculatorEngine

Re-exports PathwayCalculatorEngine as PathwayEngine for the unified
__init__.py and setup module naming convention.

The underlying engine lives in pathway_calculator_engine.py and
implements ACA, SDA, FLAG, economic/physical intensity, and supplier
engagement pathway computation.

Example:
    >>> engine = PathwayEngine(config)
    >>> pathway = engine.calculate_aca_pathway(100000, 2020, 2030, "1.5c")
"""

from .pathway_calculator_engine import PathwayCalculatorEngine as PathwayEngine

__all__ = ["PathwayEngine"]
