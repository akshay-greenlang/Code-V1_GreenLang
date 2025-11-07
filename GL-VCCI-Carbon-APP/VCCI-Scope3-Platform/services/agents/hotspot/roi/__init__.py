"""
ROI Analysis Modules
GL-VCCI Scope 3 Platform

ROI calculation and abatement curve generation.

Version: 1.0.0
"""

from .roi_calculator import ROICalculator
from .abatement_curve import AbatementCurveGenerator

__all__ = [
    "ROICalculator",
    "AbatementCurveGenerator",
]
