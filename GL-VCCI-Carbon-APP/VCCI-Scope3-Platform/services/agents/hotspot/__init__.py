"""
HotspotAnalysisAgent Package
GL-VCCI Scope 3 Platform

Emissions hotspot analysis and scenario modeling agent.

Features:
- Pareto analysis (80/20 rule)
- Multi-dimensional segmentation
- Scenario modeling framework
- ROI analysis and abatement curves
- Automated hotspot detection
- Actionable insight generation

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

from .agent import HotspotAnalysisAgent
from .models import *
from .config import *
from .exceptions import *

__version__ = "1.0.0"
__all__ = [
    "HotspotAnalysisAgent",
]
