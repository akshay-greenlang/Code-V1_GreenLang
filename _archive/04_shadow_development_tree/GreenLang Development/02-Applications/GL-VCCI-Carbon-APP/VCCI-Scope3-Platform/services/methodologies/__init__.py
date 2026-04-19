# -*- coding: utf-8 -*-
"""
Methodologies and Uncertainty Catalog

This module provides scientific methodologies for emissions calculations including:
- ILCD Pedigree Matrix (data quality assessment)
- Monte Carlo simulation (uncertainty propagation)
- DQI (Data Quality Index) calculation
- Uncertainty quantification

Key Components:
- pedigree_matrix: ILCD Pedigree Matrix implementation
- monte_carlo: Monte Carlo simulation engine
- dqi_calculator: Data Quality Index calculator
- uncertainty: Uncertainty quantification
- models: Pydantic data models
- constants: Scientific constants and lookup tables
- config: Configuration management

References:
- ILCD Handbook (2010): https://eplca.jrc.ec.europa.eu/ilcd.html
- GHG Protocol: Corporate Value Chain (Scope 3) Standard
- IPCC Guidelines for National Greenhouse Gas Inventories
- ISO 14044:2006: Environmental management - Life cycle assessment

Version: 1.0.0
Date: 2025-10-30

Example Usage:
    >>> from services.methodologies import (
    ...     PedigreeScore,
    ...     PedigreeMatrixEvaluator,
    ...     MonteCarloSimulator,
    ...     DQICalculator,
    ...     UncertaintyQuantifier,
    ... )
    >>>
    >>> # Create pedigree score
    >>> pedigree = PedigreeScore(
    ...     reliability=1,
    ...     completeness=2,
    ...     temporal=1,
    ...     geographical=2,
    ...     technological=1
    ... )
    >>>
    >>> # Calculate DQI
    >>> calculator = DQICalculator()
    >>> dqi = calculator.calculate_dqi(
    ...     pedigree_score=pedigree,
    ...     factor_source="ecoinvent",
    ...     data_tier=1
    ... )
    >>> print(f"DQI Score: {dqi.score:.2f} ({dqi.quality_label})")
    >>>
    >>> # Run Monte Carlo simulation
    >>> simulator = MonteCarloSimulator(seed=42)
    >>> result = simulator.simple_propagation(
    ...     activity_data=1000.0,
    ...     activity_uncertainty=0.1,
    ...     emission_factor=2.5,
    ...     factor_uncertainty=0.15,
    ...     iterations=10000
    ... )
    >>> print(f"Emission: {result.mean:.2f} ± {result.std_dev:.2f} kg CO2e")
    >>> print(f"95% CI: [{result.p5:.2f}, {result.p95:.2f}]")
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Models
from .models import (
    PedigreeScore,
    DQIScore,
    UncertaintyResult,
    MonteCarloInput,
    MonteCarloResult,
)

# Constants
from .constants import (
    PedigreeIndicator,
    GWPVersion,
    DistributionType,
    PEDIGREE_MATRIX,
    GWP_AR5,
    GWP_AR6,
    DEFAULT_UNCERTAINTIES,
)

# Configuration
from .config import (
    MethodologiesConfig,
    config,
    get_config,
    reload_config,
    update_config,
)

# Pedigree Matrix
from .pedigree_matrix import (
    PedigreeMatrixEvaluator,
    create_pedigree_score,
    assess_data_quality,
)

# Monte Carlo
from .monte_carlo import (
    MonteCarloSimulator,
    AnalyticalPropagator,
    run_monte_carlo,
)

# DQI Calculator
from .dqi_calculator import (
    DQICalculator,
    calculate_dqi,
    assess_factor_quality,
)

# Uncertainty
from .uncertainty import (
    UncertaintyQuantifier,
    SensitivityAnalyzer,
    quantify_uncertainty,
    propagate_uncertainty,
)


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang AI"
__date__ = "2025-10-30"


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__date__",

    # Models
    "PedigreeScore",
    "DQIScore",
    "UncertaintyResult",
    "MonteCarloInput",
    "MonteCarloResult",

    # Constants
    "PedigreeIndicator",
    "GWPVersion",
    "DistributionType",
    "PEDIGREE_MATRIX",
    "GWP_AR5",
    "GWP_AR6",
    "DEFAULT_UNCERTAINTIES",

    # Configuration
    "MethodologiesConfig",
    "config",
    "get_config",
    "reload_config",
    "update_config",

    # Pedigree Matrix
    "PedigreeMatrixEvaluator",
    "create_pedigree_score",
    "assess_data_quality",

    # Monte Carlo
    "MonteCarloSimulator",
    "AnalyticalPropagator",
    "run_monte_carlo",

    # DQI Calculator
    "DQICalculator",
    "calculate_dqi",
    "assess_factor_quality",

    # Uncertainty
    "UncertaintyQuantifier",
    "SensitivityAnalyzer",
    "quantify_uncertainty",
    "propagate_uncertainty",
]
