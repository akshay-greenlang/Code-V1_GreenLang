# -*- coding: utf-8 -*-
"""
GreenLang Agricultural Emissions Agent - AGENT-MRV-008 (GL-MRV-SCOPE1-008)

Scope 1 agricultural emission estimation covering enteric fermentation from
livestock (CH4), manure management (CH4 and N2O), cropland emissions including
direct and indirect N2O from synthetic fertilisers, organic amendments, crop
residues, and soil management practices, rice cultivation CH4, and Monte Carlo
uncertainty quantification.

Engines:
    1. AgriculturalDatabaseEngine   - Livestock categories, emission factors, crop defaults
    2. EntericFermentationEngine    - CH4 from ruminant digestive processes (Tier 1/2/3)
    3. ManureManagementEngine       - CH4 and N2O from manure storage and treatment
    4. CroplandEmissionsEngine      - Direct/indirect N2O, rice CH4, residue burning
    5. UncertaintyQuantifierEngine   - Monte Carlo uncertainty and confidence intervals
    6. ComplianceCheckerEngine       - Multi-framework regulatory compliance validation
    7. AgriculturalPipelineEngine    - End-to-end orchestration of all engines

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "AgriculturalDatabaseEngine",
    "EntericFermentationEngine",
    "ManureManagementEngine",
    "CroplandEmissionsEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "AgriculturalPipelineEngine",
]

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful engine imports - engines may not yet be implemented
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
except ImportError:
    AgriculturalDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.enteric_fermentation import (
        EntericFermentationEngine,
    )
except ImportError:
    EntericFermentationEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.manure_management import (
        ManureManagementEngine,
    )
except ImportError:
    ManureManagementEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.cropland_emissions import (
        CroplandEmissionsEngine,
    )
except ImportError:
    CroplandEmissionsEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agricultural_emissions.agricultural_pipeline import (
        AgriculturalPipelineEngine,
    )
except ImportError:
    AgriculturalPipelineEngine = None  # type: ignore[assignment,misc]
