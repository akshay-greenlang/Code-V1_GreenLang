# -*- coding: utf-8 -*-
"""
GreenLang Waste Treatment Emissions Agent - AGENT-MRV-007 (GL-MRV-SCOPE1-007)

Production-grade on-site waste treatment emissions calculation engine
for Scope 1 GHG reporting under IPCC 2006/2019 Vol 5 (Waste).

Covers biological treatment (composting, anaerobic digestion, MBT),
thermal treatment (incineration, pyrolysis, gasification), and
on-site wastewater treatment.

Engines:
    1. WasteTreatmentDatabaseEngine   - Reference data and emission factors
    2. BiologicalTreatmentEngine      - Composting, AD, MBT calculations
    3. ThermalTreatmentEngine         - Incineration, pyrolysis, gasification
    4. WastewaterTreatmentEngine      - On-site wastewater CH4/N2O
    5. UncertaintyQuantifierEngine    - Monte Carlo uncertainty analysis
    6. ComplianceCheckerEngine        - Multi-framework compliance validation
    7. WasteTreatmentPipelineEngine   - 8-stage pipeline orchestration

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 Waste Treatment Emissions (GL-MRV-SCOPE1-007)
Status: Production Ready
"""

from __future__ import annotations

__all__ = [
    "WasteTreatmentDatabaseEngine",
    "BiologicalTreatmentEngine",
    "ThermalTreatmentEngine",
    "WastewaterTreatmentEngine",
    "UncertaintyQuantifierEngine",
    "ComplianceCheckerEngine",
    "WasteTreatmentPipelineEngine",
]

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful engine imports - engines may not yet be implemented
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
except ImportError:
    WasteTreatmentDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.biological_treatment import (
        BiologicalTreatmentEngine,
    )
except ImportError:
    BiologicalTreatmentEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.thermal_treatment import (
        ThermalTreatmentEngine,
    )
except ImportError:
    ThermalTreatmentEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.wastewater_treatment import (
        WastewaterTreatmentEngine,
    )
except ImportError:
    WastewaterTreatmentEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_pipeline import (
        WasteTreatmentPipelineEngine,
    )
except ImportError:
    WasteTreatmentPipelineEngine = None  # type: ignore[assignment,misc]
