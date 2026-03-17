# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Workflows Module
=====================================================

This package provides 12 ESRS disclosure workflows for the PACK-017
ESRS Full Coverage Pack. Each workflow implements a multi-phase
async pipeline for a specific ESRS standard, with SHA-256 provenance
tracking and full audit trail support.

Workflows:
    1.  ESRS2GeneralWorkflow       - ESRS 2 general disclosures
    2.  ESRS2GovernanceWorkflow    - ESRS 2 governance structure
    3.  E2PollutionWorkflow        - E2 pollution disclosures
    4.  E3WaterWorkflow            - E3 water and marine resources
    5.  E4BiodiversityWorkflow     - E4 biodiversity and ecosystems
    6.  E5CircularEconomyWorkflow  - E5 resource use and circular economy
    7.  S1WorkforceWorkflow        - S1 own workforce
    8.  S2ValueChainWorkflow       - S2 workers in the value chain
    9.  S3CommunitiesWorkflow      - S3 affected communities
    10. S4ConsumersWorkflow        - S4 consumers and end-users
    11. G1GovernanceWorkflow       - G1 business conduct
    12. FullESRSWorkflow           - Cross-standard full coverage

Author: GreenLang Team
Version: 17.0.0
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__version__: str = "17.0.0"
__pack__: str = "PACK-017"

_loaded_workflows: list[str] = []

# ---------------------------------------------------------------------------
# Workflow imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .esrs2_general_workflow import ESRS2GeneralWorkflow
    _loaded_workflows.append("ESRS2GeneralWorkflow")
except ImportError as e:
    ESRS2GeneralWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ESRS2GeneralWorkflow not available: %s", e)

try:
    from .esrs2_governance_workflow import ESRS2GovernanceWorkflow
    _loaded_workflows.append("ESRS2GovernanceWorkflow")
except ImportError as e:
    ESRS2GovernanceWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("ESRS2GovernanceWorkflow not available: %s", e)

try:
    from .e2_pollution_workflow import E2PollutionWorkflow
    _loaded_workflows.append("E2PollutionWorkflow")
except ImportError as e:
    E2PollutionWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("E2PollutionWorkflow not available: %s", e)

try:
    from .e3_water_workflow import E3WaterWorkflow
    _loaded_workflows.append("E3WaterWorkflow")
except ImportError as e:
    E3WaterWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("E3WaterWorkflow not available: %s", e)

try:
    from .e4_biodiversity_workflow import E4BiodiversityWorkflow
    _loaded_workflows.append("E4BiodiversityWorkflow")
except ImportError as e:
    E4BiodiversityWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("E4BiodiversityWorkflow not available: %s", e)

try:
    from .e5_circular_economy_workflow import E5CircularWorkflow as E5CircularEconomyWorkflow
    _loaded_workflows.append("E5CircularEconomyWorkflow")
except ImportError as e:
    E5CircularEconomyWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("E5CircularEconomyWorkflow not available: %s", e)

try:
    from .s1_workforce_workflow import S1WorkforceWorkflow
    _loaded_workflows.append("S1WorkforceWorkflow")
except ImportError as e:
    S1WorkforceWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("S1WorkforceWorkflow not available: %s", e)

try:
    from .s2_value_chain_workflow import S2ValueChainWorkflow
    _loaded_workflows.append("S2ValueChainWorkflow")
except ImportError as e:
    S2ValueChainWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("S2ValueChainWorkflow not available: %s", e)

try:
    from .s3_communities_workflow import S3CommunitiesWorkflow
    _loaded_workflows.append("S3CommunitiesWorkflow")
except ImportError as e:
    S3CommunitiesWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("S3CommunitiesWorkflow not available: %s", e)

try:
    from .s4_consumers_workflow import S4ConsumersWorkflow
    _loaded_workflows.append("S4ConsumersWorkflow")
except ImportError as e:
    S4ConsumersWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("S4ConsumersWorkflow not available: %s", e)

try:
    from .g1_governance_workflow import G1GovernanceWorkflow
    _loaded_workflows.append("G1GovernanceWorkflow")
except ImportError as e:
    G1GovernanceWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("G1GovernanceWorkflow not available: %s", e)

try:
    from .full_esrs_workflow import FullESRSWorkflow
    _loaded_workflows.append("FullESRSWorkflow")
except ImportError as e:
    FullESRSWorkflow = None  # type: ignore[assignment,misc]
    logger.debug("FullESRSWorkflow not available: %s", e)


# ---------------------------------------------------------------------------
# Dynamic __all__
# ---------------------------------------------------------------------------

__all__: list[str] = [
    *_loaded_workflows,
    "get_loaded_workflows",
    "get_workflow_count",
    "get_standard_workflow_mapping",
]


def get_loaded_workflows() -> list[str]:
    """Return list of successfully loaded workflow class names."""
    return list(_loaded_workflows)


def get_workflow_count() -> int:
    """Return count of loaded workflows."""
    return len(_loaded_workflows)


def get_standard_workflow_mapping() -> Dict[str, str]:
    """Return mapping of ESRS standard to workflow class name."""
    return {
        "ESRS_2_GENERAL": "ESRS2GeneralWorkflow",
        "ESRS_2_GOVERNANCE": "ESRS2GovernanceWorkflow",
        "E2": "E2PollutionWorkflow",
        "E3": "E3WaterWorkflow",
        "E4": "E4BiodiversityWorkflow",
        "E5": "E5CircularEconomyWorkflow",
        "S1": "S1WorkforceWorkflow",
        "S2": "S2ValueChainWorkflow",
        "S3": "S3CommunitiesWorkflow",
        "S4": "S4ConsumersWorkflow",
        "G1": "G1GovernanceWorkflow",
        "FULL_ESRS": "FullESRSWorkflow",
    }
