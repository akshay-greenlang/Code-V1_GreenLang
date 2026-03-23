# -*- coding: utf-8 -*-
"""
Employee Commuting Agent Package - AGENT-MRV-020

GHG Protocol Scope 3, Category 7: Employee Commuting.
Calculates emissions from employees traveling between their homes and
worksites in vehicles not owned or operated by the reporting company,
including personal vehicles, public transit, active transport, carpooling,
and telework/remote work adjustments.

Agent ID: GL-MRV-S3-007
Package: greenlang.agents.mrv.employee_commuting
API: /api/v1/employee-commuting
DB Migration: V071
Metrics Prefix: gl_ec_
Table Prefix: gl_ec_

Supported Commute Modes:
    - Personal vehicle (car, SUV, pickup, motorcycle - 12 fuel types)
    - Public transit (bus, subway, light rail, commuter rail, ferry)
    - Active transport (bicycle, e-bike, walking, scooter)
    - Carpool / vanpool (2-8 occupants, shared distance allocation)
    - Telework / remote work (home office energy, equipment, heating/cooling)
    - Multi-modal commute (combined segments per trip)

Calculation Methods:
    - Survey-based (employee commute survey with modal split)
    - Distance-based (distance x mode-specific EF)
    - Average-data (national/regional averages per employee)
    - Spend-based (EEIO factors with CPI deflation)

Frameworks:
    - GHG Protocol Scope 3 Category 7
    - ISO 14064-1:2018
    - CSRD ESRS E1
    - CDP Climate Change
    - SBTi Corporate Net-Zero
    - SB 253 (California Climate Disclosure)
    - GRI 305

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "EmployeeCommutingDatabaseEngine",
    "PersonalVehicleCalculatorEngine",
    "PublicTransitCalculatorEngine",
    "ActiveTransportCalculatorEngine",
    "TeleworkCalculatorEngine",
    "ComplianceCheckerEngine",
    "EmployeeCommutingPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-007"
AGENT_COMPONENT: str = "AGENT-MRV-020"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_ec_"

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.employee_commuting.employee_commuting_database import (
        EmployeeCommutingDatabaseEngine,
    )
except ImportError:
    EmployeeCommutingDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.personal_vehicle_calculator import (
        PersonalVehicleCalculatorEngine,
    )
except ImportError:
    PersonalVehicleCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
        PublicTransitCalculatorEngine,
    )
except ImportError:
    PublicTransitCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.active_transport_calculator import (
        ActiveTransportCalculatorEngine,
    )
except ImportError:
    ActiveTransportCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.telework_calculator import (
        TeleworkCalculatorEngine,
    )
except ImportError:
    TeleworkCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.employee_commuting.employee_commuting_pipeline import (
        EmployeeCommutingPipelineEngine,
    )
except ImportError:
    EmployeeCommutingPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.agents.mrv.employee_commuting.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
