# -*- coding: utf-8 -*-
"""
AGENT-MRV-028: Investments (Scope 3 Category 15) Agent

GHG Protocol Scope 3, Category 15: Investments.
Calculates GHG emissions from the reporting company's investments not
included in Scope 1 or Scope 2. Particularly relevant for financial
institutions (banks, asset managers, PE/VC firms, insurance companies).

Uses PCAF (Partnership for Carbon Accounting Financials) methodology for
attribution of financed emissions across asset classes:
    - Listed Equity & Corporate Bonds (EVIC attribution)
    - Private Equity / Venture Capital (equity share)
    - Project Finance (pro-rata project cost)
    - Commercial Real Estate (EPC/EUI-based)
    - Mortgages (LTV-weighted property emissions)
    - Motor Vehicle Loans (per-vehicle EFs)
    - Sovereign Bonds (GDP-PPP attribution)

Agent ID: GL-MRV-S3-015
Package: greenlang.agents.mrv.investments
API: /api/v1/investments
DB Migration: V079
Metrics Prefix: gl_inv_
Table Prefix: gl_inv_

Calculation Methods:
    - Reported emissions (PCAF Score 1 -- investee-reported, verified)
    - Physical activity-based (PCAF Score 2 -- energy/production data)
    - Revenue-based EEIO (PCAF Score 3-4 -- sector average / EEIO)
    - Sector average (PCAF Score 5 -- asset class defaults)
    - Asset-specific (real estate EUI, vehicle EFs)

Compliance Frameworks:
    - GHG Protocol Scope 3 Standard (Category 15)
    - PCAF Global GHG Standard (3rd Edition)
    - ISO 14064-1:2018
    - CSRD ESRS E1 (Climate Change)
    - CDP Climate Change Questionnaire
    - SBTi Financial Institutions (SBTi-FI)
    - California SB 253
    - TCFD Recommendations
    - Net-Zero Banking Alliance (NZBA)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    # Engine classes
    "InvestmentDatabaseEngine",
    "EquityInvestmentCalculatorEngine",
    "DebtInvestmentCalculatorEngine",
    "RealAssetCalculatorEngine",
    "SovereignBondCalculatorEngine",
    "ComplianceCheckerEngine",
    "InvestmentsPipelineEngine",
    # Metadata constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Configuration helper
    "get_config",
    # Info helpers
    "get_version",
    "get_agent_info",
]

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# ---------------------------------------------------------------------------
# Graceful imports -- each engine with try/except so the package can be
# imported even when optional engine dependencies are not yet installed.
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.investments.investment_database import InvestmentDatabaseEngine
except ImportError:
    InvestmentDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.equity_investment_calculator import EquityInvestmentCalculatorEngine
except ImportError:
    EquityInvestmentCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.debt_investment_calculator import DebtInvestmentCalculatorEngine
except ImportError:
    DebtInvestmentCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.real_asset_calculator import RealAssetCalculatorEngine
except ImportError:
    RealAssetCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.sovereign_bond_calculator import SovereignBondCalculatorEngine
except ImportError:
    SovereignBondCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.investments.investments_pipeline import InvestmentsPipelineEngine
except ImportError:
    InvestmentsPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.agents.mrv.investments.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None


def get_version() -> str:
    """Return the current version string for AGENT-MRV-028.

    Returns:
        Semantic version string (e.g., ``'1.0.0'``).

    Example:
        >>> get_version()
        '1.0.0'
    """
    return VERSION


def get_agent_info() -> dict:
    """Return metadata dictionary describing this agent.

    Returns:
        Dictionary with keys ``agent_id``, ``component``, ``version``,
        ``table_prefix``, ``package``, ``scope``, ``category``,
        ``methodology``, and ``asset_classes``.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-MRV-S3-015'
    """
    return {
        "agent_id": AGENT_ID,
        "component": AGENT_COMPONENT,
        "version": VERSION,
        "table_prefix": TABLE_PREFIX,
        "package": "greenlang.agents.mrv.investments",
        "scope": "Scope 3",
        "category": "Category 15 -- Investments",
        "methodology": "PCAF Global GHG Accounting & Reporting Standard (3rd Ed.)",
        "asset_classes": [
            "listed_equity",
            "corporate_bond",
            "private_equity",
            "project_finance",
            "commercial_real_estate",
            "mortgage",
            "motor_vehicle_loan",
            "sovereign_bond",
        ],
    }
