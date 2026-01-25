# -*- coding: utf-8 -*-
# GL-VCCI Connectors Module
# ERP system integrations (SAP, Oracle, Workday)

"""
VCCI ERP Connectors
===================

Native integrations with enterprise ERP systems for automated data extraction.

Supported Systems:
-----------------
1. SAP S/4HANA (Priority #1 - 80% of market)
   - OData API integration
   - Modules: MM (Procurement), SD (Sales), FI (Finance)
   - OAuth 2.0 authentication

2. Oracle ERP Cloud
   - REST API integration
   - Modules: Procurement, Supply Chain, Finance
   - OAuth 2.0 authentication

3. Workday
   - REST API integration
   - Modules: HCM (Travel, Commuting), Finance
   - OAuth 2.0 authentication

Data Extracted:
--------------
- Procurement transactions (Category 1)
- Logistics data (Category 4, 9)
- Business travel (Category 6)
- Employee commuting (Category 7)
- Capital expenditures (Category 2)

Usage:
------
```python
from connectors import SAPConnector, OracleConnector, WorkdayConnector

# Initialize SAP connector
sap = SAPConnector(
    endpoint="https://your-sap-instance.com/sap/opu/odata/sap",
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Extract procurement data
procurement_data = sap.get_procurement_transactions(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Extract logistics data
logistics_data = sap.get_logistics_data(
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

Security:
---------
- OAuth 2.0 token-based authentication
- Encrypted credentials storage (HashiCorp Vault)
- Rate limiting (respect ERP system limits)
- Audit logging (all API calls logged)
"""

__version__ = "1.0.0"

__all__ = [
    # "SAPConnector",
    # "OracleConnector",
    # "WorkdayConnector",
]

# Connector registry
CONNECTOR_REGISTRY = {
    "sap": {
        "name": "SAP S/4HANA Connector",
        "version": "1.0.0",
        "status": "planned",
        "priority": 1,
        "market_share": 0.80,
        "week_scheduled": "19-21",
    },
    "oracle": {
        "name": "Oracle ERP Cloud Connector",
        "version": "1.0.0",
        "status": "planned",
        "priority": 2,
        "market_share": 0.12,
        "week_scheduled": "22-23",
    },
    "workday": {
        "name": "Workday Connector",
        "version": "1.0.0",
        "status": "planned",
        "priority": 3,
        "market_share": 0.08,
        "week_scheduled": "24",
    },
}


def get_connector_status(connector_name: str) -> dict:
    """Get status information for a specific connector.

    Args:
        connector_name: Name of the connector (e.g., "sap", "oracle")

    Returns:
        dict: Connector metadata including status, version, and schedule
    """
    return CONNECTOR_REGISTRY.get(connector_name, {})


def list_connectors() -> list:
    """List all available connectors.

    Returns:
        list: List of connector names
    """
    return list(CONNECTOR_REGISTRY.keys())
