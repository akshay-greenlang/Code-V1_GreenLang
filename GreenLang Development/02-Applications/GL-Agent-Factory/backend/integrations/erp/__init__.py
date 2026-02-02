"""
ERP Integration Module.

This package provides connectors for various ERP systems
to extract emissions, energy, and procurement data.

Supported ERP Systems:
- SAP (S/4HANA, ECC, Business One)
- Oracle (Cloud ERP, E-Business Suite)

Example:
    >>> from integrations.erp import SAPConnector, SAPConfig
    >>>
    >>> config = SAPConfig(
    ...     host="sap.company.com",
    ...     username="user",
    ...     password="pass"
    ... )
    >>> connector = SAPConnector(config)
    >>> await connector.connect()
    >>> records = await connector.fetch_emissions_data(date_from, date_to)
"""

from .base import (
    BaseERPConnector,
    ConnectionConfig,
    ConnectionStatus,
    DataQuery,
    ERPRecord,
    ERPType,
)
from .sap_connector import (
    SAPConnector,
    SAPConfig,
    SAPEntityMapping,
)
from .oracle_connector import (
    OracleConnector,
    OracleConfig,
)

__all__ = [
    # Base classes
    "BaseERPConnector",
    "ConnectionConfig",
    "ConnectionStatus",
    "DataQuery",
    "ERPRecord",
    "ERPType",
    # SAP
    "SAPConnector",
    "SAPConfig",
    "SAPEntityMapping",
    # Oracle
    "OracleConnector",
    "OracleConfig",
]


def get_connector(erp_type: str, config: ConnectionConfig) -> BaseERPConnector:
    """
    Factory function to get ERP connector by type.

    Args:
        erp_type: ERP type (sap, oracle)
        config: Connection configuration

    Returns:
        ERP connector instance

    Raises:
        ValueError: If ERP type not supported
    """
    connectors = {
        "sap": SAPConnector,
        "oracle": OracleConnector,
    }

    if erp_type.lower() not in connectors:
        raise ValueError(
            f"Unsupported ERP type: {erp_type}. "
            f"Supported: {list(connectors.keys())}"
        )

    return connectors[erp_type.lower()](config)
