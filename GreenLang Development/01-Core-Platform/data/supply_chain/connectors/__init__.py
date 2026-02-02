"""
Data Connectors for Supply Chain Integration.

Provides connectors for various procurement and ERP systems:
- SAP Ariba integration
- Coupa integration
- File import (CSV, Excel)
- REST API for manual entry
"""

from greenlang.supply_chain.connectors.sap_ariba_connector import (
    SAPAribaConnector,
    AribaSupplierRecord,
    AribaPORecord,
)
from greenlang.supply_chain.connectors.coupa_connector import (
    CoupaConnector,
    CoupaSupplierRecord,
    CoupaInvoiceRecord,
)
from greenlang.supply_chain.connectors.file_connector import (
    FileConnector,
    ImportResult,
    ExportFormat,
)

__all__ = [
    "SAPAribaConnector",
    "AribaSupplierRecord",
    "AribaPORecord",
    "CoupaConnector",
    "CoupaSupplierRecord",
    "CoupaInvoiceRecord",
    "FileConnector",
    "ImportResult",
    "ExportFormat",
]
