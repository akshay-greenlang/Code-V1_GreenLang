# SAP S/4HANA Connector
# OData API integration for procurement and logistics data extraction

"""
SAP S/4HANA Connector
====================

Native integration with SAP S/4HANA using OData API.

Modules Supported:
-----------------
- MM (Materials Management): Procurement transactions
- SD (Sales & Distribution): Logistics, transportation
- FI (Financial Accounting): Capital expenditures

Data Extracted:
--------------
1. Procurement (Category 1):
   - Purchase orders
   - Goods receipts
   - Vendor master data
   - Material master data

2. Logistics (Category 4):
   - Inbound deliveries
   - Transportation orders
   - Freight data

3. Capital Goods (Category 2):
   - Fixed asset acquisitions
   - Capital expenditures

Authentication:
--------------
OAuth 2.0 client credentials flow

Usage:
------
```python
from connectors.sap import SAPConnector

connector = SAPConnector(
    endpoint="https://your-sap-instance.com/sap/opu/odata/sap",
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Extract procurement data
data = connector.get_procurement_data(
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```
"""

__version__ = "1.0.0"
