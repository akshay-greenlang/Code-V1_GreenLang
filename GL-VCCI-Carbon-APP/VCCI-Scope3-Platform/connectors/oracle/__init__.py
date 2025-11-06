# Oracle ERP Cloud Connector
# REST API integration for procurement and supply chain data extraction

"""
Oracle ERP Cloud Connector
==========================

Native integration with Oracle ERP Cloud using REST API.

Modules Supported:
-----------------
- Procurement Cloud: Purchase orders, suppliers
- Supply Chain Management: Logistics, inventory
- Financials Cloud: Capital expenditures

Data Extracted:
--------------
1. Procurement (Category 1):
   - Purchase orders
   - Purchase requisitions
   - Supplier information
   - Item master data

2. Supply Chain (Category 4):
   - Shipments
   - Transportation orders
   - Logistics data

3. Capital Goods (Category 2):
   - Fixed assets
   - Capital projects

Authentication:
--------------
OAuth 2.0 client credentials flow

Usage:
------
```python
from connectors.oracle import OracleConnector

connector = OracleConnector(
    endpoint="https://your-oracle-instance.com",
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
