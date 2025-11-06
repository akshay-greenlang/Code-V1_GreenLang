# Workday Connector
# REST API integration for business travel and employee commuting data

"""
Workday Connector
================

Native integration with Workday using REST API.

Modules Supported:
-----------------
- HCM (Human Capital Management): Employee data, travel, commuting
- Financial Management: Expense reports

Data Extracted:
--------------
1. Business Travel (Category 6):
   - Expense reports (travel expenses)
   - Flight bookings
   - Hotel stays
   - Ground transportation

2. Employee Commuting (Category 7):
   - Employee locations
   - Commuting patterns
   - Work-from-home data

Authentication:
--------------
OAuth 2.0 client credentials flow

Usage:
------
```python
from connectors.workday import WorkdayConnector

connector = WorkdayConnector(
    endpoint="https://wd2-impl-services1.workday.com",
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Extract business travel data
travel_data = connector.get_travel_data(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Extract employee commuting data
commuting_data = connector.get_commuting_data(
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```
"""

__version__ = "1.0.0"
