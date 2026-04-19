# -*- coding: utf-8 -*-
# SAP Jobs
# Celery tasks for SAP data synchronization

"""
SAP Background Jobs
===================

Celery-based background jobs for SAP S/4HANA data synchronization.

Jobs:
-----
- Delta sync jobs for SAP modules (MM, SD, FI)
- Scheduled extraction jobs
- Data validation jobs
- Error recovery jobs

Usage:
------
```python
from connectors.sap.jobs import sync_purchase_orders, sync_deliveries

# Trigger sync job
result = sync_purchase_orders.delay()

# Check job status
if result.ready():
    print(result.result)
```

Scheduler:
----------
Jobs are scheduled using Celery Beat. See scheduler.py for configuration.
"""

__version__ = "1.0.0"

__all__ = [
    "sync_purchase_orders",
    "sync_deliveries",
    "sync_capital_goods",
]
