"""
GL-016 Waterguard CMMS Integration Package

Provides integration with Computerized Maintenance Management Systems (CMMS)
for work order creation, asset tracking, and maintenance scheduling.
"""

from integrations.cmms.cmms_integration import (
    CMMSConnector,
    CMMSConfig,
    WorkOrderTrigger,
)
from integrations.cmms.cmms_schemas import (
    WorkOrder,
    WorkOrderType,
    WorkOrderPriority,
    WorkOrderStatus,
    Asset,
    AssetType,
    MaintenanceTask,
)

__all__ = [
    # Connector
    "CMMSConnector",
    "CMMSConfig",
    "WorkOrderTrigger",
    # Schemas
    "WorkOrder",
    "WorkOrderType",
    "WorkOrderPriority",
    "WorkOrderStatus",
    "Asset",
    "AssetType",
    "MaintenanceTask",
]

__version__ = "1.0.0"
