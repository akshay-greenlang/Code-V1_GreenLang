"""
External System Connectors

Connectors for integrating with external enterprise systems:
- Azure IoT Hub (IoT sensors and meters)
- SAP ERP (enterprise resource planning)
- Generic ERP systems (Oracle, Microsoft Dynamics, etc.)
"""

from .azure_iot_connector import AzureIoTConnector
from .sap_connector import SAPConnector
from .generic_erp_connector import GenericERPConnector

__all__ = [
    'AzureIoTConnector',
    'SAPConnector',
    'GenericERPConnector',
]
