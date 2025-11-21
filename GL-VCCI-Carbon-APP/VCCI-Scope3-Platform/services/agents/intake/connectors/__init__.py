# -*- coding: utf-8 -*-
"""
ERP Connectors

Integration stubs for SAP, Oracle, Workday.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from .base import BaseConnector
from .sap_connector import SAPConnector
from .oracle_connector import OracleConnector
from .workday_connector import WorkdayConnector

__all__ = ["BaseConnector", "SAPConnector", "OracleConnector", "WorkdayConnector"]
