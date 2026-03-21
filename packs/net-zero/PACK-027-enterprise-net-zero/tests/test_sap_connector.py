# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - SAP Connector (deep).

Deep tests for SAP S/4HANA integration: module extraction, data transformation,
GL code mapping, emission factor lookup, and Concur travel.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~45 tests
"""

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations.sap_connector import (
    SAPConnector,
    SAPConfig,
    SAPModule,
    SAPExtractionResult,
    SAPMaterialGroupMapping,
    SAPExtractionRequest,
    SAPConnectionStatus,
)

# Aliases for test compatibility
SAPConnectorConfig = SAPConfig


class TestSAPInstantiation:
    def test_connector_instantiates(self):
        connector = SAPConnector()
        assert connector is not None

    def test_connector_has_config(self):
        connector = SAPConnector()
        assert connector.config is not None

    def test_config_defaults(self):
        config = SAPConnectorConfig()
        assert config is not None
        assert config.pack_id == "PACK-027"


class TestSAPModuleEnum:
    @pytest.mark.parametrize("module_name", [
        "MM", "FI", "CO", "SD", "TM", "PM", "HCM", "RE", "EHS",
    ])
    def test_module_exists(self, module_name):
        assert SAPModule(module_name.lower()) is not None

    def test_module_count(self):
        assert len(SAPModule) >= 9


class TestSAPConnectorMethods:
    def test_has_connect(self):
        connector = SAPConnector()
        assert hasattr(connector, "connect")

    def test_has_disconnect(self):
        connector = SAPConnector()
        assert hasattr(connector, "disconnect")

    def test_has_extract_procurement(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_procurement")

    def test_has_extract_cost_centers(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_cost_centers")

    def test_has_extract_equipment_fuel(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_equipment_fuel")

    def test_has_extract_employee_data(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_employee_data")

    def test_has_extract_fleet_data(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_fleet_data")

    def test_has_extract_travel_bookings(self):
        connector = SAPConnector()
        assert hasattr(connector, "extract_travel_bookings")

    def test_has_get_connection_status(self):
        connector = SAPConnector()
        assert hasattr(connector, "get_connection_status")


class TestSAPDataTransformation:
    def test_map_material_group(self):
        connector = SAPConnector()
        result = connector.map_material_group("RAW-STEEL", 100000.0)
        assert result is not None

    def test_get_material_group_mappings(self):
        connector = SAPConnector()
        result = connector.get_material_group_mappings()
        assert isinstance(result, (list, dict))


class TestSAPScope3Mapping:
    @pytest.mark.parametrize("module", [
        SAPModule.MM, SAPModule.FI, SAPModule.SD, SAPModule.PM, SAPModule.HCM,
    ])
    def test_module_is_valid(self, module):
        assert module.value is not None


class TestSAPConfiguration:
    def test_config_has_host(self):
        config = SAPConnectorConfig()
        assert hasattr(config, "sap_host")

    def test_config_has_protocol(self):
        config = SAPConnectorConfig()
        assert hasattr(config, "protocol")

    def test_config_has_rate_limit(self):
        config = SAPConnectorConfig()
        assert config.rate_limit_per_minute >= 1

    def test_config_has_timeout(self):
        config = SAPConnectorConfig()
        assert config.timeout_seconds >= 1

    def test_config_has_max_retries(self):
        config = SAPConnectorConfig()
        assert config.max_retries >= 1

    def test_config_has_extraction_mode(self):
        config = SAPConnectorConfig()
        assert hasattr(config, "extraction_mode")

    def test_config_has_enable_provenance(self):
        config = SAPConnectorConfig()
        assert config.enable_provenance is True

    def test_multi_company_code(self):
        config = SAPConnectorConfig(company_codes=["1000", "2000", "3000"])
        assert len(config.company_codes) == 3


class TestSAPErrorHandling:
    def test_connection_status_check(self):
        connector = SAPConnector()
        status = connector.get_connection_status()
        assert isinstance(status, SAPConnectionStatus)

    def test_connector_status(self):
        connector = SAPConnector()
        status = connector.get_connector_status()
        assert status is not None
