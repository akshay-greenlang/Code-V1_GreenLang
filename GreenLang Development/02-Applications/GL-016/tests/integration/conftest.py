# -*- coding: utf-8 -*-
"""
Integration test fixtures for GL-016 WATERGUARD.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta


@pytest.fixture
async def running_scada_server():
    """Start mock SCADA server for integration tests."""
    from tests.integration.mock_servers import MockSCADAServer

    server = MockSCADAServer(host='localhost', port=4840)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def running_analyzer():
    """Start mock water analyzer for integration tests."""
    from tests.integration.mock_servers import MockWaterAnalyzer

    analyzer = MockWaterAnalyzer(analyzer_id='ANALYZER-001')
    await analyzer.start()
    yield analyzer
    await analyzer.stop()


@pytest.fixture
async def running_dosing_system():
    """Start mock chemical dosing system for integration tests."""
    from tests.integration.mock_servers import MockChemicalDosingSystem

    dosing_system = MockChemicalDosingSystem(system_id='DOSING-001')
    await dosing_system.start()
    yield dosing_system
    await dosing_system.stop()


@pytest.fixture
async def running_erp_system():
    """Start mock ERP system for integration tests."""
    from tests.integration.mock_servers import MockERPSystem

    erp = MockERPSystem(host='localhost', port=8000)
    await erp.start()
    yield erp
    await erp.stop()


@pytest.fixture
def integration_agent_config():
    """Agent configuration for integration tests."""
    return {
        'agent_id': 'GL-016-INTEGRATION',
        'agent_name': 'IntegrationTestWaterGuard',
        'version': '1.0.0-integration',
        'scada_host': 'localhost',
        'scada_port': 4840,
        'erp_host': 'localhost',
        'erp_port': 8000,
        'enable_real_connections': False,
        'test_mode': True
    }
