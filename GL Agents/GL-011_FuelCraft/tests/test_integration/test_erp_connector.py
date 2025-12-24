# -*- coding: utf-8 -*-
"""
Integration Tests for ERP Connector

Tests ERP system integration including:
- Contract retrieval
- Order creation
- Delivery schedule synchronization
- Circuit breaker behavior
- Connection state management

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.erp_connector import (
    ERPConnector,
    ERPConfig,
    ERPSystem,
    ConnectionState,
    OrderStatus,
    DeliveryStatus,
    ProcurementOrder,
    ContractData,
    DeliverySchedule,
    SupplierData,
    ERPCircuitBreaker,
    CircuitBreakerState,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPConnectorLifecycle:
    """Tests for ERP connector lifecycle management."""

    async def test_connector_connect_disconnect(self, erp_config):
        """Test connector connect and disconnect lifecycle."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        assert connector.state == ConnectionState.DISCONNECTED

        result = await connector.connect()
        assert result is True
        assert connector.state == ConnectionState.CONNECTED
        assert connector.is_connected is True

        await connector.disconnect()
        assert connector.state == ConnectionState.DISCONNECTED

    async def test_connector_state_callback(self, erp_config):
        """Test state change callback is invoked."""
        state_changes = []

        def on_state_change(new_state):
            state_changes.append(new_state)

        config = ERPConfig(**erp_config)
        connector = ERPConnector(config, on_state_change=on_state_change)

        await connector.connect()
        await connector.disconnect()

        assert ConnectionState.CONNECTING in state_changes
        assert ConnectionState.CONNECTED in state_changes
        assert ConnectionState.DISCONNECTED in state_changes


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPContractRetrieval:
    """Tests for contract data retrieval."""

    async def test_get_contracts(self, erp_config):
        """Test retrieving contracts from ERP."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        contracts = await connector.get_contracts()

        assert isinstance(contracts, list)
        # Mock implementation returns some contracts
        for contract in contracts:
            assert isinstance(contract, ContractData)
            assert contract.contract_id is not None
            assert contract.fuel_type is not None

        await connector.disconnect()

    async def test_get_contracts_filtered_by_fuel(self, erp_config):
        """Test retrieving contracts filtered by fuel type."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        contracts = await connector.get_contracts(fuel_type="natural_gas")

        for contract in contracts:
            assert contract.fuel_type == "natural_gas"

        await connector.disconnect()

    async def test_get_single_contract(self, erp_config):
        """Test retrieving single contract by ID."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        contracts = await connector.get_contracts()
        if contracts:
            contract_id = contracts[0].contract_id
            contract = await connector.get_contract(contract_id)

            assert contract is not None
            assert contract.contract_id == contract_id

        await connector.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPOrderCreation:
    """Tests for procurement order creation."""

    async def test_create_order(self, erp_config):
        """Test creating procurement order in ERP."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        order = await connector.create_order(
            fuel_type="natural_gas",
            quantity_mmbtu=1000.0,
            supplier_id="SUPPLIER-001",
            requested_delivery_date=datetime.now(timezone.utc) + timedelta(days=7),
            unit_price_usd=3.50,
            contract_id="CONTRACT-001",
            optimization_run_id="RUN-001",
            created_by="fuelcraft",
        )

        assert order is not None
        assert order.fuel_type == "natural_gas"
        assert order.quantity_mmbtu == 1000.0
        assert order.status == OrderStatus.SUBMITTED
        assert order.external_id is not None

        await connector.disconnect()

    async def test_create_order_calculates_total(self, erp_config):
        """Test order creation calculates total price."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        order = await connector.create_order(
            fuel_type="diesel",
            quantity_mmbtu=500.0,
            supplier_id="SUPPLIER-002",
            requested_delivery_date=datetime.now(timezone.utc) + timedelta(days=3),
            unit_price_usd=15.0,
        )

        assert order.total_price_usd == 500.0 * 15.0

        await connector.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPDeliverySchedule:
    """Tests for delivery schedule synchronization."""

    async def test_get_delivery_schedule(self, erp_config):
        """Test retrieving delivery schedule from ERP."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        now = datetime.now(timezone.utc)
        deliveries = await connector.get_delivery_schedule(
            start_date=now,
            end_date=now + timedelta(days=7),
        )

        assert isinstance(deliveries, list)
        for delivery in deliveries:
            assert isinstance(delivery, DeliverySchedule)
            assert delivery.delivery_id is not None
            assert delivery.scheduled_date >= now

        await connector.disconnect()

    async def test_get_delivery_schedule_filtered(self, erp_config):
        """Test retrieving filtered delivery schedule."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        now = datetime.now(timezone.utc)
        deliveries = await connector.get_delivery_schedule(
            start_date=now,
            end_date=now + timedelta(days=7),
            fuel_type="natural_gas",
        )

        for delivery in deliveries:
            assert delivery.fuel_type == "natural_gas"

        await connector.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPSuppliers:
    """Tests for supplier data retrieval."""

    async def test_get_suppliers(self, erp_config):
        """Test retrieving suppliers from ERP."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        suppliers = await connector.get_suppliers()

        assert isinstance(suppliers, list)
        for supplier in suppliers:
            assert isinstance(supplier, SupplierData)
            assert supplier.supplier_id is not None
            assert supplier.name is not None

        await connector.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPCircuitBreaker:
    """Tests for ERP circuit breaker."""

    async def test_circuit_breaker_blocks_on_open(self, erp_config):
        """Test circuit breaker blocks connection when open."""
        config = ERPConfig(
            **erp_config,
            circuit_breaker_threshold=2,
            circuit_breaker_timeout_seconds=1,
        )
        connector = ERPConnector(config)

        # Manually open circuit breaker
        connector._circuit_breaker._failure_count = 5
        connector._circuit_breaker._state = CircuitBreakerState.OPEN

        result = await connector.connect()

        assert result is False


@pytest.mark.integration
@pytest.mark.asyncio
class TestERPStatistics:
    """Tests for ERP connector statistics."""

    async def test_get_statistics(self, erp_config):
        """Test retrieving connector statistics."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()
        await connector.get_contracts()

        stats = connector.get_statistics()

        assert "contracts_fetched" in stats
        assert "state" in stats
        assert "circuit_breaker_state" in stats

        await connector.disconnect()

    async def test_health_check(self, erp_config):
        """Test health check endpoint."""
        config = ERPConfig(**erp_config)
        connector = ERPConnector(config)

        await connector.connect()

        health = await connector.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
        assert "state" in health

        await connector.disconnect()


@pytest.mark.integration
class TestERPDataModels:
    """Tests for ERP data model classes."""

    def test_procurement_order_hash(self):
        """Test procurement order hash computation."""
        order = ProcurementOrder(
            fuel_type="natural_gas",
            quantity_mmbtu=1000.0,
            unit_price_usd=3.50,
            total_price_usd=3500.0,
            order_date=datetime.now(timezone.utc),
            requested_delivery_date=datetime.now(timezone.utc) + timedelta(days=7),
            supplier_id="SUPPLIER-001",
            supplier_name="Natural Gas Co.",
            delivery_location="Site Tank Farm",
            created_by="fuelcraft",
        )

        hash_value = order.compute_hash()

        assert hash_value is not None
        assert len(hash_value) == 16

    def test_contract_validity_check(self):
        """Test contract validity date check."""
        now = datetime.now(timezone.utc)

        contract = ContractData(
            contract_id="CONTRACT-001",
            fuel_type="natural_gas",
            supplier_id="SUPPLIER-001",
            supplier_name="Natural Gas Co.",
            total_quantity_mmbtu=50000.0,
            start_date=now - timedelta(days=30),
            end_date=now + timedelta(days=335),
            delivery_terms="DAP",
            delivery_location="Site Tank Farm",
        )

        assert contract.is_within_validity() is True
        assert contract.is_within_validity(now - timedelta(days=60)) is False

    def test_contract_available_quantity(self):
        """Test contract available quantity calculation."""
        now = datetime.now(timezone.utc)

        contract = ContractData(
            contract_id="CONTRACT-001",
            fuel_type="natural_gas",
            supplier_id="SUPPLIER-001",
            supplier_name="Natural Gas Co.",
            total_quantity_mmbtu=50000.0,
            utilized_quantity_mmbtu=15000.0,
            start_date=now - timedelta(days=30),
            end_date=now + timedelta(days=335),
            delivery_terms="DAP",
            delivery_location="Site Tank Farm",
        )

        available = contract.get_available_quantity()

        assert available == 50000.0 - 15000.0


@pytest.mark.integration
class TestERPSystemEnum:
    """Tests for ERPSystem enumeration."""

    def test_erp_system_values(self):
        """Test ERPSystem enum values."""
        assert ERPSystem.SAP_S4HANA.value == "sap_s4hana"
        assert ERPSystem.ORACLE_CLOUD.value == "oracle_cloud"
        assert ERPSystem.DYNAMICS_365.value == "dynamics_365"
        assert ERPSystem.GENERIC_REST.value == "generic_rest"


@pytest.mark.integration
class TestOrderStatusEnum:
    """Tests for OrderStatus enumeration."""

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.DRAFT.value == "draft"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.APPROVED.value == "approved"
        assert OrderStatus.CONFIRMED.value == "confirmed"
        assert OrderStatus.DELIVERED.value == "delivered"
        assert OrderStatus.CANCELLED.value == "cancelled"


@pytest.mark.integration
class TestDeliveryStatusEnum:
    """Tests for DeliveryStatus enumeration."""

    def test_delivery_status_values(self):
        """Test DeliveryStatus enum values."""
        assert DeliveryStatus.SCHEDULED.value == "scheduled"
        assert DeliveryStatus.DISPATCHED.value == "dispatched"
        assert DeliveryStatus.IN_TRANSIT.value == "in_transit"
        assert DeliveryStatus.ARRIVED.value == "arrived"
        assert DeliveryStatus.COMPLETE.value == "complete"
