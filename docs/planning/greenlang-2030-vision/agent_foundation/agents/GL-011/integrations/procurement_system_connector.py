# -*- coding: utf-8 -*-
"""
Procurement System Connector for GL-011 FUELCRAFT.

Provides integration with ERP systems (SAP, Oracle, etc.)
for fuel procurement, purchase orders, and inventory management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Purchase order status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    ORDERED = "ordered"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class PurchaseOrder:
    """Fuel purchase order."""
    order_id: str
    fuel_type: str
    quantity_kg: float
    unit_price_usd: float
    total_price_usd: float
    supplier_id: str
    supplier_name: str
    order_date: datetime
    expected_delivery: datetime
    status: OrderStatus
    notes: str = ""


@dataclass
class Supplier:
    """Fuel supplier information."""
    supplier_id: str
    name: str
    fuel_types: List[str]
    lead_time_days: int
    minimum_order_kg: float
    price_per_kg: Dict[str, float]
    rating: float
    active: bool


@dataclass
class InventoryTransaction:
    """Inventory transaction record."""
    transaction_id: str
    fuel_type: str
    quantity_kg: float
    transaction_type: str  # receipt, consumption, adjustment
    reference: str
    timestamp: datetime


class ProcurementSystemConnector:
    """
    Connector for ERP/procurement systems.

    Supports:
    - SAP (via RFC or OData)
    - Oracle
    - REST API
    - Simulation mode

    Example:
        >>> connector = ProcurementSystemConnector(config)
        >>> await connector.connect()
        >>> orders = await connector.get_open_orders()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize procurement connector.

        Args:
            config: Configuration with endpoint, credentials, etc.
        """
        self.config = config
        self.protocol = config.get('protocol', 'simulation')
        self.endpoint = config.get('endpoint', '')
        self.connected = False
        self._client = None

        # Cache
        self._suppliers: Dict[str, Supplier] = {}
        self._orders: Dict[str, PurchaseOrder] = {}

    async def connect(self) -> bool:
        """
        Establish connection to ERP system.

        Returns:
            True if connection successful
        """
        try:
            if self.protocol == 'simulation':
                self.connected = True
                self._init_simulation_data()
                logger.info("Procurement connector in simulation mode")
                return True

            elif self.protocol == 'sap':
                logger.info(f"Connecting to SAP at {self.endpoint}")
                self.connected = True  # Simulated

            elif self.protocol == 'oracle':
                logger.info(f"Connecting to Oracle at {self.endpoint}")
                self.connected = True  # Simulated

            elif self.protocol == 'rest':
                logger.info(f"Connecting to REST API at {self.endpoint}")
                self.connected = True  # Simulated

            return self.connected

        except Exception as e:
            logger.error(f"Procurement connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connection."""
        self.connected = False
        logger.info("Procurement connector disconnected")

    async def get_suppliers(self, fuel_type: Optional[str] = None) -> List[Supplier]:
        """
        Get list of suppliers.

        Args:
            fuel_type: Filter by fuel type if specified

        Returns:
            List of suppliers
        """
        if not self.connected:
            return []

        suppliers = list(self._suppliers.values())

        if fuel_type:
            suppliers = [s for s in suppliers if fuel_type in s.fuel_types]

        return suppliers

    async def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        """Get supplier by ID."""
        return self._suppliers.get(supplier_id)

    async def get_open_orders(self) -> List[PurchaseOrder]:
        """
        Get all open purchase orders.

        Returns:
            List of open orders
        """
        if not self.connected:
            return []

        open_statuses = [
            OrderStatus.DRAFT,
            OrderStatus.SUBMITTED,
            OrderStatus.APPROVED,
            OrderStatus.ORDERED,
            OrderStatus.SHIPPED
        ]

        return [o for o in self._orders.values() if o.status in open_statuses]

    async def get_order(self, order_id: str) -> Optional[PurchaseOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    async def create_order(
        self,
        fuel_type: str,
        quantity_kg: float,
        supplier_id: str
    ) -> Optional[PurchaseOrder]:
        """
        Create a new purchase order.

        Args:
            fuel_type: Type of fuel
            quantity_kg: Quantity in kg
            supplier_id: Supplier ID

        Returns:
            Created purchase order
        """
        if not self.connected:
            return None

        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            logger.error(f"Supplier not found: {supplier_id}")
            return None

        if quantity_kg < supplier.minimum_order_kg:
            logger.warning(f"Quantity below minimum: {quantity_kg} < {supplier.minimum_order_kg}")

        price = supplier.price_per_kg.get(fuel_type, 0.05)
        total = quantity_kg * price

        order_id = f"PO-{len(self._orders) + 1:06d}"
        now = datetime.now(timezone.utc)

        order = PurchaseOrder(
            order_id=order_id,
            fuel_type=fuel_type,
            quantity_kg=quantity_kg,
            unit_price_usd=price,
            total_price_usd=round(total, 2),
            supplier_id=supplier_id,
            supplier_name=supplier.name,
            order_date=now,
            expected_delivery=datetime.fromtimestamp(
                now.timestamp() + supplier.lead_time_days * 86400,
                tz=timezone.utc
            ),
            status=OrderStatus.DRAFT
        )

        self._orders[order_id] = order
        logger.info(f"Created order {order_id} for {quantity_kg} kg {fuel_type}")

        return order

    async def submit_order(self, order_id: str) -> bool:
        """
        Submit order for approval.

        Args:
            order_id: Order ID

        Returns:
            True if successful
        """
        order = self._orders.get(order_id)
        if not order:
            return False

        if order.status != OrderStatus.DRAFT:
            logger.warning(f"Order {order_id} not in draft status")
            return False

        order.status = OrderStatus.SUBMITTED
        logger.info(f"Order {order_id} submitted")
        return True

    async def get_inventory_balance(self, fuel_type: str) -> float:
        """
        Get current inventory balance from ERP.

        Args:
            fuel_type: Type of fuel

        Returns:
            Current balance in kg
        """
        if not self.connected:
            return 0.0

        # In simulation, return random balance
        import random
        random.seed(hash(fuel_type) % 2**32)
        return round(random.uniform(5000, 50000), 2)

    async def record_receipt(
        self,
        order_id: str,
        received_quantity: float
    ) -> Optional[InventoryTransaction]:
        """
        Record goods receipt.

        Args:
            order_id: Purchase order ID
            received_quantity: Quantity received

        Returns:
            Inventory transaction
        """
        order = self._orders.get(order_id)
        if not order:
            return None

        transaction = InventoryTransaction(
            transaction_id=f"GR-{order_id}",
            fuel_type=order.fuel_type,
            quantity_kg=received_quantity,
            transaction_type='receipt',
            reference=order_id,
            timestamp=datetime.now(timezone.utc)
        )

        order.status = OrderStatus.DELIVERED
        logger.info(f"Recorded receipt for {order_id}: {received_quantity} kg")

        return transaction

    def _init_simulation_data(self) -> None:
        """Initialize simulation data."""
        # Create sample suppliers
        self._suppliers = {
            'SUP-001': Supplier(
                supplier_id='SUP-001',
                name='FuelCorp International',
                fuel_types=['natural_gas', 'diesel', 'propane'],
                lead_time_days=5,
                minimum_order_kg=5000,
                price_per_kg={
                    'natural_gas': 0.045,
                    'diesel': 0.95,
                    'propane': 0.55
                },
                rating=4.5,
                active=True
            ),
            'SUP-002': Supplier(
                supplier_id='SUP-002',
                name='GreenBiomass Ltd',
                fuel_types=['biomass', 'wood_pellets', 'biogas'],
                lead_time_days=7,
                minimum_order_kg=10000,
                price_per_kg={
                    'biomass': 0.08,
                    'wood_pellets': 0.12,
                    'biogas': 0.035
                },
                rating=4.2,
                active=True
            ),
            'SUP-003': Supplier(
                supplier_id='SUP-003',
                name='CoalMine Industries',
                fuel_types=['coal'],
                lead_time_days=14,
                minimum_order_kg=50000,
                price_per_kg={'coal': 0.04},
                rating=3.8,
                active=True
            ),
            'SUP-004': Supplier(
                supplier_id='SUP-004',
                name='H2 Solutions',
                fuel_types=['hydrogen'],
                lead_time_days=3,
                minimum_order_kg=500,
                price_per_kg={'hydrogen': 5.00},
                rating=4.7,
                active=True
            )
        }

        logger.info(f"Initialized {len(self._suppliers)} simulation suppliers")
