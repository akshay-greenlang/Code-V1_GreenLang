"""
GL-011 FUELCRAFT - ERP Connector

Enterprise Resource Planning system integration for:
- Procurement system exports (purchase orders, requisitions)
- Contract data imports (terms, pricing, quantities)
- Order recommendation publishing
- Delivery schedule synchronization

Supported ERP Systems:
- SAP S/4HANA (via OData/RFC)
- Oracle ERP Cloud (via REST)
- Microsoft Dynamics 365 (via Dataverse)
- Generic REST/SOAP interfaces

Features:
- Automatic retry with exponential backoff
- Circuit breaker per IEC 61511
- Data transformation and mapping
- Audit logging for all transactions
- Idempotent operations via correlation IDs
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import hashlib
import json
import logging
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ERPSystem(str, Enum):
    """Supported ERP systems."""
    SAP_S4HANA = "sap_s4hana"
    ORACLE_CLOUD = "oracle_cloud"
    DYNAMICS_365 = "dynamics_365"
    GENERIC_REST = "generic_rest"


class OrderStatus(str, Enum):
    """Procurement order status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    CONFIRMED = "confirmed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    INVOICED = "invoiced"
    CANCELLED = "cancelled"


class DeliveryStatus(str, Enum):
    """Delivery status."""
    SCHEDULED = "scheduled"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"
    UNLOADING = "unloading"
    COMPLETE = "complete"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class ConnectionState(str, Enum):
    """ERP connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# =============================================================================
# Configuration
# =============================================================================

class ERPConfig(BaseModel):
    """ERP connector configuration."""
    system_type: ERPSystem = Field(..., description="ERP system type")
    base_url: str = Field(..., description="ERP API base URL")

    # Authentication
    auth_type: str = Field("oauth2", description="oauth2, basic, api_key")
    client_id: Optional[str] = Field(None)
    client_secret: Optional[str] = Field(None)  # From vault
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)  # From vault
    api_key: Optional[str] = Field(None)  # From vault

    # OAuth settings
    token_url: Optional[str] = Field(None)
    scope: Optional[str] = Field(None)

    # Connection settings
    timeout_seconds: int = Field(30)
    max_retries: int = Field(3)
    retry_delay_seconds: float = Field(1.0)
    retry_backoff_factor: float = Field(2.0)

    # Circuit breaker
    circuit_breaker_threshold: int = Field(5)
    circuit_breaker_timeout_seconds: int = Field(60)

    # SAP-specific settings
    sap_client: Optional[str] = Field(None, description="SAP client number")
    sap_language: str = Field("EN")

    # Mapping configuration
    company_code: str = Field(..., description="Company code in ERP")
    plant_code: Optional[str] = Field(None)
    purchasing_org: Optional[str] = Field(None)


# =============================================================================
# Data Models
# =============================================================================

class ProcurementOrder(BaseModel):
    """Procurement order model."""
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_id: Optional[str] = Field(None, description="ERP order number")
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Order details
    fuel_type: str
    quantity_mmbtu: float = Field(..., gt=0)
    unit_price_usd: float = Field(..., ge=0)
    total_price_usd: float = Field(..., ge=0)
    currency: str = Field("USD")

    # Dates
    order_date: datetime
    requested_delivery_date: datetime
    confirmed_delivery_date: Optional[datetime] = None

    # Parties
    supplier_id: str
    supplier_name: str
    delivery_location: str
    ship_to_address: Optional[str] = None

    # Contract reference
    contract_id: Optional[str] = None
    contract_line_item: Optional[str] = None

    # Status
    status: OrderStatus = Field(OrderStatus.DRAFT)

    # Metadata
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

    # Optimization reference
    optimization_run_id: Optional[str] = Field(None, description="Source optimization run")

    def compute_hash(self) -> str:
        """Compute hash for idempotency."""
        data = f"{self.fuel_type}|{self.quantity_mmbtu}|{self.supplier_id}|{self.requested_delivery_date.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ContractData(BaseModel):
    """Contract data model."""
    contract_id: str
    external_id: Optional[str] = Field(None, description="ERP contract number")

    # Contract details
    contract_type: str = Field("purchase", description="purchase, blanket, framework")
    fuel_type: str
    supplier_id: str
    supplier_name: str

    # Quantities
    total_quantity_mmbtu: float = Field(..., ge=0)
    minimum_quantity_mmbtu: float = Field(0.0, ge=0)
    maximum_quantity_mmbtu: Optional[float] = None
    take_or_pay_quantity_mmbtu: Optional[float] = None

    # Pricing
    price_type: str = Field("fixed", description="fixed, index, formula")
    base_price_usd_mmbtu: Optional[float] = None
    price_formula: Optional[str] = None
    price_index: Optional[str] = None
    price_adjustment_factor: float = Field(1.0)

    # Dates
    start_date: datetime
    end_date: datetime
    amendment_date: Optional[datetime] = None

    # Delivery
    delivery_terms: str = Field("DAP", description="Incoterms")
    delivery_location: str
    max_daily_quantity_mmbtu: Optional[float] = None
    lead_time_days: int = Field(1, ge=0)

    # Status
    is_active: bool = Field(True)
    remaining_quantity_mmbtu: Optional[float] = None

    # Usage tracking
    utilized_quantity_mmbtu: float = Field(0.0, ge=0)

    @validator("end_date")
    def end_after_start(cls, v: datetime, values: Dict[str, Any]) -> datetime:
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    def is_within_validity(self, check_date: Optional[datetime] = None) -> bool:
        """Check if contract is valid for given date."""
        check = check_date or datetime.now(timezone.utc)
        return self.start_date <= check <= self.end_date and self.is_active

    def get_available_quantity(self) -> float:
        """Get remaining available quantity."""
        if self.remaining_quantity_mmbtu is not None:
            return self.remaining_quantity_mmbtu
        return self.total_quantity_mmbtu - self.utilized_quantity_mmbtu

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DeliverySchedule(BaseModel):
    """Delivery schedule model."""
    delivery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_id: Optional[str] = Field(None, description="ERP delivery number")
    order_id: str = Field(..., description="Related order ID")

    # Delivery details
    fuel_type: str
    quantity_mmbtu: float = Field(..., gt=0)
    supplier_id: str

    # Schedule
    scheduled_date: datetime
    scheduled_time_window_start: Optional[datetime] = None
    scheduled_time_window_end: Optional[datetime] = None
    actual_arrival_time: Optional[datetime] = None

    # Logistics
    carrier: Optional[str] = None
    vehicle_id: Optional[str] = None
    driver_name: Optional[str] = None
    tracking_number: Optional[str] = None

    # Delivery location
    delivery_location: str
    tank_id: Optional[str] = None

    # Status
    status: DeliveryStatus = Field(DeliveryStatus.SCHEDULED)

    # Quality
    quality_check_required: bool = Field(True)
    quality_check_passed: Optional[bool] = None
    quality_parameters: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SupplierData(BaseModel):
    """Supplier data model."""
    supplier_id: str
    external_id: Optional[str] = Field(None, description="ERP vendor number")
    name: str
    legal_name: Optional[str] = None

    # Contact
    address: str
    city: str
    country: str
    postal_code: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None

    # Business
    fuel_types: List[str] = Field(default=[])
    payment_terms: str = Field("Net30")
    currency: str = Field("USD")

    # Status
    is_active: bool = Field(True)
    is_approved: bool = Field(True)
    credit_limit_usd: Optional[float] = None

    # Performance
    on_time_delivery_rate: Optional[float] = None
    quality_rating: Optional[float] = None


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ERPCircuitBreaker:
    """Circuit breaker for ERP connections."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count_half_open = 0

    @property
    def state(self) -> CircuitBreakerState:
        import time

        if self._state == CircuitBreakerState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout_seconds:
                self._state = CircuitBreakerState.HALF_OPEN

        return self._state

    def allow_request(self) -> bool:
        return self.state != CircuitBreakerState.OPEN

    def record_success(self) -> None:
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count_half_open += 1
            if self._success_count_half_open >= 3:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._success_count_half_open = 0
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        import time

        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            self._success_count_half_open = 0
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN


# =============================================================================
# ERP Connector
# =============================================================================

class ERPConnector:
    """
    ERP connector for procurement and contract management.

    Provides integration with ERP systems for:
    - Creating and managing procurement orders
    - Retrieving contract data
    - Synchronizing delivery schedules
    - Publishing order recommendations

    Example:
        config = ERPConfig(
            system_type=ERPSystem.SAP_S4HANA,
            base_url="https://sap.company.com/sap/opu/odata/sap/",
            company_code="1000",
        )

        connector = ERPConnector(config)
        await connector.connect()

        # Get active contracts
        contracts = await connector.get_contracts(fuel_type="natural_gas")

        # Create order from recommendation
        order = await connector.create_order(
            fuel_type="natural_gas",
            quantity_mmbtu=1000.0,
            supplier_id="SUPPLIER-001",
        )
    """

    def __init__(
        self,
        config: ERPConfig,
        vault_client: Optional[Any] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
    ) -> None:
        """Initialize ERP connector."""
        self.config = config
        self.vault_client = vault_client
        self._on_state_change = on_state_change

        # Retrieve credentials from vault
        if vault_client:
            self._load_credentials_from_vault()

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Circuit breaker
        self._circuit_breaker = ERPCircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            reset_timeout_seconds=config.circuit_breaker_timeout_seconds,
        )

        # Cache
        self._contracts_cache: Dict[str, ContractData] = {}
        self._suppliers_cache: Dict[str, SupplierData] = {}
        self._cache_ttl_seconds = 300

        # Statistics
        self._stats = {
            "requests": 0,
            "errors": 0,
            "orders_created": 0,
            "contracts_fetched": 0,
            "deliveries_synced": 0,
        }

        logger.info(f"ERP connector initialized for {config.system_type.value}")

    def _load_credentials_from_vault(self) -> None:
        """Load credentials from vault."""
        try:
            if self.config.auth_type == "oauth2":
                self.config.client_secret = self.vault_client.get_secret(
                    f"erp/{self.config.system_type.value}/client_secret"
                )
            elif self.config.auth_type == "basic":
                self.config.password = self.vault_client.get_secret(
                    f"erp/{self.config.system_type.value}/password"
                )
            elif self.config.auth_type == "api_key":
                self.config.api_key = self.vault_client.get_secret(
                    f"erp/{self.config.system_type.value}/api_key"
                )
        except Exception as e:
            logger.warning(f"Failed to load credentials from vault: {e}")

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    def _set_state(self, new_state: ConnectionState) -> None:
        old_state = self._state
        self._state = new_state
        if old_state != new_state:
            logger.info(f"ERP connector state: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                self._on_state_change(new_state)

    async def connect(self) -> bool:
        """Connect to ERP system and authenticate."""
        if not self._circuit_breaker.allow_request():
            logger.warning("Circuit breaker open, connection rejected")
            return False

        self._set_state(ConnectionState.CONNECTING)

        try:
            # Authenticate based on auth type
            if self.config.auth_type == "oauth2":
                await self._authenticate_oauth2()
            elif self.config.auth_type == "basic":
                # Basic auth doesn't require pre-authentication
                self._access_token = "basic_auth_placeholder"

            self._set_state(ConnectionState.CONNECTED)
            self._circuit_breaker.record_success()
            logger.info(f"Connected to ERP: {self.config.base_url}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            self._set_state(ConnectionState.ERROR)
            logger.error(f"ERP connection failed: {e}")
            return False

    async def _authenticate_oauth2(self) -> None:
        """Authenticate using OAuth2."""
        # In production, use httpx or aiohttp:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         self.config.token_url,
        #         data={
        #             "grant_type": "client_credentials",
        #             "client_id": self.config.client_id,
        #             "client_secret": self.config.client_secret,
        #             "scope": self.config.scope,
        #         }
        #     )
        #     token_data = response.json()
        #     self._access_token = token_data["access_token"]
        #     self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=token_data["expires_in"])

        # Mock implementation
        self._access_token = "mock_access_token"
        self._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        logger.info("OAuth2 authentication successful")

    async def disconnect(self) -> None:
        """Disconnect from ERP system."""
        self._access_token = None
        self._token_expiry = None
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Disconnected from ERP")

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token."""
        if self._state != ConnectionState.CONNECTED:
            await self.connect()

        if self.config.auth_type == "oauth2" and self._token_expiry:
            if datetime.now(timezone.utc) >= self._token_expiry - timedelta(minutes=5):
                await self._authenticate_oauth2()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to ERP with retry logic."""
        await self._ensure_authenticated()

        if not self._circuit_breaker.allow_request():
            raise ConnectionError("Circuit breaker open")

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        self._stats["requests"] += 1

        for attempt in range(self.config.max_retries):
            try:
                # In production, use httpx:
                # headers = {"Authorization": f"Bearer {self._access_token}"}
                # async with httpx.AsyncClient() as client:
                #     if method == "GET":
                #         response = await client.get(url, headers=headers, params=params, timeout=self.config.timeout_seconds)
                #     elif method == "POST":
                #         response = await client.post(url, headers=headers, json=data, timeout=self.config.timeout_seconds)
                #     response.raise_for_status()
                #     return response.json()

                # Mock implementation
                self._circuit_breaker.record_success()
                return {"success": True, "data": data or {}}

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self._stats["errors"] += 1
                    self._circuit_breaker.record_failure()
                    raise

                delay = self.config.retry_delay_seconds * (self.config.retry_backoff_factor ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    # =========================================================================
    # Contract Operations
    # =========================================================================

    async def get_contracts(
        self,
        fuel_type: Optional[str] = None,
        supplier_id: Optional[str] = None,
        active_only: bool = True,
    ) -> List[ContractData]:
        """
        Get contracts from ERP system.

        Args:
            fuel_type: Filter by fuel type
            supplier_id: Filter by supplier
            active_only: Only return active contracts

        Returns:
            List of contract data
        """
        logger.info(f"Fetching contracts: fuel_type={fuel_type}, supplier={supplier_id}")

        try:
            # Build query parameters
            params = {"company_code": self.config.company_code}
            if fuel_type:
                params["fuel_type"] = fuel_type
            if supplier_id:
                params["supplier_id"] = supplier_id
            if active_only:
                params["is_active"] = "true"

            # Make request (would vary by ERP system)
            # response = await self._make_request("GET", "contracts", params=params)

            # Return mock contracts for demonstration
            contracts = self._generate_mock_contracts(fuel_type, supplier_id)

            self._stats["contracts_fetched"] += len(contracts)

            # Update cache
            for contract in contracts:
                self._contracts_cache[contract.contract_id] = contract

            return contracts

        except Exception as e:
            logger.error(f"Failed to fetch contracts: {e}")
            raise

    def _generate_mock_contracts(
        self,
        fuel_type: Optional[str],
        supplier_id: Optional[str],
    ) -> List[ContractData]:
        """Generate mock contracts for demonstration."""
        now = datetime.now(timezone.utc)
        contracts = []

        fuel_types = [fuel_type] if fuel_type else ["natural_gas", "fuel_oil_2"]

        for i, ft in enumerate(fuel_types):
            contracts.append(ContractData(
                contract_id=f"CONTRACT-{ft.upper()}-{i+1:03d}",
                external_id=f"4500000{i+1}",
                fuel_type=ft,
                supplier_id=supplier_id or f"SUPPLIER-{i+1:03d}",
                supplier_name=f"Fuel Supplier {i+1}",
                total_quantity_mmbtu=50000.0,
                minimum_quantity_mmbtu=5000.0,
                take_or_pay_quantity_mmbtu=10000.0,
                price_type="index",
                base_price_usd_mmbtu=3.50 if ft == "natural_gas" else 15.0,
                price_index="Henry Hub" if ft == "natural_gas" else "Platts",
                start_date=now - timedelta(days=30),
                end_date=now + timedelta(days=335),
                delivery_terms="DAP",
                delivery_location="Site Tank Farm",
                max_daily_quantity_mmbtu=2000.0,
                lead_time_days=2,
                utilized_quantity_mmbtu=15000.0,
            ))

        return contracts

    async def get_contract(self, contract_id: str) -> Optional[ContractData]:
        """Get single contract by ID."""
        # Check cache first
        if contract_id in self._contracts_cache:
            return self._contracts_cache[contract_id]

        contracts = await self.get_contracts()
        for contract in contracts:
            if contract.contract_id == contract_id:
                return contract

        return None

    # =========================================================================
    # Order Operations
    # =========================================================================

    async def create_order(
        self,
        fuel_type: str,
        quantity_mmbtu: float,
        supplier_id: str,
        requested_delivery_date: datetime,
        unit_price_usd: Optional[float] = None,
        contract_id: Optional[str] = None,
        optimization_run_id: Optional[str] = None,
        created_by: str = "fuelcraft",
    ) -> ProcurementOrder:
        """
        Create procurement order in ERP.

        Args:
            fuel_type: Type of fuel
            quantity_mmbtu: Order quantity
            supplier_id: Supplier identifier
            requested_delivery_date: Requested delivery date
            unit_price_usd: Unit price (optional, may come from contract)
            contract_id: Reference contract
            optimization_run_id: Source optimization run
            created_by: User/system creating order

        Returns:
            Created procurement order
        """
        logger.info(
            f"Creating order: {quantity_mmbtu} MMBtu {fuel_type} from {supplier_id}"
        )

        # Get contract price if not provided
        if unit_price_usd is None and contract_id:
            contract = await self.get_contract(contract_id)
            if contract and contract.base_price_usd_mmbtu:
                unit_price_usd = contract.base_price_usd_mmbtu

        unit_price_usd = unit_price_usd or 5.0  # Default fallback

        # Create order object
        order = ProcurementOrder(
            fuel_type=fuel_type,
            quantity_mmbtu=quantity_mmbtu,
            unit_price_usd=unit_price_usd,
            total_price_usd=quantity_mmbtu * unit_price_usd,
            order_date=datetime.now(timezone.utc),
            requested_delivery_date=requested_delivery_date,
            supplier_id=supplier_id,
            supplier_name=f"Supplier {supplier_id}",
            delivery_location="Site Tank Farm",
            contract_id=contract_id,
            status=OrderStatus.DRAFT,
            created_by=created_by,
            optimization_run_id=optimization_run_id,
        )

        try:
            # Submit to ERP
            # response = await self._make_request("POST", "purchase_orders", data=order.dict())
            # order.external_id = response.get("order_number")
            # order.status = OrderStatus.SUBMITTED

            # Mock ERP response
            order.external_id = f"PO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            order.status = OrderStatus.SUBMITTED

            self._stats["orders_created"] += 1
            logger.info(f"Order created: {order.order_id} (ERP: {order.external_id})")

            return order

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise

    async def get_order_status(self, order_id: str) -> Optional[ProcurementOrder]:
        """Get order status from ERP."""
        logger.info(f"Fetching order status: {order_id}")
        # In production, query ERP for order status
        return None

    async def update_order_status(
        self,
        order_id: str,
        new_status: OrderStatus,
    ) -> bool:
        """Update order status in ERP."""
        logger.info(f"Updating order {order_id} status to {new_status.value}")
        # In production, update ERP order status
        return True

    async def cancel_order(self, order_id: str, reason: str) -> bool:
        """Cancel order in ERP."""
        logger.info(f"Cancelling order {order_id}: {reason}")
        return await self.update_order_status(order_id, OrderStatus.CANCELLED)

    # =========================================================================
    # Delivery Operations
    # =========================================================================

    async def get_delivery_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        fuel_type: Optional[str] = None,
    ) -> List[DeliverySchedule]:
        """
        Get delivery schedule from ERP.

        Args:
            start_date: Schedule start date
            end_date: Schedule end date
            fuel_type: Optional fuel type filter

        Returns:
            List of scheduled deliveries
        """
        logger.info(f"Fetching delivery schedule: {start_date.date()} to {end_date.date()}")

        # Mock deliveries for demonstration
        deliveries = []

        current_date = start_date
        delivery_num = 1
        while current_date <= end_date:
            deliveries.append(DeliverySchedule(
                delivery_id=f"DEL-{current_date.strftime('%Y%m%d')}-{delivery_num:03d}",
                order_id=f"ORDER-{delivery_num:03d}",
                fuel_type=fuel_type or "natural_gas",
                quantity_mmbtu=500.0 + (delivery_num * 100),
                supplier_id="SUPPLIER-001",
                scheduled_date=current_date,
                delivery_location="Site Tank Farm",
                tank_id="TANK-001",
                status=DeliveryStatus.SCHEDULED,
            ))

            current_date += timedelta(days=1)
            delivery_num += 1

        self._stats["deliveries_synced"] += len(deliveries)
        return deliveries

    async def update_delivery_status(
        self,
        delivery_id: str,
        status: DeliveryStatus,
        actual_arrival_time: Optional[datetime] = None,
        quality_parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update delivery status in ERP."""
        logger.info(f"Updating delivery {delivery_id} status to {status.value}")
        # In production, update ERP delivery record
        return True

    # =========================================================================
    # Supplier Operations
    # =========================================================================

    async def get_suppliers(
        self,
        fuel_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[SupplierData]:
        """Get suppliers from ERP."""
        logger.info(f"Fetching suppliers: fuel_type={fuel_type}")

        # Mock suppliers
        suppliers = [
            SupplierData(
                supplier_id="SUPPLIER-001",
                external_id="V10001",
                name="Natural Gas Co.",
                address="123 Energy Blvd",
                city="Houston",
                country="US",
                fuel_types=["natural_gas"],
                on_time_delivery_rate=0.95,
                quality_rating=4.5,
            ),
            SupplierData(
                supplier_id="SUPPLIER-002",
                external_id="V10002",
                name="Oil Products Inc.",
                address="456 Refinery Way",
                city="New Orleans",
                country="US",
                fuel_types=["fuel_oil_2", "fuel_oil_6", "diesel"],
                on_time_delivery_rate=0.92,
                quality_rating=4.2,
            ),
        ]

        if fuel_type:
            suppliers = [s for s in suppliers if fuel_type in s.fuel_types]

        return suppliers

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "state": self._state.value,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "contracts_cached": len(self._contracts_cache),
            "suppliers_cached": len(self._suppliers_cache),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.is_connected else "unhealthy",
            "state": self._state.value,
            "circuit_breaker": self._circuit_breaker.state.value,
            "requests": self._stats["requests"],
            "errors": self._stats["errors"],
        }
