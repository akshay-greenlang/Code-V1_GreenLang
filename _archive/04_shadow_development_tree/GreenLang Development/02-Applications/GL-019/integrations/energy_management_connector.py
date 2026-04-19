"""
Energy Management System Integration Module for GL-019 HEATSCHEDULER

Provides enterprise-grade connectivity to energy management systems for
real-time pricing, demand response, grid operator signals, and energy metering.

Integration Points:
- Real-time energy pricing feeds (LMP, wholesale markets)
- Demand response signal handling (OpenADR, proprietary)
- Grid operator integration (ISO/RTO signals)
- Energy meter data collection (Modbus, OPC UA, REST)

Supported Protocols:
- REST/HTTPS (API endpoints)
- OpenADR 2.0b (Demand response)
- Modbus TCP/RTU (Energy meters)
- OPC UA (Industrial meters)
- WebSocket (Real-time feeds)

Supported Markets/ISOs:
- PJM (PJM Interconnection)
- ERCOT (Electric Reliability Council of Texas)
- CAISO (California ISO)
- NYISO (New York ISO)
- ISO-NE (ISO New England)
- MISO (Midcontinent ISO)
- SPP (Southwest Power Pool)

Author: GreenLang Data Integration Engineering Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import time
import json
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class EMSConnectionStatus(str, Enum):
    """EMS connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    ERROR = "error"


class MeterProtocol(str, Enum):
    """Energy meter communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    REST = "rest"
    BACNET = "bacnet"
    MQTT = "mqtt"


class PriceType(str, Enum):
    """Energy price types."""
    LMP = "lmp"  # Locational Marginal Price
    DAM = "dam"  # Day-Ahead Market
    RTM = "rtm"  # Real-Time Market
    ANCILLARY = "ancillary"
    CAPACITY = "capacity"
    TRANSMISSION = "transmission"
    RETAIL = "retail"
    TOU = "tou"  # Time of Use


class DemandResponseLevel(str, Enum):
    """Demand response signal levels."""
    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


class DemandResponseEventType(str, Enum):
    """Demand response event types."""
    PRICE_SIGNAL = "price_signal"
    LOAD_CURTAILMENT = "load_curtailment"
    LOAD_SHEDDING = "load_shedding"
    CAPACITY_BIDDING = "capacity_bidding"
    FREQUENCY_REGULATION = "frequency_regulation"
    SPINNING_RESERVE = "spinning_reserve"
    ANCILLARY_SERVICES = "ancillary_services"


class ISOMarket(str, Enum):
    """ISO/RTO markets."""
    PJM = "pjm"
    ERCOT = "ercot"
    CAISO = "caiso"
    NYISO = "nyiso"
    ISO_NE = "iso_ne"
    MISO = "miso"
    SPP = "spp"


class GridSignalType(str, Enum):
    """Grid operator signal types."""
    FREQUENCY = "frequency"
    ACE = "ace"  # Area Control Error
    LMP = "lmp"
    LOAD_FORECAST = "load_forecast"
    GENERATION_DISPATCH = "generation_dispatch"
    EMERGENCY_ALERT = "emergency_alert"
    RESERVE_STATUS = "reserve_status"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class EMSConfig(BaseModel):
    """Base configuration for EMS connectors."""

    model_config = ConfigDict(extra="forbid")

    # Connection settings
    host: str = Field(..., description="Server host")
    port: int = Field(default=443, ge=1, le=65535, description="Server port")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")

    # Authentication
    api_key: Optional[str] = Field(default=None, description="API key")
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")

    # Timeouts
    connection_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    request_timeout: float = Field(default=60.0, ge=5.0, le=300.0)

    # Retry settings
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=2.0, ge=0.5, le=30.0)

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)

    # Caching
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=60, ge=10, le=3600)


class PricingConfig(EMSConfig):
    """Real-time pricing connector configuration."""

    # Pricing source
    iso_market: Optional[ISOMarket] = Field(default=None, description="ISO/RTO market")
    pricing_node: Optional[str] = Field(default=None, description="Pricing node/zone")
    utility_id: Optional[str] = Field(default=None, description="Utility identifier")

    # Data settings
    price_types: List[PriceType] = Field(
        default=[PriceType.LMP, PriceType.RTM],
        description="Price types to retrieve"
    )
    update_interval_seconds: int = Field(
        default=300, ge=60, le=3600, description="Price update interval"
    )

    # Historical data
    historical_days: int = Field(default=7, ge=1, le=365, description="Historical data days")


class DemandResponseConfig(EMSConfig):
    """Demand response connector configuration."""

    # OpenADR settings
    ven_id: Optional[str] = Field(default=None, description="VEN (Virtual End Node) ID")
    vtn_url: Optional[HttpUrl] = Field(default=None, description="VTN URL")
    resource_id: Optional[str] = Field(default=None, description="Resource ID")

    # Program settings
    program_id: Optional[str] = Field(default=None, description="DR program ID")
    program_name: Optional[str] = Field(default=None, description="DR program name")

    # Response settings
    auto_opt_in: bool = Field(default=False, description="Automatic opt-in to events")
    max_curtailment_kw: float = Field(
        default=0.0, ge=0, description="Maximum curtailment (kW)"
    )
    min_notification_minutes: int = Field(
        default=15, ge=0, le=1440, description="Minimum notification time"
    )


class GridOperatorConfig(EMSConfig):
    """Grid operator connector configuration."""

    iso_market: ISOMarket = Field(..., description="ISO/RTO market")
    zone: Optional[str] = Field(default=None, description="Zone/area")
    node_id: Optional[str] = Field(default=None, description="Node ID")

    # Signal subscriptions
    subscribe_frequency: bool = Field(default=True, description="Subscribe to frequency")
    subscribe_lmp: bool = Field(default=True, description="Subscribe to LMP")
    subscribe_alerts: bool = Field(default=True, description="Subscribe to alerts")


class EnergyMeterConfig(BaseModel):
    """Energy meter connector configuration."""

    model_config = ConfigDict(extra="forbid")

    # Connection
    protocol: MeterProtocol = Field(..., description="Communication protocol")
    host: str = Field(..., description="Meter host/address")
    port: int = Field(default=502, ge=1, le=65535, description="Port")

    # Meter identification
    meter_id: str = Field(..., description="Meter identifier")
    meter_name: Optional[str] = Field(default=None, description="Meter name")

    # Modbus settings
    modbus_unit_id: int = Field(default=1, ge=1, le=247, description="Modbus unit ID")
    modbus_timeout: float = Field(default=3.0, ge=0.5, le=30.0)

    # OPC UA settings
    namespace_index: int = Field(default=2, ge=0, description="OPC UA namespace")

    # Polling
    poll_interval_seconds: float = Field(default=5.0, ge=1.0, le=300.0)

    # Register mapping
    register_map: Dict[str, int] = Field(
        default_factory=dict, description="Parameter to register mapping"
    )


# =============================================================================
# Pydantic Models - Data
# =============================================================================


class PricePoint(BaseModel):
    """Single price data point."""

    model_config = ConfigDict(extra="allow")

    timestamp: datetime = Field(..., description="Price timestamp")
    price: float = Field(..., description="Price value ($/MWh or cents/kWh)")
    price_type: PriceType = Field(..., description="Price type")
    unit: str = Field(default="$/MWh", description="Price unit")

    # Location
    node: Optional[str] = Field(default=None, description="Pricing node")
    zone: Optional[str] = Field(default=None, description="Pricing zone")

    # Components
    energy_component: Optional[float] = Field(default=None, description="Energy component")
    congestion_component: Optional[float] = Field(default=None, description="Congestion")
    loss_component: Optional[float] = Field(default=None, description="Losses")

    # Metadata
    source: Optional[str] = Field(default=None, description="Data source")
    quality: str = Field(default="good", description="Data quality")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class PricingFeed(BaseModel):
    """Collection of price data."""

    model_config = ConfigDict(extra="allow")

    feed_id: str = Field(..., description="Feed identifier")
    price_type: PriceType = Field(..., description="Price type")
    node: str = Field(..., description="Pricing node")

    # Current price
    current_price: Optional[PricePoint] = Field(default=None, description="Current price")

    # Historical prices
    prices: List[PricePoint] = Field(default_factory=list, description="Historical prices")

    # Forecast
    forecast_prices: List[PricePoint] = Field(
        default_factory=list, description="Forecast prices"
    )

    # Statistics
    min_price_24h: Optional[float] = Field(default=None, description="24h min price")
    max_price_24h: Optional[float] = Field(default=None, description="24h max price")
    avg_price_24h: Optional[float] = Field(default=None, description="24h avg price")

    # Timestamps
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class DemandResponseSignal(BaseModel):
    """Demand response signal."""

    model_config = ConfigDict(extra="allow")

    signal_id: str = Field(..., description="Signal identifier")
    signal_type: str = Field(default="level", description="Signal type")

    # Signal value
    level: DemandResponseLevel = Field(..., description="Signal level")
    payload: float = Field(default=0.0, description="Signal payload value")

    # Timing
    start_time: datetime = Field(..., description="Signal start time")
    end_time: Optional[datetime] = Field(default=None, description="Signal end time")
    duration_minutes: Optional[int] = Field(default=None, description="Duration")

    # Source
    source: str = Field(default="unknown", description="Signal source")
    program_id: Optional[str] = Field(default=None, description="Program ID")

    # Status
    active: bool = Field(default=True, description="Signal active")
    acknowledged: bool = Field(default=False, description="Signal acknowledged")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class DemandResponseEvent(BaseModel):
    """Demand response event."""

    model_config = ConfigDict(extra="allow")

    event_id: str = Field(..., description="Event identifier")
    event_type: DemandResponseEventType = Field(..., description="Event type")

    # Timing
    notification_time: datetime = Field(..., description="Notification time")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    duration_minutes: int = Field(default=0, ge=0, description="Duration")

    # Parameters
    target_reduction_kw: Optional[float] = Field(default=None, description="Target reduction")
    price_signal: Optional[float] = Field(default=None, description="Price signal")
    penalty_price: Optional[float] = Field(default=None, description="Penalty price")

    # Status
    status: str = Field(default="pending", description="Event status")
    opt_in: bool = Field(default=False, description="Opted in")
    response_required: bool = Field(default=True, description="Response required")

    # Program
    program_id: Optional[str] = Field(default=None, description="Program ID")
    program_name: Optional[str] = Field(default=None, description="Program name")

    # Response
    actual_reduction_kw: Optional[float] = Field(default=None, description="Actual reduction")
    compliance_score: Optional[float] = Field(default=None, description="Compliance score")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ISOSignal(BaseModel):
    """Grid operator / ISO signal."""

    model_config = ConfigDict(extra="allow")

    signal_id: str = Field(..., description="Signal identifier")
    signal_type: GridSignalType = Field(..., description="Signal type")
    iso_market: ISOMarket = Field(..., description="ISO market")

    # Value
    value: float = Field(..., description="Signal value")
    unit: str = Field(default="", description="Unit")

    # Location
    zone: Optional[str] = Field(default=None, description="Zone")
    node: Optional[str] = Field(default=None, description="Node")

    # Timing
    timestamp: datetime = Field(..., description="Signal timestamp")
    valid_until: Optional[datetime] = Field(default=None, description="Valid until")

    # Status
    quality: str = Field(default="good", description="Data quality")
    is_forecast: bool = Field(default=False, description="Is forecast")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class GridFrequency(BaseModel):
    """Grid frequency measurement."""

    timestamp: datetime = Field(..., description="Measurement timestamp")
    frequency_hz: float = Field(..., description="Grid frequency (Hz)")
    deviation_hz: float = Field(default=0.0, description="Deviation from nominal")
    nominal_hz: float = Field(default=60.0, description="Nominal frequency")

    zone: Optional[str] = Field(default=None, description="Zone")
    iso_market: Optional[ISOMarket] = Field(default=None, description="ISO market")

    # Status
    status: str = Field(default="normal", description="Frequency status")
    alert_level: Optional[str] = Field(default=None, description="Alert level")


class EnergyMeterReading(BaseModel):
    """Energy meter reading."""

    model_config = ConfigDict(extra="allow")

    meter_id: str = Field(..., description="Meter identifier")
    timestamp: datetime = Field(..., description="Reading timestamp")

    # Power measurements
    active_power_kw: Optional[float] = Field(default=None, description="Active power (kW)")
    reactive_power_kvar: Optional[float] = Field(default=None, description="Reactive power")
    apparent_power_kva: Optional[float] = Field(default=None, description="Apparent power")
    power_factor: Optional[float] = Field(default=None, description="Power factor")

    # Energy measurements
    active_energy_kwh: Optional[float] = Field(default=None, description="Active energy")
    reactive_energy_kvarh: Optional[float] = Field(default=None, description="Reactive energy")

    # Voltage and current
    voltage_v: Optional[float] = Field(default=None, description="Voltage (V)")
    current_a: Optional[float] = Field(default=None, description="Current (A)")

    # Phase measurements (3-phase)
    voltage_l1: Optional[float] = Field(default=None, description="L1 voltage")
    voltage_l2: Optional[float] = Field(default=None, description="L2 voltage")
    voltage_l3: Optional[float] = Field(default=None, description="L3 voltage")
    current_l1: Optional[float] = Field(default=None, description="L1 current")
    current_l2: Optional[float] = Field(default=None, description="L2 current")
    current_l3: Optional[float] = Field(default=None, description="L3 current")

    # Demand
    demand_kw: Optional[float] = Field(default=None, description="Demand (kW)")
    peak_demand_kw: Optional[float] = Field(default=None, description="Peak demand")

    # Quality
    quality: str = Field(default="good", description="Data quality")

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Abstract Base Class
# =============================================================================


class EMSConnectorBase(ABC):
    """Abstract base class for EMS connectors."""

    def __init__(self, config: EMSConfig) -> None:
        """Initialize EMS connector."""
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._status = EMSConnectionStatus.DISCONNECTED
        self._client: Any = None

        # Cache
        self._cache: Dict[str, Tuple[Any, float]] = {}

        # Statistics
        self._stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "cache_hits": 0,
            "last_request": None,
            "last_error": None,
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect."""
        pass

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._status in [
            EMSConnectionStatus.CONNECTED,
            EMSConnectionStatus.AUTHENTICATED,
            EMSConnectionStatus.SUBSCRIBED,
        ]

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, cache_time = self._cache[key]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "status": self._status.value,
            "cache_size": len(self._cache),
        }


# =============================================================================
# Real-Time Pricing Connector
# =============================================================================


class RealTimePricingConnector(EMSConnectorBase):
    """
    Real-time energy pricing connector.

    Supports:
    - ISO/RTO LMP feeds (PJM, ERCOT, CAISO, etc.)
    - Day-ahead and real-time market prices
    - Price forecasts
    - Historical price data
    """

    def __init__(self, config: PricingConfig) -> None:
        """Initialize pricing connector."""
        super().__init__(config)
        self._pricing_config = config
        self._current_prices: Dict[str, PricePoint] = {}
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._subscription_task: Optional[asyncio.Task] = None
        self._price_callbacks: List[Callable[[PricePoint], None]] = []

    async def connect(self) -> bool:
        """Connect to pricing feed."""
        try:
            import httpx

            base_url = self._get_base_url()

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
                verify=self._config.use_ssl,
            )

            self._status = EMSConnectionStatus.CONNECTED

            # Authenticate if needed
            if self._config.api_key:
                self._status = EMSConnectionStatus.AUTHENTICATED

            self._logger.info(f"Connected to pricing feed: {base_url}")
            return True

        except Exception as e:
            self._logger.error(f"Pricing connection failed: {e}")
            self._status = EMSConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from pricing feed."""
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        self._status = EMSConnectionStatus.DISCONNECTED

    def _get_base_url(self) -> str:
        """Get base URL for ISO market."""
        iso_urls = {
            ISOMarket.PJM: "https://api.pjm.com",
            ISOMarket.ERCOT: "https://www.ercot.com",
            ISOMarket.CAISO: "https://oasis.caiso.com",
            ISOMarket.NYISO: "https://mis.nyiso.com",
            ISOMarket.ISO_NE: "https://webservices.iso-ne.com",
            ISOMarket.MISO: "https://api.misoenergy.org",
            ISOMarket.SPP: "https://marketplace.spp.org",
        }

        if self._pricing_config.iso_market:
            return iso_urls.get(
                self._pricing_config.iso_market,
                f"https://{self._config.host}:{self._config.port}"
            )

        return f"https://{self._config.host}:{self._config.port}"

    async def get_current_price(
        self,
        price_type: PriceType = PriceType.LMP,
        node: Optional[str] = None
    ) -> Optional[PricePoint]:
        """
        Get current energy price.

        Args:
            price_type: Type of price to retrieve
            node: Pricing node (defaults to configured node)

        Returns:
            Current price point
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to pricing feed")

        node = node or self._pricing_config.pricing_node or "default"
        cache_key = f"price:{price_type.value}:{node}"

        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Fetch price from API
            price_point = await self._fetch_current_price(price_type, node)

            if price_point:
                self._current_prices[f"{price_type.value}:{node}"] = price_point
                self._price_history[f"{price_type.value}:{node}"].append(price_point)
                self._set_cached(cache_key, price_point)

            return price_point

        except Exception as e:
            self._logger.error(f"Failed to get current price: {e}")
            self._stats["failures"] += 1
            raise

    async def _fetch_current_price(
        self,
        price_type: PriceType,
        node: str
    ) -> Optional[PricePoint]:
        """Fetch current price from API."""
        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            # ISO-specific endpoints
            if self._pricing_config.iso_market == ISOMarket.PJM:
                url = f"/api/prices/lmp?node={node}"
            elif self._pricing_config.iso_market == ISOMarket.ERCOT:
                url = f"/misapp/GetRtmPrices?zone={node}"
            elif self._pricing_config.iso_market == ISOMarket.CAISO:
                url = f"/oasisapi/SingleZip?node={node}&marketType=RTM"
            else:
                url = f"/api/prices/{price_type.value}?node={node}"

            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            self._stats["requests"] += 1
            self._stats["successes"] += 1
            self._stats["last_request"] = datetime.now(timezone.utc)

            # Parse response (ISO-specific)
            return self._parse_price_response(data, price_type, node)

        except Exception as e:
            self._logger.error(f"Price fetch failed: {e}")
            self._stats["failures"] += 1
            raise

    def _parse_price_response(
        self,
        data: Dict[str, Any],
        price_type: PriceType,
        node: str
    ) -> Optional[PricePoint]:
        """Parse price response from API."""
        # Generic parsing - would be customized per ISO
        if "price" in data:
            return PricePoint(
                timestamp=datetime.now(timezone.utc),
                price=float(data["price"]),
                price_type=price_type,
                node=node,
                source=self._pricing_config.iso_market.value if self._pricing_config.iso_market else "unknown",
            )
        elif "lmp" in data:
            return PricePoint(
                timestamp=datetime.now(timezone.utc),
                price=float(data["lmp"]),
                price_type=price_type,
                node=node,
                energy_component=float(data.get("energy", 0)),
                congestion_component=float(data.get("congestion", 0)),
                loss_component=float(data.get("loss", 0)),
                source=self._pricing_config.iso_market.value if self._pricing_config.iso_market else "unknown",
            )

        return None

    async def get_price_forecast(
        self,
        hours_ahead: int = 24,
        price_type: PriceType = PriceType.DAM,
        node: Optional[str] = None
    ) -> List[PricePoint]:
        """
        Get price forecast.

        Args:
            hours_ahead: Hours of forecast
            price_type: Price type
            node: Pricing node

        Returns:
            List of forecast price points
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to pricing feed")

        node = node or self._pricing_config.pricing_node or "default"

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = f"/api/prices/forecast?node={node}&hours={hours_ahead}&type={price_type.value}"
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            forecasts = []

            for item in data.get("forecasts", []):
                forecasts.append(PricePoint(
                    timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                    price=float(item["price"]),
                    price_type=price_type,
                    node=node,
                    quality="forecast",
                ))

            return forecasts

        except Exception as e:
            self._logger.error(f"Price forecast fetch failed: {e}")
            raise

    async def get_historical_prices(
        self,
        start_time: datetime,
        end_time: datetime,
        price_type: PriceType = PriceType.LMP,
        node: Optional[str] = None,
        interval_minutes: int = 5
    ) -> List[PricePoint]:
        """
        Get historical prices.

        Args:
            start_time: Start time
            end_time: End time
            price_type: Price type
            node: Pricing node
            interval_minutes: Data interval

        Returns:
            List of historical price points
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to pricing feed")

        node = node or self._pricing_config.pricing_node or "default"

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = (
                f"/api/prices/historical?"
                f"node={node}&"
                f"start={start_time.isoformat()}&"
                f"end={end_time.isoformat()}&"
                f"type={price_type.value}&"
                f"interval={interval_minutes}"
            )
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            prices = []

            for item in data.get("prices", []):
                prices.append(PricePoint(
                    timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                    price=float(item["price"]),
                    price_type=price_type,
                    node=node,
                ))

            return prices

        except Exception as e:
            self._logger.error(f"Historical price fetch failed: {e}")
            raise

    async def subscribe_to_prices(
        self,
        callback: Callable[[PricePoint], None],
        price_types: Optional[List[PriceType]] = None,
        node: Optional[str] = None
    ) -> None:
        """
        Subscribe to real-time price updates.

        Args:
            callback: Callback function for price updates
            price_types: Price types to subscribe to
            node: Pricing node
        """
        self._price_callbacks.append(callback)

        if not self._subscription_task:
            self._subscription_task = asyncio.create_task(
                self._price_subscription_loop(price_types, node)
            )
            self._status = EMSConnectionStatus.SUBSCRIBED

    async def _price_subscription_loop(
        self,
        price_types: Optional[List[PriceType]],
        node: Optional[str]
    ) -> None:
        """Background task for price subscription."""
        price_types = price_types or self._pricing_config.price_types
        interval = self._pricing_config.update_interval_seconds

        while True:
            try:
                for price_type in price_types:
                    price = await self.get_current_price(price_type, node)
                    if price:
                        for callback in self._price_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(price)
                                else:
                                    callback(price)
                            except Exception as e:
                                self._logger.error(f"Callback error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Subscription error: {e}")
                await asyncio.sleep(interval)

    def get_price_statistics(
        self,
        price_type: PriceType = PriceType.LMP,
        node: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, float]:
        """Get price statistics for specified period."""
        node = node or self._pricing_config.pricing_node or "default"
        key = f"{price_type.value}:{node}"

        history = list(self._price_history.get(key, []))
        if not history:
            return {}

        # Filter to requested hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        history = [p for p in history if p.timestamp >= cutoff]

        if not history:
            return {}

        prices = [p.price for p in history]
        return {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "current": prices[-1] if prices else 0.0,
            "count": len(prices),
        }


# =============================================================================
# Demand Response Connector
# =============================================================================


class DemandResponseConnector(EMSConnectorBase):
    """
    Demand response connector.

    Supports:
    - OpenADR 2.0b protocol
    - Proprietary DR platforms
    - Event notification and response
    - Automated opt-in/opt-out
    """

    def __init__(self, config: DemandResponseConfig) -> None:
        """Initialize demand response connector."""
        super().__init__(config)
        self._dr_config = config
        self._active_events: Dict[str, DemandResponseEvent] = {}
        self._current_signal: Optional[DemandResponseSignal] = None
        self._event_callbacks: List[Callable[[DemandResponseEvent], None]] = []
        self._signal_callbacks: List[Callable[[DemandResponseSignal], None]] = []
        self._poll_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to demand response platform."""
        try:
            import httpx

            # Use VTN URL for OpenADR or generic endpoint
            base_url = str(self._dr_config.vtn_url) if self._dr_config.vtn_url else (
                f"https://{self._config.host}:{self._config.port}"
            )

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
                verify=self._config.use_ssl,
            )

            # Register VEN if using OpenADR
            if self._dr_config.ven_id:
                await self._register_ven()

            self._status = EMSConnectionStatus.CONNECTED
            self._logger.info(f"Connected to DR platform: {base_url}")
            return True

        except Exception as e:
            self._logger.error(f"DR connection failed: {e}")
            self._status = EMSConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from demand response platform."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        self._status = EMSConnectionStatus.DISCONNECTED

    async def _register_ven(self) -> bool:
        """Register VEN with VTN (OpenADR)."""
        try:
            # OpenADR registration
            registration_data = {
                "venID": self._dr_config.ven_id,
                "resourceID": self._dr_config.resource_id,
                "programID": self._dr_config.program_id,
            }

            response = await self._client.post(
                "/EiRegisterParty",
                json=registration_data
            )
            response.raise_for_status()

            self._logger.info(f"VEN registered: {self._dr_config.ven_id}")
            return True

        except Exception as e:
            self._logger.error(f"VEN registration failed: {e}")
            return False

    async def get_current_signal(self) -> Optional[DemandResponseSignal]:
        """
        Get current demand response signal.

        Returns:
            Current DR signal or None
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to DR platform")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = "/api/signals/current"
            if self._dr_config.ven_id:
                url = f"/EiEvent?venID={self._dr_config.ven_id}"

            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            self._stats["requests"] += 1
            self._stats["successes"] += 1

            if data:
                signal = DemandResponseSignal(
                    signal_id=data.get("signalID", str(time.time())),
                    level=DemandResponseLevel(data.get("level", "normal")),
                    payload=float(data.get("payload", 0)),
                    start_time=datetime.fromisoformat(
                        data.get("startTime", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")
                    ),
                    end_time=datetime.fromisoformat(
                        data.get("endTime", "").replace("Z", "+00:00")
                    ) if data.get("endTime") else None,
                    source=data.get("source", "unknown"),
                    program_id=self._dr_config.program_id,
                )
                self._current_signal = signal
                return signal

            return None

        except Exception as e:
            self._logger.error(f"Failed to get DR signal: {e}")
            self._stats["failures"] += 1
            raise

    async def get_active_events(self) -> List[DemandResponseEvent]:
        """
        Get active demand response events.

        Returns:
            List of active DR events
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to DR platform")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = "/api/events/active"
            if self._dr_config.ven_id:
                url = f"/EiEvent?venID={self._dr_config.ven_id}&status=active"

            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            events = []

            for item in data.get("events", []):
                event = DemandResponseEvent(
                    event_id=item.get("eventID", ""),
                    event_type=DemandResponseEventType(
                        item.get("eventType", "load_curtailment")
                    ),
                    notification_time=datetime.fromisoformat(
                        item.get("notificationTime", "").replace("Z", "+00:00")
                    ),
                    start_time=datetime.fromisoformat(
                        item.get("startTime", "").replace("Z", "+00:00")
                    ),
                    end_time=datetime.fromisoformat(
                        item.get("endTime", "").replace("Z", "+00:00")
                    ),
                    target_reduction_kw=float(item.get("targetReduction", 0)),
                    status=item.get("status", "pending"),
                    program_id=self._dr_config.program_id,
                    program_name=self._dr_config.program_name,
                )
                events.append(event)
                self._active_events[event.event_id] = event

            return events

        except Exception as e:
            self._logger.error(f"Failed to get DR events: {e}")
            raise

    async def opt_in_event(self, event_id: str) -> bool:
        """
        Opt in to a demand response event.

        Args:
            event_id: Event identifier

        Returns:
            True if opt-in successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to DR platform")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            response = await self._client.post(
                f"/api/events/{event_id}/optin",
                json={"venID": self._dr_config.ven_id},
                headers=headers
            )
            response.raise_for_status()

            if event_id in self._active_events:
                self._active_events[event_id].opt_in = True

            self._logger.info(f"Opted in to DR event: {event_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to opt in to event {event_id}: {e}")
            return False

    async def opt_out_event(self, event_id: str, reason: str = "") -> bool:
        """
        Opt out of a demand response event.

        Args:
            event_id: Event identifier
            reason: Opt-out reason

        Returns:
            True if opt-out successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to DR platform")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            response = await self._client.post(
                f"/api/events/{event_id}/optout",
                json={"venID": self._dr_config.ven_id, "reason": reason},
                headers=headers
            )
            response.raise_for_status()

            if event_id in self._active_events:
                self._active_events[event_id].opt_in = False

            self._logger.info(f"Opted out of DR event: {event_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to opt out of event {event_id}: {e}")
            return False

    async def report_compliance(
        self,
        event_id: str,
        actual_reduction_kw: float
    ) -> bool:
        """
        Report compliance for a DR event.

        Args:
            event_id: Event identifier
            actual_reduction_kw: Actual load reduction achieved

        Returns:
            True if report successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to DR platform")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            response = await self._client.post(
                f"/api/events/{event_id}/report",
                json={
                    "venID": self._dr_config.ven_id,
                    "actualReduction": actual_reduction_kw,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                headers=headers
            )
            response.raise_for_status()

            if event_id in self._active_events:
                self._active_events[event_id].actual_reduction_kw = actual_reduction_kw

            self._logger.info(f"Reported compliance for event {event_id}: {actual_reduction_kw} kW")
            return True

        except Exception as e:
            self._logger.error(f"Failed to report compliance for event {event_id}: {e}")
            return False

    async def subscribe_to_events(
        self,
        callback: Callable[[DemandResponseEvent], None]
    ) -> None:
        """Subscribe to demand response events."""
        self._event_callbacks.append(callback)

        if not self._poll_task:
            self._poll_task = asyncio.create_task(self._event_poll_loop())
            self._status = EMSConnectionStatus.SUBSCRIBED

    async def subscribe_to_signals(
        self,
        callback: Callable[[DemandResponseSignal], None]
    ) -> None:
        """Subscribe to demand response signals."""
        self._signal_callbacks.append(callback)

        if not self._poll_task:
            self._poll_task = asyncio.create_task(self._event_poll_loop())
            self._status = EMSConnectionStatus.SUBSCRIBED

    async def _event_poll_loop(self) -> None:
        """Background task for polling DR events and signals."""
        poll_interval = 60  # Poll every minute

        while True:
            try:
                # Get current signal
                signal = await self.get_current_signal()
                if signal:
                    for callback in self._signal_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(signal)
                            else:
                                callback(signal)
                        except Exception as e:
                            self._logger.error(f"Signal callback error: {e}")

                # Get active events
                events = await self.get_active_events()
                for event in events:
                    for callback in self._event_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)
                        except Exception as e:
                            self._logger.error(f"Event callback error: {e}")

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Event poll error: {e}")
                await asyncio.sleep(poll_interval)


# =============================================================================
# Grid Operator Connector
# =============================================================================


class GridOperatorConnector(EMSConnectorBase):
    """
    Grid operator / ISO connector.

    Supports:
    - Real-time grid frequency
    - LMP data
    - Emergency alerts
    - Load forecasts
    """

    def __init__(self, config: GridOperatorConfig) -> None:
        """Initialize grid operator connector."""
        super().__init__(config)
        self._grid_config = config
        self._current_frequency: Optional[GridFrequency] = None
        self._current_lmp: Optional[float] = None
        self._alerts: List[ISOSignal] = []
        self._subscription_task: Optional[asyncio.Task] = None
        self._signal_callbacks: List[Callable[[ISOSignal], None]] = []

    async def connect(self) -> bool:
        """Connect to grid operator."""
        try:
            import httpx

            base_url = self._get_iso_base_url()

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
                verify=self._config.use_ssl,
            )

            self._status = EMSConnectionStatus.CONNECTED
            self._logger.info(f"Connected to grid operator: {self._grid_config.iso_market.value}")
            return True

        except Exception as e:
            self._logger.error(f"Grid operator connection failed: {e}")
            self._status = EMSConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from grid operator."""
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        self._status = EMSConnectionStatus.DISCONNECTED

    def _get_iso_base_url(self) -> str:
        """Get base URL for ISO."""
        iso_urls = {
            ISOMarket.PJM: "https://api.pjm.com",
            ISOMarket.ERCOT: "https://www.ercot.com",
            ISOMarket.CAISO: "https://oasis.caiso.com",
            ISOMarket.NYISO: "https://mis.nyiso.com",
            ISOMarket.ISO_NE: "https://webservices.iso-ne.com",
            ISOMarket.MISO: "https://api.misoenergy.org",
            ISOMarket.SPP: "https://marketplace.spp.org",
        }
        return iso_urls.get(
            self._grid_config.iso_market,
            f"https://{self._config.host}:{self._config.port}"
        )

    async def get_grid_frequency(self) -> GridFrequency:
        """
        Get current grid frequency.

        Returns:
            Current grid frequency
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to grid operator")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = "/api/frequency/current"
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            frequency = GridFrequency(
                timestamp=datetime.now(timezone.utc),
                frequency_hz=float(data.get("frequency", 60.0)),
                deviation_hz=float(data.get("deviation", 0.0)),
                nominal_hz=60.0 if self._grid_config.iso_market != ISOMarket.CAISO else 60.0,
                iso_market=self._grid_config.iso_market,
                zone=self._grid_config.zone,
            )

            # Determine status
            if abs(frequency.deviation_hz) < 0.02:
                frequency.status = "normal"
            elif abs(frequency.deviation_hz) < 0.05:
                frequency.status = "warning"
            else:
                frequency.status = "alert"

            self._current_frequency = frequency
            return frequency

        except Exception as e:
            self._logger.error(f"Failed to get grid frequency: {e}")
            raise

    async def get_current_lmp(self, node: Optional[str] = None) -> ISOSignal:
        """
        Get current LMP for node.

        Args:
            node: Pricing node (defaults to configured node)

        Returns:
            LMP signal
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to grid operator")

        node = node or self._grid_config.node_id or "default"

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = f"/api/lmp/current?node={node}"
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            signal = ISOSignal(
                signal_id=f"lmp-{node}-{time.time()}",
                signal_type=GridSignalType.LMP,
                iso_market=self._grid_config.iso_market,
                value=float(data.get("lmp", 0.0)),
                unit="$/MWh",
                node=node,
                timestamp=datetime.now(timezone.utc),
            )

            self._current_lmp = signal.value
            return signal

        except Exception as e:
            self._logger.error(f"Failed to get LMP: {e}")
            raise

    async def get_alerts(self) -> List[ISOSignal]:
        """
        Get current grid alerts.

        Returns:
            List of alert signals
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to grid operator")

        try:
            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            url = "/api/alerts/current"
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            alerts = []

            for item in data.get("alerts", []):
                alert = ISOSignal(
                    signal_id=item.get("alertID", str(time.time())),
                    signal_type=GridSignalType.EMERGENCY_ALERT,
                    iso_market=self._grid_config.iso_market,
                    value=float(item.get("severity", 0)),
                    timestamp=datetime.fromisoformat(
                        item.get("timestamp", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")
                    ),
                    metadata={"message": item.get("message", "")},
                )
                alerts.append(alert)

            self._alerts = alerts
            return alerts

        except Exception as e:
            self._logger.error(f"Failed to get alerts: {e}")
            raise

    async def subscribe_to_signals(
        self,
        callback: Callable[[ISOSignal], None],
        signal_types: Optional[List[GridSignalType]] = None
    ) -> None:
        """Subscribe to grid signals."""
        self._signal_callbacks.append(callback)

        if not self._subscription_task:
            self._subscription_task = asyncio.create_task(
                self._signal_subscription_loop(signal_types)
            )
            self._status = EMSConnectionStatus.SUBSCRIBED

    async def _signal_subscription_loop(
        self,
        signal_types: Optional[List[GridSignalType]]
    ) -> None:
        """Background task for signal subscription."""
        poll_interval = 10  # Poll every 10 seconds

        while True:
            try:
                signals = []

                # Get frequency if subscribed
                if self._grid_config.subscribe_frequency:
                    freq = await self.get_grid_frequency()
                    signals.append(ISOSignal(
                        signal_id=f"freq-{time.time()}",
                        signal_type=GridSignalType.FREQUENCY,
                        iso_market=self._grid_config.iso_market,
                        value=freq.frequency_hz,
                        unit="Hz",
                        timestamp=freq.timestamp,
                    ))

                # Get LMP if subscribed
                if self._grid_config.subscribe_lmp:
                    lmp = await self.get_current_lmp()
                    signals.append(lmp)

                # Get alerts if subscribed
                if self._grid_config.subscribe_alerts:
                    alerts = await self.get_alerts()
                    signals.extend(alerts)

                # Filter by signal types if specified
                if signal_types:
                    signals = [s for s in signals if s.signal_type in signal_types]

                # Notify callbacks
                for signal in signals:
                    for callback in self._signal_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(signal)
                            else:
                                callback(signal)
                        except Exception as e:
                            self._logger.error(f"Signal callback error: {e}")

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Signal subscription error: {e}")
                await asyncio.sleep(poll_interval)


# =============================================================================
# Energy Meter Connector
# =============================================================================


class EnergyMeterConnector:
    """
    Energy meter data collection connector.

    Supports:
    - Modbus TCP/RTU meters
    - OPC UA meters
    - REST API meters
    - Real-time power monitoring
    """

    def __init__(self, config: EnergyMeterConfig) -> None:
        """Initialize energy meter connector."""
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.EnergyMeterConnector")
        self._connected = False
        self._client: Any = None
        self._last_reading: Optional[EnergyMeterReading] = None
        self._reading_history: deque = deque(maxlen=10000)
        self._poll_task: Optional[asyncio.Task] = None
        self._reading_callbacks: List[Callable[[EnergyMeterReading], None]] = []

    async def connect(self) -> bool:
        """Connect to energy meter."""
        try:
            if self._config.protocol == MeterProtocol.MODBUS_TCP:
                return await self._connect_modbus_tcp()
            elif self._config.protocol == MeterProtocol.MODBUS_RTU:
                return await self._connect_modbus_rtu()
            elif self._config.protocol == MeterProtocol.OPC_UA:
                return await self._connect_opcua()
            elif self._config.protocol == MeterProtocol.REST:
                return await self._connect_rest()
            else:
                raise ValueError(f"Unsupported protocol: {self._config.protocol}")

        except Exception as e:
            self._logger.error(f"Meter connection failed: {e}")
            return False

    async def _connect_modbus_tcp(self) -> bool:
        """Connect via Modbus TCP."""
        try:
            from pymodbus.client import AsyncModbusTcpClient

            self._client = AsyncModbusTcpClient(
                host=self._config.host,
                port=self._config.port,
                timeout=self._config.modbus_timeout,
            )
            await self._client.connect()

            if self._client.connected:
                self._connected = True
                self._logger.info(f"Connected to Modbus meter: {self._config.meter_id}")
                return True

            return False

        except ImportError:
            self._logger.error("pymodbus not installed. Install with: pip install pymodbus")
            return False
        except Exception as e:
            self._logger.error(f"Modbus TCP connection failed: {e}")
            return False

    async def _connect_modbus_rtu(self) -> bool:
        """Connect via Modbus RTU."""
        try:
            from pymodbus.client import AsyncModbusSerialClient

            self._client = AsyncModbusSerialClient(
                port=self._config.host,  # Serial port path
                timeout=self._config.modbus_timeout,
            )
            await self._client.connect()

            if self._client.connected:
                self._connected = True
                self._logger.info(f"Connected to Modbus RTU meter: {self._config.meter_id}")
                return True

            return False

        except Exception as e:
            self._logger.error(f"Modbus RTU connection failed: {e}")
            return False

    async def _connect_opcua(self) -> bool:
        """Connect via OPC UA."""
        try:
            from asyncua import Client

            url = f"opc.tcp://{self._config.host}:{self._config.port}"
            self._client = Client(url=url)
            await self._client.connect()

            self._connected = True
            self._logger.info(f"Connected to OPC UA meter: {self._config.meter_id}")
            return True

        except ImportError:
            self._logger.error("asyncua not installed. Install with: pip install asyncua")
            return False
        except Exception as e:
            self._logger.error(f"OPC UA connection failed: {e}")
            return False

    async def _connect_rest(self) -> bool:
        """Connect via REST API."""
        try:
            import httpx

            base_url = f"https://{self._config.host}:{self._config.port}"
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=30.0,
            )

            self._connected = True
            self._logger.info(f"Connected to REST meter: {self._config.meter_id}")
            return True

        except Exception as e:
            self._logger.error(f"REST connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from meter."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._client:
            if self._config.protocol in [MeterProtocol.MODBUS_TCP, MeterProtocol.MODBUS_RTU]:
                self._client.close()
            elif self._config.protocol == MeterProtocol.OPC_UA:
                await self._client.disconnect()
            elif self._config.protocol == MeterProtocol.REST:
                await self._client.aclose()

            self._client = None

        self._connected = False
        self._logger.info(f"Disconnected from meter: {self._config.meter_id}")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def read_meter(self) -> EnergyMeterReading:
        """
        Read current meter values.

        Returns:
            Energy meter reading
        """
        if not self._connected:
            raise ConnectionError("Not connected to meter")

        if self._config.protocol in [MeterProtocol.MODBUS_TCP, MeterProtocol.MODBUS_RTU]:
            return await self._read_modbus()
        elif self._config.protocol == MeterProtocol.OPC_UA:
            return await self._read_opcua()
        elif self._config.protocol == MeterProtocol.REST:
            return await self._read_rest()
        else:
            raise ValueError(f"Unsupported protocol: {self._config.protocol}")

    async def _read_modbus(self) -> EnergyMeterReading:
        """Read meter via Modbus."""
        reading = EnergyMeterReading(
            meter_id=self._config.meter_id,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            register_map = self._config.register_map or self._get_default_register_map()

            for param, address in register_map.items():
                result = await self._client.read_holding_registers(
                    address=address,
                    count=2,  # Most values are 32-bit (2 registers)
                    slave=self._config.modbus_unit_id
                )

                if not result.isError():
                    # Combine registers for 32-bit value
                    value = (result.registers[0] << 16) + result.registers[1]
                    value = value / 1000.0  # Scale factor

                    setattr(reading, param, value)

        except Exception as e:
            self._logger.error(f"Modbus read error: {e}")
            reading.quality = "bad"

        self._last_reading = reading
        self._reading_history.append(reading)
        return reading

    async def _read_opcua(self) -> EnergyMeterReading:
        """Read meter via OPC UA."""
        reading = EnergyMeterReading(
            meter_id=self._config.meter_id,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            # Read common power meter nodes
            params = [
                ("active_power_kw", "ActivePower"),
                ("reactive_power_kvar", "ReactivePower"),
                ("voltage_v", "Voltage"),
                ("current_a", "Current"),
                ("power_factor", "PowerFactor"),
                ("active_energy_kwh", "ActiveEnergy"),
            ]

            for attr, node_name in params:
                try:
                    node_id = f"ns={self._config.namespace_index};s={node_name}"
                    node = self._client.get_node(node_id)
                    value = await node.read_value()
                    setattr(reading, attr, float(value))
                except Exception:
                    pass

        except Exception as e:
            self._logger.error(f"OPC UA read error: {e}")
            reading.quality = "bad"

        self._last_reading = reading
        self._reading_history.append(reading)
        return reading

    async def _read_rest(self) -> EnergyMeterReading:
        """Read meter via REST API."""
        reading = EnergyMeterReading(
            meter_id=self._config.meter_id,
            timestamp=datetime.now(timezone.utc),
        )

        try:
            response = await self._client.get(f"/api/meters/{self._config.meter_id}/reading")
            response.raise_for_status()

            data = response.json()

            reading.active_power_kw = float(data.get("activePower", 0))
            reading.reactive_power_kvar = float(data.get("reactivePower", 0))
            reading.voltage_v = float(data.get("voltage", 0))
            reading.current_a = float(data.get("current", 0))
            reading.power_factor = float(data.get("powerFactor", 0))
            reading.active_energy_kwh = float(data.get("activeEnergy", 0))

        except Exception as e:
            self._logger.error(f"REST read error: {e}")
            reading.quality = "bad"

        self._last_reading = reading
        self._reading_history.append(reading)
        return reading

    def _get_default_register_map(self) -> Dict[str, int]:
        """Get default Modbus register map."""
        # Common power meter register layout
        return {
            "active_power_kw": 0,
            "reactive_power_kvar": 2,
            "apparent_power_kva": 4,
            "power_factor": 6,
            "voltage_v": 8,
            "current_a": 10,
            "active_energy_kwh": 12,
        }

    async def start_polling(
        self,
        callback: Optional[Callable[[EnergyMeterReading], None]] = None
    ) -> None:
        """
        Start continuous polling.

        Args:
            callback: Optional callback for readings
        """
        if callback:
            self._reading_callbacks.append(callback)

        if not self._poll_task:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        """Stop continuous polling."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        interval = self._config.poll_interval_seconds

        while True:
            try:
                reading = await self.read_meter()

                for callback in self._reading_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(reading)
                        else:
                            callback(reading)
                    except Exception as e:
                        self._logger.error(f"Reading callback error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(interval)

    def get_last_reading(self) -> Optional[EnergyMeterReading]:
        """Get last meter reading."""
        return self._last_reading

    def get_reading_history(self, count: int = 100) -> List[EnergyMeterReading]:
        """Get reading history."""
        return list(self._reading_history)[-count:]


# =============================================================================
# Factory Functions
# =============================================================================


def create_pricing_connector(
    iso_market: Optional[ISOMarket] = None,
    pricing_node: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> RealTimePricingConnector:
    """Create pricing connector."""
    config = PricingConfig(
        host=kwargs.get("host", "localhost"),
        iso_market=iso_market,
        pricing_node=pricing_node,
        api_key=api_key,
        **{k: v for k, v in kwargs.items() if k not in ["host"]}
    )
    return RealTimePricingConnector(config)


def create_demand_response_connector(
    ven_id: Optional[str] = None,
    vtn_url: Optional[str] = None,
    program_id: Optional[str] = None,
    **kwargs
) -> DemandResponseConnector:
    """Create demand response connector."""
    config = DemandResponseConfig(
        host=kwargs.get("host", "localhost"),
        ven_id=ven_id,
        vtn_url=vtn_url,
        program_id=program_id,
        **{k: v for k, v in kwargs.items() if k not in ["host"]}
    )
    return DemandResponseConnector(config)


def create_grid_operator_connector(
    iso_market: ISOMarket,
    zone: Optional[str] = None,
    node_id: Optional[str] = None,
    **kwargs
) -> GridOperatorConnector:
    """Create grid operator connector."""
    config = GridOperatorConfig(
        host=kwargs.get("host", "localhost"),
        iso_market=iso_market,
        zone=zone,
        node_id=node_id,
        **{k: v for k, v in kwargs.items() if k not in ["host"]}
    )
    return GridOperatorConnector(config)


def create_energy_meter_connector(
    protocol: MeterProtocol,
    host: str,
    meter_id: str,
    **kwargs
) -> EnergyMeterConnector:
    """Create energy meter connector."""
    config = EnergyMeterConfig(
        protocol=protocol,
        host=host,
        meter_id=meter_id,
        **kwargs
    )
    return EnergyMeterConnector(config)
