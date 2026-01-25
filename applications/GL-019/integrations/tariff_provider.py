"""
Energy Tariff Data Provider Module for GL-019 HEATSCHEDULER

Provides enterprise-grade connectivity to utility tariff APIs, real-time pricing
feeds, wholesale markets, and time-of-use rate schedules for optimal heat scheduling.

Supported Data Sources:
- Utility tariff APIs (OpenEI, utility-specific)
- Real-time pricing feeds (LMP, wholesale)
- Time-of-use rate schedules
- Demand charge tracking
- Rate change notifications

Features:
- Tariff rate retrieval
- Time-of-use schedule management
- Demand charge calculations
- Rate change detection and alerts
- Historical rate analysis
- Cost forecasting

Author: GreenLang Data Integration Engineering Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date, time as dt_time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import time as time_module
import json
import calendar
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class TariffType(str, Enum):
    """Types of energy tariffs."""
    FIXED = "fixed"
    TOU = "time_of_use"
    TIERED = "tiered"
    REAL_TIME = "real_time"
    CRITICAL_PEAK = "critical_peak"
    DEMAND = "demand"
    INTERRUPTIBLE = "interruptible"
    WHOLESALE = "wholesale"


class RatePeriod(str, Enum):
    """Time-of-use rate periods."""
    OFF_PEAK = "off_peak"
    MID_PEAK = "mid_peak"
    ON_PEAK = "on_peak"
    SUPER_PEAK = "super_peak"
    CRITICAL_PEAK = "critical_peak"


class SeasonType(str, Enum):
    """Seasonal rate types."""
    SUMMER = "summer"
    WINTER = "winter"
    SPRING = "spring"
    FALL = "fall"
    SHOULDER = "shoulder"


class DayType(str, Enum):
    """Day types for rate schedules."""
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class ChargeType(str, Enum):
    """Types of charges."""
    ENERGY = "energy"
    DEMAND = "demand"
    FIXED = "fixed"
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"
    GENERATION = "generation"
    CAPACITY = "capacity"
    ANCILLARY = "ancillary"
    TAX = "tax"
    SURCHARGE = "surcharge"


class RateChangeType(str, Enum):
    """Types of rate changes."""
    SCHEDULED = "scheduled"
    TOU_PERIOD = "tou_period"
    SEASON = "season"
    REGULATORY = "regulatory"
    EMERGENCY = "emergency"
    MARKET = "market"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class TariffProviderConfig(BaseModel):
    """Configuration for tariff data providers."""

    model_config = ConfigDict(extra="forbid")

    # Provider identification
    provider_type: str = Field(..., description="Provider type (utility, openei, wholesale)")
    provider_name: str = Field(..., description="Provider name")

    # Connection settings
    base_url: Optional[HttpUrl] = Field(default=None, description="API base URL")
    api_key: Optional[str] = Field(default=None, description="API key")

    # Utility identification
    utility_id: Optional[str] = Field(default=None, description="Utility identifier")
    tariff_id: Optional[str] = Field(default=None, description="Tariff ID")
    rate_schedule: Optional[str] = Field(default=None, description="Rate schedule name")

    # Location
    zip_code: Optional[str] = Field(default=None, description="ZIP code")
    state: Optional[str] = Field(default=None, description="State")
    country: str = Field(default="US", description="Country code")

    # Timeouts
    connection_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    request_timeout: float = Field(default=60.0, ge=5.0, le=300.0)

    # Caching
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)

    # Rate change monitoring
    monitor_rate_changes: bool = Field(default=True)
    rate_change_check_interval: int = Field(default=300, ge=60, le=3600)


# =============================================================================
# Pydantic Models - Data
# =============================================================================


class TariffRate(BaseModel):
    """Single tariff rate."""

    model_config = ConfigDict(extra="allow")

    rate_id: str = Field(..., description="Rate identifier")
    charge_type: ChargeType = Field(..., description="Type of charge")

    # Rate value
    rate: float = Field(..., description="Rate value")
    unit: str = Field(default="$/kWh", description="Rate unit")
    currency: str = Field(default="USD", description="Currency")

    # Applicability
    tariff_type: TariffType = Field(default=TariffType.FIXED)
    rate_period: Optional[RatePeriod] = Field(default=None)
    season: Optional[SeasonType] = Field(default=None)
    day_type: Optional[DayType] = Field(default=None)

    # Time bounds
    effective_date: Optional[date] = Field(default=None)
    expiration_date: Optional[date] = Field(default=None)
    start_hour: Optional[int] = Field(default=None, ge=0, le=23)
    end_hour: Optional[int] = Field(default=None, ge=0, le=23)

    # Tiered rates
    tier: Optional[int] = Field(default=None, ge=1)
    tier_min_kwh: Optional[float] = Field(default=None)
    tier_max_kwh: Optional[float] = Field(default=None)

    # Demand charges
    demand_rate: Optional[float] = Field(default=None, description="$/kW")
    min_demand_kw: Optional[float] = Field(default=None)
    max_demand_kw: Optional[float] = Field(default=None)

    description: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimeOfUseRate(BaseModel):
    """Time-of-use rate definition."""

    model_config = ConfigDict(extra="allow")

    rate_period: RatePeriod = Field(..., description="Rate period")
    rate: float = Field(..., description="Energy rate ($/kWh)")

    # Time window
    start_hour: int = Field(..., ge=0, le=23, description="Start hour")
    end_hour: int = Field(..., ge=0, le=23, description="End hour")
    start_minute: int = Field(default=0, ge=0, le=59)
    end_minute: int = Field(default=0, ge=0, le=59)

    # Day applicability
    weekdays: bool = Field(default=True, description="Applies to weekdays")
    weekends: bool = Field(default=False, description="Applies to weekends")
    holidays: bool = Field(default=False, description="Applies to holidays")
    specific_days: List[int] = Field(
        default_factory=list,
        description="Specific days of week (0=Mon, 6=Sun)"
    )

    # Season
    season: Optional[SeasonType] = Field(default=None)
    season_months: List[int] = Field(
        default_factory=list,
        description="Months this rate applies (1-12)"
    )

    # Additional charges
    demand_rate: Optional[float] = Field(default=None, description="$/kW for this period")

    description: Optional[str] = Field(default=None)


class DemandCharge(BaseModel):
    """Demand charge definition."""

    model_config = ConfigDict(extra="allow")

    charge_id: str = Field(..., description="Charge identifier")
    charge_type: str = Field(default="monthly_max", description="Demand charge type")

    # Rate
    rate_per_kw: float = Field(..., description="$/kW")

    # Thresholds
    min_demand_kw: float = Field(default=0.0, ge=0)
    max_demand_kw: Optional[float] = Field(default=None)
    included_kw: float = Field(default=0.0, ge=0, description="Included kW")

    # Measurement
    measurement_interval_minutes: int = Field(default=15, ge=1, le=60)
    averaging_method: str = Field(default="peak_15min")

    # TOU demand
    tou_period: Optional[RatePeriod] = Field(default=None)
    season: Optional[SeasonType] = Field(default=None)

    # Ratchet
    ratchet_enabled: bool = Field(default=False)
    ratchet_months: int = Field(default=12, ge=1, le=24)
    ratchet_percentage: float = Field(default=100.0, ge=0, le=100)

    description: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TariffSchedule(BaseModel):
    """Complete tariff schedule."""

    model_config = ConfigDict(extra="allow")

    schedule_id: str = Field(..., description="Schedule identifier")
    schedule_name: str = Field(..., description="Schedule name")
    utility_name: str = Field(..., description="Utility name")
    tariff_type: TariffType = Field(..., description="Tariff type")

    # Effective dates
    effective_date: date = Field(..., description="Effective date")
    expiration_date: Optional[date] = Field(default=None)

    # Energy rates
    energy_rates: List[TariffRate] = Field(default_factory=list)
    tou_rates: List[TimeOfUseRate] = Field(default_factory=list)

    # Demand charges
    demand_charges: List[DemandCharge] = Field(default_factory=list)

    # Fixed charges
    monthly_fixed_charge: float = Field(default=0.0, ge=0)
    daily_fixed_charge: float = Field(default=0.0, ge=0)
    minimum_bill: float = Field(default=0.0, ge=0)

    # Taxes and surcharges
    tax_rate_percent: float = Field(default=0.0, ge=0, le=100)
    surcharges: List[TariffRate] = Field(default_factory=list)

    # Holiday schedule
    holidays: List[date] = Field(default_factory=list)

    # Season definitions
    summer_months: List[int] = Field(default=[6, 7, 8, 9])
    winter_months: List[int] = Field(default=[12, 1, 2])

    description: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class RateChange(BaseModel):
    """Rate change notification."""

    model_config = ConfigDict(extra="allow")

    change_id: str = Field(..., description="Change identifier")
    change_type: RateChangeType = Field(..., description="Type of rate change")

    # Timing
    effective_time: datetime = Field(..., description="When rate changes")
    detected_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Rate details
    rate_period: Optional[RatePeriod] = Field(default=None)
    old_rate: float = Field(..., description="Previous rate")
    new_rate: float = Field(..., description="New rate")
    rate_change_percent: float = Field(default=0.0)
    unit: str = Field(default="$/kWh")

    # Duration
    duration_hours: Optional[float] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)

    # Impact
    estimated_cost_impact: Optional[float] = Field(default=None)

    description: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Abstract Base Class
# =============================================================================


class TariffProviderBase(ABC):
    """Abstract base class for tariff data providers."""

    def __init__(self, config: TariffProviderConfig) -> None:
        """Initialize tariff provider."""
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._connected = False
        self._client: Any = None

        # Cache
        self._tariff_cache: Optional[Tuple[TariffSchedule, float]] = None
        self._rate_cache: Dict[str, Tuple[float, float]] = {}

        # Rate monitoring
        self._current_rate: Optional[float] = None
        self._current_period: Optional[RatePeriod] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._rate_change_callbacks: List[Callable[[RateChange], None]] = []

        # Statistics
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "rate_changes_detected": 0,
            "last_request": None,
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to tariff provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from provider."""
        pass

    @abstractmethod
    async def fetch_tariff_schedule(self) -> TariffSchedule:
        """Fetch complete tariff schedule."""
        pass

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def _get_cached_tariff(self) -> Optional[TariffSchedule]:
        """Get cached tariff if not expired."""
        if self._tariff_cache:
            tariff, cache_time = self._tariff_cache
            if time_module.time() - cache_time < self._config.cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return tariff
        return None

    def _set_cached_tariff(self, tariff: TariffSchedule) -> None:
        """Set cached tariff."""
        self._tariff_cache = (tariff, time_module.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "provider": self._config.provider_name,
            "current_rate": self._current_rate,
            "current_period": self._current_period.value if self._current_period else None,
        }


# =============================================================================
# Utility Tariff Connector
# =============================================================================


class UtilityTariffConnector(TariffProviderBase):
    """
    Utility tariff API connector.

    Supports:
    - OpenEI Utility Rate Database
    - Direct utility APIs
    - Custom rate schedules
    """

    def __init__(self, config: TariffProviderConfig) -> None:
        """Initialize utility tariff connector."""
        super().__init__(config)
        self._tariff_schedule: Optional[TariffSchedule] = None

    async def connect(self) -> bool:
        """Connect to utility API."""
        try:
            import httpx

            base_url = str(self._config.base_url) if self._config.base_url else (
                "https://api.openei.org/utility_rates"
            )

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
            )

            self._connected = True
            self._logger.info(f"Connected to utility tariff API: {base_url}")

            # Start rate monitoring if enabled
            if self._config.monitor_rate_changes:
                self._monitor_task = asyncio.create_task(self._rate_monitor_loop())

            return True

        except Exception as e:
            self._logger.error(f"Utility API connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from utility API."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False

    async def fetch_tariff_schedule(self) -> TariffSchedule:
        """Fetch complete tariff schedule from utility API."""
        # Check cache
        cached = self._get_cached_tariff()
        if cached:
            return cached

        if not self._connected:
            raise ConnectionError("Not connected to utility API")

        try:
            params = {
                "api_key": self._config.api_key,
                "format": "json",
            }

            if self._config.tariff_id:
                params["getpage"] = self._config.tariff_id
            elif self._config.utility_id:
                params["eia"] = self._config.utility_id
            elif self._config.zip_code:
                params["address"] = self._config.zip_code

            response = await self._client.get("/", params=params)
            response.raise_for_status()

            data = response.json()
            self._stats["requests"] += 1
            self._stats["last_request"] = datetime.now(timezone.utc)

            tariff = self._parse_tariff_response(data)
            self._tariff_schedule = tariff
            self._set_cached_tariff(tariff)

            return tariff

        except Exception as e:
            self._logger.error(f"Failed to fetch tariff schedule: {e}")
            raise

    def _parse_tariff_response(self, data: Dict[str, Any]) -> TariffSchedule:
        """Parse tariff data from API response."""
        items = data.get("items", [{}])
        tariff_data = items[0] if items else {}

        schedule = TariffSchedule(
            schedule_id=tariff_data.get("label", "unknown"),
            schedule_name=tariff_data.get("name", "Unknown Tariff"),
            utility_name=tariff_data.get("utility", "Unknown Utility"),
            tariff_type=TariffType.TOU if "tou" in str(tariff_data).lower() else TariffType.FIXED,
            effective_date=date.today(),
        )

        # Parse energy rates
        if "energyratestructure" in tariff_data:
            for period_idx, period_rates in enumerate(tariff_data["energyratestructure"]):
                for tier_idx, tier_data in enumerate(period_rates):
                    rate = TariffRate(
                        rate_id=f"energy_{period_idx}_{tier_idx}",
                        charge_type=ChargeType.ENERGY,
                        rate=float(tier_data.get("rate", 0)),
                        unit="$/kWh",
                        tier=tier_idx + 1 if len(period_rates) > 1 else None,
                        tier_max_kwh=float(tier_data.get("max", 0)) if tier_data.get("max") else None,
                    )
                    schedule.energy_rates.append(rate)

        # Parse demand charges
        if "demandratestructure" in tariff_data:
            for period_idx, period_rates in enumerate(tariff_data["demandratestructure"]):
                for tier_data in period_rates:
                    charge = DemandCharge(
                        charge_id=f"demand_{period_idx}",
                        rate_per_kw=float(tier_data.get("rate", 0)),
                        max_demand_kw=float(tier_data.get("max")) if tier_data.get("max") else None,
                    )
                    schedule.demand_charges.append(charge)

        # Parse fixed charges
        if "fixedchargefirstmeter" in tariff_data:
            schedule.monthly_fixed_charge = float(tariff_data["fixedchargefirstmeter"])

        return schedule

    async def get_current_rate(self, timestamp: Optional[datetime] = None) -> TariffRate:
        """
        Get current applicable rate.

        Args:
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Current applicable rate
        """
        tariff = await self.fetch_tariff_schedule()
        timestamp = timestamp or datetime.now(timezone.utc)

        # Determine current period
        current_period = self._determine_rate_period(tariff, timestamp)

        # Find matching rate
        for tou_rate in tariff.tou_rates:
            if tou_rate.rate_period == current_period:
                if self._is_rate_applicable(tou_rate, timestamp):
                    rate = TariffRate(
                        rate_id=f"current_{current_period.value}",
                        charge_type=ChargeType.ENERGY,
                        rate=tou_rate.rate,
                        unit="$/kWh",
                        rate_period=current_period,
                        tariff_type=TariffType.TOU,
                    )
                    self._current_rate = rate.rate
                    self._current_period = current_period
                    return rate

        # Default to first energy rate
        if tariff.energy_rates:
            rate = tariff.energy_rates[0]
            self._current_rate = rate.rate
            return rate

        raise ValueError("No applicable rate found")

    def _determine_rate_period(
        self,
        tariff: TariffSchedule,
        timestamp: datetime
    ) -> RatePeriod:
        """Determine rate period for given timestamp."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        is_holiday = timestamp.date() in tariff.holidays
        month = timestamp.month

        # Determine season
        is_summer = month in tariff.summer_months

        # Check TOU periods
        for tou_rate in tariff.tou_rates:
            if self._is_rate_applicable(tou_rate, timestamp):
                return tou_rate.rate_period

        # Default periods based on time and season
        if is_weekend or is_holiday:
            return RatePeriod.OFF_PEAK

        if is_summer:
            if 14 <= hour < 19:
                return RatePeriod.ON_PEAK
            elif 12 <= hour < 14 or 19 <= hour < 21:
                return RatePeriod.MID_PEAK
            else:
                return RatePeriod.OFF_PEAK
        else:
            if 17 <= hour < 21:
                return RatePeriod.ON_PEAK
            elif 7 <= hour < 17:
                return RatePeriod.MID_PEAK
            else:
                return RatePeriod.OFF_PEAK

    def _is_rate_applicable(
        self,
        tou_rate: TimeOfUseRate,
        timestamp: datetime
    ) -> bool:
        """Check if TOU rate is applicable for given timestamp."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5

        # Check day type
        if is_weekend and not tou_rate.weekends:
            return False
        if not is_weekend and not tou_rate.weekdays:
            return False
        if tou_rate.specific_days and day_of_week not in tou_rate.specific_days:
            return False

        # Check time window
        if tou_rate.start_hour <= tou_rate.end_hour:
            if not (tou_rate.start_hour <= hour < tou_rate.end_hour):
                return False
        else:  # Crosses midnight
            if not (hour >= tou_rate.start_hour or hour < tou_rate.end_hour):
                return False

        # Check season
        if tou_rate.season_months:
            if timestamp.month not in tou_rate.season_months:
                return False

        return True

    async def get_rate_forecast(
        self,
        hours_ahead: int = 24,
        interval_minutes: int = 60
    ) -> List[Tuple[datetime, float, RatePeriod]]:
        """
        Get rate forecast for specified period.

        Args:
            hours_ahead: Hours to forecast
            interval_minutes: Interval between forecasts

        Returns:
            List of (timestamp, rate, period) tuples
        """
        tariff = await self.fetch_tariff_schedule()
        forecast = []

        now = datetime.now(timezone.utc)
        intervals = (hours_ahead * 60) // interval_minutes

        for i in range(intervals):
            timestamp = now + timedelta(minutes=i * interval_minutes)
            period = self._determine_rate_period(tariff, timestamp)

            # Get rate for period
            rate = 0.0
            for tou_rate in tariff.tou_rates:
                if tou_rate.rate_period == period:
                    rate = tou_rate.rate
                    break

            if rate == 0.0 and tariff.energy_rates:
                rate = tariff.energy_rates[0].rate

            forecast.append((timestamp, rate, period))

        return forecast

    async def calculate_cost(
        self,
        energy_kwh: float,
        peak_demand_kw: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate energy cost.

        Args:
            energy_kwh: Energy consumption (kWh)
            peak_demand_kw: Peak demand (kW)
            timestamp: Timestamp for rate calculation

        Returns:
            Cost breakdown dictionary
        """
        tariff = await self.fetch_tariff_schedule()
        rate = await self.get_current_rate(timestamp)

        costs = {
            "energy_cost": energy_kwh * rate.rate,
            "demand_cost": 0.0,
            "fixed_cost": tariff.monthly_fixed_charge / 30,  # Daily portion
            "taxes": 0.0,
            "total": 0.0,
        }

        # Calculate demand charges
        for charge in tariff.demand_charges:
            billable_kw = max(0, peak_demand_kw - charge.included_kw)
            if charge.max_demand_kw:
                billable_kw = min(billable_kw, charge.max_demand_kw)
            costs["demand_cost"] += billable_kw * charge.rate_per_kw / 30  # Daily portion

        # Calculate taxes
        subtotal = costs["energy_cost"] + costs["demand_cost"] + costs["fixed_cost"]
        costs["taxes"] = subtotal * (tariff.tax_rate_percent / 100)

        costs["total"] = subtotal + costs["taxes"]

        return costs

    async def subscribe_to_rate_changes(
        self,
        callback: Callable[[RateChange], None]
    ) -> None:
        """Subscribe to rate change notifications."""
        self._rate_change_callbacks.append(callback)

    async def _rate_monitor_loop(self) -> None:
        """Background task to monitor rate changes."""
        last_rate = self._current_rate
        last_period = self._current_period

        while True:
            try:
                await asyncio.sleep(self._config.rate_change_check_interval)

                # Get current rate
                rate = await self.get_current_rate()

                # Check for changes
                if last_rate is not None and last_period is not None:
                    if rate.rate != last_rate or rate.rate_period != last_period:
                        change = RateChange(
                            change_id=f"change_{time_module.time()}",
                            change_type=RateChangeType.TOU_PERIOD,
                            effective_time=datetime.now(timezone.utc),
                            rate_period=rate.rate_period,
                            old_rate=last_rate,
                            new_rate=rate.rate,
                            rate_change_percent=(
                                ((rate.rate - last_rate) / last_rate * 100)
                                if last_rate > 0 else 0
                            ),
                        )

                        self._stats["rate_changes_detected"] += 1

                        # Notify callbacks
                        for callback in self._rate_change_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(change)
                                else:
                                    callback(change)
                            except Exception as e:
                                self._logger.error(f"Rate change callback error: {e}")

                last_rate = rate.rate
                last_period = rate.rate_period

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Rate monitor error: {e}")


# =============================================================================
# Wholesale Market Connector
# =============================================================================


class WholesaleMarketConnector(TariffProviderBase):
    """
    Wholesale energy market connector.

    Supports:
    - Day-ahead market (DAM) prices
    - Real-time market (RTM) prices
    - Ancillary service prices
    """

    def __init__(self, config: TariffProviderConfig) -> None:
        """Initialize wholesale market connector."""
        super().__init__(config)
        self._current_prices: Dict[str, float] = {}
        self._price_history: deque = deque(maxlen=10000)

    async def connect(self) -> bool:
        """Connect to wholesale market API."""
        try:
            import httpx

            base_url = str(self._config.base_url) if self._config.base_url else (
                "https://api.iso-market.example.com"  # Placeholder
            )

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
            )

            self._connected = True
            self._logger.info(f"Connected to wholesale market: {base_url}")
            return True

        except Exception as e:
            self._logger.error(f"Wholesale market connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from wholesale market."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def fetch_tariff_schedule(self) -> TariffSchedule:
        """Fetch wholesale market schedule (rates are dynamic)."""
        schedule = TariffSchedule(
            schedule_id="wholesale",
            schedule_name="Wholesale Market Rates",
            utility_name="Wholesale",
            tariff_type=TariffType.WHOLESALE,
            effective_date=date.today(),
        )

        # Wholesale rates are fetched dynamically
        current_price = await self.get_current_lmp()

        schedule.energy_rates.append(TariffRate(
            rate_id="wholesale_lmp",
            charge_type=ChargeType.ENERGY,
            rate=current_price,
            unit="$/MWh",
            tariff_type=TariffType.REAL_TIME,
        ))

        return schedule

    async def get_current_lmp(
        self,
        node: Optional[str] = None
    ) -> float:
        """
        Get current Locational Marginal Price.

        Args:
            node: Pricing node (defaults to configured node)

        Returns:
            Current LMP ($/MWh)
        """
        if not self._connected:
            raise ConnectionError("Not connected to wholesale market")

        try:
            params = {}
            if self._config.api_key:
                params["api_key"] = self._config.api_key
            if node:
                params["node"] = node

            response = await self._client.get("/lmp/current", params=params)
            response.raise_for_status()

            data = response.json()
            self._stats["requests"] += 1
            self._stats["last_request"] = datetime.now(timezone.utc)

            lmp = float(data.get("lmp", 0))
            self._current_prices["lmp"] = lmp

            # Add to history
            self._price_history.append({
                "timestamp": datetime.now(timezone.utc),
                "lmp": lmp,
            })

            return lmp

        except Exception as e:
            self._logger.error(f"Failed to get LMP: {e}")
            # Return last known price if available
            if "lmp" in self._current_prices:
                return self._current_prices["lmp"]
            raise

    async def get_dam_prices(
        self,
        trade_date: Optional[date] = None,
        node: Optional[str] = None
    ) -> List[Tuple[datetime, float]]:
        """
        Get day-ahead market prices.

        Args:
            trade_date: Trade date (defaults to tomorrow)
            node: Pricing node

        Returns:
            List of (hour, price) tuples
        """
        if not self._connected:
            raise ConnectionError("Not connected to wholesale market")

        trade_date = trade_date or date.today() + timedelta(days=1)

        try:
            params = {
                "date": trade_date.isoformat(),
            }
            if self._config.api_key:
                params["api_key"] = self._config.api_key
            if node:
                params["node"] = node

            response = await self._client.get("/dam/prices", params=params)
            response.raise_for_status()

            data = response.json()
            prices = []

            for item in data.get("prices", []):
                hour = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                price = float(item["price"])
                prices.append((hour, price))

            return prices

        except Exception as e:
            self._logger.error(f"Failed to get DAM prices: {e}")
            raise

    async def get_price_forecast(
        self,
        hours_ahead: int = 24
    ) -> List[Tuple[datetime, float]]:
        """
        Get price forecast.

        Args:
            hours_ahead: Hours to forecast

        Returns:
            List of (timestamp, forecasted_price) tuples
        """
        # Try DAM prices first
        try:
            dam_prices = await self.get_dam_prices()
            if dam_prices:
                return dam_prices[:hours_ahead]
        except Exception:
            pass

        # Generate forecast based on historical patterns
        forecast = []
        now = datetime.now(timezone.utc)

        # Simple forecast based on recent history
        recent_prices = [p["lmp"] for p in list(self._price_history)[-24:]]
        if not recent_prices:
            recent_prices = [50.0]  # Default $/MWh

        avg_price = sum(recent_prices) / len(recent_prices)

        for hour in range(hours_ahead):
            timestamp = now + timedelta(hours=hour)
            # Simple pattern: higher during day
            if 6 <= timestamp.hour <= 22:
                price = avg_price * 1.2
            else:
                price = avg_price * 0.8
            forecast.append((timestamp, price))

        return forecast


# =============================================================================
# LMP Pricing Connector
# =============================================================================


class LMPPricingConnector(TariffProviderBase):
    """
    Locational Marginal Price connector.

    Provides real-time and historical LMP data from ISO/RTO markets.
    """

    def __init__(self, config: TariffProviderConfig) -> None:
        """Initialize LMP connector."""
        super().__init__(config)
        self._current_lmp: Optional[float] = None
        self._lmp_history: deque = deque(maxlen=10000)
        self._subscription_task: Optional[asyncio.Task] = None
        self._lmp_callbacks: List[Callable[[float, datetime], None]] = []

    async def connect(self) -> bool:
        """Connect to LMP data source."""
        try:
            import httpx

            base_url = str(self._config.base_url) if self._config.base_url else (
                "https://api.iso-lmp.example.com"  # Placeholder
            )

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._config.request_timeout,
            )

            self._connected = True
            self._logger.info(f"Connected to LMP data source: {base_url}")
            return True

        except Exception as e:
            self._logger.error(f"LMP connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from LMP source."""
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        self._connected = False

    async def fetch_tariff_schedule(self) -> TariffSchedule:
        """Fetch LMP-based tariff (dynamic rates)."""
        schedule = TariffSchedule(
            schedule_id="lmp",
            schedule_name="LMP Real-Time Pricing",
            utility_name="ISO/RTO",
            tariff_type=TariffType.REAL_TIME,
            effective_date=date.today(),
        )

        current_lmp = await self.get_current_lmp()

        schedule.energy_rates.append(TariffRate(
            rate_id="lmp_current",
            charge_type=ChargeType.ENERGY,
            rate=current_lmp / 1000,  # Convert $/MWh to $/kWh
            unit="$/kWh",
            tariff_type=TariffType.REAL_TIME,
        ))

        return schedule

    async def get_current_lmp(
        self,
        node: Optional[str] = None
    ) -> float:
        """
        Get current LMP.

        Args:
            node: Pricing node

        Returns:
            Current LMP ($/MWh)
        """
        if not self._connected:
            raise ConnectionError("Not connected to LMP source")

        cache_key = f"lmp:{node or 'default'}"
        if cache_key in self._rate_cache:
            cached_value, cache_time = self._rate_cache[cache_key]
            if time_module.time() - cache_time < 60:  # 1-minute cache for real-time
                return cached_value

        try:
            params = {}
            if self._config.api_key:
                params["api_key"] = self._config.api_key
            if node:
                params["node"] = node

            response = await self._client.get("/lmp", params=params)
            response.raise_for_status()

            data = response.json()
            self._stats["requests"] += 1
            self._stats["last_request"] = datetime.now(timezone.utc)

            lmp = float(data.get("lmp", data.get("price", 0)))

            # Update cache and history
            self._rate_cache[cache_key] = (lmp, time_module.time())
            self._current_lmp = lmp
            self._lmp_history.append({
                "timestamp": datetime.now(timezone.utc),
                "lmp": lmp,
                "node": node,
            })

            return lmp

        except Exception as e:
            self._logger.error(f"Failed to get LMP: {e}")
            if self._current_lmp is not None:
                return self._current_lmp
            raise

    async def get_historical_lmp(
        self,
        start_time: datetime,
        end_time: datetime,
        node: Optional[str] = None,
        interval_minutes: int = 5
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical LMP data.

        Args:
            start_time: Start time
            end_time: End time
            node: Pricing node
            interval_minutes: Data interval

        Returns:
            List of (timestamp, lmp) tuples
        """
        if not self._connected:
            raise ConnectionError("Not connected to LMP source")

        try:
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "interval": interval_minutes,
            }
            if self._config.api_key:
                params["api_key"] = self._config.api_key
            if node:
                params["node"] = node

            response = await self._client.get("/lmp/historical", params=params)
            response.raise_for_status()

            data = response.json()
            prices = []

            for item in data.get("prices", []):
                timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                lmp = float(item["lmp"])
                prices.append((timestamp, lmp))

            return prices

        except Exception as e:
            self._logger.error(f"Failed to get historical LMP: {e}")
            raise

    async def subscribe_to_lmp(
        self,
        callback: Callable[[float, datetime], None],
        interval_seconds: int = 60
    ) -> None:
        """
        Subscribe to LMP updates.

        Args:
            callback: Callback function (lmp, timestamp)
            interval_seconds: Update interval
        """
        self._lmp_callbacks.append(callback)

        if not self._subscription_task:
            self._subscription_task = asyncio.create_task(
                self._lmp_subscription_loop(interval_seconds)
            )

    async def _lmp_subscription_loop(self, interval: int) -> None:
        """Background task for LMP subscription."""
        while True:
            try:
                lmp = await self.get_current_lmp()
                timestamp = datetime.now(timezone.utc)

                for callback in self._lmp_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(lmp, timestamp)
                        else:
                            callback(lmp, timestamp)
                    except Exception as e:
                        self._logger.error(f"LMP callback error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"LMP subscription error: {e}")
                await asyncio.sleep(interval)

    def get_lmp_statistics(self, hours: int = 24) -> Dict[str, float]:
        """Get LMP statistics for specified period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [p["lmp"] for p in self._lmp_history if p["timestamp"] >= cutoff]

        if not recent:
            return {}

        return {
            "min": min(recent),
            "max": max(recent),
            "avg": sum(recent) / len(recent),
            "current": recent[-1] if recent else 0.0,
            "count": len(recent),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_tariff_connector(
    utility_id: Optional[str] = None,
    tariff_id: Optional[str] = None,
    zip_code: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> UtilityTariffConnector:
    """
    Create utility tariff connector.

    Args:
        utility_id: Utility identifier
        tariff_id: Tariff ID
        zip_code: ZIP code for rate lookup
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Configured utility tariff connector
    """
    config = TariffProviderConfig(
        provider_type="utility",
        provider_name=kwargs.get("provider_name", "OpenEI"),
        utility_id=utility_id,
        tariff_id=tariff_id,
        zip_code=zip_code,
        api_key=api_key,
        **{k: v for k, v in kwargs.items() if k not in ["provider_name"]}
    )
    return UtilityTariffConnector(config)


def create_wholesale_connector(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> WholesaleMarketConnector:
    """
    Create wholesale market connector.

    Args:
        base_url: Market API URL
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Configured wholesale market connector
    """
    config = TariffProviderConfig(
        provider_type="wholesale",
        provider_name=kwargs.get("provider_name", "Wholesale Market"),
        base_url=base_url,
        api_key=api_key,
        **{k: v for k, v in kwargs.items() if k not in ["provider_name"]}
    )
    return WholesaleMarketConnector(config)


def create_lmp_connector(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LMPPricingConnector:
    """
    Create LMP pricing connector.

    Args:
        base_url: LMP data source URL
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Configured LMP connector
    """
    config = TariffProviderConfig(
        provider_type="lmp",
        provider_name=kwargs.get("provider_name", "LMP Provider"),
        base_url=base_url,
        api_key=api_key,
        **{k: v for k, v in kwargs.items() if k not in ["provider_name"]}
    )
    return LMPPricingConnector(config)
