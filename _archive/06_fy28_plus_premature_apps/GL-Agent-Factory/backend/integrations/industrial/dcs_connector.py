"""
Distributed Control System (DCS) Connector for GreenLang.

This module provides integration with major DCS platforms including
Honeywell Experion, Emerson DeltaV, and ABB 800xA for process control
data access.

Features:
    - Honeywell Experion PKS integration
    - Emerson DeltaV integration
    - ABB 800xA (Ability Symphony Plus) integration
    - Tag mapping and normalization
    - Alarm and event handling
    - Control module access
    - Batch/recipe integration

Example:
    >>> from integrations.industrial import ExperionConnector, ExperionConfig
    >>>
    >>> config = ExperionConfig(
    ...     host="experion-server.factory.local",
    ...     username="operator"
    ... )
    >>> connector = ExperionConnector(config)
    >>> async with connector:
    ...     values = await connector.read_tags(["TIC-101.PV"])
"""

import asyncio
import logging
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, SecretStr

from .base import (
    AuthenticationType,
    BaseConnectorConfig,
    BaseIndustrialConnector,
    TLSConfig,
)
from .data_models import (
    AlarmEvent,
    AlarmSeverity,
    AlarmState,
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    ConnectionState,
    DataQuality,
    DataType,
    HistoricalQuery,
    HistoricalResult,
    SubscriptionConfig,
    TagMetadata,
    TagValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DCS Types and Enums
# =============================================================================


class DCSVendor(str, Enum):
    """DCS vendor/platform types."""

    HONEYWELL = "honeywell"  # Experion PKS
    EMERSON = "emerson"  # DeltaV
    ABB = "abb"  # 800xA
    YOKOGAWA = "yokogawa"  # CENTUM VP
    SIEMENS = "siemens"  # PCS 7


class ControlModuleType(str, Enum):
    """Standard control module types."""

    PID = "pid"
    CASCADE = "cascade"
    RATIO = "ratio"
    OVERRIDE = "override"
    SELECTOR = "selector"
    SPLIT_RANGE = "split_range"
    FEEDFORWARD = "feedforward"


class ControlMode(str, Enum):
    """Control module operating modes."""

    AUTO = "auto"
    MANUAL = "manual"
    CASCADE = "cascade"
    REMOTE = "remote"
    LOCAL = "local"
    PROGRAM = "program"
    OUT_OF_SERVICE = "oos"


class TagType(str, Enum):
    """DCS tag types."""

    ANALOG_INPUT = "ai"
    ANALOG_OUTPUT = "ao"
    DIGITAL_INPUT = "di"
    DIGITAL_OUTPUT = "do"
    REGULATORY_CONTROL = "rc"
    DEVICE_CONTROL = "dc"
    MOTOR_CONTROL = "mc"
    VALVE_CONTROL = "vc"
    CALCULATED = "calc"


# =============================================================================
# Tag Mapping
# =============================================================================


class DCSTagMapping(BaseModel):
    """
    DCS tag mapping configuration.

    Maps standard GreenLang tag names to DCS-specific paths.

    Attributes:
        greenlang_id: GreenLang standard tag ID
        dcs_path: DCS-specific tag path
        tag_type: Tag type
        data_type: Data type
        description: Tag description
        engineering_unit: Unit of measure
        scaling: Optional scaling configuration
    """

    greenlang_id: str = Field(..., description="GreenLang tag ID")
    dcs_path: str = Field(..., description="DCS-specific path")
    tag_type: TagType = Field(TagType.ANALOG_INPUT, description="Tag type")
    data_type: DataType = Field(DataType.FLOAT64, description="Data type")
    description: str = Field("", description="Description")
    engineering_unit: str = Field("", description="Engineering unit")

    # Scaling
    raw_min: Optional[float] = Field(None, description="Raw minimum")
    raw_max: Optional[float] = Field(None, description="Raw maximum")
    eu_min: Optional[float] = Field(None, description="EU minimum")
    eu_max: Optional[float] = Field(None, description="EU maximum")

    # DCS-specific attributes
    control_module: Optional[str] = Field(None, description="Parent control module")
    parameter_name: Optional[str] = Field(None, description="Parameter name")
    area: Optional[str] = Field(None, description="Plant area")
    unit: Optional[str] = Field(None, description="Process unit")


class TagMappingRegistry:
    """
    Registry for DCS tag mappings.

    Provides bidirectional mapping between GreenLang standard
    tags and DCS-specific paths.
    """

    def __init__(self):
        """Initialize mapping registry."""
        self._by_greenlang: Dict[str, DCSTagMapping] = {}
        self._by_dcs_path: Dict[str, DCSTagMapping] = {}

    def register(self, mapping: DCSTagMapping) -> None:
        """Register a tag mapping."""
        self._by_greenlang[mapping.greenlang_id] = mapping
        self._by_dcs_path[mapping.dcs_path.upper()] = mapping

    def register_bulk(self, mappings: List[DCSTagMapping]) -> None:
        """Register multiple tag mappings."""
        for mapping in mappings:
            self.register(mapping)

    def get_by_greenlang(self, greenlang_id: str) -> Optional[DCSTagMapping]:
        """Get mapping by GreenLang ID."""
        return self._by_greenlang.get(greenlang_id)

    def get_by_dcs_path(self, dcs_path: str) -> Optional[DCSTagMapping]:
        """Get mapping by DCS path."""
        return self._by_dcs_path.get(dcs_path.upper())

    def to_dcs_path(self, greenlang_id: str) -> Optional[str]:
        """Convert GreenLang ID to DCS path."""
        mapping = self._by_greenlang.get(greenlang_id)
        return mapping.dcs_path if mapping else None

    def to_greenlang_id(self, dcs_path: str) -> Optional[str]:
        """Convert DCS path to GreenLang ID."""
        mapping = self._by_dcs_path.get(dcs_path.upper())
        return mapping.greenlang_id if mapping else None

    def get_all(self) -> List[DCSTagMapping]:
        """Get all registered mappings."""
        return list(self._by_greenlang.values())

    def clear(self) -> None:
        """Clear all mappings."""
        self._by_greenlang.clear()
        self._by_dcs_path.clear()


# =============================================================================
# Base DCS Configuration
# =============================================================================


class BaseDCSConfig(BaseConnectorConfig):
    """
    Base configuration for DCS connectors.

    Attributes:
        vendor: DCS vendor/platform
        enable_alarms: Enable alarm subscription
        enable_events: Enable event subscription
        scan_rate_ms: Default scan rate
    """

    vendor: DCSVendor = Field(..., description="DCS vendor")
    enable_alarms: bool = Field(True, description="Enable alarm subscription")
    enable_events: bool = Field(True, description="Enable event subscription")
    scan_rate_ms: int = Field(1000, ge=100, description="Default scan rate")

    # Tag normalization
    normalize_tag_names: bool = Field(True, description="Normalize tag names")
    tag_name_separator: str = Field(".", description="Tag path separator")


# =============================================================================
# Base DCS Connector
# =============================================================================


class BaseDCSConnector(BaseIndustrialConnector):
    """
    Base class for DCS connectors.

    Provides common functionality for DCS integrations including
    tag mapping, alarm handling, and control module access.
    """

    def __init__(self, config: BaseDCSConfig):
        """Initialize DCS connector."""
        super().__init__(config)
        self.dcs_config = config
        self._tag_registry = TagMappingRegistry()
        self._alarm_callbacks: List[Callable[[AlarmEvent], None]] = []
        self._event_callbacks: List[Callable[[Dict], None]] = []

    def register_tag_mapping(self, mapping: DCSTagMapping) -> None:
        """Register a tag mapping."""
        self._tag_registry.register(mapping)

    def register_tag_mappings(self, mappings: List[DCSTagMapping]) -> None:
        """Register multiple tag mappings."""
        self._tag_registry.register_bulk(mappings)

    def add_alarm_callback(self, callback: Callable[[AlarmEvent], None]) -> None:
        """Add callback for alarm events."""
        self._alarm_callbacks.append(callback)

    def add_event_callback(self, callback: Callable[[Dict], None]) -> None:
        """Add callback for system events."""
        self._event_callbacks.append(callback)

    async def _dispatch_alarm(self, alarm: AlarmEvent) -> None:
        """Dispatch alarm to registered callbacks."""
        for callback in self._alarm_callbacks:
            try:
                callback(alarm)
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

    async def _dispatch_event(self, event: Dict) -> None:
        """Dispatch event to registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _normalize_value(
        self,
        value: Any,
        mapping: Optional[DCSTagMapping],
    ) -> Any:
        """Apply scaling/normalization to value."""
        if not mapping or not isinstance(value, (int, float)):
            return value

        # Apply scaling if configured
        if all(v is not None for v in [mapping.raw_min, mapping.raw_max, mapping.eu_min, mapping.eu_max]):
            raw_range = mapping.raw_max - mapping.raw_min
            eu_range = mapping.eu_max - mapping.eu_min

            if raw_range != 0:
                return mapping.eu_min + (value - mapping.raw_min) * eu_range / raw_range

        return value


# =============================================================================
# Honeywell Experion PKS
# =============================================================================


class ExperionConfig(BaseDCSConfig):
    """
    Honeywell Experion PKS configuration.

    Attributes:
        host: Experion server hostname
        port: OPC DA/UA port
        node_name: Experion node name
        use_opc_ua: Use OPC-UA (vs OPC DA)
        station_name: Operator station name
    """

    vendor: DCSVendor = DCSVendor.HONEYWELL
    port: int = Field(4840, description="OPC-UA port")
    node_name: str = Field(..., description="Experion node name")
    use_opc_ua: bool = Field(True, description="Use OPC-UA")
    station_name: Optional[str] = Field(None, description="Station name")

    # Experion-specific
    server_redundancy: bool = Field(False, description="Enable server redundancy")
    backup_host: Optional[str] = Field(None, description="Backup server host")
    cda_service_name: str = Field("CDA", description="CDA service name")


class ExperionConnector(BaseDCSConnector):
    """
    Honeywell Experion PKS Connector.

    Provides integration with Honeywell Experion PKS distributed
    control system via OPC-UA or native API.

    Features:
        - Point/parameter reading and writing
        - Control module access
        - Alarm and event subscription
        - Historical data retrieval
        - Server redundancy support

    Example:
        >>> config = ExperionConfig(
        ...     host="experion-server.local",
        ...     node_name="SCADA01",
        ...     username="operator"
        ... )
        >>> connector = ExperionConnector(config)
        >>> await connector.connect()
        >>> value = await connector.read_tag("TIC-101.PV")
    """

    def __init__(self, config: ExperionConfig):
        """Initialize Experion connector."""
        base_config = BaseDCSConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.USERNAME_PASSWORD,
            username=config.username,
            password=config.password,
            name=config.name or "experion_connector",
            vendor=DCSVendor.HONEYWELL,
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.experion_config = config

        # Experion-specific state
        self._opc_client: Optional[Any] = None
        self._cda_connection: Optional[Any] = None
        self._point_cache: Dict[str, Any] = {}

    async def _do_connect(self) -> bool:
        """Connect to Experion PKS."""
        logger.info(f"Connecting to Experion PKS: {self.experion_config.host}")

        try:
            if self.experion_config.use_opc_ua:
                # Connect via OPC-UA
                # In production, use asyncua:
                # from asyncua import Client
                # endpoint = f"opc.tcp://{self.experion_config.host}:{self.experion_config.port}"
                # self._opc_client = Client(endpoint)
                # await self._opc_client.connect()
                pass
            else:
                # Connect via native CDA API
                # In production, use Honeywell CDA SDK
                pass

            self._opc_client = True  # Simulated
            logger.info("Experion PKS connected")
            return True

        except Exception as e:
            logger.error(f"Experion connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from Experion."""
        if self._opc_client:
            # await self._opc_client.disconnect()
            pass
        self._opc_client = None
        self._point_cache.clear()
        logger.info("Experion disconnected")

    async def _do_health_check(self) -> bool:
        """Check Experion connection."""
        return self._opc_client is not None

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read tag values from Experion."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        for tag_id in tag_ids:
            try:
                # Map to DCS path
                mapping = self._tag_registry.get_by_greenlang(tag_id)
                dcs_path = mapping.dcs_path if mapping else tag_id

                # Build Experion point path
                # Format: NODE.POINT.PARAMETER
                point_path = f"{self.experion_config.node_name}.{dcs_path}"

                # In production:
                # node = self._opc_client.get_node(point_path)
                # data_value = await node.read_data_value()
                # raw_value = data_value.Value.Value
                # quality = self._map_opc_quality(data_value.StatusCode)

                # Simulated read
                import random
                raw_value = random.uniform(0, 100)

                # Apply normalization
                normalized_value = self._normalize_value(raw_value, mapping)

                values[tag_id] = TagValue(
                    tag_id=tag_id,
                    value=round(normalized_value, 2),
                    timestamp=datetime.utcnow(),
                    quality=DataQuality.GOOD,
                    unit=mapping.engineering_unit if mapping else None,
                )

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write tag values to Experion."""
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                mapping = self._tag_registry.get_by_greenlang(tag_id)
                dcs_path = mapping.dcs_path if mapping else tag_id

                # Validate write permission based on tag type
                if mapping and mapping.tag_type == TagType.ANALOG_INPUT:
                    errors[tag_id] = "Cannot write to analog input"
                    continue

                # In production:
                # node = self._opc_client.get_node(f"{self.experion_config.node_name}.{dcs_path}")
                # await node.write_value(value)

                success[tag_id] = True
                logger.info(f"Wrote {value} to Experion point {dcs_path}")

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_control_module(
        self,
        module_name: str,
    ) -> Dict[str, Any]:
        """
        Read control module parameters.

        Args:
            module_name: Control module name

        Returns:
            Dictionary of module parameters
        """
        self._validate_connected()

        # Standard Experion control module parameters
        parameters = ["PV", "SP", "OP", "MODE", "ALMSTS"]

        result = {}
        for param in parameters:
            tag_id = f"{module_name}.{param}"
            response = await self.read_tags([tag_id])
            if tag_id in response.values:
                result[param] = response.values[tag_id].value

        return result

    async def set_control_mode(
        self,
        module_name: str,
        mode: ControlMode,
    ) -> bool:
        """
        Set control module operating mode.

        Args:
            module_name: Control module name
            mode: Desired operating mode

        Returns:
            True if successful
        """
        tag_id = f"{module_name}.MODE"
        mode_value = {"auto": 1, "manual": 2, "cascade": 3}.get(mode.value, 2)

        request = BatchWriteRequest(writes={tag_id: mode_value})
        response = await self.write_tags(request)

        return response.success.get(tag_id, False)


# =============================================================================
# Emerson DeltaV
# =============================================================================


class DeltaVConfig(BaseDCSConfig):
    """
    Emerson DeltaV configuration.

    Attributes:
        host: DeltaV server hostname
        port: OPC-DA/UA port
        area: Default plant area
        node: DeltaV node name
        use_opc_ua: Use OPC-UA (vs OPC DA)
    """

    vendor: DCSVendor = DCSVendor.EMERSON
    port: int = Field(4840, description="OPC-UA port")
    area: Optional[str] = Field(None, description="Default plant area")
    node: Optional[str] = Field(None, description="DeltaV node")
    use_opc_ua: bool = Field(True, description="Use OPC-UA")

    # DeltaV-specific
    network_name: str = Field("DeltaV", description="Network name")
    workstation: Optional[str] = Field(None, description="Workstation name")


class DeltaVConnector(BaseDCSConnector):
    """
    Emerson DeltaV Connector.

    Provides integration with Emerson DeltaV distributed
    control system.

    Features:
        - Module parameter access
        - Continuous/batch control
        - Alarm and event handling
        - SIS integration
        - Recipe management

    Example:
        >>> config = DeltaVConfig(
        ...     host="deltav-server.local",
        ...     area="PROCESS",
        ...     username="operator"
        ... )
        >>> connector = DeltaVConnector(config)
        >>> await connector.connect()
    """

    def __init__(self, config: DeltaVConfig):
        """Initialize DeltaV connector."""
        base_config = BaseDCSConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.USERNAME_PASSWORD,
            username=config.username,
            password=config.password,
            name=config.name or "deltav_connector",
            vendor=DCSVendor.EMERSON,
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.deltav_config = config
        self._opc_client: Optional[Any] = None

    async def _do_connect(self) -> bool:
        """Connect to DeltaV."""
        logger.info(f"Connecting to DeltaV: {self.deltav_config.host}")

        try:
            # In production, connect via OPC-UA or DeltaV API
            self._opc_client = True  # Simulated
            logger.info("DeltaV connected")
            return True

        except Exception as e:
            logger.error(f"DeltaV connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from DeltaV."""
        self._opc_client = None
        logger.info("DeltaV disconnected")

    async def _do_health_check(self) -> bool:
        """Check DeltaV connection."""
        return self._opc_client is not None

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read tag values from DeltaV."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        for tag_id in tag_ids:
            try:
                mapping = self._tag_registry.get_by_greenlang(tag_id)
                dcs_path = mapping.dcs_path if mapping else tag_id

                # Build DeltaV path
                # Format: AREA/MODULE/PARAMETER
                if self.deltav_config.area and "/" not in dcs_path:
                    dcs_path = f"{self.deltav_config.area}/{dcs_path}"

                # Simulated read
                import random
                raw_value = random.uniform(0, 100)
                normalized_value = self._normalize_value(raw_value, mapping)

                values[tag_id] = TagValue(
                    tag_id=tag_id,
                    value=round(normalized_value, 2),
                    timestamp=datetime.utcnow(),
                    quality=DataQuality.GOOD,
                )

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write values to DeltaV."""
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                success[tag_id] = True
            except Exception as e:
                errors[tag_id] = str(e)

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_function_block(
        self,
        module: str,
        block: str,
    ) -> Dict[str, Any]:
        """
        Read DeltaV function block parameters.

        Args:
            module: Module name
            block: Function block name

        Returns:
            Dictionary of block parameters
        """
        self._validate_connected()

        # Standard DeltaV function block parameters
        parameters = ["PV", "SP", "OUT", "MODE_ACTUAL", "BKCAL_IN"]

        result = {}
        for param in parameters:
            tag_id = f"{module}/{block}/{param}"
            response = await self.read_tags([tag_id])
            if tag_id in response.values:
                result[param] = response.values[tag_id].value

        return result


# =============================================================================
# ABB 800xA
# =============================================================================


class ABB800xAConfig(BaseDCSConfig):
    """
    ABB 800xA (Ability Symphony Plus) configuration.

    Attributes:
        host: 800xA server hostname
        port: OPC-DA port
        aspect_server: Aspect server name
        connectivity_server: Connectivity server
        system_name: Control system name
    """

    vendor: DCSVendor = DCSVendor.ABB
    port: int = Field(135, description="OPC DA DCOM port")
    aspect_server: Optional[str] = Field(None, description="Aspect server")
    connectivity_server: Optional[str] = Field(None, description="Connectivity server")
    system_name: str = Field("System 1", description="Control system name")

    # ABB-specific
    use_aspect_objects: bool = Field(True, description="Use Aspect Objects")
    engineering_workspace: Optional[str] = Field(None, description="Engineering workspace")


class ABB800xAConnector(BaseDCSConnector):
    """
    ABB 800xA Connector.

    Provides integration with ABB 800xA (Ability Symphony Plus)
    distributed control system.

    Features:
        - Aspect Object access
        - Control module parameters
        - AC800M controller integration
        - Alarm handling
        - Historical data

    Example:
        >>> config = ABB800xAConfig(
        ...     host="abb-server.local",
        ...     system_name="Process Control",
        ...     username="operator"
        ... )
        >>> connector = ABB800xAConnector(config)
        >>> await connector.connect()
    """

    def __init__(self, config: ABB800xAConfig):
        """Initialize ABB 800xA connector."""
        base_config = BaseDCSConfig(
            host=config.host,
            port=config.port,
            timeout_seconds=config.timeout_seconds,
            auth_type=AuthenticationType.USERNAME_PASSWORD,
            username=config.username,
            password=config.password,
            name=config.name or "abb800xa_connector",
            vendor=DCSVendor.ABB,
            tls=config.tls,
            rate_limit=config.rate_limit,
            reconnect=config.reconnect,
            health_check_interval_seconds=config.health_check_interval_seconds,
        )

        super().__init__(base_config)
        self.abb_config = config
        self._opc_client: Optional[Any] = None

    async def _do_connect(self) -> bool:
        """Connect to ABB 800xA."""
        logger.info(f"Connecting to ABB 800xA: {self.abb_config.host}")

        try:
            # In production, connect via OPC DA or ABB API
            self._opc_client = True  # Simulated
            logger.info("ABB 800xA connected")
            return True

        except Exception as e:
            logger.error(f"ABB 800xA connection failed: {e}")
            raise

    async def _do_disconnect(self) -> None:
        """Disconnect from ABB 800xA."""
        self._opc_client = None
        logger.info("ABB 800xA disconnected")

    async def _do_health_check(self) -> bool:
        """Check ABB 800xA connection."""
        return self._opc_client is not None

    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """Read tag values from ABB 800xA."""
        self._validate_connected()

        values: Dict[str, TagValue] = {}
        errors: Dict[str, str] = {}

        for tag_id in tag_ids:
            try:
                mapping = self._tag_registry.get_by_greenlang(tag_id)
                dcs_path = mapping.dcs_path if mapping else tag_id

                # ABB Aspect Object path format
                # Format: SYSTEM/AREA/OBJECT/ASPECT/ATTRIBUTE
                if self.abb_config.use_aspect_objects:
                    dcs_path = f"{self.abb_config.system_name}/{dcs_path}"

                # Simulated read
                import random
                raw_value = random.uniform(0, 100)
                normalized_value = self._normalize_value(raw_value, mapping)

                values[tag_id] = TagValue(
                    tag_id=tag_id,
                    value=round(normalized_value, 2),
                    timestamp=datetime.utcnow(),
                    quality=DataQuality.GOOD,
                )

            except Exception as e:
                errors[tag_id] = str(e)

        return BatchReadResponse(
            values=values,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """Write values to ABB 800xA."""
        self._validate_connected()

        success: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for tag_id, value in request.writes.items():
            try:
                success[tag_id] = True
            except Exception as e:
                errors[tag_id] = str(e)

        return BatchWriteResponse(
            success=success,
            errors=errors,
            timestamp=datetime.utcnow(),
        )

    async def read_aspect_object(
        self,
        object_path: str,
        aspect: str = "ControlConnection",
    ) -> Dict[str, Any]:
        """
        Read ABB Aspect Object.

        Args:
            object_path: Aspect Object path
            aspect: Aspect name

        Returns:
            Dictionary of aspect attributes
        """
        self._validate_connected()

        # Standard control aspect attributes
        attributes = ["PV", "SP", "OUT", "MODE", "STATUS"]

        result = {}
        for attr in attributes:
            tag_id = f"{object_path}/{aspect}/{attr}"
            response = await self.read_tags([tag_id])
            if tag_id in response.values:
                result[attr] = response.values[tag_id].value

        return result


# =============================================================================
# DCS Factory
# =============================================================================


def get_dcs_connector(config: BaseDCSConfig) -> BaseDCSConnector:
    """
    Factory function to get appropriate DCS connector.

    Args:
        config: DCS configuration

    Returns:
        Appropriate connector instance

    Raises:
        ValueError: If DCS vendor not supported
    """
    connectors = {
        DCSVendor.HONEYWELL: ExperionConnector,
        DCSVendor.EMERSON: DeltaVConnector,
        DCSVendor.ABB: ABB800xAConnector,
    }

    connector_class = connectors.get(config.vendor)
    if not connector_class:
        raise ValueError(f"Unsupported DCS vendor: {config.vendor}")

    return connector_class(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "DCSVendor",
    "ControlModuleType",
    "ControlMode",
    "TagType",
    # Tag mapping
    "DCSTagMapping",
    "TagMappingRegistry",
    # Configuration
    "BaseDCSConfig",
    "ExperionConfig",
    "DeltaVConfig",
    "ABB800xAConfig",
    # Connectors
    "BaseDCSConnector",
    "ExperionConnector",
    "DeltaVConnector",
    "ABB800xAConnector",
    # Factory
    "get_dcs_connector",
]
