# -*- coding: utf-8 -*-
"""
PCI-DSS Cardholder Data Mapper - SEC-010 Phase 5

Maps cardholder data (CHD) flows and identifies the Cardholder Data
Environment (CDE) scope for PCI-DSS v4.0 compliance assessment.

PCI-DSS v4.0 Requirements:
- Req 1.2.4: Accurate network diagrams
- Req 1.2.5: Accurate data flow diagrams
- Req 12.5.2: Maintain CHD scope documentation

Classes:
    - CardDataMapper: Main mapper for CHD flows and CDE scope.
    - DataFlowDiagram: Representation of CHD data flows.
    - CDEComponent: Component within the CDE.
    - CDEScopeResult: Result of CDE scope assessment.

Example:
    >>> mapper = CardDataMapper()
    >>> flow = await mapper.map_cardholder_data_flow()
    >>> scope = await mapper.identify_cde_scope()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CardDataType(str, Enum):
    """Types of cardholder data (PCI-DSS defined)."""

    PAN = "pan"  # Primary Account Number (required to encrypt)
    CARDHOLDER_NAME = "cardholder_name"
    SERVICE_CODE = "service_code"
    EXPIRATION_DATE = "expiration_date"
    CVV = "cvv"  # Sensitive Authentication Data - never store
    PIN = "pin"  # Sensitive Authentication Data - never store
    TRACK_DATA = "track_data"  # Sensitive Authentication Data - never store


class ComponentType(str, Enum):
    """Types of components that may be in CDE scope."""

    DATABASE = "database"
    APPLICATION = "application"
    API_GATEWAY = "api_gateway"
    NETWORK_DEVICE = "network_device"
    SERVER = "server"
    CONTAINER = "container"
    STORAGE = "storage"
    LOGGING_SYSTEM = "logging_system"
    THIRD_PARTY = "third_party"


class DataFlowDirection(str, Enum):
    """Direction of data flow."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class ScopeCategory(str, Enum):
    """CDE scope categories per PCI-DSS."""

    CDE = "cde"  # Cardholder Data Environment - stores/processes/transmits CHD
    CONNECTED = "connected"  # Connected to CDE, in scope
    SECURITY_IMPACTING = "security_impacting"  # Could impact CDE security
    OUT_OF_SCOPE = "out_of_scope"  # Properly segmented, not in scope


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CDEComponent(BaseModel):
    """A component within or connected to the CDE.

    Attributes:
        id: Unique component identifier.
        name: Component name.
        component_type: Type of component.
        description: Component description.
        ip_address: IP address or hostname.
        scope_category: CDE scope category.
        handles_pan: Whether component handles PAN.
        data_types: Types of card data handled.
        network_segment: Network segment/VLAN.
        owner: Component owner/team.
        last_scanned: Last vulnerability scan date.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    component_type: ComponentType
    description: str = ""
    ip_address: Optional[str] = None
    scope_category: ScopeCategory = ScopeCategory.OUT_OF_SCOPE
    handles_pan: bool = False
    data_types: List[CardDataType] = Field(default_factory=list)
    network_segment: str = ""
    owner: str = ""
    last_scanned: Optional[datetime] = None


class DataFlowConnection(BaseModel):
    """Connection between components in the data flow.

    Attributes:
        id: Unique connection identifier.
        source_id: Source component ID.
        destination_id: Destination component ID.
        direction: Flow direction.
        data_types: Types of card data flowing.
        encrypted: Whether connection is encrypted.
        encryption_protocol: Encryption protocol used.
        port: Network port used.
        description: Connection description.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    destination_id: str
    direction: DataFlowDirection = DataFlowDirection.INTERNAL
    data_types: List[CardDataType] = Field(default_factory=list)
    encrypted: bool = True
    encryption_protocol: Optional[str] = "TLS 1.3"
    port: Optional[int] = None
    description: str = ""


class DataFlowDiagram(BaseModel):
    """Complete cardholder data flow diagram.

    Attributes:
        id: Unique diagram identifier.
        name: Diagram name.
        version: Diagram version.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        components: List of components.
        connections: List of connections.
        entry_points: External entry points.
        exit_points: External exit points.
        notes: Additional notes.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = "Cardholder Data Flow Diagram"
    version: str = "1.0"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    components: List[CDEComponent] = Field(default_factory=list)
    connections: List[DataFlowConnection] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    exit_points: List[str] = Field(default_factory=list)
    notes: str = ""


class CDEScopeResult(BaseModel):
    """Result of CDE scope assessment.

    Attributes:
        assessed_at: When the assessment was performed.
        total_components: Total components assessed.
        cde_components: Components in CDE.
        connected_components: Connected components (in scope).
        security_impacting_components: Security-impacting components.
        out_of_scope_components: Components out of scope.
        scope_summary: Summary by component type.
        recommendations: Scope reduction recommendations.
    """

    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_components: int = 0
    cde_components: int = 0
    connected_components: int = 0
    security_impacting_components: int = 0
    out_of_scope_components: int = 0
    scope_summary: Dict[str, int] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Card Data Mapper
# ---------------------------------------------------------------------------


class CardDataMapper:
    """Maps cardholder data flows and CDE scope.

    Provides tools for documenting and assessing the cardholder data
    environment (CDE) as required by PCI-DSS v4.0.

    Attributes:
        components: Registry of CDE components.
        data_flow_diagram: Current data flow diagram.

    Example:
        >>> mapper = CardDataMapper()
        >>> mapper.add_component(CDEComponent(
        ...     name="Payment API",
        ...     component_type=ComponentType.APPLICATION,
        ...     handles_pan=True,
        ...     scope_category=ScopeCategory.CDE,
        ... ))
        >>> flow = await mapper.map_cardholder_data_flow()
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the card data mapper.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self.components: Dict[str, CDEComponent] = {}
        self.connections: Dict[str, DataFlowConnection] = {}
        self.data_flow_diagram: Optional[DataFlowDiagram] = None

        # Initialize with default GreenLang components
        self._initialize_default_components()

        logger.info("Initialized CardDataMapper with %d components", len(self.components))

    def add_component(self, component: CDEComponent) -> None:
        """Add a component to the CDE inventory.

        Args:
            component: The component to add.
        """
        self.components[component.id] = component
        logger.debug("Added component: %s (%s)", component.name, component.component_type.value)

    def remove_component(self, component_id: str) -> bool:
        """Remove a component from the inventory.

        Args:
            component_id: The component ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if component_id in self.components:
            del self.components[component_id]
            return True
        return False

    def add_connection(self, connection: DataFlowConnection) -> None:
        """Add a data flow connection.

        Args:
            connection: The connection to add.
        """
        self.connections[connection.id] = connection
        logger.debug(
            "Added connection: %s -> %s",
            connection.source_id,
            connection.destination_id,
        )

    async def map_cardholder_data_flow(self) -> DataFlowDiagram:
        """Generate the cardholder data flow diagram.

        Creates a complete data flow diagram showing how cardholder
        data moves through the environment.

        Returns:
            DataFlowDiagram with all components and connections.
        """
        logger.info("Mapping cardholder data flow")

        # Auto-discover components that handle PAN
        pan_components = await self._discover_pan_handlers()

        # Auto-discover connections
        connections = await self._discover_connections()

        # Identify entry and exit points
        entry_points = self._identify_entry_points()
        exit_points = self._identify_exit_points()

        # Create diagram
        self.data_flow_diagram = DataFlowDiagram(
            name="GreenLang Cardholder Data Flow",
            version="1.0",
            components=list(self.components.values()),
            connections=list(self.connections.values()) + connections,
            entry_points=entry_points,
            exit_points=exit_points,
        )

        logger.info(
            "Data flow diagram created: %d components, %d connections",
            len(self.data_flow_diagram.components),
            len(self.data_flow_diagram.connections),
        )

        return self.data_flow_diagram

    async def identify_cde_scope(self) -> CDEScopeResult:
        """Identify the CDE scope.

        Categorizes all components into CDE, connected, security-impacting,
        or out-of-scope categories.

        Returns:
            CDEScopeResult with scope assessment.
        """
        logger.info("Identifying CDE scope")

        result = CDEScopeResult(
            total_components=len(self.components),
        )

        scope_counts: Dict[str, int] = {}

        for component in self.components.values():
            # Categorize component
            scope = self._categorize_component(component)
            component.scope_category = scope

            # Count by scope
            scope_name = scope.value
            scope_counts[scope_name] = scope_counts.get(scope_name, 0) + 1

            # Count by type
            type_name = component.component_type.value
            result.scope_summary[type_name] = result.scope_summary.get(type_name, 0) + 1

        result.cde_components = scope_counts.get(ScopeCategory.CDE.value, 0)
        result.connected_components = scope_counts.get(ScopeCategory.CONNECTED.value, 0)
        result.security_impacting_components = scope_counts.get(
            ScopeCategory.SECURITY_IMPACTING.value, 0
        )
        result.out_of_scope_components = scope_counts.get(
            ScopeCategory.OUT_OF_SCOPE.value, 0
        )

        # Generate recommendations
        result.recommendations = self._generate_scope_recommendations(result)

        logger.info(
            "CDE scope identified: %d in CDE, %d connected, %d out of scope",
            result.cde_components,
            result.connected_components,
            result.out_of_scope_components,
        )

        return result

    def get_component(self, component_id: str) -> Optional[CDEComponent]:
        """Get a component by ID.

        Args:
            component_id: The component ID.

        Returns:
            The CDEComponent or None.
        """
        return self.components.get(component_id)

    def get_components_by_scope(
        self,
        scope: ScopeCategory,
    ) -> List[CDEComponent]:
        """Get all components with a specific scope category.

        Args:
            scope: The scope category to filter by.

        Returns:
            List of components in that scope.
        """
        return [
            c for c in self.components.values()
            if c.scope_category == scope
        ]

    def get_pan_handlers(self) -> List[CDEComponent]:
        """Get all components that handle PAN.

        Returns:
            List of components that handle PAN.
        """
        return [c for c in self.components.values() if c.handles_pan]

    async def export_diagram(self, format: str = "json") -> str:
        """Export the data flow diagram.

        Args:
            format: Export format (json, mermaid).

        Returns:
            Exported diagram as string.
        """
        if self.data_flow_diagram is None:
            await self.map_cardholder_data_flow()

        if format == "json":
            return self.data_flow_diagram.model_dump_json(indent=2)
        elif format == "mermaid":
            return self._to_mermaid()
        else:
            return self.data_flow_diagram.model_dump_json(indent=2)

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _initialize_default_components(self) -> None:
        """Initialize default GreenLang CDE components."""
        # Note: GreenLang is primarily a sustainability platform, not a payment
        # processor. These are hypothetical components if payment processing
        # were to be added.

        default_components = [
            CDEComponent(
                name="Payment Gateway Integration",
                component_type=ComponentType.THIRD_PARTY,
                description="Third-party payment gateway (Stripe/PayPal)",
                scope_category=ScopeCategory.CONNECTED,
                handles_pan=False,  # Tokenized, never sees raw PAN
                data_types=[],
            ),
            CDEComponent(
                name="Subscription Service",
                component_type=ComponentType.APPLICATION,
                description="Handles subscription management",
                scope_category=ScopeCategory.CONNECTED,
                handles_pan=False,
                data_types=[],
            ),
            CDEComponent(
                name="API Gateway (Kong)",
                component_type=ComponentType.API_GATEWAY,
                description="Kong API Gateway",
                scope_category=ScopeCategory.SECURITY_IMPACTING,
                handles_pan=False,
                data_types=[],
            ),
            CDEComponent(
                name="PostgreSQL Database",
                component_type=ComponentType.DATABASE,
                description="Primary database (no PAN stored)",
                scope_category=ScopeCategory.OUT_OF_SCOPE,
                handles_pan=False,
                data_types=[],
            ),
            CDEComponent(
                name="Audit Logging (Loki)",
                component_type=ComponentType.LOGGING_SYSTEM,
                description="Centralized logging",
                scope_category=ScopeCategory.SECURITY_IMPACTING,
                handles_pan=False,
                data_types=[],
            ),
        ]

        for component in default_components:
            self.components[component.id] = component

    async def _discover_pan_handlers(self) -> List[CDEComponent]:
        """Auto-discover components that handle PAN.

        In production, this would scan configurations and code.
        """
        pan_handlers: List[CDEComponent] = []

        for component in self.components.values():
            # Check if component handles PAN based on configuration
            # In production, this would analyze actual configs
            if component.handles_pan:
                pan_handlers.append(component)

        return pan_handlers

    async def _discover_connections(self) -> List[DataFlowConnection]:
        """Auto-discover data flow connections.

        In production, this would analyze network configurations.
        """
        discovered: List[DataFlowConnection] = []

        # Add connections based on component types
        api_gateways = [
            c for c in self.components.values()
            if c.component_type == ComponentType.API_GATEWAY
        ]
        applications = [
            c for c in self.components.values()
            if c.component_type == ComponentType.APPLICATION
        ]
        databases = [
            c for c in self.components.values()
            if c.component_type == ComponentType.DATABASE
        ]

        # API Gateway -> Application connections
        for gateway in api_gateways:
            for app in applications:
                discovered.append(DataFlowConnection(
                    source_id=gateway.id,
                    destination_id=app.id,
                    direction=DataFlowDirection.INBOUND,
                    encrypted=True,
                    encryption_protocol="TLS 1.3",
                ))

        # Application -> Database connections
        for app in applications:
            for db in databases:
                discovered.append(DataFlowConnection(
                    source_id=app.id,
                    destination_id=db.id,
                    direction=DataFlowDirection.INTERNAL,
                    encrypted=True,
                    encryption_protocol="TLS 1.2",
                ))

        return discovered

    def _identify_entry_points(self) -> List[str]:
        """Identify external entry points to the CDE."""
        entry_points: List[str] = []

        for component in self.components.values():
            if component.component_type in (
                ComponentType.API_GATEWAY,
                ComponentType.THIRD_PARTY,
            ):
                entry_points.append(component.name)

        return entry_points

    def _identify_exit_points(self) -> List[str]:
        """Identify external exit points from the CDE."""
        exit_points: List[str] = []

        for component in self.components.values():
            if component.component_type == ComponentType.THIRD_PARTY:
                exit_points.append(component.name)

        return exit_points

    def _categorize_component(self, component: CDEComponent) -> ScopeCategory:
        """Categorize a component's PCI-DSS scope.

        Args:
            component: The component to categorize.

        Returns:
            The appropriate ScopeCategory.
        """
        # Direct PAN handlers are in CDE
        if component.handles_pan or CardDataType.PAN in component.data_types:
            return ScopeCategory.CDE

        # Components connected to CDE
        if self._is_connected_to_cde(component):
            return ScopeCategory.CONNECTED

        # Security controls that impact CDE
        if component.component_type in (
            ComponentType.NETWORK_DEVICE,
            ComponentType.API_GATEWAY,
            ComponentType.LOGGING_SYSTEM,
        ):
            return ScopeCategory.SECURITY_IMPACTING

        return ScopeCategory.OUT_OF_SCOPE

    def _is_connected_to_cde(self, component: CDEComponent) -> bool:
        """Check if component is connected to CDE.

        Args:
            component: The component to check.

        Returns:
            True if connected to CDE.
        """
        # Check connections to/from CDE components
        cde_ids = {
            c.id for c in self.components.values()
            if c.handles_pan
        }

        for conn in self.connections.values():
            if conn.source_id == component.id and conn.destination_id in cde_ids:
                return True
            if conn.destination_id == component.id and conn.source_id in cde_ids:
                return True

        return False

    def _generate_scope_recommendations(
        self,
        result: CDEScopeResult,
    ) -> List[str]:
        """Generate scope reduction recommendations."""
        recommendations: List[str] = []

        if result.cde_components > 0:
            recommendations.append(
                "Consider using tokenization to reduce PAN exposure"
            )

        if result.connected_components > 5:
            recommendations.append(
                "Implement network segmentation to reduce connected system count"
            )

        if result.security_impacting_components > 10:
            recommendations.append(
                "Review security-impacting systems for potential scope reduction"
            )

        return recommendations

    def _to_mermaid(self) -> str:
        """Convert diagram to Mermaid format."""
        if self.data_flow_diagram is None:
            return ""

        lines = ["flowchart TD"]

        # Add components
        for comp in self.data_flow_diagram.components:
            shape = "([{}])" if comp.handles_pan else "[{}]"
            lines.append(f"    {comp.id[:8]}{shape.format(comp.name)}")

        # Add connections
        for conn in self.data_flow_diagram.connections:
            arrow = "-->" if conn.encrypted else "-.->"
            lines.append(f"    {conn.source_id[:8]} {arrow} {conn.destination_id[:8]}")

        return "\n".join(lines)


__all__ = [
    "CardDataMapper",
    "CDEComponent",
    "DataFlowConnection",
    "DataFlowDiagram",
    "CDEScopeResult",
    "CardDataType",
    "ComponentType",
    "ScopeCategory",
    "DataFlowDirection",
]
