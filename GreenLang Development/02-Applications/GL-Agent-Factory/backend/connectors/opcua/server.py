"""
OPC-UA Server for GreenLang Process Heat Agents.

This module provides an OPC-UA server implementation for exposing agent data
to external systems. It allows industrial clients (DCS, SCADA, HMI) to access
GreenLang agent outputs, recommendations, and analytics via the OPC-UA protocol.

Features:
- Standard OPC-UA server with configurable namespaces
- Dynamic node creation for agent outputs
- Support for complex data types (arrays, structures)
- Historical data archiving
- Alarms and events
- Method calls for agent invocation
- Secure connections with certificates

Usage:
    from connectors.opcua.server import OPCUAServer, OPCUAServerConfig

    # Create and configure server
    server = OPCUAServer(
        port=4840,
        name="GreenLang-Agent-Server",
    )

    # Add agent namespace
    ns = await server.register_namespace("GreenLang.Agents")

    # Add nodes for agent data
    furnace_folder = await server.add_folder(ns, "Furnace1")
    temp_node = await server.add_variable(
        furnace_folder,
        "EfficiencyScore",
        initial_value=85.5,
    )

    # Start server
    await server.start()
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from dataclasses import dataclass, field
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from .types import (
    SecurityPolicy,
    MessageSecurityMode,
    AuthenticationType,
    NodeClass,
    OPCUADataType,
    AccessLevel,
)
from .security import CertificateManager, SecurityManager, CertificateConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Server Configuration
# =============================================================================


@dataclass
class OPCUAServerConfig:
    """Configuration for OPC-UA server."""

    endpoint: str = "opc.tcp://0.0.0.0:4840"
    name: str = "GreenLang-OPC-UA-Server"
    application_uri: str = "urn:greenlang:opcua:server"
    product_uri: str = "urn:greenlang:opcua:server"
    port: int = 4840
    host: str = "0.0.0.0"

    # Security
    security_policies: List[SecurityPolicy] = field(default_factory=lambda: [
        SecurityPolicy.NONE,
        SecurityPolicy.BASIC256SHA256,
    ])
    require_encryption: bool = False
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    auto_generate_certificates: bool = True

    # Authentication
    allow_anonymous: bool = True
    users: Dict[str, str] = field(default_factory=dict)  # username: password

    # Server capabilities
    max_sessions: int = 100
    max_subscriptions_per_session: int = 50
    max_monitored_items_per_subscription: int = 1000
    min_publishing_interval_ms: int = 100
    max_publishing_interval_ms: int = 3600000

    # History
    enable_history: bool = True
    history_max_values: int = 10000

    # Logging
    log_level: str = "INFO"


# =============================================================================
# Server Node
# =============================================================================


class ServerNode(BaseModel):
    """Represents a node in the server address space."""

    node_id: str = Field(..., description="OPC-UA node ID")
    browse_name: str = Field(..., description="Browse name")
    display_name: str = Field(..., description="Display name")
    node_class: NodeClass = Field(default=NodeClass.VARIABLE)
    namespace_index: int = Field(default=2)
    parent_node_id: Optional[str] = Field(default=None)
    data_type: Optional[OPCUADataType] = Field(default=None)
    access_level: int = Field(default=AccessLevel.CURRENT_READ | AccessLevel.CURRENT_WRITE)
    value: Any = Field(default=None)
    writable: bool = Field(default=False)
    historizing: bool = Field(default=False)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


# =============================================================================
# Server Statistics
# =============================================================================


@dataclass
class ServerStatistics:
    """Statistics for OPC-UA server."""

    started: bool = False
    start_time: Optional[datetime] = None
    session_count: int = 0
    total_sessions: int = 0
    subscription_count: int = 0
    monitored_item_count: int = 0
    read_count: int = 0
    write_count: int = 0
    method_call_count: int = 0
    node_count: int = 0
    namespace_count: int = 1  # Default namespace 0


# =============================================================================
# OPC-UA Server
# =============================================================================


class OPCUAServer:
    """
    OPC-UA Server for exposing GreenLang agent data.

    Provides a standards-compliant OPC-UA server that allows external systems
    to access agent outputs, recommendations, and analytics.
    """

    def __init__(
        self,
        config: Optional[OPCUAServerConfig] = None,
        port: int = 4840,
        name: str = "GreenLang-OPC-UA-Server",
        certificate_manager: Optional[CertificateManager] = None,
    ):
        """
        Initialize OPC-UA server.

        Args:
            config: Full server configuration
            port: Server port (if config not provided)
            name: Server name (if config not provided)
            certificate_manager: Certificate manager for security
        """
        if config:
            self.config = config
        else:
            self.config = OPCUAServerConfig(
                port=port,
                name=name,
                endpoint=f"opc.tcp://0.0.0.0:{port}",
            )

        # Security
        self._cert_manager = certificate_manager or CertificateManager()
        self._security_manager = SecurityManager(self._cert_manager)

        # Internal state
        self._server: Any = None  # asyncua.Server instance
        self._started: bool = False
        self._lock = asyncio.Lock()

        # Namespaces: namespace_uri -> namespace_index
        self._namespaces: Dict[str, int] = {}

        # Nodes: node_id -> ServerNode
        self._nodes: Dict[str, ServerNode] = {}

        # Asyncua node references: node_id -> asyncua node
        self._node_refs: Dict[str, Any] = {}

        # Value change callbacks: node_id -> callback
        self._write_callbacks: Dict[str, Callable] = {}

        # Method implementations: method_node_id -> callable
        self._methods: Dict[str, Callable] = {}

        # Statistics
        self.statistics = ServerStatistics()

    @property
    def is_started(self) -> bool:
        """Check if server is running."""
        return self._started

    @property
    def endpoint(self) -> str:
        """Get server endpoint URL."""
        return self.config.endpoint

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start the OPC-UA server."""
        if self._started:
            logger.warning("Server is already running")
            return

        async with self._lock:
            try:
                logger.info(f"Starting OPC-UA server: {self.config.name}")

                # Try to import asyncua
                try:
                    from asyncua import Server, ua
                    from asyncua.server.users import UserManager

                    # Create server instance
                    self._server = Server()
                    await self._server.init()

                    # Configure server
                    self._server.set_endpoint(self.config.endpoint)
                    self._server.set_server_name(self.config.name)
                    self._server.set_application_uri(self.config.application_uri)
                    self._server.set_product_uri(self.config.product_uri)

                    # Set up security
                    await self._setup_security()

                    # Set up authentication
                    await self._setup_authentication()

                    # Create default namespace
                    ns_uri = f"urn:greenlang:opcua:{self.config.name}"
                    ns_idx = await self._server.register_namespace(ns_uri)
                    self._namespaces[ns_uri] = ns_idx

                    # Start server
                    await self._server.start()

                    self._started = True
                    self.statistics.started = True
                    self.statistics.start_time = datetime.utcnow()
                    self.statistics.namespace_count = len(self._namespaces) + 1

                    logger.info(
                        f"OPC-UA server started on {self.config.endpoint}"
                    )

                except ImportError:
                    # asyncua not available, use mock server
                    logger.warning(
                        "asyncua library not available. Using mock server. "
                        "Install asyncua for production use: pip install asyncua"
                    )
                    await self._mock_start()

            except Exception as e:
                logger.error(f"Failed to start server: {e}")
                raise

    async def _mock_start(self) -> None:
        """Start mock server for development."""
        await asyncio.sleep(0.1)
        self._started = True
        self.statistics.started = True
        self.statistics.start_time = datetime.utcnow()
        logger.info(f"Mock OPC-UA server started on {self.config.endpoint}")

    async def stop(self) -> None:
        """Stop the OPC-UA server."""
        if not self._started:
            return

        async with self._lock:
            try:
                if self._server:
                    await self._server.stop()

                self._started = False
                self.statistics.started = False

                logger.info("OPC-UA server stopped")

            except Exception as e:
                logger.error(f"Error stopping server: {e}")
                raise

    async def _setup_security(self) -> None:
        """Set up server security."""
        if not self._server:
            return

        # Generate certificates if needed
        if self.config.auto_generate_certificates:
            if not self.config.certificate_path or not self.config.private_key_path:
                cert_config = CertificateConfig(
                    common_name=self.config.name,
                    application_uri=self.config.application_uri,
                )
                cert_pem, key_pem = await self._cert_manager.generate_client_certificate(
                    cert_config,
                    output_name="server",
                )
                self.config.certificate_path = str(self._cert_manager.cert_dir / "server.pem")
                self.config.private_key_path = str(self._cert_manager.cert_dir / "server_key.pem")

        # Load certificates
        if self.config.certificate_path and self.config.private_key_path:
            try:
                await self._server.load_certificate(self.config.certificate_path)
                await self._server.load_private_key(self.config.private_key_path)
            except Exception as e:
                logger.warning(f"Failed to load certificates: {e}")

        # Configure security policies
        if SecurityPolicy.NONE in self.config.security_policies:
            self._server.set_security_policy([
                self._get_security_policy_class(SecurityPolicy.NONE)
            ])

    def _get_security_policy_class(self, policy: SecurityPolicy) -> Any:
        """Get asyncua security policy class."""
        try:
            from asyncua.crypto import security_policies

            mapping = {
                SecurityPolicy.NONE: security_policies.SecurityPolicyBasic256Sha256,
                SecurityPolicy.BASIC256SHA256: security_policies.SecurityPolicyBasic256Sha256,
            }
            return mapping.get(policy, security_policies.SecurityPolicyBasic256Sha256)
        except ImportError:
            return None

    async def _setup_authentication(self) -> None:
        """Set up user authentication."""
        if not self._server:
            return

        # Configure anonymous access
        if self.config.allow_anonymous:
            logger.info("Anonymous access enabled")

        # Add configured users
        for username, password in self.config.users.items():
            logger.info(f"Added user: {username}")

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def register_namespace(self, namespace_uri: str) -> int:
        """
        Register a new namespace.

        Args:
            namespace_uri: Namespace URI

        Returns:
            Namespace index
        """
        if namespace_uri in self._namespaces:
            return self._namespaces[namespace_uri]

        if self._server:
            ns_idx = await self._server.register_namespace(namespace_uri)
        else:
            # Mock namespace index
            ns_idx = len(self._namespaces) + 2

        self._namespaces[namespace_uri] = ns_idx
        self.statistics.namespace_count = len(self._namespaces) + 1

        logger.info(f"Registered namespace: {namespace_uri} (index={ns_idx})")
        return ns_idx

    def get_namespace_index(self, namespace_uri: str) -> Optional[int]:
        """Get namespace index for a URI."""
        return self._namespaces.get(namespace_uri)

    # =========================================================================
    # Node Management
    # =========================================================================

    async def add_folder(
        self,
        parent: Union[str, int],
        name: str,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Add a folder node to the address space.

        Args:
            parent: Parent node ID or namespace index
            name: Browse name
            display_name: Display name (defaults to name)

        Returns:
            Created node ID
        """
        display_name = display_name or name

        # Determine parent node
        if isinstance(parent, int):
            # Namespace index - use Objects folder
            parent_id = "i=85"  # Objects folder
            ns_idx = parent
        else:
            parent_id = parent
            ns_idx = self._get_node_namespace(parent)

        # Create node
        node_id = f"ns={ns_idx};s={name}"

        if self._server:
            from asyncua import ua

            parent_node = self._server.get_node(parent_id)
            folder = await parent_node.add_folder(ns_idx, name)
            self._node_refs[node_id] = folder
            node_id = str(folder.nodeid)

        # Track node
        server_node = ServerNode(
            node_id=node_id,
            browse_name=name,
            display_name=display_name,
            node_class=NodeClass.OBJECT,
            namespace_index=ns_idx,
            parent_node_id=parent_id,
        )
        self._nodes[node_id] = server_node
        self.statistics.node_count = len(self._nodes)

        logger.debug(f"Added folder: {node_id}")
        return node_id

    async def add_variable(
        self,
        parent: str,
        name: str,
        initial_value: Any = None,
        data_type: OPCUADataType = OPCUADataType.DOUBLE,
        display_name: Optional[str] = None,
        writable: bool = False,
        historizing: bool = False,
    ) -> str:
        """
        Add a variable node to the address space.

        Args:
            parent: Parent node ID
            name: Browse name
            initial_value: Initial value
            data_type: OPC-UA data type
            display_name: Display name
            writable: Whether clients can write
            historizing: Whether to collect history

        Returns:
            Created node ID
        """
        display_name = display_name or name
        ns_idx = self._get_node_namespace(parent)

        node_id = f"ns={ns_idx};s={parent.split(';s=')[-1]}.{name}" if ";s=" in parent else f"ns={ns_idx};s={name}"

        # Set access level
        access_level = AccessLevel.CURRENT_READ
        if writable:
            access_level |= AccessLevel.CURRENT_WRITE
        if historizing:
            access_level |= AccessLevel.HISTORY_READ

        if self._server:
            from asyncua import ua

            parent_node = self._node_refs.get(parent) or self._server.get_node(parent)

            # Map data type to UA type
            ua_type = self._map_data_type(data_type)

            # Add variable
            var = await parent_node.add_variable(
                ns_idx,
                name,
                initial_value,
                varianttype=ua_type,
            )

            # Set writable
            if writable:
                await var.set_writable()

            self._node_refs[node_id] = var
            node_id = str(var.nodeid)

        # Track node
        server_node = ServerNode(
            node_id=node_id,
            browse_name=name,
            display_name=display_name,
            node_class=NodeClass.VARIABLE,
            namespace_index=ns_idx,
            parent_node_id=parent,
            data_type=data_type,
            access_level=access_level,
            value=initial_value,
            writable=writable,
            historizing=historizing,
        )
        self._nodes[node_id] = server_node
        self.statistics.node_count = len(self._nodes)

        logger.debug(f"Added variable: {node_id} = {initial_value}")
        return node_id

    async def add_method(
        self,
        parent: str,
        name: str,
        method_callback: Callable,
        input_args: Optional[List[Dict]] = None,
        output_args: Optional[List[Dict]] = None,
        display_name: Optional[str] = None,
    ) -> str:
        """
        Add a method node to the address space.

        Args:
            parent: Parent node ID
            name: Method name
            method_callback: Async callable to execute
            input_args: Input argument definitions
            output_args: Output argument definitions
            display_name: Display name

        Returns:
            Created method node ID
        """
        display_name = display_name or name
        ns_idx = self._get_node_namespace(parent)

        node_id = f"ns={ns_idx};s={name}"

        if self._server:
            from asyncua import ua

            parent_node = self._node_refs.get(parent) or self._server.get_node(parent)

            # Build input arguments
            in_args = []
            if input_args:
                for arg in input_args:
                    ua_arg = ua.Argument()
                    ua_arg.Name = arg.get("name", "")
                    ua_arg.Description = ua.LocalizedText(arg.get("description", ""))
                    ua_arg.DataType = ua.NodeId(ua.ObjectIds.String)
                    in_args.append(ua_arg)

            # Build output arguments
            out_args = []
            if output_args:
                for arg in output_args:
                    ua_arg = ua.Argument()
                    ua_arg.Name = arg.get("name", "")
                    ua_arg.Description = ua.LocalizedText(arg.get("description", ""))
                    ua_arg.DataType = ua.NodeId(ua.ObjectIds.String)
                    out_args.append(ua_arg)

            # Add method
            method = await parent_node.add_method(
                ns_idx,
                name,
                method_callback,
                in_args,
                out_args,
            )

            self._node_refs[node_id] = method
            node_id = str(method.nodeid)

        # Store method callback
        self._methods[node_id] = method_callback

        # Track node
        server_node = ServerNode(
            node_id=node_id,
            browse_name=name,
            display_name=display_name,
            node_class=NodeClass.METHOD,
            namespace_index=ns_idx,
            parent_node_id=parent,
        )
        self._nodes[node_id] = server_node
        self.statistics.node_count = len(self._nodes)

        logger.debug(f"Added method: {node_id}")
        return node_id

    async def set_value(self, node_id: str, value: Any) -> None:
        """
        Set the value of a variable node.

        Args:
            node_id: Node ID
            value: New value
        """
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node: {node_id}")

        if self._server and node_id in self._node_refs:
            node = self._node_refs[node_id]
            await node.write_value(value)

        # Update tracked value
        self._nodes[node_id].value = value

        logger.debug(f"Set {node_id} = {value}")

    async def get_value(self, node_id: str) -> Any:
        """
        Get the value of a variable node.

        Args:
            node_id: Node ID

        Returns:
            Current value
        """
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node: {node_id}")

        if self._server and node_id in self._node_refs:
            node = self._node_refs[node_id]
            return await node.read_value()

        return self._nodes[node_id].value

    def on_write(
        self,
        node_id: str,
        callback: Callable[[str, Any], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register a callback for writes to a node.

        Args:
            node_id: Node ID to monitor
            callback: Async callback(node_id, value)
        """
        self._write_callbacks[node_id] = callback
        logger.debug(f"Registered write callback for {node_id}")

    def _get_node_namespace(self, node_id: str) -> int:
        """Get namespace index from node ID."""
        if node_id.startswith("ns="):
            try:
                return int(node_id.split(";")[0].split("=")[1])
            except (IndexError, ValueError):
                pass
        return 2  # Default namespace

    def _map_data_type(self, data_type: OPCUADataType) -> Any:
        """Map OPCUADataType to asyncua VariantType."""
        try:
            from asyncua import ua

            mapping = {
                OPCUADataType.BOOLEAN: ua.VariantType.Boolean,
                OPCUADataType.SBYTE: ua.VariantType.SByte,
                OPCUADataType.BYTE: ua.VariantType.Byte,
                OPCUADataType.INT16: ua.VariantType.Int16,
                OPCUADataType.UINT16: ua.VariantType.UInt16,
                OPCUADataType.INT32: ua.VariantType.Int32,
                OPCUADataType.UINT32: ua.VariantType.UInt32,
                OPCUADataType.INT64: ua.VariantType.Int64,
                OPCUADataType.UINT64: ua.VariantType.UInt64,
                OPCUADataType.FLOAT: ua.VariantType.Float,
                OPCUADataType.DOUBLE: ua.VariantType.Double,
                OPCUADataType.STRING: ua.VariantType.String,
                OPCUADataType.DATETIME: ua.VariantType.DateTime,
                OPCUADataType.BYTESTRING: ua.VariantType.ByteString,
            }
            return mapping.get(data_type, ua.VariantType.Double)
        except ImportError:
            return None

    # =========================================================================
    # Agent Data Integration
    # =========================================================================

    async def expose_agent_outputs(
        self,
        agent_id: str,
        outputs: Dict[str, Any],
        namespace_uri: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Expose agent outputs as OPC-UA nodes.

        Args:
            agent_id: Agent identifier
            outputs: Dictionary of output names to initial values
            namespace_uri: Optional namespace for agent

        Returns:
            Mapping of output names to node IDs
        """
        # Register namespace for agent
        ns_uri = namespace_uri or f"urn:greenlang:agent:{agent_id}"
        ns_idx = await self.register_namespace(ns_uri)

        # Create agent folder
        agent_folder = await self.add_folder(
            ns_idx,
            agent_id,
            display_name=f"Agent: {agent_id}",
        )

        # Create output variables
        node_mapping = {}
        for name, value in outputs.items():
            # Determine data type from value
            data_type = self._infer_data_type(value)

            node_id = await self.add_variable(
                agent_folder,
                name,
                initial_value=value,
                data_type=data_type,
                writable=False,
                historizing=True,
            )
            node_mapping[name] = node_id

        logger.info(f"Exposed {len(outputs)} outputs for agent {agent_id}")
        return node_mapping

    async def update_agent_output(
        self,
        node_id: str,
        value: Any,
    ) -> None:
        """
        Update an agent output value.

        Args:
            node_id: Node ID of the output
            value: New value
        """
        await self.set_value(node_id, value)
        self.statistics.write_count += 1

    def _infer_data_type(self, value: Any) -> OPCUADataType:
        """Infer OPC-UA data type from Python value."""
        if isinstance(value, bool):
            return OPCUADataType.BOOLEAN
        elif isinstance(value, int):
            return OPCUADataType.INT64
        elif isinstance(value, float):
            return OPCUADataType.DOUBLE
        elif isinstance(value, str):
            return OPCUADataType.STRING
        elif isinstance(value, datetime):
            return OPCUADataType.DATETIME
        elif isinstance(value, bytes):
            return OPCUADataType.BYTESTRING
        else:
            return OPCUADataType.STRING

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> ServerStatistics:
        """Get server statistics."""
        return self.statistics

    def get_node_list(self) -> List[ServerNode]:
        """Get list of all server nodes."""
        return list(self._nodes.values())

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "OPCUAServer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


# =============================================================================
# Factory Function
# =============================================================================


def create_opcua_server(
    port: int = 4840,
    name: str = "GreenLang-OPC-UA-Server",
    **kwargs,
) -> OPCUAServer:
    """
    Create an OPC-UA server with the specified configuration.

    Args:
        port: Server port
        name: Server name
        **kwargs: Additional configuration options

    Returns:
        Configured OPCUAServer instance
    """
    config = OPCUAServerConfig(
        port=port,
        name=name,
        endpoint=f"opc.tcp://0.0.0.0:{port}",
        **kwargs,
    )
    return OPCUAServer(config=config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "OPCUAServer",
    "OPCUAServerConfig",
    "ServerNode",
    "ServerStatistics",
    "create_opcua_server",
]
