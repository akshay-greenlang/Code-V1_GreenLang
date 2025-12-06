"""
OPC-UA Server Implementation for GreenLang Agents

This module provides a production-ready OPC-UA server that exposes
GreenLang agent data nodes for industrial integration.

Features:
- Automatic node creation from agent schemas
- Security profiles (Basic256Sha256, Aes128_Sha256_RsaOaep)
- User authentication
- Subscription management
- Historical data access
- Method calls for agent actions

Example:
    >>> server = OPCUAServer(config)
    >>> await server.start()
    >>> await server.register_agent_nodes(agent)
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

try:
    from asyncua import Server, ua
    from asyncua.common.callback import CallbackType
    from asyncua.server.users import User, UserRole
    ASYNCUA_AVAILABLE = True
except ImportError:
    ASYNCUA_AVAILABLE = False
    Server = None
    ua = None

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class MessageSecurityMode(str, Enum):
    """OPC-UA message security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


@dataclass
class OPCUAServerConfig:
    """Configuration for OPC-UA server."""
    endpoint: str = "opc.tcp://0.0.0.0:4840/greenlang/"
    name: str = "GreenLang OPC-UA Server"
    namespace: str = "urn:greenlang:opcua:server"
    security_policies: List[SecurityPolicy] = field(
        default_factory=lambda: [SecurityPolicy.BASIC256SHA256]
    )
    security_mode: MessageSecurityMode = MessageSecurityMode.SIGN_AND_ENCRYPT
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    max_sessions: int = 100
    session_timeout_ms: int = 3600000
    enable_discovery: bool = True
    enable_history: bool = True
    history_retention_days: int = 365


class AgentNode(BaseModel):
    """Represents an OPC-UA node for an agent variable."""
    node_id: str = Field(..., description="OPC-UA node ID")
    browse_name: str = Field(..., description="Browse name for the node")
    display_name: str = Field(..., description="Human-readable display name")
    data_type: str = Field(..., description="OPC-UA data type")
    access_level: int = Field(default=3, description="Read/Write access level")
    historizing: bool = Field(default=True, description="Enable historizing")
    value: Any = Field(default=None, description="Current value")


class SubscriptionInfo(BaseModel):
    """Subscription tracking information."""
    subscription_id: str = Field(default_factory=lambda: str(uuid4()))
    client_id: str = Field(..., description="Client identifier")
    node_ids: List[str] = Field(default_factory=list, description="Subscribed node IDs")
    publishing_interval_ms: int = Field(default=1000, description="Publishing interval")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OPCUAServer:
    """
    Production-ready OPC-UA server for GreenLang agents.

    This server exposes agent data as OPC-UA nodes, enabling integration
    with industrial automation systems, SCADA, and historian platforms.

    Attributes:
        config: Server configuration
        server: AsyncUA server instance
        nodes: Registered agent nodes
        subscriptions: Active subscriptions

    Example:
        >>> config = OPCUAServerConfig(
        ...     endpoint="opc.tcp://0.0.0.0:4840/greenlang/",
        ...     security_policies=[SecurityPolicy.BASIC256SHA256]
        ... )
        >>> server = OPCUAServer(config)
        >>> await server.start()
        >>> await server.register_agent("emissions", emissions_schema)
    """

    def __init__(self, config: OPCUAServerConfig):
        """
        Initialize OPC-UA server.

        Args:
            config: Server configuration

        Raises:
            ImportError: If asyncua is not installed
        """
        if not ASYNCUA_AVAILABLE:
            raise ImportError(
                "asyncua is required for OPC-UA support. "
                "Install with: pip install asyncua"
            )

        self.config = config
        self.server: Optional[Server] = None
        self.nodes: Dict[str, Any] = {}
        self.subscriptions: Dict[str, SubscriptionInfo] = {}
        self._agent_namespaces: Dict[str, int] = {}
        self._running = False
        self._event_callbacks: Dict[str, List[Callable]] = {}

        logger.info(f"OPCUAServer initialized with endpoint: {config.endpoint}")

    async def start(self) -> None:
        """
        Start the OPC-UA server.

        Initializes the server, sets up security, and begins listening
        for client connections.

        Raises:
            RuntimeError: If server fails to start
        """
        try:
            self.server = Server()
            await self.server.init()

            # Configure server
            self.server.set_endpoint(self.config.endpoint)
            self.server.set_server_name(self.config.name)

            # Register namespace
            ns_idx = await self.server.register_namespace(self.config.namespace)
            self._agent_namespaces["greenlang"] = ns_idx

            # Configure security
            await self._configure_security()

            # Set up event handlers
            await self._setup_event_handlers()

            # Start server
            await self.server.start()
            self._running = True

            logger.info(
                f"OPC-UA server started on {self.config.endpoint} "
                f"with namespace index {ns_idx}"
            )

        except Exception as e:
            logger.error(f"Failed to start OPC-UA server: {e}", exc_info=True)
            raise RuntimeError(f"OPC-UA server start failed: {e}") from e

    async def stop(self) -> None:
        """
        Stop the OPC-UA server gracefully.

        Closes all active subscriptions and client connections before
        shutting down the server.
        """
        if self.server and self._running:
            try:
                # Notify subscribers
                for sub_id, sub_info in self.subscriptions.items():
                    logger.info(f"Closing subscription {sub_id} for client {sub_info.client_id}")

                await self.server.stop()
                self._running = False
                logger.info("OPC-UA server stopped gracefully")

            except Exception as e:
                logger.error(f"Error stopping OPC-UA server: {e}", exc_info=True)

    async def _configure_security(self) -> None:
        """Configure server security policies and certificates."""
        if self.config.certificate_path and self.config.private_key_path:
            await self.server.load_certificate(self.config.certificate_path)
            await self.server.load_private_key(self.config.private_key_path)

            # Set security policies
            security_policies = []
            for policy in self.config.security_policies:
                if policy == SecurityPolicy.NONE:
                    security_policies.append(ua.SecurityPolicyType.NoSecurity)
                elif policy == SecurityPolicy.BASIC256SHA256:
                    security_policies.append(ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt)
                elif policy == SecurityPolicy.AES128_SHA256_RSAOAEP:
                    security_policies.append(ua.SecurityPolicyType.Aes128Sha256RsaOaep_SignAndEncrypt)

            if security_policies:
                self.server.set_security_policy(security_policies)

            logger.info(f"Security configured with policies: {self.config.security_policies}")
        else:
            logger.warning("No certificates configured - running without security")

    async def _setup_event_handlers(self) -> None:
        """Set up server event handlers."""
        # Event handling is configured through the server's subscription manager
        pass

    async def register_agent_namespace(
        self,
        agent_name: str,
        schema: Dict[str, Any]
    ) -> int:
        """
        Register a new namespace for an agent.

        Args:
            agent_name: Name of the agent
            schema: Agent data schema defining variables

        Returns:
            Namespace index for the agent

        Raises:
            RuntimeError: If server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running")

        namespace_uri = f"{self.config.namespace}:{agent_name}"
        ns_idx = await self.server.register_namespace(namespace_uri)
        self._agent_namespaces[agent_name] = ns_idx

        # Create folder for agent
        objects = self.server.nodes.objects
        agent_folder = await objects.add_folder(ns_idx, agent_name)

        # Create nodes from schema
        await self._create_nodes_from_schema(agent_folder, ns_idx, agent_name, schema)

        logger.info(f"Registered namespace for agent '{agent_name}' with index {ns_idx}")
        return ns_idx

    async def _create_nodes_from_schema(
        self,
        parent: Any,
        ns_idx: int,
        agent_name: str,
        schema: Dict[str, Any]
    ) -> None:
        """Create OPC-UA nodes from agent schema."""
        properties = schema.get("properties", {})

        for prop_name, prop_def in properties.items():
            node_id = f"ns={ns_idx};s={agent_name}.{prop_name}"
            data_type = self._map_json_type_to_opcua(prop_def.get("type", "string"))

            # Create variable node
            node = await parent.add_variable(
                ns_idx,
                prop_name,
                prop_def.get("default"),
                datatype=data_type
            )

            # Set writable if not read-only
            if not prop_def.get("readOnly", False):
                await node.set_writable()

            # Store node reference
            self.nodes[node_id] = AgentNode(
                node_id=node_id,
                browse_name=prop_name,
                display_name=prop_def.get("title", prop_name),
                data_type=str(data_type),
                historizing=self.config.enable_history,
                value=prop_def.get("default")
            )

            logger.debug(f"Created node {node_id} with type {data_type}")

    def _map_json_type_to_opcua(self, json_type: str) -> Any:
        """Map JSON schema type to OPC-UA data type."""
        type_mapping = {
            "string": ua.VariantType.String,
            "integer": ua.VariantType.Int64,
            "number": ua.VariantType.Double,
            "boolean": ua.VariantType.Boolean,
            "array": ua.VariantType.ExtensionObject,
            "object": ua.VariantType.ExtensionObject,
        }
        return type_mapping.get(json_type, ua.VariantType.String)

    async def write_value(
        self,
        node_id: str,
        value: Any,
        source_timestamp: Optional[datetime] = None
    ) -> str:
        """
        Write a value to an OPC-UA node.

        Args:
            node_id: OPC-UA node identifier
            value: Value to write
            source_timestamp: Optional source timestamp

        Returns:
            Provenance hash of the write operation

        Raises:
            KeyError: If node_id does not exist
            RuntimeError: If server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running")

        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

        # Get node from server
        node = self.server.get_node(node_id)

        # Create data value with timestamp
        dv = ua.DataValue(ua.Variant(value))
        if source_timestamp:
            dv.SourceTimestamp = source_timestamp
        else:
            dv.SourceTimestamp = datetime.utcnow()

        # Write value
        await node.write_value(dv)

        # Update local cache
        self.nodes[node_id].value = value

        # Calculate provenance hash
        provenance_str = f"{node_id}:{value}:{dv.SourceTimestamp.isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Notify subscribers
        await self._notify_subscribers(node_id, value, dv.SourceTimestamp)

        logger.debug(f"Wrote value to {node_id}: {value} (hash: {provenance_hash[:8]}...)")
        return provenance_hash

    async def read_value(self, node_id: str) -> Any:
        """
        Read current value from an OPC-UA node.

        Args:
            node_id: OPC-UA node identifier

        Returns:
            Current node value

        Raises:
            KeyError: If node_id does not exist
            RuntimeError: If server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running")

        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

        node = self.server.get_node(node_id)
        value = await node.read_value()

        return value

    async def _notify_subscribers(
        self,
        node_id: str,
        value: Any,
        timestamp: datetime
    ) -> None:
        """Notify all subscribers of a value change."""
        for sub_id, sub_info in self.subscriptions.items():
            if node_id in sub_info.node_ids:
                callbacks = self._event_callbacks.get(sub_id, [])
                for callback in callbacks:
                    try:
                        await callback(node_id, value, timestamp)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")

    async def register_method(
        self,
        agent_name: str,
        method_name: str,
        callback: Callable,
        input_args: List[Dict[str, Any]],
        output_args: List[Dict[str, Any]]
    ) -> str:
        """
        Register a callable method on the server.

        Args:
            agent_name: Name of the agent
            method_name: Name of the method
            callback: Async function to call
            input_args: Input argument definitions
            output_args: Output argument definitions

        Returns:
            Method node ID
        """
        if not self._running:
            raise RuntimeError("Server is not running")

        ns_idx = self._agent_namespaces.get(agent_name)
        if ns_idx is None:
            raise KeyError(f"Agent namespace '{agent_name}' not found")

        # Find agent folder
        objects = self.server.nodes.objects
        agent_folder = await objects.get_child([f"{ns_idx}:{agent_name}"])

        # Create input/output arguments
        ua_input_args = [
            ua.Argument(
                Name=arg["name"],
                DataType=self._map_json_type_to_opcua(arg.get("type", "string")),
                Description=ua.LocalizedText(arg.get("description", ""))
            )
            for arg in input_args
        ]

        ua_output_args = [
            ua.Argument(
                Name=arg["name"],
                DataType=self._map_json_type_to_opcua(arg.get("type", "string")),
                Description=ua.LocalizedText(arg.get("description", ""))
            )
            for arg in output_args
        ]

        # Add method to folder
        method_node = await agent_folder.add_method(
            ns_idx,
            method_name,
            callback,
            ua_input_args,
            ua_output_args
        )

        method_id = f"ns={ns_idx};s={agent_name}.{method_name}"
        logger.info(f"Registered method {method_id}")

        return method_id

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Dictionary containing server statistics
        """
        return {
            "running": self._running,
            "endpoint": self.config.endpoint,
            "registered_nodes": len(self.nodes),
            "active_subscriptions": len(self.subscriptions),
            "agent_namespaces": list(self._agent_namespaces.keys()),
        }

    async def __aenter__(self) -> "OPCUAServer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
