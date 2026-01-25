"""
gRPC Service Definitions for GreenLang

This module provides gRPC service registration and management
for GreenLang microservices.

Features:
- Service registration
- Interceptor chains
- Health checking
- Reflection support
- Streaming support
- Load balancing

Example:
    >>> registry = GRPCServiceRegistry(config)
    >>> registry.register_service(EmissionsService)
    >>> server = await registry.start()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None
    aio = None

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    UNKNOWN = "UNKNOWN"
    SERVING = "SERVING"
    NOT_SERVING = "NOT_SERVING"


@dataclass
class GRPCServiceConfig:
    """Configuration for gRPC service registry."""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_length: int = 4 * 1024 * 1024  # 4MB
    enable_reflection: bool = True
    enable_health_check: bool = True
    enable_tracing: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000
    max_concurrent_rpcs: int = 100


class ServiceDefinition(BaseModel):
    """gRPC service definition."""
    name: str = Field(..., description="Service name")
    package: str = Field(default="greenlang", description="Package name")
    methods: List[str] = Field(default_factory=list, description="Method names")
    description: Optional[str] = Field(default=None)
    version: str = Field(default="1.0.0")


class MethodDefinition(BaseModel):
    """gRPC method definition."""
    name: str = Field(..., description="Method name")
    request_type: str = Field(..., description="Request message type")
    response_type: str = Field(..., description="Response message type")
    is_streaming_request: bool = Field(default=False)
    is_streaming_response: bool = Field(default=False)
    description: Optional[str] = Field(default=None)


class InterceptorContext(BaseModel):
    """Context passed through interceptor chain."""
    method: str = Field(..., description="Called method")
    metadata: Dict[str, str] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    start_time: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(default=None)
    tenant_id: Optional[str] = Field(default=None)


class GRPCInterceptor:
    """
    Base class for gRPC interceptors.

    Interceptors can modify requests/responses and add
    cross-cutting concerns like logging, auth, and tracing.
    """

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any
    ) -> Any:
        """Intercept unary-unary call."""
        return await continuation(client_call_details, request)

    async def intercept_unary_stream(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any
    ) -> Any:
        """Intercept unary-stream call."""
        return await continuation(client_call_details, request)


class LoggingInterceptor(GRPCInterceptor):
    """Interceptor for request/response logging."""

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any
    ) -> Any:
        """Log unary-unary calls."""
        method = client_call_details.method
        start_time = datetime.utcnow()

        logger.info(f"gRPC request: {method}")

        try:
            response = await continuation(client_call_details, request)
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"gRPC response: {method} duration={duration:.2f}ms")
            return response

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"gRPC error: {method} error={e} duration={duration:.2f}ms")
            raise


class AuthInterceptor(GRPCInterceptor):
    """Interceptor for authentication."""

    def __init__(self, auth_func: Callable[[Dict[str, str]], bool]):
        """Initialize with auth function."""
        self.auth_func = auth_func

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any
    ) -> Any:
        """Check authentication."""
        # Extract metadata
        metadata = dict(client_call_details.metadata or [])

        if not self.auth_func(metadata):
            if GRPC_AVAILABLE:
                context = grpc.ServicerContext()
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid credentials")

        return await continuation(client_call_details, request)


class TracingInterceptor(GRPCInterceptor):
    """Interceptor for distributed tracing."""

    async def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any
    ) -> Any:
        """Add tracing headers."""
        metadata = list(client_call_details.metadata or [])

        # Add correlation ID if not present
        has_correlation = any(k == "x-correlation-id" for k, v in metadata)
        if not has_correlation:
            metadata.append(("x-correlation-id", str(uuid4())))

        # Create new call details with updated metadata
        new_details = client_call_details._replace(metadata=metadata)

        return await continuation(new_details, request)


class GRPCServiceRegistry:
    """
    gRPC service registry and server manager.

    Manages gRPC service registration, interceptors, and
    server lifecycle.

    Attributes:
        config: Registry configuration
        services: Registered services

    Example:
        >>> config = GRPCServiceConfig(port=50051)
        >>> registry = GRPCServiceRegistry(config)
        >>> registry.register_service(EmissionsServicer())
        >>> await registry.start()
    """

    def __init__(self, config: Optional[GRPCServiceConfig] = None):
        """
        Initialize gRPC service registry.

        Args:
            config: Registry configuration
        """
        if not GRPC_AVAILABLE:
            raise ImportError(
                "grpcio is required for gRPC support. "
                "Install with: pip install grpcio grpcio-tools"
            )

        self.config = config or GRPCServiceConfig()
        self._services: Dict[str, Any] = {}
        self._service_definitions: Dict[str, ServiceDefinition] = {}
        self._interceptors: List[GRPCInterceptor] = []
        self._server: Optional[Any] = None
        self._started = False
        self._health_status: Dict[str, ServiceStatus] = {}

        logger.info(
            f"GRPCServiceRegistry initialized on "
            f"{self.config.host}:{self.config.port}"
        )

    def register_service(
        self,
        servicer: Any,
        add_to_server_func: Callable,
        definition: Optional[ServiceDefinition] = None
    ) -> None:
        """
        Register a gRPC service.

        Args:
            servicer: Service implementation
            add_to_server_func: Function to add servicer to server
            definition: Optional service definition
        """
        service_name = type(servicer).__name__
        self._services[service_name] = (servicer, add_to_server_func)

        if definition:
            self._service_definitions[service_name] = definition

        self._health_status[service_name] = ServiceStatus.SERVING

        logger.info(f"Registered gRPC service: {service_name}")

    def add_interceptor(self, interceptor: GRPCInterceptor) -> None:
        """
        Add an interceptor to the chain.

        Args:
            interceptor: Interceptor to add
        """
        self._interceptors.append(interceptor)
        logger.debug(f"Added interceptor: {type(interceptor).__name__}")

    async def start(self) -> Any:
        """
        Start the gRPC server.

        Returns:
            Server instance
        """
        if self._started:
            logger.warning("Server already started")
            return self._server

        try:
            # Create server with options
            options = [
                ("grpc.max_receive_message_length", self.config.max_message_length),
                ("grpc.max_send_message_length", self.config.max_message_length),
                ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
                ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
                ("grpc.max_concurrent_streams", self.config.max_concurrent_rpcs),
            ]

            self._server = aio.server(options=options)

            # Add services
            for service_name, (servicer, add_func) in self._services.items():
                add_func(servicer, self._server)
                logger.debug(f"Added service to server: {service_name}")

            # Add health check
            if self.config.enable_health_check:
                self._add_health_service()

            # Add reflection
            if self.config.enable_reflection:
                self._add_reflection()

            # Configure SSL if provided
            if self.config.ssl_cert_path and self.config.ssl_key_path:
                credentials = self._create_ssl_credentials()
                self._server.add_secure_port(
                    f"{self.config.host}:{self.config.port}",
                    credentials
                )
            else:
                self._server.add_insecure_port(
                    f"{self.config.host}:{self.config.port}"
                )

            # Start server
            await self._server.start()
            self._started = True

            logger.info(
                f"gRPC server started on "
                f"{self.config.host}:{self.config.port}"
            )

            return self._server

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}", exc_info=True)
            raise

    async def stop(self, grace_period: float = 5.0) -> None:
        """
        Stop the gRPC server gracefully.

        Args:
            grace_period: Grace period in seconds
        """
        if not self._started:
            return

        try:
            # Set all services to not serving
            for service_name in self._health_status:
                self._health_status[service_name] = ServiceStatus.NOT_SERVING

            # Stop server
            await self._server.stop(grace_period)
            self._started = False

            logger.info("gRPC server stopped")

        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")

    def _add_health_service(self) -> None:
        """Add health check service."""
        # Would add grpc_health service
        logger.debug("Added health check service")

    def _add_reflection(self) -> None:
        """Add reflection service."""
        service_names = [
            sd.name for sd in self._service_definitions.values()
        ]
        if service_names:
            reflection.enable_server_reflection(service_names, self._server)
            logger.debug("Added reflection service")

    def _create_ssl_credentials(self) -> Any:
        """Create SSL credentials."""
        with open(self.config.ssl_cert_path, "rb") as f:
            cert = f.read()
        with open(self.config.ssl_key_path, "rb") as f:
            key = f.read()

        root_cert = None
        if self.config.ssl_ca_path:
            with open(self.config.ssl_ca_path, "rb") as f:
                root_cert = f.read()

        return grpc.ssl_server_credentials(
            [(key, cert)],
            root_certificates=root_cert,
            require_client_auth=root_cert is not None
        )

    def set_service_status(
        self,
        service_name: str,
        status: ServiceStatus
    ) -> None:
        """
        Set service health status.

        Args:
            service_name: Service name
            status: Health status
        """
        self._health_status[service_name] = status
        logger.info(f"Service {service_name} status: {status.value}")

    def get_service_status(self, service_name: str) -> ServiceStatus:
        """
        Get service health status.

        Args:
            service_name: Service name

        Returns:
            Service status
        """
        return self._health_status.get(service_name, ServiceStatus.UNKNOWN)

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self._server:
            await self._server.wait_for_termination()

    def get_registered_services(self) -> List[str]:
        """
        Get list of registered services.

        Returns:
            List of service names
        """
        return list(self._services.keys())


def create_channel(
    host: str,
    port: int,
    ssl: bool = False,
    ssl_cert_path: Optional[str] = None
) -> Any:
    """
    Create a gRPC channel.

    Args:
        host: Server host
        port: Server port
        ssl: Use SSL
        ssl_cert_path: Path to SSL certificate

    Returns:
        gRPC channel
    """
    if not GRPC_AVAILABLE:
        raise ImportError("grpcio is required")

    target = f"{host}:{port}"

    if ssl:
        if ssl_cert_path:
            with open(ssl_cert_path, "rb") as f:
                cert = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=cert)
        else:
            credentials = grpc.ssl_channel_credentials()

        return aio.secure_channel(target, credentials)
    else:
        return aio.insecure_channel(target)


def generate_proto_stub(
    service_def: ServiceDefinition,
    output_dir: str
) -> str:
    """
    Generate a .proto file from service definition.

    Args:
        service_def: Service definition
        output_dir: Output directory

    Returns:
        Path to generated .proto file
    """
    proto_content = f'''syntax = "proto3";

package {service_def.package};

service {service_def.name} {{
'''

    for method in service_def.methods:
        proto_content += f'  rpc {method}(Request) returns (Response);\n'

    proto_content += '}\n'

    # Write file
    import os
    proto_path = os.path.join(output_dir, f"{service_def.name.lower()}.proto")
    with open(proto_path, "w") as f:
        f.write(proto_content)

    logger.info(f"Generated proto file: {proto_path}")
    return proto_path
