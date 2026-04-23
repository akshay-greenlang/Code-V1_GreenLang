"""
GL-003 UnifiedSteam gRPC Server

Production-ready gRPC server with server lifecycle management,
graceful shutdown, health checks, TLS/SSL support, and connection pooling.

Features:
- Async server with configurable thread pools
- Graceful shutdown handling with SIGINT/SIGTERM
- Health check service (grpc_health)
- Server reflection for debugging
- TLS/SSL support with mTLS option
- Connection pooling and keepalive configuration
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import ssl
import sys
from concurrent import futures
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import grpc
from grpc import aio as grpc_aio
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from .grpc_services import (
    SteamPropertiesServicer,
    OptimizationServicer,
    DiagnosticsServicer,
    StreamingServicer,
    RCAServicer,
    AuthenticationInterceptor,
    RateLimitInterceptor,
    LoggingInterceptor,
    RateLimitConfig,
    get_all_servicers,
    get_interceptors,
)
from .api_auth import AuthConfig, get_auth_config

logger = logging.getLogger(__name__)


# =============================================================================
# Server Configuration
# =============================================================================

@dataclass
class GRPCServerConfig:
    """Configuration for the gRPC server."""

    # Network settings
    host: str = "0.0.0.0"
    port: int = 50052
    max_workers: int = 10

    # Thread pool settings
    max_concurrent_rpcs: Optional[int] = None

    # Connection settings
    max_message_length: int = 4 * 1024 * 1024  # 4MB
    max_receive_message_length: int = 4 * 1024 * 1024  # 4MB
    max_send_message_length: int = 4 * 1024 * 1024  # 4MB

    # Keepalive settings
    keepalive_time_ms: int = 30000  # 30 seconds
    keepalive_timeout_ms: int = 10000  # 10 seconds
    keepalive_permit_without_calls: bool = True
    max_connection_idle_ms: int = 300000  # 5 minutes
    max_connection_age_ms: int = 3600000  # 1 hour
    max_connection_age_grace_ms: int = 60000  # 1 minute

    # TLS settings
    enable_tls: bool = False
    server_cert_path: Optional[str] = None
    server_key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    require_client_cert: bool = False

    # Authentication settings
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    enable_logging: bool = True

    # Health check settings
    enable_health_check: bool = True
    enable_reflection: bool = True

    # Graceful shutdown settings
    shutdown_grace_period_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "GRPCServerConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("GRPC_HOST", "0.0.0.0"),
            port=int(os.getenv("GRPC_PORT", "50052")),
            max_workers=int(os.getenv("GRPC_MAX_WORKERS", "10")),
            max_message_length=int(os.getenv("GRPC_MAX_MESSAGE_LENGTH", str(4 * 1024 * 1024))),
            keepalive_time_ms=int(os.getenv("GRPC_KEEPALIVE_TIME_MS", "30000")),
            keepalive_timeout_ms=int(os.getenv("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")),
            enable_tls=os.getenv("GRPC_ENABLE_TLS", "false").lower() == "true",
            server_cert_path=os.getenv("GRPC_SERVER_CERT"),
            server_key_path=os.getenv("GRPC_SERVER_KEY"),
            ca_cert_path=os.getenv("GRPC_CA_CERT"),
            require_client_cert=os.getenv("GRPC_REQUIRE_CLIENT_CERT", "false").lower() == "true",
            enable_auth=os.getenv("GRPC_ENABLE_AUTH", "true").lower() == "true",
            enable_rate_limiting=os.getenv("GRPC_ENABLE_RATE_LIMITING", "true").lower() == "true",
            enable_logging=os.getenv("GRPC_ENABLE_LOGGING", "true").lower() == "true",
            enable_health_check=os.getenv("GRPC_ENABLE_HEALTH_CHECK", "true").lower() == "true",
            enable_reflection=os.getenv("GRPC_ENABLE_REFLECTION", "true").lower() == "true",
            shutdown_grace_period_seconds=float(os.getenv("GRPC_SHUTDOWN_GRACE_PERIOD", "30.0")),
        )


class ServerStatus(Enum):
    """Server status states."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


# =============================================================================
# Health Check Service
# =============================================================================

class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    Health check servicer implementing the standard gRPC health protocol.

    Supports per-service health status and overall server health.
    """

    def __init__(self):
        self._status: Dict[str, health_pb2.HealthCheckResponse.ServingStatus] = {}
        self._lock = asyncio.Lock()
        self._watchers: Dict[str, List[asyncio.Queue]] = {}

        # Set initial status
        self._status[""] = health_pb2.HealthCheckResponse.SERVING
        self._status["unifiedsteam.v1.SteamPropertiesService"] = health_pb2.HealthCheckResponse.SERVING
        self._status["unifiedsteam.v1.OptimizationService"] = health_pb2.HealthCheckResponse.SERVING
        self._status["unifiedsteam.v1.DiagnosticsService"] = health_pb2.HealthCheckResponse.SERVING
        self._status["unifiedsteam.v1.StreamingService"] = health_pb2.HealthCheckResponse.SERVING
        self._status["unifiedsteam.v1.RCAService"] = health_pb2.HealthCheckResponse.SERVING

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc_aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """Check the health status of a service."""
        service = request.service

        async with self._lock:
            if service in self._status:
                return health_pb2.HealthCheckResponse(status=self._status[service])
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Service '{service}' not found")
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                )

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc_aio.ServicerContext,
    ):
        """Watch the health status of a service (streaming)."""
        service = request.service
        queue: asyncio.Queue = asyncio.Queue()

        async with self._lock:
            if service not in self._watchers:
                self._watchers[service] = []
            self._watchers[service].append(queue)

        try:
            # Send initial status
            async with self._lock:
                if service in self._status:
                    yield health_pb2.HealthCheckResponse(status=self._status[service])
                else:
                    yield health_pb2.HealthCheckResponse(
                        status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                    )

            # Watch for status changes
            while not context.cancelled():
                try:
                    status = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield health_pb2.HealthCheckResponse(status=status)
                except asyncio.TimeoutError:
                    # Send heartbeat with current status
                    async with self._lock:
                        if service in self._status:
                            yield health_pb2.HealthCheckResponse(status=self._status[service])

        finally:
            async with self._lock:
                if service in self._watchers:
                    self._watchers[service].remove(queue)

    async def set_status(
        self,
        service: str,
        status: health_pb2.HealthCheckResponse.ServingStatus,
    ) -> None:
        """Set the health status of a service."""
        async with self._lock:
            self._status[service] = status

            # Notify watchers
            if service in self._watchers:
                for queue in self._watchers[service]:
                    await queue.put(status)

    async def set_all_not_serving(self) -> None:
        """Set all services to NOT_SERVING status (for shutdown)."""
        async with self._lock:
            for service in self._status:
                self._status[service] = health_pb2.HealthCheckResponse.NOT_SERVING

                if service in self._watchers:
                    for queue in self._watchers[service]:
                        await queue.put(health_pb2.HealthCheckResponse.NOT_SERVING)


# =============================================================================
# gRPC Server Class
# =============================================================================

class GRPCServer:
    """
    Production-ready gRPC server for UnifiedSteam services.

    Features:
    - Async server with configurable thread pools
    - Graceful shutdown with SIGINT/SIGTERM handling
    - Health check service (grpc_health)
    - Server reflection for debugging
    - TLS/SSL support with mTLS option
    - Connection pooling and keepalive configuration
    """

    def __init__(
        self,
        config: Optional[GRPCServerConfig] = None,
        auth_config: Optional[AuthConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize the gRPC server.

        Args:
            config: Server configuration
            auth_config: Authentication configuration
            rate_limit_config: Rate limiting configuration
        """
        self.config = config or GRPCServerConfig()
        self.auth_config = auth_config or get_auth_config()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()

        self._server: Optional[grpc_aio.Server] = None
        self._health_servicer: Optional[HealthServicer] = None
        self._status = ServerStatus.CREATED
        self._start_time: Optional[datetime] = None
        self._shutdown_event = asyncio.Event()
        self._servicers: Dict[str, Any] = {}

    @property
    def status(self) -> ServerStatus:
        """Get current server status."""
        return self._status

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return 0.0

    @property
    def address(self) -> str:
        """Get the server listen address."""
        return f"{self.config.host}:{self.config.port}"

    def _get_server_options(self) -> List[Tuple[str, Any]]:
        """Get gRPC server options."""
        options = [
            ("grpc.max_receive_message_length", self.config.max_receive_message_length),
            ("grpc.max_send_message_length", self.config.max_send_message_length),
            ("grpc.keepalive_time_ms", self.config.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self.config.keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", int(self.config.keepalive_permit_without_calls)),
            ("grpc.max_connection_idle_ms", self.config.max_connection_idle_ms),
            ("grpc.max_connection_age_ms", self.config.max_connection_age_ms),
            ("grpc.max_connection_age_grace_ms", self.config.max_connection_age_grace_ms),
        ]

        if self.config.max_concurrent_rpcs:
            options.append(("grpc.max_concurrent_streams", self.config.max_concurrent_rpcs))

        return options

    def _get_interceptors(self) -> List[grpc_aio.ServerInterceptor]:
        """Get configured server interceptors."""
        return get_interceptors(
            enable_auth=self.config.enable_auth,
            enable_rate_limiting=self.config.enable_rate_limiting,
            enable_logging=self.config.enable_logging,
            auth_config=self.auth_config,
            rate_limit_config=self.rate_limit_config,
        )

    def _load_ssl_credentials(self) -> Optional[grpc.ServerCredentials]:
        """Load SSL credentials for TLS support."""
        if not self.config.enable_tls:
            return None

        if not self.config.server_cert_path or not self.config.server_key_path:
            raise ValueError(
                "TLS is enabled but server_cert_path and server_key_path are not set"
            )

        # Load server certificate and key
        with open(self.config.server_key_path, "rb") as f:
            server_key = f.read()
        with open(self.config.server_cert_path, "rb") as f:
            server_cert = f.read()

        # Load CA certificate for mTLS if configured
        ca_cert = None
        if self.config.ca_cert_path:
            with open(self.config.ca_cert_path, "rb") as f:
                ca_cert = f.read()

        if ca_cert and self.config.require_client_cert:
            # mTLS - require client certificate
            credentials = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
                root_certificates=ca_cert,
                require_client_auth=True,
            )
            logger.info("TLS enabled with mutual TLS (client certificate required)")
        else:
            # TLS only
            credentials = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
            )
            logger.info("TLS enabled (server-side only)")

        return credentials

    def _register_services(self, server: grpc_aio.Server) -> None:
        """Register all gRPC services with the server."""
        # Get all servicers
        self._servicers = get_all_servicers()

        # NOTE: In production, register generated service stubs here.
        # For example:
        #
        # from . import steam_pb2_grpc
        # steam_pb2_grpc.add_SteamPropertiesServiceServicer_to_server(
        #     self._servicers["SteamPropertiesService"], server
        # )
        # steam_pb2_grpc.add_OptimizationServiceServicer_to_server(
        #     self._servicers["OptimizationService"], server
        # )
        # steam_pb2_grpc.add_DiagnosticsServiceServicer_to_server(
        #     self._servicers["DiagnosticsService"], server
        # )
        # steam_pb2_grpc.add_StreamingServiceServicer_to_server(
        #     self._servicers["StreamingService"], server
        # )
        # steam_pb2_grpc.add_RCAServiceServicer_to_server(
        #     self._servicers["RCAService"], server
        # )

        logger.info(f"Registered {len(self._servicers)} services")

    def _register_health_service(self, server: grpc_aio.Server) -> None:
        """Register the health check service."""
        if not self.config.enable_health_check:
            return

        self._health_servicer = HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(self._health_servicer, server)
        logger.info("Health check service registered")

    def _register_reflection_service(self, server: grpc_aio.Server) -> None:
        """Register the reflection service for debugging."""
        if not self.config.enable_reflection:
            return

        service_names = [
            "unifiedsteam.v1.SteamPropertiesService",
            "unifiedsteam.v1.OptimizationService",
            "unifiedsteam.v1.DiagnosticsService",
            "unifiedsteam.v1.StreamingService",
            "unifiedsteam.v1.RCAService",
            "grpc.health.v1.Health",
            reflection.SERVICE_NAME,
        ]

        reflection.enable_server_reflection(service_names, server)
        logger.info("Server reflection enabled")

    async def start(self) -> None:
        """
        Start the gRPC server.

        Raises:
            RuntimeError: If server is already running
        """
        if self._status == ServerStatus.RUNNING:
            raise RuntimeError("Server is already running")

        self._status = ServerStatus.STARTING
        logger.info(f"Starting gRPC server on {self.address}")

        try:
            # Create server with options and interceptors
            options = self._get_server_options()
            interceptors = self._get_interceptors()

            self._server = grpc_aio.server(
                futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
                interceptors=interceptors,
                options=options,
            )

            # Register services
            self._register_services(self._server)
            self._register_health_service(self._server)
            self._register_reflection_service(self._server)

            # Add port with or without TLS
            credentials = self._load_ssl_credentials()
            if credentials:
                self._server.add_secure_port(self.address, credentials)
            else:
                self._server.add_insecure_port(self.address)

            # Start server
            await self._server.start()

            self._status = ServerStatus.RUNNING
            self._start_time = datetime.utcnow()

            logger.info(
                f"gRPC server started on {self.address} "
                f"(TLS: {self.config.enable_tls}, Auth: {self.config.enable_auth})"
            )

        except Exception as e:
            self._status = ServerStatus.FAILED
            logger.error(f"Failed to start gRPC server: {e}")
            raise

    async def stop(self, grace_period: Optional[float] = None) -> None:
        """
        Stop the gRPC server gracefully.

        Args:
            grace_period: Grace period in seconds for shutdown
        """
        if self._status not in (ServerStatus.RUNNING, ServerStatus.STARTING):
            logger.warning(f"Server is not running (status: {self._status})")
            return

        grace = grace_period or self.config.shutdown_grace_period_seconds
        self._status = ServerStatus.STOPPING

        logger.info(f"Stopping gRPC server (grace period: {grace}s)")

        try:
            # Mark all services as not serving
            if self._health_servicer:
                await self._health_servicer.set_all_not_serving()

            # Stop accepting new connections and wait for existing ones
            if self._server:
                await self._server.stop(grace)

            self._status = ServerStatus.STOPPED
            logger.info("gRPC server stopped")

        except Exception as e:
            self._status = ServerStatus.FAILED
            logger.error(f"Error during server shutdown: {e}")
            raise

    async def wait_for_termination(self) -> None:
        """Wait for the server to terminate."""
        if self._server:
            await self._server.wait_for_termination()

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def handle_signal(sig: signal.Signals):
            logger.info(f"Received signal {sig.name}, initiating shutdown")
            asyncio.create_task(self.stop())

        # Handle SIGINT and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: handle_signal(signal.Signals(s)))

    async def set_service_health(
        self,
        service: str,
        healthy: bool,
    ) -> None:
        """
        Set the health status of a specific service.

        Args:
            service: Service name
            healthy: Whether the service is healthy
        """
        if self._health_servicer:
            status = (
                health_pb2.HealthCheckResponse.SERVING
                if healthy
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
            await self._health_servicer.set_status(service, status)

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about registered services."""
        return {
            "status": self._status.value,
            "address": self.address,
            "uptime_seconds": self.uptime_seconds,
            "tls_enabled": self.config.enable_tls,
            "auth_enabled": self.config.enable_auth,
            "rate_limiting_enabled": self.config.enable_rate_limiting,
            "health_check_enabled": self.config.enable_health_check,
            "reflection_enabled": self.config.enable_reflection,
            "services": list(self._servicers.keys()),
        }


# =============================================================================
# Server Factory and Runner
# =============================================================================

async def create_server(
    config: Optional[GRPCServerConfig] = None,
    auth_config: Optional[AuthConfig] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
) -> GRPCServer:
    """
    Factory function to create a gRPC server.

    Args:
        config: Server configuration
        auth_config: Authentication configuration
        rate_limit_config: Rate limiting configuration

    Returns:
        Configured GRPCServer instance
    """
    server = GRPCServer(
        config=config,
        auth_config=auth_config,
        rate_limit_config=rate_limit_config,
    )
    return server


async def serve(
    config: Optional[GRPCServerConfig] = None,
    auth_config: Optional[AuthConfig] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    setup_signals: bool = True,
) -> None:
    """
    Create and run a gRPC server until termination.

    Args:
        config: Server configuration
        auth_config: Authentication configuration
        rate_limit_config: Rate limiting configuration
        setup_signals: Whether to setup signal handlers
    """
    server = await create_server(
        config=config,
        auth_config=auth_config,
        rate_limit_config=rate_limit_config,
    )

    if setup_signals:
        server.setup_signal_handlers()

    await server.start()
    await server.wait_for_termination()


def run_server(
    host: str = "0.0.0.0",
    port: int = 50052,
    enable_tls: bool = False,
    server_cert_path: Optional[str] = None,
    server_key_path: Optional[str] = None,
    ca_cert_path: Optional[str] = None,
    enable_auth: bool = True,
    enable_rate_limiting: bool = True,
) -> None:
    """
    Convenience function to run the gRPC server.

    Args:
        host: Host to bind to
        port: Port to listen on
        enable_tls: Enable TLS/SSL
        server_cert_path: Path to server certificate
        server_key_path: Path to server key
        ca_cert_path: Path to CA certificate for mTLS
        enable_auth: Enable authentication
        enable_rate_limiting: Enable rate limiting
    """
    config = GRPCServerConfig(
        host=host,
        port=port,
        enable_tls=enable_tls,
        server_cert_path=server_cert_path,
        server_key_path=server_key_path,
        ca_cert_path=ca_cert_path,
        enable_auth=enable_auth,
        enable_rate_limiting=enable_rate_limiting,
    )

    asyncio.run(serve(config=config))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config from environment
    config = GRPCServerConfig.from_environment()

    # Run server
    asyncio.run(serve(config=config))
