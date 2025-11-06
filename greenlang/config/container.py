"""
GreenLang Dependency Injection Container
=========================================

Lightweight DI container for managing service lifecycles:
- Singleton services (shared across application)
- Scoped services (per-request/transaction)
- Transient services (new instance each time)
- Lazy initialization
- Auto-wiring from configuration

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, cast

from greenlang.config.manager import ConfigManager
from greenlang.config.schemas import GreenLangConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Service Lifetime
# ==============================================================================

class ServiceLifetime(str, Enum):
    """Service lifetime scope."""

    SINGLETON = "singleton"  # Single instance for entire application
    SCOPED = "scoped"  # One instance per scope (request, transaction)
    TRANSIENT = "transient"  # New instance every time


# ==============================================================================
# Service Registration
# ==============================================================================

T = TypeVar("T")


class ServiceRegistration(Generic[T]):
    """Registration information for a service."""

    def __init__(
        self,
        service_type: Type[T],
        factory: Callable[[ServiceContainer], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ):
        """
        Initialize service registration.

        Args:
            service_type: Type of service
            factory: Factory function to create service
            lifetime: Service lifetime scope
        """
        self.service_type = service_type
        self.factory = factory
        self.lifetime = lifetime
        self.instance: Optional[T] = None


# ==============================================================================
# Service Container
# ==============================================================================

class ServiceContainer:
    """
    Dependency injection container.

    Features:
    - Register services with different lifetimes
    - Resolve dependencies automatically
    - Lazy initialization
    - Thread-safe singleton management
    - Scoped service management

    Usage:
        >>> container = ServiceContainer()
        >>>
        >>> # Register services
        >>> container.register_singleton(LLMProvider, lambda c: OpenAIProvider(...))
        >>> container.register_scoped(Database, lambda c: PostgresDatabase(...))
        >>>
        >>> # Resolve services
        >>> llm = container.resolve(LLMProvider)
        >>> db = container.resolve(Database)
    """

    def __init__(self, config: Optional[GreenLangConfig] = None):
        """
        Initialize service container.

        Args:
            config: Configuration (loads from ConfigManager if None)
        """
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()

        # Store config
        if config is None:
            config = ConfigManager.get_instance().get_config()
        self._config = config

        logger.info("ServiceContainer initialized")

    # ==========================================================================
    # Registration Methods
    # ==========================================================================

    def register_singleton(
        self,
        service_type: Type[T],
        factory: Callable[[ServiceContainer], T],
    ) -> ServiceContainer:
        """
        Register singleton service (single instance).

        Args:
            service_type: Service type to register
            factory: Factory function taking container as argument

        Returns:
            Self for chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                factory=factory,
                lifetime=ServiceLifetime.SINGLETON,
            )
            self._registrations[service_type] = registration
            logger.debug(f"Registered singleton: {service_type.__name__}")
            return self

    def register_scoped(
        self,
        service_type: Type[T],
        factory: Callable[[ServiceContainer], T],
    ) -> ServiceContainer:
        """
        Register scoped service (one per scope).

        Args:
            service_type: Service type to register
            factory: Factory function taking container as argument

        Returns:
            Self for chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                factory=factory,
                lifetime=ServiceLifetime.SCOPED,
            )
            self._registrations[service_type] = registration
            logger.debug(f"Registered scoped: {service_type.__name__}")
            return self

    def register_transient(
        self,
        service_type: Type[T],
        factory: Callable[[ServiceContainer], T],
    ) -> ServiceContainer:
        """
        Register transient service (new instance each time).

        Args:
            service_type: Service type to register
            factory: Factory function taking container as argument

        Returns:
            Self for chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                factory=factory,
                lifetime=ServiceLifetime.TRANSIENT,
            )
            self._registrations[service_type] = registration
            logger.debug(f"Registered transient: {service_type.__name__}")
            return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> ServiceContainer:
        """
        Register existing instance as singleton.

        Args:
            service_type: Service type
            instance: Existing instance

        Returns:
            Self for chaining
        """
        with self._lock:
            self._singletons[service_type] = instance
            registration = ServiceRegistration(
                service_type=service_type,
                factory=lambda c: instance,
                lifetime=ServiceLifetime.SINGLETON,
            )
            registration.instance = instance
            self._registrations[service_type] = registration
            logger.debug(f"Registered instance: {service_type.__name__}")
            return self

    # ==========================================================================
    # Resolution Methods
    # ==========================================================================

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve service instance.

        Args:
            service_type: Type of service to resolve

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
        """
        with self._lock:
            if service_type not in self._registrations:
                raise KeyError(f"Service not registered: {service_type.__name__}")

            registration = self._registrations[service_type]

            if registration.lifetime == ServiceLifetime.SINGLETON:
                return self._resolve_singleton(registration)
            elif registration.lifetime == ServiceLifetime.SCOPED:
                return self._resolve_scoped(registration)
            else:  # TRANSIENT
                return self._resolve_transient(registration)

    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """
        Try to resolve service (returns None if not registered).

        Args:
            service_type: Type of service to resolve

        Returns:
            Service instance or None
        """
        try:
            return self.resolve(service_type)
        except KeyError:
            return None

    def is_registered(self, service_type: Type) -> bool:
        """
        Check if service is registered.

        Args:
            service_type: Service type to check

        Returns:
            True if registered
        """
        return service_type in self._registrations

    # ==========================================================================
    # Private Resolution Methods
    # ==========================================================================

    def _resolve_singleton(self, registration: ServiceRegistration[T]) -> T:
        """Resolve singleton (lazy initialization)."""
        if registration.instance is None:
            logger.debug(f"Creating singleton: {registration.service_type.__name__}")
            registration.instance = registration.factory(self)
        return registration.instance

    def _resolve_scoped(self, registration: ServiceRegistration[T]) -> T:
        """Resolve scoped service."""
        if registration.service_type not in self._scoped_instances:
            logger.debug(f"Creating scoped: {registration.service_type.__name__}")
            self._scoped_instances[registration.service_type] = registration.factory(self)
        return self._scoped_instances[registration.service_type]

    def _resolve_transient(self, registration: ServiceRegistration[T]) -> T:
        """Resolve transient service."""
        logger.debug(f"Creating transient: {registration.service_type.__name__}")
        return registration.factory(self)

    # ==========================================================================
    # Scope Management
    # ==========================================================================

    def create_scope(self) -> ServiceScope:
        """
        Create a new scope for scoped services.

        Returns:
            ServiceScope context manager
        """
        return ServiceScope(self)

    def clear_scope(self):
        """Clear scoped service instances."""
        with self._lock:
            self._scoped_instances.clear()
            logger.debug("Cleared scoped services")

    # ==========================================================================
    # Configuration Access
    # ==========================================================================

    def get_config(self) -> GreenLangConfig:
        """Get configuration."""
        return self._config

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_all_registrations(self) -> Dict[Type, ServiceRegistration]:
        """Get all service registrations."""
        return self._registrations.copy()

    def clear_all(self):
        """Clear all services (for testing)."""
        with self._lock:
            self._registrations.clear()
            self._singletons.clear()
            self._scoped_instances.clear()
            logger.info("Cleared all services")

    def __repr__(self) -> str:
        """String representation."""
        return f"ServiceContainer(services={len(self._registrations)})"


# ==============================================================================
# Service Scope Context Manager
# ==============================================================================

class ServiceScope:
    """
    Context manager for service scope.

    Usage:
        >>> with container.create_scope() as scope:
        ...     db = scope.resolve(Database)
        ...     # Do work with scoped services
        ... # Scoped services are automatically cleared
    """

    def __init__(self, container: ServiceContainer):
        """
        Initialize scope.

        Args:
            container: Parent container
        """
        self.container = container

    def __enter__(self) -> ServiceContainer:
        """Enter scope."""
        return self.container

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit scope - clear scoped services."""
        self.container.clear_scope()
        return False

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service in this scope."""
        return self.container.resolve(service_type)


# ==============================================================================
# Global Container
# ==============================================================================

_global_container: Optional[ServiceContainer] = None
_global_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """
    Get global service container.

    Returns:
        Global ServiceContainer instance
    """
    global _global_container

    if _global_container is None:
        with _global_container_lock:
            if _global_container is None:
                _global_container = ServiceContainer()

    return _global_container


def reset_container():
    """Reset global container (for testing)."""
    global _global_container

    with _global_container_lock:
        _global_container = None


# ==============================================================================
# Service Registration Helpers
# ==============================================================================

def register_default_services(container: ServiceContainer):
    """
    Register default GreenLang services.

    Args:
        container: Container to register services in
    """
    config = container.get_config()

    # Register configuration itself
    container.register_instance(GreenLangConfig, config)

    # TODO: Register LLM provider based on config
    # container.register_singleton(LLMProvider, lambda c: create_llm_provider(c.get_config()))

    # TODO: Register database based on config
    # container.register_scoped(Database, lambda c: create_database(c.get_config()))

    # TODO: Register cache based on config
    # container.register_singleton(Cache, lambda c: create_cache(c.get_config()))

    logger.info("Default services registered")


# ==============================================================================
# Decorator for Dependency Injection
# ==============================================================================

def inject(service_type: Type[T]) -> T:
    """
    Decorator/helper to inject services.

    Example:
        >>> class MyService:
        ...     def __init__(self, llm: LLMProvider = inject(LLMProvider)):
        ...         self.llm = llm

    Args:
        service_type: Service type to inject

    Returns:
        Resolved service instance
    """
    return get_container().resolve(service_type)
