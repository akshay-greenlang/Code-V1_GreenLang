"""
IoC Dependency Injection Container

This module implements an Inversion of Control (IoC) container for managing
dependencies throughout the GL-Agent-Factory application. It supports:

- Service registration with multiple lifetimes (singleton, scoped, transient)
- Constructor injection with automatic resolution
- Factory functions for complex instantiation
- Async initialization support
- Scoped dependency management for request handling
- Type-safe dependency resolution

Usage:
    from core.container import Container, ServiceLifetime, inject

    container = Container()

    # Register services
    container.register(IDatabase, PostgresDatabase, lifetime=ServiceLifetime.SINGLETON)
    container.register(ICache, RedisCache, lifetime=ServiceLifetime.SINGLETON)
    container.register(AgentRegistry, lifetime=ServiceLifetime.SINGLETON)

    # Resolve dependencies
    registry = container.resolve(AgentRegistry)

    # Use decorator for automatic injection
    @inject
    def calculate_emissions(db: IDatabase, cache: ICache):
        ...
"""
import asyncio
import functools
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    get_origin,
    get_args,
)
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")
TService = TypeVar("TService")


# =============================================================================
# Service Lifetime
# =============================================================================


class ServiceLifetime(Enum):
    """Defines how long a service instance lives."""

    SINGLETON = "singleton"
    """Single instance for the entire application lifetime."""

    SCOPED = "scoped"
    """Single instance per scope (e.g., per HTTP request)."""

    TRANSIENT = "transient"
    """New instance every time the service is resolved."""


# =============================================================================
# Service Registration
# =============================================================================


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""

    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[Type] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    name: Optional[str] = None

    def __post_init__(self):
        """Validate the descriptor."""
        if self.implementation_type is None and self.factory is None and self.instance is None:
            # Use service_type as implementation
            self.implementation_type = self.service_type


@dataclass
class ScopeContext:
    """Context for a dependency scope."""

    scope_id: str
    instances: Dict[Type, Any] = field(default_factory=dict)
    parent: Optional["ScopeContext"] = None


# =============================================================================
# Exceptions
# =============================================================================


class ContainerError(Exception):
    """Base exception for container errors."""

    pass


class ServiceNotFoundError(ContainerError):
    """Raised when a service cannot be resolved."""

    def __init__(self, service_type: Type):
        self.service_type = service_type
        super().__init__(f"Service not found: {service_type.__name__}")


class CircularDependencyError(ContainerError):
    """Raised when circular dependencies are detected."""

    def __init__(self, chain: List[Type]):
        self.chain = chain
        names = " -> ".join(t.__name__ for t in chain)
        super().__init__(f"Circular dependency detected: {names}")


class RegistrationError(ContainerError):
    """Raised when service registration fails."""

    pass


# =============================================================================
# Container Implementation
# =============================================================================


class Container:
    """
    IoC Dependency Injection Container.

    Provides dependency injection capabilities for the application with
    support for different service lifetimes and automatic resolution.
    """

    def __init__(self):
        """Initialize an empty container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._named_services: Dict[str, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._current_scope: Optional[ScopeContext] = None
        self._scope_stack: List[ScopeContext] = []
        self._resolving: Set[Type] = set()  # For circular dependency detection
        self._interceptors: List[Callable[[Type, Any], Any]] = []
        self._initializers: List[Callable[[Any], None]] = []

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        service_type: Type[TService],
        implementation: Optional[Type[TService]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        factory: Optional[Callable[..., TService]] = None,
        instance: Optional[TService] = None,
        name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> "Container":
        """
        Register a service with the container.

        Args:
            service_type: The service interface/type to register
            implementation: The concrete implementation class
            lifetime: Service lifetime (singleton, scoped, transient)
            factory: Factory function for creating instances
            instance: Pre-created instance (implies singleton)
            name: Optional name for named resolution
            tags: Optional tags for filtering

        Returns:
            Self for fluent chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation,
                factory=factory,
                instance=instance,
                lifetime=lifetime if instance is None else ServiceLifetime.SINGLETON,
                tags=tags or set(),
                name=name,
            )

            # If instance provided, store as singleton immediately
            if instance is not None:
                self._singletons[service_type] = instance

            # Analyze dependencies if implementation provided
            if implementation is not None:
                descriptor.dependencies = self._get_dependencies(implementation)

            self._services[service_type] = descriptor

            if name:
                self._named_services[name] = descriptor

            logger.debug(
                f"Registered {service_type.__name__} "
                f"(lifetime={lifetime.value}, name={name})"
            )

        return self

    def register_singleton(
        self,
        service_type: Type[TService],
        implementation: Optional[Type[TService]] = None,
        factory: Optional[Callable[..., TService]] = None,
    ) -> "Container":
        """Convenience method to register a singleton service."""
        return self.register(
            service_type,
            implementation,
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory,
        )

    def register_scoped(
        self,
        service_type: Type[TService],
        implementation: Optional[Type[TService]] = None,
        factory: Optional[Callable[..., TService]] = None,
    ) -> "Container":
        """Convenience method to register a scoped service."""
        return self.register(
            service_type,
            implementation,
            lifetime=ServiceLifetime.SCOPED,
            factory=factory,
        )

    def register_transient(
        self,
        service_type: Type[TService],
        implementation: Optional[Type[TService]] = None,
        factory: Optional[Callable[..., TService]] = None,
    ) -> "Container":
        """Convenience method to register a transient service."""
        return self.register(
            service_type,
            implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            factory=factory,
        )

    def register_instance(
        self, service_type: Type[TService], instance: TService, name: Optional[str] = None
    ) -> "Container":
        """Register a pre-created instance."""
        return self.register(service_type, instance=instance, name=name)

    def register_factory(
        self,
        service_type: Type[TService],
        factory: Callable[..., TService],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "Container":
        """Register a factory function for creating instances."""
        return self.register(service_type, factory=factory, lifetime=lifetime)

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve(self, service_type: Type[TService], name: Optional[str] = None) -> TService:
        """
        Resolve a service from the container.

        Args:
            service_type: The service type to resolve
            name: Optional name for named resolution

        Returns:
            The resolved service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependencies detected
        """
        with self._lock:
            return self._resolve_internal(service_type, name)

    def _resolve_internal(
        self, service_type: Type[TService], name: Optional[str] = None
    ) -> TService:
        """Internal resolution logic."""
        # Check for circular dependency
        if service_type in self._resolving:
            chain = list(self._resolving) + [service_type]
            raise CircularDependencyError(chain)

        # Get descriptor
        descriptor = self._get_descriptor(service_type, name)
        if descriptor is None:
            raise ServiceNotFoundError(service_type)

        # Check lifetime and return cached if applicable
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if self._current_scope and service_type in self._current_scope.instances:
                return self._current_scope.instances[service_type]

        # Mark as resolving (for circular dependency detection)
        self._resolving.add(service_type)

        try:
            # Create instance
            instance = self._create_instance(descriptor)

            # Run interceptors
            for interceptor in self._interceptors:
                instance = interceptor(service_type, instance)

            # Run initializers
            for initializer in self._initializers:
                initializer(instance)

            # Cache based on lifetime
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                self._singletons[service_type] = instance

            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if self._current_scope:
                    self._current_scope.instances[service_type] = instance

            return instance

        finally:
            self._resolving.discard(service_type)

    def _get_descriptor(
        self, service_type: Type, name: Optional[str]
    ) -> Optional[ServiceDescriptor]:
        """Get service descriptor by type or name."""
        if name:
            return self._named_services.get(name)
        return self._services.get(service_type)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of a service."""
        # Instance already provided
        if descriptor.instance is not None:
            return descriptor.instance

        # Use factory if provided
        if descriptor.factory is not None:
            # Inject factory dependencies
            factory_deps = self._get_dependencies(descriptor.factory)
            args = [self._resolve_internal(dep) for dep in factory_deps]
            return descriptor.factory(*args)

        # Use implementation type
        impl = descriptor.implementation_type or descriptor.service_type
        deps = self._get_dependencies(impl)
        args = [self._resolve_internal(dep) for dep in deps]
        return impl(*args)

    def _get_dependencies(self, target: Union[Type, Callable]) -> List[Type]:
        """Extract dependencies from constructor or callable signature."""
        deps = []

        try:
            hints = get_type_hints(target.__init__ if inspect.isclass(target) else target)
        except Exception:
            hints = {}

        sig = inspect.signature(target.__init__ if inspect.isclass(target) else target)

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            if param.annotation != inspect.Parameter.empty:
                dep_type = hints.get(param_name, param.annotation)
                # Handle Optional types
                origin = get_origin(dep_type)
                if origin is Union:
                    args = get_args(dep_type)
                    # Get the non-None type from Optional
                    dep_type = next((a for a in args if a is not type(None)), args[0])
                deps.append(dep_type)

        return deps

    def try_resolve(
        self, service_type: Type[TService], name: Optional[str] = None
    ) -> Optional[TService]:
        """Try to resolve a service, returning None if not found."""
        try:
            return self.resolve(service_type, name)
        except ServiceNotFoundError:
            return None

    def resolve_all(self, service_type: Type[TService]) -> List[TService]:
        """Resolve all registered implementations of a service type."""
        results = []
        for desc in self._services.values():
            if issubclass(desc.implementation_type or desc.service_type, service_type):
                results.append(self.resolve(desc.service_type))
        return results

    def resolve_by_tag(self, tag: str) -> List[Any]:
        """Resolve all services with a specific tag."""
        results = []
        for desc in self._services.values():
            if tag in desc.tags:
                results.append(self.resolve(desc.service_type))
        return results

    # =========================================================================
    # Scoping
    # =========================================================================

    @contextmanager
    def scope(self, scope_id: Optional[str] = None):
        """
        Create a new dependency scope.

        Usage:
            with container.scope():
                service = container.resolve(MyScopedService)
        """
        scope_id = scope_id or f"scope_{id(threading.current_thread())}"
        scope = ScopeContext(scope_id=scope_id, parent=self._current_scope)

        self._scope_stack.append(self._current_scope)
        self._current_scope = scope

        try:
            yield scope
        finally:
            self._current_scope = self._scope_stack.pop()

    @asynccontextmanager
    async def async_scope(self, scope_id: Optional[str] = None):
        """Async version of scope context manager."""
        with self.scope(scope_id) as scope:
            yield scope

    # =========================================================================
    # Interceptors & Initializers
    # =========================================================================

    def add_interceptor(
        self, interceptor: Callable[[Type, Any], Any]
    ) -> "Container":
        """
        Add an interceptor that runs after instance creation.

        Interceptors can modify or wrap instances.
        """
        self._interceptors.append(interceptor)
        return self

    def add_initializer(self, initializer: Callable[[Any], None]) -> "Container":
        """
        Add an initializer that runs after instance creation.

        Initializers can perform additional setup.
        """
        self._initializers.append(initializer)
        return self

    # =========================================================================
    # Validation & Utilities
    # =========================================================================

    def validate(self) -> List[str]:
        """
        Validate all registrations can be resolved.

        Returns list of error messages, empty if all valid.
        """
        errors = []

        for service_type, descriptor in self._services.items():
            try:
                # Check dependencies can be resolved
                deps = descriptor.dependencies
                for dep in deps:
                    if dep not in self._services:
                        errors.append(
                            f"{service_type.__name__} depends on unregistered "
                            f"service: {dep.__name__}"
                        )
            except Exception as e:
                errors.append(f"Error validating {service_type.__name__}: {e}")

        return errors

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services

    def get_registrations(self) -> Dict[str, ServiceDescriptor]:
        """Get all service registrations."""
        return {t.__name__: d for t, d in self._services.items()}

    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        with self._lock:
            self._services.clear()
            self._named_services.clear()
            self._singletons.clear()
            self._scope_stack.clear()
            self._current_scope = None

    def __contains__(self, service_type: Type) -> bool:
        """Check if service is registered using 'in' operator."""
        return self.is_registered(service_type)


# =============================================================================
# Decorators
# =============================================================================

# Global container instance
_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def set_container(container: Container) -> None:
    """Set the global container instance."""
    global _container
    _container = container


def inject(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for automatic dependency injection.

    Usage:
        @inject
        def my_function(db: IDatabase, cache: ICache):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        container = get_container()
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build kwargs with injected dependencies
        injected = dict(kwargs)
        for param_name, param in sig.parameters.items():
            if param_name in injected or param_name in ("self", "cls"):
                continue
            if param_name in hints:
                try:
                    injected[param_name] = container.resolve(hints[param_name])
                except ServiceNotFoundError:
                    if param.default == inspect.Parameter.empty:
                        raise

        return func(*args, **injected)

    return wrapper


def injectable(
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    name: Optional[str] = None,
    tags: Optional[Set[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator to auto-register with the container.

    Usage:
        @injectable(lifetime=ServiceLifetime.SINGLETON)
        class MyService:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Register with global container when class is defined
        container = get_container()
        container.register(cls, lifetime=lifetime, name=name, tags=tags)
        return cls

    return decorator


# =============================================================================
# Service Provider (FastAPI Integration)
# =============================================================================


class ServiceProvider:
    """
    Service provider for FastAPI dependency injection integration.

    Usage with FastAPI:
        provider = ServiceProvider(container)

        @app.get("/")
        async def endpoint(db: IDatabase = Depends(provider.provide(IDatabase))):
            ...
    """

    def __init__(self, container: Container):
        self.container = container

    def provide(self, service_type: Type[T]) -> Callable[[], T]:
        """Create a FastAPI dependency provider for a service type."""
        def provider() -> T:
            return self.container.resolve(service_type)
        return provider

    def scoped_provider(
        self, service_type: Type[T]
    ) -> Callable[[], T]:
        """Create a scoped provider for request-scoped services."""
        def provider() -> T:
            with self.container.scope():
                return self.container.resolve(service_type)
        return provider


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Container",
    "ServiceLifetime",
    "ServiceDescriptor",
    "ScopeContext",
    "ContainerError",
    "ServiceNotFoundError",
    "CircularDependencyError",
    "RegistrationError",
    "get_container",
    "set_container",
    "inject",
    "injectable",
    "ServiceProvider",
]
