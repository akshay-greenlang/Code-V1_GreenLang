# -*- coding: utf-8 -*-
"""
Backend Factory for GreenLang Runtime
"""

import logging
from typing import Dict, Any, Optional, Type

from .base import Backend
from .k8s import KubernetesBackend
from .docker import DockerBackend
from .local import LocalBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory for creating execution backends"""

    # Registry of available backends
    _backends: Dict[str, Type[Backend]] = {
        "kubernetes": KubernetesBackend,
        "k8s": KubernetesBackend,
        "docker": DockerBackend,
        "local": LocalBackend,
    }

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]):
        """
        Register a new backend type

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._backends[name.lower()] = backend_class
        logger.info(f"Registered backend: {name}")

    @classmethod
    def create(
        cls, backend_type: str, config: Optional[Dict[str, Any]] = None
    ) -> Backend:
        """
        Create a backend instance

        Args:
            backend_type: Type of backend
            config: Backend configuration

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is not supported
        """
        backend_type = backend_type.lower()

        if backend_type not in cls._backends:
            raise ValueError(
                f"Unsupported backend type: {backend_type}. "
                f"Available: {list(cls._backends.keys())}"
            )

        backend_class = cls._backends[backend_type]

        try:
            backend = backend_class(config or {})
            logger.info(f"Created {backend_type} backend")
            return backend
        except Exception as e:
            logger.error(f"Failed to create {backend_type} backend: {e}")
            raise

    @classmethod
    def list_backends(cls) -> list:
        """
        List available backend types

        Returns:
            List of backend names
        """
        return list(cls._backends.keys())

    @classmethod
    def get_backend_info(cls, backend_type: str) -> Dict[str, Any]:
        """
        Get information about a backend type

        Args:
            backend_type: Backend type

        Returns:
            Backend information
        """
        backend_type = backend_type.lower()

        if backend_type not in cls._backends:
            return {"error": f"Unknown backend: {backend_type}"}

        backend_class = cls._backends[backend_type]

        return {
            "name": backend_type,
            "class": backend_class.__name__,
            "module": backend_class.__module__,
            "doc": backend_class.__doc__ or "No documentation available",
        }


class BackendManager:
    """Manager for backend lifecycle and operations"""

    def __init__(
        self,
        default_backend: str = "local",
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize backend manager

        Args:
            default_backend: Default backend type
            default_config: Default backend configuration
        """
        self.default_backend = default_backend
        self.default_config = default_config or {}
        self.backends: Dict[str, Backend] = {}

        logger.info(f"BackendManager initialized with default: {default_backend}")

    def get_backend(
        self,
        backend_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Backend:
        """
        Get or create a backend instance

        Args:
            backend_type: Backend type (uses default if not specified)
            config: Backend configuration (uses default if not specified)

        Returns:
            Backend instance
        """
        backend_type = backend_type or self.default_backend
        config = config or self.default_config

        # Create cache key
        cache_key = f"{backend_type}_{hash(str(config))}"

        # Return cached backend if available
        if cache_key in self.backends:
            return self.backends[cache_key]

        # Create new backend
        backend = BackendFactory.create(backend_type, config)
        self.backends[cache_key] = backend

        return backend

    def execute_pipeline(
        self,
        pipeline,
        context,
        backend_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute a pipeline using specified backend

        Args:
            pipeline: Pipeline to execute
            context: Execution context
            backend_type: Backend type
            config: Backend configuration

        Returns:
            ExecutionResult
        """
        backend = self.get_backend(backend_type, config)
        return backend.execute(pipeline, context)

    def cleanup_all(self):
        """Cleanup all backends"""
        for backend in self.backends.values():
            if hasattr(backend, "cleanup_all"):
                backend.cleanup_all()

        self.backends.clear()
        logger.info("Cleaned up all backends")


# Global backend manager instance
_manager = None


def get_backend_manager() -> BackendManager:
    """Get global backend manager instance"""
    global _manager

    if _manager is None:
        _manager = BackendManager()

    return _manager


def set_default_backend(backend_type: str, config: Optional[Dict[str, Any]] = None):
    """
    Set default backend for execution

    Args:
        backend_type: Backend type
        config: Backend configuration
    """
    manager = get_backend_manager()
    manager.default_backend = backend_type
    manager.default_config = config or {}

    logger.info(f"Set default backend to: {backend_type}")
