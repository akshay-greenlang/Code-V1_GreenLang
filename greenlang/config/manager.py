"""
GreenLang Configuration Manager
================================

Centralized configuration management with:
- Environment-based config loading
- Hot-reload support
- Config validation
- Override mechanism for testing
- Thread-safe singleton pattern

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # Dummy base class
    Observer = None

from greenlang.config.schemas import (
    Environment,
    GreenLangConfig,
    load_config_from_env,
    load_config_from_file,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Config File Watcher
# ==============================================================================

class ConfigFileWatcher(FileSystemEventHandler):
    """Watch config file for changes and trigger reload."""

    def __init__(self, config_manager: ConfigManager, config_path: Path):
        """
        Initialize watcher.

        Args:
            config_manager: ConfigManager instance to notify
            config_path: Path to config file to watch
        """
        self.config_manager = config_manager
        self.config_path = config_path
        self.last_modified = time.time()

    def on_modified(self, event):
        """Handle file modification event."""
        if event.src_path == str(self.config_path):
            # Debounce: only reload if >1s since last reload
            now = time.time()
            if now - self.last_modified > 1.0:
                self.last_modified = now
                logger.info(f"Config file modified: {self.config_path}")
                self.config_manager.reload()


# ==============================================================================
# Configuration Manager
# ==============================================================================

class ConfigManager:
    """
    Centralized configuration manager with hot-reload support.

    Features:
    - Singleton pattern (one config per process)
    - Environment-based loading
    - Hot-reload from file changes
    - Override mechanism for testing
    - Thread-safe operations
    - Config validation

    Usage:
        >>> manager = ConfigManager.get_instance()
        >>> config = manager.get_config()
        >>> print(config.llm.model)
        >>>
        >>> # Enable hot-reload
        >>> manager.enable_hot_reload("config.yaml")
        >>>
        >>> # Override for testing
        >>> with manager.override(debug=True):
        ...     config = manager.get_config()
        ...     assert config.debug is True
    """

    _instance: Optional[ConfigManager] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize config manager (use get_instance() instead)."""
        self._config: Optional[GreenLangConfig] = None
        self._config_path: Optional[Path] = None
        self._observer: Optional[Observer] = None
        self._reload_callbacks: List[Callable[[GreenLangConfig], None]] = []
        self._override_stack: List[Dict[str, Any]] = []
        self._config_lock = threading.RLock()

        logger.info("ConfigManager initialized")

    @classmethod
    def get_instance(cls) -> ConfigManager:
        """
        Get singleton instance of ConfigManager.

        Returns:
            ConfigManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ConfigManager()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls._instance and cls._instance._observer:
                cls._instance._observer.stop()
                cls._instance._observer.join()
            cls._instance = None

    def load_from_env(self) -> GreenLangConfig:
        """
        Load configuration from environment variables.

        Returns:
            Loaded configuration
        """
        with self._config_lock:
            self._config = load_config_from_env()
            logger.info(f"Config loaded from environment: {self._config.environment}")
            self._notify_reload_callbacks()
            return self._config

    def load_from_file(self, config_path: Path | str) -> GreenLangConfig:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to config file

        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)

        with self._config_lock:
            self._config = load_config_from_file(config_path)
            self._config_path = config_path
            logger.info(f"Config loaded from file: {config_path}")
            self._notify_reload_callbacks()
            return self._config

    def get_config(self) -> GreenLangConfig:
        """
        Get current configuration.

        Returns:
            Current configuration (loads from env if not loaded)
        """
        with self._config_lock:
            if self._config is None:
                self.load_from_env()

            # Apply overrides if any
            if self._override_stack:
                return self._apply_overrides(self._config)

            return self._config

    def reload(self):
        """Reload configuration from source."""
        with self._config_lock:
            if self._config_path:
                logger.info(f"Reloading config from {self._config_path}")
                self.load_from_file(self._config_path)
            else:
                logger.info("Reloading config from environment")
                self.load_from_env()

    def enable_hot_reload(self, config_path: Path | str):
        """
        Enable hot-reload of config file.

        When the config file changes, it will be automatically reloaded.

        Args:
            config_path: Path to config file to watch

        Raises:
            ImportError: If watchdog is not installed
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog package is required for hot-reload. "
                "Install with: pip install watchdog"
            )

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load initial config
        self.load_from_file(config_path)

        # Set up file watcher
        if self._observer:
            self._observer.stop()
            self._observer.join()

        event_handler = ConfigFileWatcher(self, config_path)
        self._observer = Observer()
        self._observer.schedule(event_handler, str(config_path.parent), recursive=False)
        self._observer.start()

        logger.info(f"Hot-reload enabled for {config_path}")

    def disable_hot_reload(self):
        """Disable hot-reload."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Hot-reload disabled")

    def add_reload_callback(self, callback: Callable[[GreenLangConfig], None]):
        """
        Add callback to be called when config reloads.

        Args:
            callback: Function that takes GreenLangConfig as argument
        """
        self._reload_callbacks.append(callback)

    def _notify_reload_callbacks(self):
        """Notify all reload callbacks."""
        for callback in self._reload_callbacks:
            try:
                callback(self._config)
            except Exception as e:
                logger.error(f"Reload callback failed: {e}", exc_info=True)

    def _apply_overrides(self, config: GreenLangConfig) -> GreenLangConfig:
        """
        Apply override stack to config.

        Args:
            config: Base configuration

        Returns:
            Configuration with overrides applied
        """
        # Merge all overrides
        merged_overrides = {}
        for overrides in self._override_stack:
            merged_overrides.update(overrides)

        # Create new config with overrides
        config_dict = config.model_dump()
        self._deep_update(config_dict, merged_overrides)

        return GreenLangConfig(**config_dict)

    def _deep_update(self, base: dict, updates: dict):
        """Deep update dict (modifies base in-place)."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    # ==========================================================================
    # Context Manager for Overrides
    # ==========================================================================

    def override(self, **overrides) -> ConfigOverrideContext:
        """
        Create context manager for temporary config overrides.

        Example:
            >>> manager = ConfigManager.get_instance()
            >>> with manager.override(debug=True, llm={"model": "gpt-3.5"}):
            ...     config = manager.get_config()
            ...     assert config.debug is True
            ...     assert config.llm.model == "gpt-3.5"

        Args:
            **overrides: Config fields to override

        Returns:
            Context manager
        """
        return ConfigOverrideContext(self, overrides)

    def _push_overrides(self, overrides: Dict[str, Any]):
        """Push overrides onto stack."""
        with self._config_lock:
            self._override_stack.append(overrides)

    def _pop_overrides(self):
        """Pop overrides from stack."""
        with self._config_lock:
            if self._override_stack:
                self._override_stack.pop()

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    def get_environment(self) -> Environment:
        """Get current environment."""
        return self.get_config().environment

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.get_config().is_production()

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.get_config().is_development()

    def is_test(self) -> bool:
        """Check if running in tests."""
        return self.get_config().is_test()

    def get_llm_config(self):
        """Get LLM configuration."""
        return self.get_config().llm

    def get_database_config(self):
        """Get database configuration."""
        return self.get_config().database

    def get_cache_config(self):
        """Get cache configuration."""
        return self.get_config().cache

    def __repr__(self) -> str:
        """String representation."""
        config = self._config
        if config:
            return f"ConfigManager(env={config.environment}, config_path={self._config_path})"
        return "ConfigManager(not loaded)"


# ==============================================================================
# Config Override Context Manager
# ==============================================================================

class ConfigOverrideContext:
    """Context manager for temporary config overrides."""

    def __init__(self, manager: ConfigManager, overrides: Dict[str, Any]):
        """
        Initialize override context.

        Args:
            manager: ConfigManager instance
            overrides: Config fields to override
        """
        self.manager = manager
        self.overrides = overrides

    def __enter__(self) -> GreenLangConfig:
        """Enter context - push overrides."""
        self.manager._push_overrides(self.overrides)
        return self.manager.get_config()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - pop overrides."""
        self.manager._pop_overrides()
        return False


# ==============================================================================
# Global Convenience Functions
# ==============================================================================

def get_config() -> GreenLangConfig:
    """
    Get current configuration.

    Returns:
        Current GreenLangConfig instance
    """
    return ConfigManager.get_instance().get_config()


def reload_config():
    """Reload configuration from source."""
    ConfigManager.get_instance().reload()


def override_config(**overrides) -> ConfigOverrideContext:
    """
    Override configuration temporarily.

    Example:
        >>> with override_config(debug=True):
        ...     config = get_config()
        ...     assert config.debug is True

    Args:
        **overrides: Config fields to override

    Returns:
        Context manager
    """
    return ConfigManager.get_instance().override(**overrides)


# ==============================================================================
# Environment Detection
# ==============================================================================

def detect_environment() -> Environment:
    """
    Detect environment from various sources.

    Priority:
    1. GL_ENVIRONMENT env var
    2. ENVIRONMENT env var
    3. Default to DEVELOPMENT

    Returns:
        Detected environment
    """
    env_str = os.getenv("GL_ENVIRONMENT") or os.getenv("ENVIRONMENT") or "development"
    env_str = env_str.lower()

    if env_str in ["prod", "production"]:
        return Environment.PRODUCTION
    elif env_str in ["staging", "stage"]:
        return Environment.STAGING
    elif env_str in ["test", "testing"]:
        return Environment.TEST
    else:
        return Environment.DEVELOPMENT


# ==============================================================================
# Config Validation
# ==============================================================================

def validate_config(config: GreenLangConfig) -> List[str]:
    """
    Validate configuration and return list of warnings.

    Args:
        config: Configuration to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Production checks
    if config.is_production():
        if config.debug:
            warnings.append("Debug mode enabled in production")

        if not config.llm.api_key:
            warnings.append("LLM API key not set in production")

        if config.database.provider == "memory":
            warnings.append("Using memory database in production")

        if config.cache.provider == "memory":
            warnings.append("Using memory cache in production")

        if config.logging.level == "DEBUG":
            warnings.append("Debug logging enabled in production")

    # Security checks
    if config.security.enable_authentication and not config.security.jwt_secret:
        warnings.append("Authentication enabled but JWT secret not set")

    # Database checks
    if config.database.provider != "memory" and not config.database.password:
        warnings.append("Database password not set")

    return warnings
