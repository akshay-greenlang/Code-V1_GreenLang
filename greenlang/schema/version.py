"""
Version information for GL-FOUND-X-002: GreenLang Schema Compiler & Validator.

This module provides version and identification constants for the schema
compiler and validator component of the GreenLang framework.

Example:
    >>> from greenlang.schema.version import __version__, __agent_id__
    >>> print(f"Running {__agent_id__} v{__version__}")
    Running GL-FOUND-X-002 v0.1.0
"""

__version__ = "0.1.0"
__agent_id__ = "GL-FOUND-X-002"
__agent_name__ = "GreenLang Schema Compiler & Validator"

# Compiler version used for cache key generation
__compiler_version__ = "0.1.0"

# Supported JSON Schema dialect
__json_schema_dialect__ = "https://json-schema.org/draft/2020-12/schema"


def get_version_info() -> dict:
    """
    Get complete version information.

    Returns:
        Dictionary containing version, agent ID, name, and compiler version.

    Example:
        >>> info = get_version_info()
        >>> print(info['version'])
        0.1.0
    """
    return {
        "version": __version__,
        "agent_id": __agent_id__,
        "agent_name": __agent_name__,
        "compiler_version": __compiler_version__,
        "json_schema_dialect": __json_schema_dialect__,
    }
