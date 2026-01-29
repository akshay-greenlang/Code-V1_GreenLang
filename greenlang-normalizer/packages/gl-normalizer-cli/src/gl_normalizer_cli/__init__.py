"""
GL-FOUND-X-003: GreenLang Unit & Reference Normalizer - Command Line Interface

This module provides a command-line interface for the GreenLang Normalizer,
supporting unit normalization, batch processing, vocabulary management,
and configuration operations.

Example:
    >>> # Normalize a single value
    >>> glnorm normalize 100 kg --to metric_ton

    >>> # Batch process a file
    >>> glnorm batch input.csv --output output.json

    >>> # Search vocabularies
    >>> glnorm vocab search "natural gas"
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gl-normalizer-cli")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
