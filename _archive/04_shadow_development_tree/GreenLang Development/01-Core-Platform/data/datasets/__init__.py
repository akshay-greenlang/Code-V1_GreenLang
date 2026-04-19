"""
GreenLang Datasets Module
=========================

Dataset loading and registry for GreenLang applications.

This module provides utilities for loading, registering, and managing
datasets used in sustainability and carbon accounting applications.

Example:
    >>> from greenlang.datasets import load_dataset, DatasetRegistry
    >>> dataset = load_dataset("emission_factors")
    >>> registry = DatasetRegistry()
    >>> registry.register("my_data", dataset)
"""

from greenlang.datasets.loader import (
    DatasetLoader,
    load_dataset,
    load_csv,
    load_json,
    load_yaml,
    load_excel,
)
from greenlang.datasets.registry import (
    DatasetRegistry,
    Dataset,
    DatasetMetadata,
)

__all__ = [
    # Loader functions
    'DatasetLoader',
    'load_dataset',
    'load_csv',
    'load_json',
    'load_yaml',
    'load_excel',
    # Registry classes
    'DatasetRegistry',
    'Dataset',
    'DatasetMetadata',
]

__version__ = '1.0.0'