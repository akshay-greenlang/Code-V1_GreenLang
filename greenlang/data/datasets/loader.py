"""
Dataset Loader
==============

Utilities for loading datasets from various sources.

Author: Data Team
Created: 2025-11-21
"""

import json
import csv
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for dataset loading."""
    cache_enabled: bool = True
    validate_schema: bool = True
    encoding: str = 'utf-8'
    delimiter: str = ','


class DatasetLoader:
    """
    Loads datasets from various file formats and sources.

    Supports CSV, JSON, YAML, Excel, and custom formats.
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize dataset loader."""
        self.config = config or LoaderConfig()
        self._cache: Dict[str, Any] = {}

    def load(self, path: Union[str, Path], format: Optional[str] = None) -> Any:
        """
        Load dataset from file.

        Args:
            path: Path to dataset file
            format: File format (auto-detected if None)

        Returns:
            Loaded dataset
        """
        path = Path(path)

        # Check cache
        cache_key = str(path.absolute())
        if self.config.cache_enabled and cache_key in self._cache:
            logger.debug(f"Loading from cache: {path}")
            return self._cache[cache_key]

        # Auto-detect format
        if format is None:
            format = self._detect_format(path)

        # Load based on format
        logger.info(f"Loading dataset: {path} (format: {format})")

        if format == 'csv':
            data = self.load_csv(path)
        elif format == 'json':
            data = self.load_json(path)
        elif format == 'yaml':
            data = self.load_yaml(path)
        elif format in ['xlsx', 'xls']:
            data = self.load_excel(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Cache if enabled
        if self.config.cache_enabled:
            self._cache[cache_key] = data

        return data

    def load_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file as DataFrame.

        Args:
            path: Path to CSV file

        Returns:
            Pandas DataFrame
        """
        path = Path(path)
        try:
            df = pd.read_csv(
                path,
                encoding=self.config.encoding,
                delimiter=self.config.delimiter
            )
            logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV {path}: {str(e)}")
            raise

    def load_json(self, path: Union[str, Path]) -> Union[Dict, List]:
        """
        Load JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON data
        """
        path = Path(path)
        try:
            with open(path, 'r', encoding=self.config.encoding) as f:
                data = json.load(f)
            logger.info(f"Loaded JSON from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {str(e)}")
            raise

    def load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML data
        """
        path = Path(path)
        try:
            with open(path, 'r', encoding=self.config.encoding) as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded YAML from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load YAML {path}: {str(e)}")
            raise

    def load_excel(self, path: Union[str, Path], sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Excel file as DataFrame.

        Args:
            path: Path to Excel file
            sheet_name: Name of sheet to load (first sheet if None)

        Returns:
            Pandas DataFrame
        """
        path = Path(path)
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            logger.info(f"Loaded Excel: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel {path}: {str(e)}")
            raise

    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xlsx': 'xlsx',
            '.xls': 'xls',
        }
        return format_map.get(suffix, 'unknown')

    def clear_cache(self) -> None:
        """Clear the loader cache."""
        self._cache.clear()
        logger.debug("Loader cache cleared")


# Convenience functions
_default_loader = DatasetLoader()


def load_dataset(name: str, path: Optional[Union[str, Path]] = None) -> Any:
    """
    Load a dataset by name or path.

    Args:
        name: Dataset name or identifier
        path: Optional path override

    Returns:
        Loaded dataset
    """
    # If path provided, use it
    if path:
        return _default_loader.load(path)

    # Otherwise, look for standard datasets
    standard_datasets = {
        "emission_factors": "data/emission_factors.csv",
        "carbon_intensity": "data/carbon_intensity.json",
        "supplier_data": "data/suppliers.csv",
        "activity_data": "data/activities.csv",
    }

    if name in standard_datasets:
        dataset_path = Path(standard_datasets[name])
        if dataset_path.exists():
            return _default_loader.load(dataset_path)

    # Try as direct path
    path = Path(name)
    if path.exists():
        return _default_loader.load(path)

    raise FileNotFoundError(f"Dataset not found: {name}")


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV file."""
    return _default_loader.load_csv(path)


def load_json(path: Union[str, Path]) -> Union[Dict, List]:
    """Load JSON file."""
    return _default_loader.load_json(path)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    return _default_loader.load_yaml(path)


def load_excel(path: Union[str, Path], sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load Excel file."""
    return _default_loader.load_excel(path, sheet_name)