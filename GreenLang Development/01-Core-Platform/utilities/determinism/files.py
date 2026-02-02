"""
GreenLang Deterministic File Operations - Sorted Directory Operations

This module provides deterministic file system operations with sorted results.

Features:
- Sorted directory listing
- Sorted glob patterns
- Sorted path iteration
- Ensures consistent ordering across platforms

Author: GreenLang Team
Date: 2025-11-21
"""

import os
import glob as glob_module
from pathlib import Path
from typing import Union, List


def sorted_listdir(path: Union[str, Path]) -> List[str]:
    """
    List directory contents in sorted order.

    Args:
        path: Directory path

    Returns:
        Sorted list of filenames
    """
    return sorted(os.listdir(path))


def sorted_glob(pattern: str, recursive: bool = False) -> List[str]:
    """
    Glob files in sorted order.

    Args:
        pattern: Glob pattern
        recursive: Enable recursive globbing

    Returns:
        Sorted list of matching paths
    """
    return sorted(glob_module.glob(pattern, recursive=recursive))


def sorted_iterdir(path: Union[str, Path]) -> List[Path]:
    """
    Iterate directory contents in sorted order.

    Args:
        path: Directory path

    Returns:
        Sorted list of Path objects
    """
    path = Path(path) if isinstance(path, str) else path
    return sorted(path.iterdir())


__all__ = [
    'sorted_listdir',
    'sorted_glob',
    'sorted_iterdir',
]
