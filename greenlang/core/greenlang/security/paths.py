# -*- coding: utf-8 -*-
"""
Path Security Module
===================

Provides security functions for file path operations including:
- Path traversal protection
- Safe archive extraction
- Directory validation
"""

import os
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


def validate_safe_path(
    base_path: Union[str, Path], target_path: Union[str, Path]
) -> Path:
    """
    Validate that target path is safely within base path (no traversal)

    Args:
        base_path: Base directory path
        target_path: Target path to validate

    Returns:
        Resolved safe path

    Raises:
        ValueError: If path traversal is detected
    """
    base = Path(base_path).resolve()
    target = Path(target_path)

    # If target is relative, join with base
    if not target.is_absolute():
        target = base / target

    # Resolve to absolute path
    resolved = target.resolve()

    # Check if resolved path is within base
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Path traversal detected! "
            f"Target path '{target}' resolves outside base directory '{base}'"
        )

    return resolved


def safe_extract_tar(
    archive_path: Union[str, Path], extract_path: Union[str, Path]
) -> None:
    """
    Safely extract tar archive with path traversal protection

    Args:
        archive_path: Path to tar archive
        extract_path: Directory to extract to

    Raises:
        ValueError: If unsafe paths are detected in archive
    """
    archive_path = Path(archive_path)
    extract_path = Path(extract_path).resolve()

    if not extract_path.exists():
        extract_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as tar:
        # Check all members for safety
        for member in tar.getmembers():
            # Check for absolute paths
            if member.name.startswith("/"):
                raise ValueError(f"Absolute path in archive not allowed: {member.name}")

            # Check for path traversal
            try:
                validate_safe_path(extract_path, member.name)
            except ValueError as e:
                raise ValueError(f"Unsafe path in tar archive: {member.name}") from e

            # Additional checks for symbolic links
            if member.issym() or member.islnk():
                # Get the link target
                link_target = member.linkname
                if link_target.startswith("/"):
                    raise ValueError(
                        f"Absolute symlink in archive not allowed: {member.name} -> {link_target}"
                    )

                # Validate link target is within extract path
                try:
                    validate_safe_path(
                        extract_path, Path(member.name).parent / link_target
                    )
                except ValueError:
                    raise ValueError(
                        f"Symlink points outside extraction directory: {member.name} -> {link_target}"
                    )

        # All checks passed, extract
        tar.extractall(extract_path)
        logger.info(f"Safely extracted tar archive to: {extract_path}")


def safe_extract_zip(
    archive_path: Union[str, Path], extract_path: Union[str, Path]
) -> None:
    """
    Safely extract zip archive with path traversal protection

    Args:
        archive_path: Path to zip archive
        extract_path: Directory to extract to

    Raises:
        ValueError: If unsafe paths are detected in archive
    """
    archive_path = Path(archive_path)
    extract_path = Path(extract_path).resolve()

    if not extract_path.exists():
        extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        # Check all members for safety
        for member in zip_ref.namelist():
            # Check for absolute paths
            if member.startswith("/"):
                raise ValueError(f"Absolute path in archive not allowed: {member}")

            # Check for path traversal
            try:
                validate_safe_path(extract_path, member)
            except ValueError as e:
                raise ValueError(f"Unsafe path in zip archive: {member}") from e

        # All checks passed, extract
        zip_ref.extractall(extract_path)
        logger.info(f"Safely extracted zip archive to: {extract_path}")


def safe_extract_archive(
    archive_path: Union[str, Path], extract_path: Union[str, Path]
) -> None:
    """
    Safely extract archive (auto-detect format) with path traversal protection

    Args:
        archive_path: Path to archive
        extract_path: Directory to extract to

    Raises:
        ValueError: If unsafe paths are detected or format unsupported
    """
    archive_path = Path(archive_path)

    # Determine archive type and extract
    if archive_path.suffix in [
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
    ] or archive_path.name.endswith(".tar.gz"):
        safe_extract_tar(archive_path, extract_path)
    elif archive_path.suffix == ".zip":
        safe_extract_zip(archive_path, extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def validate_pack_structure(pack_path: Union[str, Path]) -> None:
    """
    Validate pack directory structure for security

    Args:
        pack_path: Path to pack directory

    Raises:
        ValueError: If pack structure is invalid or unsafe
    """
    pack_path = Path(pack_path)

    if not pack_path.exists():
        raise ValueError(f"Pack directory does not exist: {pack_path}")

    # Check for required files
    manifest_path = pack_path / "pack.yaml"
    if not manifest_path.exists():
        raise ValueError(f"Missing pack.yaml in: {pack_path}")

    # Check for suspicious files
    suspicious_patterns = [
        "*.exe",
        "*.dll",
        "*.so",
        "*.dylib",  # Binaries
        ".git",
        ".svn",  # Version control
        "__pycache__",  # Python cache
        ".env",
        ".env.local",  # Environment files
        "*.key",
        "*.pem",
        "*.crt",  # Certificates/keys
    ]

    for pattern in suspicious_patterns:
        suspicious_files = list(pack_path.rglob(pattern))
        if suspicious_files:
            logger.warning(f"Suspicious files found in pack: {suspicious_files}")

    # Check file permissions (on Unix systems)
    if os.name != "nt":  # Not Windows
        for file_path in pack_path.rglob("*"):
            if file_path.is_file():
                # Check for executable files
                if os.access(file_path, os.X_OK):
                    logger.warning(f"Executable file found in pack: {file_path}")


def safe_create_directory(
    dir_path: Union[str, Path], parent_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Safely create a directory with validation

    Args:
        dir_path: Directory path to create
        parent_path: Optional parent path for validation

    Returns:
        Created directory path

    Raises:
        ValueError: If path is unsafe
    """
    dir_path = Path(dir_path)

    # Validate against parent if provided
    if parent_path:
        dir_path = validate_safe_path(parent_path, dir_path)

    # Create directory
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directory: {dir_path}")

    return dir_path
