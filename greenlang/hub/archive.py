"""
Pack Archive Utilities for GreenLang Hub
"""

import tarfile
import gzip
import json
import logging
from pathlib import Path
from typing import Union, List, Optional
import tempfile
import shutil
import hashlib

logger = logging.getLogger(__name__)


def create_pack_archive(pack_path: Path, 
                       output_path: Optional[Path] = None,
                       compression: str = 'gz',
                       exclude_patterns: Optional[List[str]] = None) -> Path:
    """
    Create a compressed archive of a pack
    
    Args:
        pack_path: Path to pack directory
        output_path: Output archive path (optional)
        compression: Compression type ('gz', 'bz2', 'xz', or None)
        exclude_patterns: Patterns to exclude from archive
        
    Returns:
        Path to created archive
    """
    pack_path = Path(pack_path)
    
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack not found: {pack_path}")
    
    if not pack_path.is_dir():
        raise ValueError(f"Pack path must be a directory: {pack_path}")
    
    # Default exclude patterns
    default_excludes = [
        '__pycache__',
        '*.pyc',
        '.git',
        '.gitignore',
        '.DS_Store',
        'Thumbs.db',
        '*.swp',
        '*.bak',
        '~*'
    ]
    
    exclude_patterns = exclude_patterns or []
    exclude_patterns.extend(default_excludes)
    
    # Determine output path
    if output_path is None:
        suffix = f".tar.{compression}" if compression else ".tar"
        output_path = Path(tempfile.gettempdir()) / f"{pack_path.name}{suffix}"
    else:
        output_path = Path(output_path)
    
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine compression mode
    mode_map = {
        'gz': 'w:gz',
        'bz2': 'w:bz2',
        'xz': 'w:xz',
        None: 'w'
    }
    mode = mode_map.get(compression, 'w:gz')
    
    logger.info(f"Creating archive: {output_path}")
    
    try:
        with tarfile.open(output_path, mode) as tar:
            # Add files to archive
            for item in pack_path.iterdir():
                # Check if item should be excluded
                if _should_exclude(item, exclude_patterns):
                    logger.debug(f"Excluding: {item}")
                    continue
                
                # Add to archive with relative path
                arcname = item.relative_to(pack_path.parent)
                logger.debug(f"Adding: {arcname}")
                tar.add(item, arcname=str(arcname), recursive=True, 
                       filter=lambda x: _tar_filter(x, exclude_patterns))
        
        # Calculate checksum
        checksum = calculate_checksum(output_path)
        logger.info(f"Archive created: {output_path} (SHA256: {checksum})")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def extract_pack_archive(archive_data: Union[bytes, Path],
                        output_dir: Path,
                        verify_checksum: Optional[str] = None) -> Path:
    """
    Extract a pack archive
    
    Args:
        archive_data: Archive data (bytes) or path to archive file
        output_dir: Directory to extract to
        verify_checksum: Optional checksum to verify
        
    Returns:
        Path to extracted pack directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle different input types
    if isinstance(archive_data, bytes):
        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
            tmp.write(archive_data)
            archive_path = Path(tmp.name)
    else:
        archive_path = Path(archive_data)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    # Verify checksum if provided
    if verify_checksum:
        actual_checksum = calculate_checksum(archive_path)
        if actual_checksum != verify_checksum:
            raise ValueError(f"Checksum mismatch: expected {verify_checksum}, got {actual_checksum}")
    
    logger.info(f"Extracting archive to: {output_dir}")
    
    try:
        # Detect compression
        compression = detect_compression(archive_path)
        
        # Open and extract archive
        with tarfile.open(archive_path, f'r:{compression}' if compression else 'r') as tar:
            # Security check: ensure no path traversal
            for member in tar.getmembers():
                if _is_path_traversal(member.name):
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            
            # Extract all
            tar.extractall(output_dir)
        
        # Find the actual pack directory (might be nested)
        pack_dir = _find_pack_root(output_dir)
        
        logger.info(f"Archive extracted successfully to: {pack_dir}")
        return pack_dir
        
    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        raise
    finally:
        # Cleanup temporary file if created
        if isinstance(archive_data, bytes) and archive_path.exists():
            archive_path.unlink()


def calculate_checksum(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Calculate checksum of a file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of checksum
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def detect_compression(archive_path: Path) -> Optional[str]:
    """
    Detect compression type of archive
    
    Args:
        archive_path: Path to archive
        
    Returns:
        Compression type ('gz', 'bz2', 'xz') or None
    """
    # Check file extension
    suffixes = archive_path.suffixes
    if '.gz' in suffixes:
        return 'gz'
    elif '.bz2' in suffixes:
        return 'bz2'
    elif '.xz' in suffixes:
        return 'xz'
    
    # Check file header
    with open(archive_path, 'rb') as f:
        header = f.read(8)
        
        if header[:2] == b'\x1f\x8b':  # gzip
            return 'gz'
        elif header[:3] == b'BZh':  # bzip2
            return 'bz2'
        elif header[:6] == b'\xfd7zXZ\x00':  # xz
            return 'xz'
    
    return None


def _should_exclude(path: Path, patterns: List[str]) -> bool:
    """Check if path should be excluded based on patterns"""
    import fnmatch
    
    name = path.name
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
    
    return False


def _tar_filter(tarinfo: tarfile.TarInfo, exclude_patterns: List[str]) -> Optional[tarfile.TarInfo]:
    """Filter function for tarfile.add"""
    import fnmatch
    
    # Check exclusion patterns
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(tarinfo.name, pattern):
            return None
    
    return tarinfo


def _is_path_traversal(path: str) -> bool:
    """Check if path contains directory traversal"""
    import os
    
    # Normalize path
    normalized = os.path.normpath(path)
    
    # Check for traversal patterns
    if normalized.startswith('..') or '/..' in normalized:
        return True
    
    # Check for absolute paths
    if os.path.isabs(normalized):
        return True
    
    return False


def _find_pack_root(extract_dir: Path) -> Path:
    """
    Find the actual pack root directory after extraction
    
    Args:
        extract_dir: Directory where archive was extracted
        
    Returns:
        Path to pack root
    """
    # Check if manifest exists in extract_dir
    if (extract_dir / 'manifest.json').exists():
        return extract_dir
    
    # Check immediate subdirectories
    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    
    if len(subdirs) == 1:
        # Single subdirectory, likely the pack root
        subdir = subdirs[0]
        if (subdir / 'manifest.json').exists():
            return subdir
    
    # Look for manifest in any subdirectory
    for subdir in subdirs:
        if (subdir / 'manifest.json').exists():
            return subdir
    
    # Default to extract_dir
    return extract_dir


def create_incremental_archive(pack_path: Path,
                              base_archive: Optional[Path] = None,
                              output_path: Optional[Path] = None) -> Path:
    """
    Create an incremental archive (only changed files since base)
    
    Args:
        pack_path: Path to pack directory
        base_archive: Previous archive to compare against
        output_path: Output archive path
        
    Returns:
        Path to incremental archive
    """
    # TODO: Implement incremental archiving
    # This would compare against a base archive and only include changed files
    logger.warning("Incremental archiving not yet implemented, creating full archive")
    return create_pack_archive(pack_path, output_path)


def verify_archive_integrity(archive_path: Path) -> bool:
    """
    Verify integrity of an archive
    
    Args:
        archive_path: Path to archive
        
    Returns:
        True if archive is valid
    """
    try:
        compression = detect_compression(archive_path)
        mode = f'r:{compression}' if compression else 'r'
        
        with tarfile.open(archive_path, mode) as tar:
            # Test archive by reading all members
            tar.getmembers()
        
        return True
        
    except Exception as e:
        logger.error(f"Archive integrity check failed: {e}")
        return False