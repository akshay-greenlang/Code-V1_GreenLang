"""
Pack Manifest Module
====================

Provides the entry point for GreenLang to discover this pack when
installed via pip.
"""

from pathlib import Path

def get_manifest_path() -> str:
    """
    Return the path to the pack.yaml manifest file.
    
    This function is registered as an entry point in pyproject.toml
    under the "greenlang.packs" group, allowing GreenLang to discover
    this pack when it's installed via pip.
    
    Returns:
        str: Absolute path to the pack.yaml file
    
    Example:
        When this pack is installed via pip:
        
        $ pip install greenlang-boiler-solar
        
        GreenLang will discover it automatically:
        
        >>> from greenlang.packs.loader import PackLoader
        >>> loader = PackLoader()
        >>> loader.list_available()
        ['boiler-solar', ...]
    """
    # Get the directory where this module is located
    module_dir = Path(__file__).parent
    
    # The pack.yaml is in the parent directory
    pack_yaml = module_dir.parent / "pack.yaml"
    
    # Return as string for compatibility
    return str(pack_yaml)