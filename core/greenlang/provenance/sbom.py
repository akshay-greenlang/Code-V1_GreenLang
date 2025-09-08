"""
SBOM (Software Bill of Materials) Generator
============================================

Generates SBOM for packs to track dependencies and components.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pkg_resources


def generate_sbom(pack_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Generate SBOM for a pack
    
    Args:
        pack_path: Path to pack directory
        output_path: Where to save SBOM
    
    Returns:
        SBOM dictionary
    """
    # Load pack manifest
    manifest_path = pack_path / "pack.yaml"
    
    if manifest_path.exists():
        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    else:
        manifest = {}
    
    # Create SBOM in CycloneDX format
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{_generate_uuid()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {
                    "vendor": "GreenLang",
                    "name": "greenlang-sbom",
                    "version": "0.1.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": manifest.get("name", "unknown"),
                "name": manifest.get("name", "unknown"),
                "version": manifest.get("version", "0.0.0"),
                "description": manifest.get("description", ""),
                "licenses": [
                    {"license": {"id": manifest.get("license", "MIT")}}
                ]
            }
        },
        "components": []
    }
    
    # Add Python dependencies
    components = []
    
    # Get requirements
    requirements = manifest.get("requirements", [])
    for req in requirements:
        component = _parse_requirement(req)
        if component:
            components.append(component)
    
    # Scan for actual installed packages
    for dist in pkg_resources.working_set:
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{dist.project_name}@{dist.version}",
            "name": dist.project_name,
            "version": dist.version,
            "purl": f"pkg:pypi/{dist.project_name}@{dist.version}",
            "scope": "required"
        }
        components.append(component)
    
    # Add pack components (agents, pipelines, datasets)
    if "exports" in manifest:
        for export_type, items in manifest["exports"].items():
            for item in items:
                component = {
                    "type": "application",
                    "bom-ref": f"{manifest.get('name')}/{item.get('name')}",
                    "name": item.get("name"),
                    "version": manifest.get("version", "0.0.0"),
                    "description": item.get("description", ""),
                    "scope": export_type
                }
                components.append(component)
    
    # Add file hashes
    hashes = _calculate_hashes(pack_path)
    for file_path, file_hash in hashes.items():
        component = {
            "type": "file",
            "bom-ref": str(file_path),
            "name": file_path.name,
            "hashes": [
                {
                    "alg": "SHA-256",
                    "content": file_hash
                }
            ]
        }
        components.append(component)
    
    sbom["components"] = components
    
    # Add vulnerabilities section (empty for now)
    sbom["vulnerabilities"] = []
    
    # Save SBOM
    with open(output_path, "w") as f:
        json.dump(sbom, f, indent=2)
    
    return sbom


def verify_sbom(sbom_path: Path, pack_path: Path) -> bool:
    """
    Verify SBOM against actual pack contents
    
    Args:
        sbom_path: Path to SBOM file
        pack_path: Path to pack directory
    
    Returns:
        True if SBOM matches pack contents
    """
    with open(sbom_path) as f:
        sbom = json.load(f)
    
    # Verify file hashes
    current_hashes = _calculate_hashes(pack_path)
    
    for component in sbom.get("components", []):
        if component.get("type") == "file":
            file_path = Path(component.get("bom-ref", ""))
            
            if file_path in current_hashes:
                expected_hash = None
                for h in component.get("hashes", []):
                    if h.get("alg") == "SHA-256":
                        expected_hash = h.get("content")
                        break
                
                if expected_hash != current_hashes[file_path]:
                    return False
    
    return True


def _generate_uuid() -> str:
    """Generate UUID for SBOM"""
    from uuid import uuid4
    return str(uuid4())


def _parse_requirement(req: str) -> Dict[str, Any]:
    """Parse Python requirement string"""
    # Simple parsing (real implementation would use packaging library)
    if ">=" in req:
        name, version = req.split(">=")
        version = f">={version}"
    elif "==" in req:
        name, version = req.split("==")
    else:
        name = req
        version = "*"
    
    return {
        "type": "library",
        "bom-ref": f"pkg:pypi/{name}",
        "name": name.strip(),
        "version": version.strip(),
        "purl": f"pkg:pypi/{name}",
        "scope": "required"
    }


def _calculate_hashes(directory: Path) -> Dict[Path, str]:
    """Calculate SHA-256 hashes for all files"""
    hashes = {}
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Skip certain files
            if file_path.suffix in ['.pyc', '.pyo']:
                continue
            if '__pycache__' in str(file_path):
                continue
            
            # Calculate hash
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
            
            relative_path = file_path.relative_to(directory)
            hashes[relative_path] = hasher.hexdigest()
    
    return hashes


def merge_sboms(sboms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple SBOMs into one
    
    Args:
        sboms: List of SBOM dictionaries
    
    Returns:
        Merged SBOM
    """
    if not sboms:
        return {}
    
    # Start with first SBOM
    merged = sboms[0].copy()
    
    # Merge components from all SBOMs
    all_components = []
    seen_refs = set()
    
    for sbom in sboms:
        for component in sbom.get("components", []):
            bom_ref = component.get("bom-ref")
            if bom_ref not in seen_refs:
                all_components.append(component)
                seen_refs.add(bom_ref)
    
    merged["components"] = all_components
    
    # Update metadata
    merged["metadata"]["timestamp"] = datetime.now().isoformat()
    merged["serialNumber"] = f"urn:uuid:{_generate_uuid()}"
    
    return merged