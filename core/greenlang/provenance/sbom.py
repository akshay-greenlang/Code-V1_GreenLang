"""
SBOM (Software Bill of Materials) Generator
============================================

Generates SBOM for packs to track dependencies and components.
"""

import json
import hashlib
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pkg_resources
except ImportError:
    pkg_resources = None
import logging

# Set up Windows encoding support for this module
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

logger = logging.getLogger(__name__)


def generate_sbom(
    path: str, output_path: Optional[str] = None, format: str = "spdx"
) -> str:
    """
    Generate SBOM for a pack in SPDX or CycloneDX format

    Args:
        path: Path to pack directory or requirements file
        output_path: Where to save SBOM (defaults to path/sbom.spdx.json)
        format: SBOM format - "spdx" or "cyclonedx"

    Returns:
        Path to generated SBOM file
    """
    pack_path = Path(path)
    if output_path is None:
        ext = "spdx.json" if format == "spdx" else "cdx.json"
        output_path = pack_path / f"sbom.{ext}"
    else:
        output_path = Path(output_path)

    if format == "spdx":
        logger.info("Generating SPDX format SBOM")
        return _generate_spdx_sbom(pack_path, output_path)
    elif format == "cyclonedx":
        # Try to use CycloneDX if available
        if _has_cyclonedx():
            logger.info("Using CycloneDX to generate SBOM")
            return _generate_with_cyclonedx(pack_path, output_path)
        else:
            logger.info("Using fallback CycloneDX generation")
            return _generate_manual_sbom(pack_path, output_path)
    else:
        raise ValueError(f"Unknown SBOM format: {format}. Use 'spdx' or 'cyclonedx'")

    # Create SBOM in CycloneDX format
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{_generate_uuid()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {"vendor": "GreenLang", "name": "greenlang-sbom", "version": "0.1.0"}
            ],
            "component": {
                "type": "application",
                "bom-ref": manifest.get("name", "unknown"),
                "name": manifest.get("name", "unknown"),
                "version": manifest.get("version", "0.0.0"),
                "description": manifest.get("description", ""),
                "licenses": [{"license": {"id": manifest.get("license", "MIT")}}],
            },
        },
        "components": [],
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
    if pkg_resources:
        for dist in pkg_resources.working_set:
            component = {
                "type": "library",
                "bom-ref": f"pkg:pypi/{dist.project_name}@{dist.version}",
                "name": dist.project_name,
                "version": dist.version,
                "purl": f"pkg:pypi/{dist.project_name}@{dist.version}",
                "scope": "required",
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
                    "scope": export_type,
                }
                components.append(component)

    # Add file hashes
    hashes = _calculate_hashes(pack_path)
    for file_path, file_hash in hashes.items():
        component = {
            "type": "file",
            "bom-ref": str(file_path),
            "name": file_path.name,
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
        }
        components.append(component)

    sbom["components"] = components

    # Add vulnerabilities section (empty for now)
    sbom["vulnerabilities"] = []

    # Save SBOM
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2, ensure_ascii=False)

    return str(output_path)


def _generate_spdx_sbom(pack_path: Path, output_path: Path) -> str:
    """
    Generate SBOM in SPDX format

    Args:
        pack_path: Path to pack directory
        output_path: Where to save SBOM

    Returns:
        Path to generated SBOM file
    """
    from ..packs.manifest import load_manifest
    import yaml

    # Load pack manifest
    try:
        manifest = load_manifest(pack_path)
        # Convert to dict if it's a Pydantic model
        if hasattr(manifest, "model_dump"):
            manifest_dict = manifest.model_dump()
        elif hasattr(manifest, "dict"):
            manifest_dict = manifest.dict()
        else:
            manifest_dict = manifest.__dict__
    except:
        # Fallback to basic manifest
        manifest_path = pack_path / "pack.yaml"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
                manifest_dict = yaml.safe_load(f)
        else:
            manifest_dict = {}

    # Extract manifest fields
    pack_name = manifest_dict.get("name", "unknown")
    pack_version = manifest_dict.get("version", "0.0.0")
    pack_license = manifest_dict.get("license", "NOASSERTION")
    pack_description = manifest_dict.get("description", "")
    pack_org = manifest_dict.get("org", "greenlang")

    # Create SPDX SBOM
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{pack_name}-{pack_version}",
        "documentNamespace": f"https://greenlang.io/sbom/{pack_org}/{pack_name}-{pack_version}-{_generate_uuid()[:8]}",
        "creationInfo": {
            "created": datetime.now().isoformat(),
            "creators": [
                "Tool: GreenLang-SBOM-Generator-0.1.0",
                f"Organization: {pack_org}",
            ],
            "licenseListVersion": "3.19",
        },
        "packages": [],
    }

    # Add main package
    main_package = {
        "SPDXID": f"SPDXRef-Package-{pack_name}",
        "name": pack_name,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": True,
        "licenseConcluded": pack_license,
        "licenseDeclared": pack_license,
        "copyrightText": f"Copyright (c) {datetime.now().year} {pack_org}",
        "versionInfo": pack_version,
        "description": pack_description,
        "externalRefs": [],
    }

    # Add dependencies
    dependencies = manifest_dict.get("dependencies", [])
    dep_packages = []
    relationships = []

    for dep in dependencies:
        if isinstance(dep, str):
            dep_name = dep
            dep_version = "*"
        elif isinstance(dep, dict):
            dep_name = dep.get("name", "unknown")
            dep_version = dep.get("version", "*")
        else:
            continue

        dep_package = {
            "SPDXID": f"SPDXRef-Package-{dep_name}",
            "name": dep_name,
            "downloadLocation": f"https://pypi.org/project/{dep_name}/",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "versionInfo": dep_version,
        }
        dep_packages.append(dep_package)

        # Add relationship
        relationships.append(
            {
                "spdxElementId": f"SPDXRef-Package-{pack_name}",
                "relatedSpdxElement": f"SPDXRef-Package-{dep_name}",
                "relationshipType": "DEPENDS_ON",
            }
        )

    # Add file information
    files = []
    file_hashes = _calculate_hashes(pack_path)

    for file_path, file_hash in file_hashes.items():
        file_info = {
            "SPDXID": f"SPDXRef-File-{file_hash[:8]}",
            "fileName": f"./{file_path}",
            "checksums": [{"algorithm": "SHA256", "checksumValue": file_hash}],
            "licenseConcluded": "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
        files.append(file_info)

        # Add file relationship
        relationships.append(
            {
                "spdxElementId": f"SPDXRef-Package-{pack_name}",
                "relatedSpdxElement": f"SPDXRef-File-{file_hash[:8]}",
                "relationshipType": "CONTAINS",
            }
        )

    # Add files to main package
    main_package["files"] = files
    main_package["packageVerificationCode"] = {
        "packageVerificationCodeValue": _calculate_package_verification_code(
            file_hashes
        )
    }

    # Add all packages
    sbom["packages"] = [main_package] + dep_packages

    # Add relationships
    sbom["relationships"] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relatedSpdxElement": f"SPDXRef-Package-{pack_name}",
            "relationshipType": "DESCRIBES",
        }
    ] + relationships

    # Add extracted licenses if custom
    if pack_license not in ["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0", "LGPL-3.0"]:
        sbom["hasExtractedLicensingInfos"] = [
            {
                "licenseId": f"LicenseRef-{pack_license}",
                "extractedText": f"Custom license: {pack_license}",
                "name": pack_license,
            }
        ]

    # Save SBOM
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated SPDX SBOM: {output_path}")
    return str(output_path)


def _calculate_package_verification_code(file_hashes: Dict[Path, str]) -> str:
    """Calculate package verification code per SPDX spec"""
    # Sort hashes and concatenate
    sorted_hashes = sorted(file_hashes.values())
    combined = "".join(sorted_hashes)

    # Calculate SHA256 of combined hashes
    hasher = hashlib.sha256()
    hasher.update(combined.encode())
    return hasher.hexdigest()


def _has_cyclonedx() -> bool:
    """Check if CycloneDX is installed"""
    try:
        result = subprocess.run(
            ["cyclonedx-py", "--version"],
            capture_output=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _generate_with_cyclonedx(pack_path: Path, output_path: Path) -> str:
    """Generate SBOM using CycloneDX"""
    # Look for requirements files
    req_files = [
        pack_path / "requirements.txt",
        pack_path / "pyproject.toml",
        pack_path / "setup.py",
        pack_path / "Pipfile",
    ]

    req_file = None
    for f in req_files:
        if f.exists():
            req_file = f
            break

    if not req_file:
        # Create temporary requirements from current environment
        req_file = pack_path / ".temp_requirements.txt"
        with open(req_file, "w", encoding="utf-8") as stdout_file:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                stdout=stdout_file,
                cwd=pack_path,
                encoding="utf-8",
                errors="replace",
            )

    try:
        # Generate SBOM with CycloneDX
        cmd = [
            "cyclonedx-py",
            "-i",
            str(req_file),
            "-o",
            str(output_path),
            "--format",
            "json",
            "--schema-version",
            "1.4",
        ]

        if req_file.suffix == ".toml":
            cmd.extend(["--pyproject", str(req_file)])
        elif req_file.name == "Pipfile":
            cmd.extend(["--pipfile", str(req_file)])

        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

        if result.returncode != 0:
            logger.error(f"CycloneDX failed: {result.stderr}")
            raise RuntimeError(f"CycloneDX generation failed: {result.stderr}")

        logger.info(f"Generated SBOM with CycloneDX: {output_path}")

        # Enhance with additional metadata
        _enhance_sbom(output_path, pack_path)

        return str(output_path)

    finally:
        # Clean up temp file if created
        if req_file.name == ".temp_requirements.txt":
            req_file.unlink(missing_ok=True)


def _generate_manual_sbom(pack_path: Path, output_path: Path) -> str:
    """Manual SBOM generation (fallback)"""
    # Load pack manifest
    manifest_path = pack_path / "pack.yaml"

    if manifest_path.exists():
        import yaml

        with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
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
                {"vendor": "GreenLang", "name": "greenlang-sbom", "version": "0.1.0"}
            ],
            "component": {
                "type": "application",
                "bom-ref": manifest.get("name", "unknown"),
                "name": manifest.get("name", "unknown"),
                "version": manifest.get("version", "0.0.0"),
                "description": manifest.get("description", ""),
                "licenses": [{"license": {"id": manifest.get("license", "MIT")}}],
            },
        },
        "components": [],
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
    if pkg_resources:
        for dist in pkg_resources.working_set:
            component = {
                "type": "library",
                "bom-ref": f"pkg:pypi/{dist.project_name}@{dist.version}",
                "name": dist.project_name,
                "version": dist.version,
                "purl": f"pkg:pypi/{dist.project_name}@{dist.version}",
                "scope": "required",
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
                    "scope": export_type,
                }
                components.append(component)

    # Add file hashes
    hashes = _calculate_hashes(pack_path)
    for file_path, file_hash in hashes.items():
        component = {
            "type": "file",
            "bom-ref": str(file_path),
            "name": file_path.name,
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
        }
        components.append(component)

    sbom["components"] = components

    # Add vulnerabilities section (empty for now)
    sbom["vulnerabilities"] = []

    # Save SBOM
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2, ensure_ascii=False)

    return str(output_path)


def _enhance_sbom(sbom_path: Path, pack_path: Path):
    """Enhance SBOM with additional metadata"""
    with open(sbom_path, "r", encoding="utf-8", errors="replace") as f:
        sbom = json.load(f)

    # Add file hashes
    hashes = _calculate_hashes(pack_path)

    # Add file components
    for file_path, file_hash in hashes.items():
        component = {
            "type": "file",
            "bom-ref": str(file_path),
            "name": file_path.name,
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
        }
        sbom.setdefault("components", []).append(component)

    # Add GreenLang metadata
    sbom.setdefault("metadata", {}).setdefault("properties", []).extend(
        [
            {"name": "greenlang:generated_by", "value": "greenlang-sbom"},
            {"name": "greenlang:pack_path", "value": str(pack_path)},
            {"name": "greenlang:timestamp", "value": datetime.now().isoformat()},
        ]
    )

    # Save enhanced SBOM
    with open(sbom_path, "w", encoding="utf-8") as f:
        json.dump(sbom, f, indent=2)


def verify_sbom(sbom_path: Path, pack_path: Path) -> bool:
    """
    Verify SBOM against actual pack contents

    Args:
        sbom_path: Path to SBOM file
        pack_path: Path to pack directory

    Returns:
        True if SBOM matches pack contents
    """
    with open(sbom_path, "r", encoding="utf-8", errors="replace") as f:
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
        "scope": "required",
    }


def _calculate_hashes(directory: Path) -> Dict[Path, str]:
    """Calculate SHA-256 hashes for all files"""
    hashes = {}

    for file_path in directory.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Skip certain files
            if file_path.suffix in [".pyc", ".pyo"]:
                continue
            if "__pycache__" in str(file_path):
                continue

            # Calculate hash
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
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
