"""
Pack Installer with Capability Validation

Validates and installs packs with security capability checks.
"""

import json
import logging
import shutil
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .manifest import PackManifest, Capabilities
from ..provenance import (
    SignatureVerifier,
    DevKeyVerifier,
    UnsignedPackError,
    verify_pack_signature,
    create_verifier
)
from ..auth.audit import audit_log as _audit_log_impl

# Wrapper for audit logging with string event types
def audit_log(event_type: str, data: Dict[str, Any]):
    """Log audit event with string type for simplicity"""
    try:
        _audit_log_impl(event_type, **data)
    except Exception as e:
        logger.warning(f"Failed to log audit event: {e}")

logger = logging.getLogger(__name__)


class PackInstaller:
    """
    Handles pack installation with capability validation
    """

    def __init__(self, packs_dir: Optional[Path] = None):
        """
        Initialize pack installer

        Args:
            packs_dir: Directory where packs are installed
        """
        self.packs_dir = packs_dir or Path.home() / ".greenlang" / "packs"
        self.packs_dir.mkdir(parents=True, exist_ok=True)

    def validate_manifest(self, manifest_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate pack manifest including capabilities

        Args:
            manifest_path: Path to pack.yaml or manifest.yaml

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        try:
            # Load and parse manifest
            manifest = PackManifest.from_file(manifest_path)

            # Check capabilities if present
            if manifest.capabilities:
                cap_issues = self._validate_capabilities(manifest.capabilities)
                issues.extend(cap_issues)

            # Check for dangerous patterns
            security_issues = self._check_security_issues(manifest)
            issues.extend(security_issues)

            # Get warnings for best practices
            warnings = manifest.get_warnings()
            for warning in warnings:
                issues.append(f"Warning: {warning}")

            # Validate referenced files exist
            base_path = manifest_path.parent
            missing_files = manifest.validate_files_exist(base_path)
            issues.extend(missing_files)

        except Exception as e:
            issues.append(f"Error parsing manifest: {e}")
            return False, issues

        return len([i for i in issues if not i.startswith("Warning:")]) == 0, issues

    def _validate_capabilities(self, capabilities: Capabilities) -> List[str]:
        """
        Validate capability declarations

        Args:
            capabilities: Capabilities object from manifest

        Returns:
            List of validation issues
        """
        issues = []

        # Check network capabilities
        if capabilities.net and capabilities.net.allow:
            if capabilities.net.outbound:
                allowlist = capabilities.net.outbound.get('allowlist', [])
                if not allowlist:
                    issues.append("Network capability enabled but no allowlist specified")

                for domain in allowlist:
                    if domain == '*':
                        issues.append("Wildcard '*' domain is too permissive")
                    elif domain.startswith('http://'):
                        issues.append(f"Insecure HTTP in allowlist: {domain}")
                    elif '..' in domain:
                        issues.append(f"Invalid domain pattern: {domain}")

        # Check filesystem capabilities
        if capabilities.fs and capabilities.fs.allow:
            # Check read allowlist
            if capabilities.fs.read:
                read_list = capabilities.fs.read.get('allowlist', [])
                if not read_list:
                    issues.append("Filesystem read capability enabled but no allowlist specified")

                for path in read_list:
                    if path == '/**':
                        issues.append("Root filesystem read access is too permissive")
                    elif path == '${HOME}/**':
                        issues.append("Home directory read access is too permissive")
                    elif '/etc/**' in path:
                        issues.append("System configuration read access is dangerous")
                    elif '..' in path:
                        issues.append(f"Path traversal pattern detected: {path}")

            # Check write allowlist
            if capabilities.fs.write:
                write_list = capabilities.fs.write.get('allowlist', [])
                if not write_list:
                    issues.append("Filesystem write capability enabled but no allowlist specified")

                for path in write_list:
                    if path == '/**':
                        issues.append("Root filesystem write access is not allowed")
                    elif '${HOME}' in path and '${RUN_TMP}' not in path:
                        issues.append("Writing to home directory outside RUN_TMP is dangerous")
                    elif '/etc' in path:
                        issues.append("Writing to system configuration is not allowed")
                    elif '/usr' in path or '/bin' in path:
                        issues.append("Writing to system directories is not allowed")
                    elif '..' in path:
                        issues.append(f"Path traversal pattern detected: {path}")

        # Check subprocess capabilities
        if capabilities.subprocess and capabilities.subprocess.allow:
            binaries = capabilities.subprocess.allowlist
            if not binaries:
                issues.append("Subprocess capability enabled but no binaries specified")

            dangerous_binaries = [
                '/bin/sh', '/bin/bash', '/bin/zsh', '/usr/bin/sh',
                '/usr/bin/bash', '/usr/bin/zsh', '/bin/dash',
                '/usr/bin/python', '/usr/bin/python3', '/usr/bin/perl',
                '/usr/bin/ruby', '/usr/bin/node', '/usr/bin/php',
                '/bin/nc', '/usr/bin/nc', '/usr/bin/netcat',
                '/usr/bin/curl', '/usr/bin/wget',
                '/usr/bin/sudo', '/bin/su', '/usr/bin/su',
                '/sbin/iptables', '/usr/sbin/iptables',
            ]

            for binary in binaries:
                if not binary.startswith('/'):
                    issues.append(f"Binary path must be absolute: {binary}")
                elif binary in dangerous_binaries:
                    issues.append(f"Dangerous binary in allowlist: {binary}")
                elif '*' in binary:
                    issues.append(f"Wildcards not allowed in binary paths: {binary}")

        return issues

    def _check_security_issues(self, manifest: PackManifest) -> List[str]:
        """
        Check for security issues in manifest

        Args:
            manifest: PackManifest object

        Returns:
            List of security issues
        """
        issues = []

        # Check if capabilities are missing (implies deny-all)
        if not manifest.capabilities:
            issues.append("Info: No capabilities specified - all access denied by default")

        # Check license
        if manifest.license in ['UNLICENSED', 'Proprietary']:
            issues.append("Warning: Proprietary license may have legal implications")

        # Check for SBOM
        if not manifest.security or not manifest.security.sbom:
            issues.append("Warning: No SBOM (Software Bill of Materials) provided")

        # Check for signatures
        if not manifest.security or not manifest.security.signatures:
            issues.append("Warning: No digital signatures provided")

        return issues

    def lint_capabilities(self, manifest_path: Path) -> str:
        """
        Lint capabilities and provide a formatted report

        Args:
            manifest_path: Path to pack manifest

        Returns:
            Formatted lint report
        """
        try:
            manifest = PackManifest.from_file(manifest_path)
        except Exception as e:
            return f"Error loading manifest: {e}"

        report = []
        report.append(f"Pack: {manifest.name} v{manifest.version}")
        report.append("=" * 50)
        report.append("\nCapabilities Summary:")
        report.append("-" * 20)

        if not manifest.capabilities:
            report.append("✓ No capabilities requested (deny-all by default)")
        else:
            caps = manifest.capabilities

            # Network
            if caps.net and caps.net.allow:
                report.append("⚠️  NETWORK: Enabled")
                if caps.net.outbound:
                    domains = caps.net.outbound.get('allowlist', [])
                    report.append(f"   Allowed domains: {len(domains)}")
                    for domain in domains[:5]:  # Show first 5
                        report.append(f"     - {domain}")
                    if len(domains) > 5:
                        report.append(f"     ... and {len(domains) - 5} more")
            else:
                report.append("✓ NETWORK: Disabled")

            # Filesystem
            if caps.fs and caps.fs.allow:
                report.append("⚠️  FILESYSTEM: Enabled")
                if caps.fs.read:
                    read_paths = caps.fs.read.get('allowlist', [])
                    report.append(f"   Read paths: {len(read_paths)}")
                    for path in read_paths[:3]:
                        report.append(f"     - {path}")
                if caps.fs.write:
                    write_paths = caps.fs.write.get('allowlist', [])
                    report.append(f"   Write paths: {len(write_paths)}")
                    for path in write_paths[:3]:
                        report.append(f"     - {path}")
            else:
                report.append("✓ FILESYSTEM: Disabled")

            # Clock
            if caps.clock and caps.clock.allow:
                report.append("⚠️  CLOCK: Real-time enabled")
            else:
                report.append("✓ CLOCK: Frozen/deterministic")

            # Subprocess
            if caps.subprocess and caps.subprocess.allow:
                report.append("⚠️  SUBPROCESS: Enabled")
                binaries = caps.subprocess.allowlist
                report.append(f"   Allowed binaries: {len(binaries)}")
                for binary in binaries[:3]:
                    report.append(f"     - {binary}")
            else:
                report.append("✓ SUBPROCESS: Disabled")

        # Validation issues
        is_valid, issues = self.validate_manifest(manifest_path)
        if issues:
            report.append("\n\nValidation Results:")
            report.append("-" * 20)
            for issue in issues:
                if issue.startswith("Warning:"):
                    report.append(f"⚠️  {issue}")
                elif issue.startswith("Info:"):
                    report.append(f"ℹ️  {issue}")
                else:
                    report.append(f"❌ {issue}")

        if is_valid:
            report.append("\n✅ Manifest is valid")
        else:
            report.append("\n❌ Manifest has errors")

        return "\n".join(report)

    def install_pack(self, pack_path: Path,
                      validate_capabilities: bool = True,
                      allow_unsigned: bool = False,
                      verifier: Optional[SignatureVerifier] = None) -> Tuple[bool, str]:
        """
        Install a pack with capability and signature validation

        Args:
            pack_path: Path to pack directory or archive
            validate_capabilities: Whether to validate capabilities
            allow_unsigned: Whether to allow unsigned packs (DANGEROUS!)
            verifier: Optional signature verifier to use

        Returns:
            (success, message)

        Raises:
            UnsignedPackError: If pack is unsigned and allow_unsigned is False
        """
        # SECURITY: Verify pack signature (default-deny unsigned packs)
        if not allow_unsigned:
            if not verifier:
                verifier = create_verifier("dev")  # Default to dev verifier for now

            # Check for pack archive or manifest
            if pack_path.is_file():
                # Archive file - verify signature
                is_valid, sig_msg = verify_pack_signature(pack_path, verifier)
                if not is_valid:
                    error_msg = f"SECURITY: Signature verification failed - {sig_msg}"
                    logger.error(error_msg)
                    audit_log("PACK_INSTALL_DENIED", {
                        "pack": str(pack_path),
                        "reason": "signature_verification_failed",
                        "message": sig_msg
                    })
                    raise UnsignedPackError(f"{error_msg}\nUse --allow-unsigned to bypass (not recommended)")
                logger.info(f"✅ Signature verified for {pack_path.name}")
            else:
                # Directory install - check for signature file
                sig_file = pack_path / "pack.sig"
                if not sig_file.exists():
                    error_msg = "SECURITY: No signature file found (pack.sig required)"
                    logger.error(error_msg)
                    audit_log("PACK_INSTALL_DENIED", {
                        "pack": str(pack_path),
                        "reason": "no_signature_file"
                    })
                    raise UnsignedPackError(f"{error_msg}\nUse --allow-unsigned to bypass (not recommended)")
        else:
            # SECURITY WARNING: Bypassing signature verification
            logger.warning("⚠️  SECURITY WARNING: Installing unsigned pack (--allow-unsigned used)")
            logger.warning("⚠️  This pack has not been cryptographically verified!")
            audit_log("PACK_INSTALL_UNSIGNED", {
                "pack": str(pack_path),
                "warning": "signature_verification_bypassed",
                "allow_unsigned": True
            })

        # Find manifest file
        if pack_path.is_dir():
            manifest_path = pack_path / "pack.yaml"
            if not manifest_path.exists():
                manifest_path = pack_path / "manifest.yaml"
                if not manifest_path.exists():
                    return False, "No pack.yaml or manifest.yaml found"
        else:
            # Extract archive first if it's a file
            import tarfile
            import zipfile

            temp_dir = Path(tempfile.mkdtemp())
            try:
                if pack_path.suffix == '.tar' or pack_path.suffixes[-2:] == ['.tar', '.gz']:
                    with tarfile.open(pack_path) as tar:
                        tar.extractall(temp_dir)
                elif pack_path.suffix == '.zip':
                    with zipfile.ZipFile(pack_path) as zip_file:
                        zip_file.extractall(temp_dir)
                else:
                    return False, f"Unsupported archive format: {pack_path.suffix}"

                # Find manifest in extracted files
                pack_path = temp_dir
                manifest_path = pack_path / "pack.yaml"
                if not manifest_path.exists():
                    manifest_path = pack_path / "manifest.yaml"
                    if not manifest_path.exists():
                        return False, "No pack.yaml or manifest.yaml found in archive"
            except Exception as e:
                return False, f"Failed to extract archive: {e}"

        # Validate manifest
        if validate_capabilities:
            is_valid, issues = self.validate_manifest(manifest_path)
            if not is_valid:
                error_issues = [i for i in issues if not i.startswith("Warning:")]
                return False, f"Manifest validation failed:\n" + "\n".join(error_issues)

        try:
            # Load manifest
            manifest = PackManifest.from_file(manifest_path)

            # Create installation directory
            install_dir = self.packs_dir / manifest.name / manifest.version
            if install_dir.exists():
                return False, f"Pack {manifest.name} v{manifest.version} already installed"

            install_dir.mkdir(parents=True)

            # Copy pack files
            shutil.copytree(pack_path, install_dir, dirs_exist_ok=True)

            # Write installation metadata
            metadata = {
                'name': manifest.name,
                'version': manifest.version,
                'installed_at': datetime.datetime.utcnow().isoformat(),
                'capabilities': manifest.capabilities.model_dump() if manifest.capabilities else {},
                'signature_verified': not allow_unsigned,
                'verifier_type': verifier.get_verifier_info()['type'] if verifier else 'none',
            }

            with open(install_dir / '.installation.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Installed pack {manifest.name} v{manifest.version}")
            return True, f"Successfully installed {manifest.name} v{manifest.version}"

        except Exception as e:
            logger.error(f"Failed to install pack: {e}")
            return False, f"Installation failed: {e}"

    def list_installed_packs(self) -> List[Dict]:
        """
        List all installed packs with their capabilities

        Returns:
            List of pack info dictionaries
        """
        packs = []

        for pack_dir in self.packs_dir.iterdir():
            if pack_dir.is_dir():
                for version_dir in pack_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_file = version_dir / '.installation.json'
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                                packs.append(metadata)

        return packs

    def uninstall_pack(self, name: str, version: Optional[str] = None) -> Tuple[bool, str]:
        """
        Uninstall a pack

        Args:
            name: Pack name
            version: Pack version (if None, uninstall all versions)

        Returns:
            (success, message)
        """
        pack_dir = self.packs_dir / name

        if not pack_dir.exists():
            return False, f"Pack {name} not found"

        if version:
            version_dir = pack_dir / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
                # Remove pack dir if empty
                if not any(pack_dir.iterdir()):
                    pack_dir.rmdir()
                return True, f"Uninstalled {name} v{version}"
            else:
                return False, f"Pack {name} v{version} not found"
        else:
            shutil.rmtree(pack_dir)
            return True, f"Uninstalled all versions of {name}"

