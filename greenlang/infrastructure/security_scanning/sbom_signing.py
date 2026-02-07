"""
SBOM Signing Module

Provides functionality to sign CycloneDX SBOMs with Cosign and attach
them as OCI attestations to container images.

Features:
    - Sign CycloneDX SBOM with Cosign keyless signing
    - Attach SBOM as OCI attestation to container images
    - Store signed SBOMs in OCI registry
    - Verify SBOM signatures

Example:
    >>> from greenlang.infrastructure.security_scanning.sbom_signing import (
    ...     SBOMSigner,
    ...     SBOMSigningConfig,
    ... )
    >>> config = SBOMSigningConfig(
    ...     oidc_issuer="https://token.actions.githubusercontent.com"
    ... )
    >>> signer = SBOMSigner(config)
    >>> result = await signer.sign_sbom("sbom.json", "ghcr.io/org/image:tag")

Author: GreenLang Security Team
Version: 1.0.0
Compliance: SOC 2 CC7.1, SLSA Level 2
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SBOMFormat(str, Enum):
    """Supported SBOM formats."""

    CYCLONEDX_JSON = "cyclonedx-json"
    CYCLONEDX_XML = "cyclonedx-xml"
    SPDX_JSON = "spdx-json"
    SPDX_TAG_VALUE = "spdx-tag-value"


class SigningMethod(str, Enum):
    """Cosign signing methods."""

    KEYLESS = "keyless"
    KEY_FILE = "key-file"
    KMS = "kms"


class SBOMSigningConfig(BaseModel):
    """Configuration for SBOM signing operations.

    Attributes:
        oidc_issuer: OIDC issuer URL for keyless signing
        rekor_url: Rekor transparency log URL
        signing_method: Method to use for signing
        key_path: Path to signing key (if using KEY_FILE method)
        kms_key: KMS key URI (if using KMS method)
        timeout_seconds: Timeout for signing operations
        verify_after_sign: Whether to verify signature after signing
    """

    oidc_issuer: str = Field(
        default="https://token.actions.githubusercontent.com",
        description="OIDC issuer for keyless signing",
    )
    rekor_url: str = Field(
        default="https://rekor.sigstore.dev",
        description="Rekor transparency log URL",
    )
    signing_method: SigningMethod = Field(
        default=SigningMethod.KEYLESS,
        description="Signing method to use",
    )
    key_path: Optional[str] = Field(
        default=None,
        description="Path to signing key file",
    )
    kms_key: Optional[str] = Field(
        default=None,
        description="KMS key URI (e.g., awskms://arn:aws:kms:...)",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for signing operations",
    )
    verify_after_sign: bool = Field(
        default=True,
        description="Verify signature after signing",
    )
    annotations: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional annotations to include in signature",
    )

    @validator("key_path")
    def validate_key_path(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate key path is provided for KEY_FILE method."""
        if values.get("signing_method") == SigningMethod.KEY_FILE and not v:
            raise ValueError("key_path required for KEY_FILE signing method")
        return v

    @validator("kms_key")
    def validate_kms_key(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate KMS key is provided for KMS method."""
        if values.get("signing_method") == SigningMethod.KMS and not v:
            raise ValueError("kms_key required for KMS signing method")
        return v


class SBOMSigningResult(BaseModel):
    """Result of an SBOM signing operation.

    Attributes:
        success: Whether signing was successful
        image_reference: Full image reference with digest
        sbom_digest: SHA-256 digest of the signed SBOM
        signature_digest: Digest of the signature
        rekor_log_id: Rekor transparency log entry ID
        signed_at: Timestamp of signing
        error_message: Error message if signing failed
        verification_passed: Whether post-sign verification passed
    """

    success: bool = Field(..., description="Whether signing succeeded")
    image_reference: str = Field(..., description="Image reference")
    sbom_digest: str = Field(default="", description="SBOM content digest")
    signature_digest: str = Field(default="", description="Signature digest")
    rekor_log_id: str = Field(default="", description="Rekor log entry ID")
    signed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Signing timestamp",
    )
    error_message: str = Field(default="", description="Error message if failed")
    verification_passed: bool = Field(
        default=False,
        description="Post-sign verification result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )


@dataclass
class SBOMSigner:
    """Signs CycloneDX SBOMs with Cosign and attaches them as OCI attestations.

    This class provides functionality to sign SBOMs using Cosign's keyless
    signing via Sigstore/Fulcio and attach them as attestations to container
    images in OCI registries.

    Attributes:
        config: Signing configuration

    Example:
        >>> config = SBOMSigningConfig()
        >>> signer = SBOMSigner(config)
        >>> result = await signer.sign_sbom("sbom.json", "ghcr.io/org/image:tag")
        >>> if result.success:
        ...     print(f"Signed at: {result.signed_at}")
    """

    config: SBOMSigningConfig = field(default_factory=SBOMSigningConfig)
    _cosign_available: Optional[bool] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate Cosign is available."""
        self._check_cosign_availability()

    def _check_cosign_availability(self) -> bool:
        """Check if Cosign CLI is available.

        Returns:
            True if Cosign is available, False otherwise.
        """
        if self._cosign_available is not None:
            return self._cosign_available

        try:
            result = subprocess.run(
                ["cosign", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._cosign_available = result.returncode == 0
            if self._cosign_available:
                logger.info(f"Cosign available: {result.stdout.strip()}")
            else:
                logger.warning("Cosign not available or not functioning")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Cosign check failed: {e}")
            self._cosign_available = False

        return self._cosign_available

    async def sign_sbom(
        self,
        sbom_path: Union[str, Path],
        image_reference: str,
        sbom_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    ) -> SBOMSigningResult:
        """Sign an SBOM and attach it as an attestation to a container image.

        Args:
            sbom_path: Path to the SBOM file
            image_reference: Container image reference (with tag or digest)
            sbom_format: Format of the SBOM file

        Returns:
            SBOMSigningResult with signing details

        Raises:
            ValueError: If SBOM file doesn't exist or is invalid
            RuntimeError: If Cosign is not available
        """
        start_time = datetime.utcnow()

        # Validate prerequisites
        if not self._check_cosign_availability():
            return SBOMSigningResult(
                success=False,
                image_reference=image_reference,
                error_message="Cosign CLI is not available",
            )

        sbom_path = Path(sbom_path)
        if not sbom_path.exists():
            return SBOMSigningResult(
                success=False,
                image_reference=image_reference,
                error_message=f"SBOM file not found: {sbom_path}",
            )

        try:
            # Calculate SBOM digest
            sbom_content = sbom_path.read_bytes()
            sbom_digest = hashlib.sha256(sbom_content).hexdigest()

            # Validate SBOM format
            if sbom_format in [SBOMFormat.CYCLONEDX_JSON, SBOMFormat.SPDX_JSON]:
                sbom_data = json.loads(sbom_content)
                self._validate_sbom(sbom_data, sbom_format)

            # Sign and attest
            result = await self._sign_and_attest(
                sbom_path=sbom_path,
                image_reference=image_reference,
                sbom_format=sbom_format,
            )

            # Verify if configured
            verification_passed = False
            if self.config.verify_after_sign and result.get("success"):
                verification_passed = await self._verify_attestation(
                    image_reference=image_reference,
                    sbom_format=sbom_format,
                )

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SBOMSigningResult(
                success=result.get("success", False),
                image_reference=image_reference,
                sbom_digest=f"sha256:{sbom_digest}",
                signature_digest=result.get("signature_digest", ""),
                rekor_log_id=result.get("rekor_log_id", ""),
                signed_at=datetime.utcnow(),
                verification_passed=verification_passed,
                processing_time_ms=processing_time,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid SBOM JSON: {e}")
            return SBOMSigningResult(
                success=False,
                image_reference=image_reference,
                error_message=f"Invalid SBOM JSON: {e}",
            )
        except Exception as e:
            logger.error(f"SBOM signing failed: {e}", exc_info=True)
            return SBOMSigningResult(
                success=False,
                image_reference=image_reference,
                error_message=str(e),
            )

    def _validate_sbom(self, sbom_data: Dict[str, Any], sbom_format: SBOMFormat) -> None:
        """Validate SBOM structure.

        Args:
            sbom_data: Parsed SBOM data
            sbom_format: Expected SBOM format

        Raises:
            ValueError: If SBOM is invalid
        """
        if sbom_format == SBOMFormat.CYCLONEDX_JSON:
            if "bomFormat" not in sbom_data or sbom_data["bomFormat"] != "CycloneDX":
                raise ValueError("Invalid CycloneDX SBOM: missing or invalid bomFormat")
            if "specVersion" not in sbom_data:
                raise ValueError("Invalid CycloneDX SBOM: missing specVersion")
            if "components" not in sbom_data:
                logger.warning("CycloneDX SBOM has no components")

        elif sbom_format == SBOMFormat.SPDX_JSON:
            if "spdxVersion" not in sbom_data:
                raise ValueError("Invalid SPDX SBOM: missing spdxVersion")
            if "SPDXID" not in sbom_data:
                raise ValueError("Invalid SPDX SBOM: missing SPDXID")

    async def _sign_and_attest(
        self,
        sbom_path: Path,
        image_reference: str,
        sbom_format: SBOMFormat,
    ) -> Dict[str, Any]:
        """Execute Cosign attest command.

        Args:
            sbom_path: Path to SBOM file
            image_reference: Container image reference
            sbom_format: SBOM format

        Returns:
            Dict with signing result details
        """
        # Determine predicate type based on format
        if sbom_format in [SBOMFormat.CYCLONEDX_JSON, SBOMFormat.CYCLONEDX_XML]:
            predicate_type = "cyclonedx"
        else:
            predicate_type = "spdx"

        # Build Cosign command
        cmd = [
            "cosign",
            "attest",
            "--yes",
            "--predicate",
            str(sbom_path),
            "--type",
            predicate_type,
        ]

        # Add signing method specific args
        if self.config.signing_method == SigningMethod.KEYLESS:
            cmd.extend([
                "--oidc-issuer",
                self.config.oidc_issuer,
            ])
            os.environ["COSIGN_EXPERIMENTAL"] = "true"

        elif self.config.signing_method == SigningMethod.KEY_FILE:
            cmd.extend(["--key", self.config.key_path])

        elif self.config.signing_method == SigningMethod.KMS:
            cmd.extend(["--key", self.config.kms_key])

        # Add annotations
        for key, value in self.config.annotations.items():
            cmd.extend(["--annotation", f"{key}={value}"])

        # Add image reference
        cmd.append(image_reference)

        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                ),
            )

            if result.returncode == 0:
                logger.info(f"SBOM attestation successful for {image_reference}")
                # Parse Rekor log ID from output if available
                rekor_log_id = self._extract_rekor_id(result.stderr)
                return {
                    "success": True,
                    "rekor_log_id": rekor_log_id,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                logger.error(f"SBOM attestation failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"SBOM attestation timed out after {self.config.timeout_seconds}s")
            return {
                "success": False,
                "error": f"Timeout after {self.config.timeout_seconds} seconds",
            }

    def _extract_rekor_id(self, output: str) -> str:
        """Extract Rekor log entry ID from Cosign output.

        Args:
            output: Cosign command output

        Returns:
            Rekor log entry ID or empty string
        """
        # Look for Rekor log entry URL/ID in output
        import re

        patterns = [
            r"tlog entry created with index: (\d+)",
            r"rekor\.sigstore\.dev/api/v1/log/entries\?logIndex=(\d+)",
            r"LogIndex:\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)

        return ""

    async def _verify_attestation(
        self,
        image_reference: str,
        sbom_format: SBOMFormat,
    ) -> bool:
        """Verify SBOM attestation on image.

        Args:
            image_reference: Container image reference
            sbom_format: SBOM format

        Returns:
            True if verification passed
        """
        predicate_type = "cyclonedx" if "cyclonedx" in sbom_format.value else "spdx"

        cmd = [
            "cosign",
            "verify-attestation",
            "--type",
            predicate_type,
            "--certificate-oidc-issuer",
            self.config.oidc_issuer,
            image_reference,
        ]

        if self.config.signing_method == SigningMethod.KEYLESS:
            os.environ["COSIGN_EXPERIMENTAL"] = "true"

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                ),
            )

            if result.returncode == 0:
                logger.info(f"Attestation verification passed for {image_reference}")
                return True
            else:
                logger.warning(f"Attestation verification failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Attestation verification timed out")
            return False

    async def generate_sbom(
        self,
        image_reference: str,
        output_path: Union[str, Path],
        sbom_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    ) -> bool:
        """Generate SBOM for a container image using Syft.

        Args:
            image_reference: Container image reference
            output_path: Path to save generated SBOM
            sbom_format: Desired SBOM format

        Returns:
            True if generation succeeded
        """
        format_map = {
            SBOMFormat.CYCLONEDX_JSON: "cyclonedx-json",
            SBOMFormat.CYCLONEDX_XML: "cyclonedx-xml",
            SBOMFormat.SPDX_JSON: "spdx-json",
            SBOMFormat.SPDX_TAG_VALUE: "spdx-tag-value",
        }

        cmd = [
            "syft",
            image_reference,
            "-o",
            f"{format_map[sbom_format]}={output_path}",
        ]

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # SBOM generation can take time
                ),
            )

            if result.returncode == 0:
                logger.info(f"SBOM generated successfully: {output_path}")
                return True
            else:
                logger.error(f"SBOM generation failed: {result.stderr}")
                return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"SBOM generation failed: {e}")
            return False
