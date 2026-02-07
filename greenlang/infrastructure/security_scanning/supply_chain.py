"""
Supply Chain Verification Module

Provides comprehensive supply chain security verification for container images,
including signature verification, SBOM attestation validation, and SLSA
provenance checking.

Features:
    - Verify container image signatures (Cosign keyless/keyed)
    - Verify SBOM attestations
    - Verify SLSA provenance attestations
    - Integration with Rekor transparency log

Example:
    >>> from greenlang.infrastructure.security_scanning.supply_chain import (
    ...     SupplyChainVerifier,
    ...     SupplyChainConfig,
    ... )
    >>> config = SupplyChainConfig()
    >>> verifier = SupplyChainVerifier(config)
    >>> result = await verifier.verify_image("ghcr.io/org/image@sha256:...")
    >>> if result.signature_valid and result.sbom_valid:
    ...     print("Image verified successfully")

Author: GreenLang Security Team
Version: 1.0.0
Compliance: SOC 2 CC7.1, SLSA Level 2
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class VerificationLevel(str, Enum):
    """Verification strictness levels."""

    NONE = "none"  # No verification
    SIGNATURE_ONLY = "signature"  # Verify signature only
    SBOM = "sbom"  # Verify signature and SBOM
    FULL = "full"  # Verify signature, SBOM, and provenance


class TrustPolicy(str, Enum):
    """Trust policy for image sources."""

    REJECT_ALL = "reject"  # Reject all unsigned images
    WARN = "warn"  # Warn but allow unsigned
    ALLOW_UNSIGNED = "allow"  # Allow unsigned images


class SupplyChainConfig(BaseModel):
    """Configuration for supply chain verification.

    Attributes:
        verification_level: Level of verification to perform
        trust_policy: Policy for handling unsigned images
        oidc_issuer: Expected OIDC issuer for keyless signatures
        certificate_identity_regexp: Regex to match certificate identity
        rekor_url: Rekor transparency log URL
        trusted_registries: List of trusted container registries
        timeout_seconds: Timeout for verification operations
    """

    verification_level: VerificationLevel = Field(
        default=VerificationLevel.FULL,
        description="Level of verification to perform",
    )
    trust_policy: TrustPolicy = Field(
        default=TrustPolicy.REJECT_ALL,
        description="Policy for unsigned images",
    )
    oidc_issuer: str = Field(
        default="https://token.actions.githubusercontent.com",
        description="Expected OIDC issuer for signatures",
    )
    certificate_identity_regexp: str = Field(
        default="https://github.com/greenlang-io/.*",
        description="Regex to match certificate identity",
    )
    rekor_url: str = Field(
        default="https://rekor.sigstore.dev",
        description="Rekor transparency log URL",
    )
    trusted_registries: List[str] = Field(
        default_factory=lambda: [
            "ghcr.io/greenlang-io",
            "gcr.io/greenlang",
            "docker.io/greenlang",
        ],
        description="Trusted container registries",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Verification timeout",
    )
    cache_results: bool = Field(
        default=True,
        description="Cache verification results",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds",
    )


class SignatureInfo(BaseModel):
    """Information about an image signature.

    Attributes:
        verified: Whether signature is valid
        issuer: OIDC issuer of the signature
        subject: Certificate subject (identity)
        signed_at: Timestamp of signature
        rekor_log_index: Rekor log entry index
        certificate_fingerprint: Certificate fingerprint
        annotations: Signature annotations
    """

    verified: bool = Field(..., description="Signature validity")
    issuer: str = Field(default="", description="OIDC issuer")
    subject: str = Field(default="", description="Certificate subject")
    signed_at: Optional[datetime] = Field(None, description="Signing timestamp")
    rekor_log_index: Optional[int] = Field(None, description="Rekor log index")
    certificate_fingerprint: str = Field(default="", description="Certificate fingerprint")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")


class SBOMInfo(BaseModel):
    """Information about an SBOM attestation.

    Attributes:
        present: Whether SBOM attestation exists
        verified: Whether SBOM signature is valid
        format: SBOM format (cyclonedx, spdx)
        spec_version: SBOM specification version
        component_count: Number of components in SBOM
        components: List of component names/versions
        vulnerabilities: Number of known vulnerabilities
    """

    present: bool = Field(..., description="SBOM attestation exists")
    verified: bool = Field(default=False, description="SBOM signature valid")
    format: str = Field(default="", description="SBOM format")
    spec_version: str = Field(default="", description="Spec version")
    component_count: int = Field(default=0, description="Component count")
    components: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Top components",
    )
    vulnerabilities: int = Field(default=0, description="Known vulnerabilities")


class ProvenanceInfo(BaseModel):
    """Information about SLSA provenance attestation.

    Attributes:
        present: Whether provenance attestation exists
        verified: Whether provenance signature is valid
        slsa_level: SLSA compliance level
        builder_id: Builder identity
        build_type: Type of build
        source_uri: Source code URI
        source_digest: Source commit digest
        materials: Build materials
    """

    present: bool = Field(..., description="Provenance exists")
    verified: bool = Field(default=False, description="Provenance valid")
    slsa_level: int = Field(default=0, description="SLSA level (0-4)")
    builder_id: str = Field(default="", description="Builder identity")
    build_type: str = Field(default="", description="Build type")
    source_uri: str = Field(default="", description="Source URI")
    source_digest: str = Field(default="", description="Source commit")
    materials: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Build materials",
    )


class VerificationResult(BaseModel):
    """Complete verification result for an image.

    Attributes:
        image_reference: Full image reference
        image_digest: Image digest
        verified: Overall verification status
        signature: Signature verification details
        sbom: SBOM attestation details
        provenance: Provenance attestation details
        trusted_registry: Whether from trusted registry
        verification_time_ms: Verification duration
        errors: List of verification errors
        warnings: List of verification warnings
    """

    image_reference: str = Field(..., description="Image reference")
    image_digest: str = Field(default="", description="Image digest")
    verified: bool = Field(..., description="Overall verification passed")
    signature: SignatureInfo = Field(..., description="Signature details")
    sbom: Optional[SBOMInfo] = Field(None, description="SBOM details")
    provenance: Optional[ProvenanceInfo] = Field(None, description="Provenance details")
    trusted_registry: bool = Field(default=False, description="From trusted registry")
    verification_time_ms: float = Field(default=0.0, description="Verification time")
    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    verified_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Verification timestamp",
    )


@dataclass
class SupplyChainVerifier:
    """Verifies supply chain security for container images.

    This class provides comprehensive verification of container image
    signatures, SBOM attestations, and SLSA provenance attestations
    using Cosign and the Sigstore ecosystem.

    Attributes:
        config: Verification configuration

    Example:
        >>> config = SupplyChainConfig()
        >>> verifier = SupplyChainVerifier(config)
        >>> result = await verifier.verify_image("ghcr.io/org/image@sha256:...")
        >>> if result.verified:
        ...     print(f"Image verified with SLSA level {result.provenance.slsa_level}")
    """

    config: SupplyChainConfig = field(default_factory=SupplyChainConfig)
    _cache: Dict[str, Tuple[VerificationResult, datetime]] = field(
        default_factory=dict,
        init=False,
    )
    _cosign_available: Optional[bool] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize verifier and check Cosign availability."""
        self._check_cosign_availability()

    def _check_cosign_availability(self) -> bool:
        """Check if Cosign CLI is available."""
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
                logger.info("Cosign CLI available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._cosign_available = False
            logger.warning("Cosign CLI not available")

        return self._cosign_available

    async def verify_image(
        self,
        image_reference: str,
        skip_cache: bool = False,
    ) -> VerificationResult:
        """Verify supply chain security for a container image.

        Args:
            image_reference: Container image reference (with tag or digest)
            skip_cache: Skip cache lookup

        Returns:
            VerificationResult with complete verification details

        Example:
            >>> result = await verifier.verify_image("ghcr.io/org/image:v1.0.0")
            >>> if result.verified:
            ...     print("Image is trusted")
        """
        start_time = datetime.utcnow()

        # Check cache
        if not skip_cache and self.config.cache_results:
            cached = self._get_cached_result(image_reference)
            if cached:
                logger.debug(f"Cache hit for {image_reference}")
                return cached

        # Check Cosign availability
        if not self._check_cosign_availability():
            return VerificationResult(
                image_reference=image_reference,
                verified=False,
                signature=SignatureInfo(verified=False),
                errors=["Cosign CLI is not available"],
            )

        errors: List[str] = []
        warnings: List[str] = []

        # Check trusted registry
        trusted_registry = self._is_trusted_registry(image_reference)
        if not trusted_registry:
            warnings.append(f"Image not from trusted registry: {image_reference}")

        # Resolve digest if needed
        image_digest = await self._resolve_digest(image_reference)

        # Verify signature
        signature_info = await self._verify_signature(image_reference)
        if not signature_info.verified:
            if self.config.trust_policy == TrustPolicy.REJECT_ALL:
                errors.append("Signature verification failed")
            else:
                warnings.append("Signature verification failed")

        # Verify SBOM if configured
        sbom_info: Optional[SBOMInfo] = None
        if self.config.verification_level in [
            VerificationLevel.SBOM,
            VerificationLevel.FULL,
        ]:
            sbom_info = await self._verify_sbom_attestation(image_reference)
            if not sbom_info.verified and sbom_info.present:
                warnings.append("SBOM attestation signature invalid")

        # Verify provenance if configured
        provenance_info: Optional[ProvenanceInfo] = None
        if self.config.verification_level == VerificationLevel.FULL:
            provenance_info = await self._verify_provenance(image_reference)
            if not provenance_info.verified and provenance_info.present:
                warnings.append("Provenance attestation signature invalid")

        # Determine overall verification status
        verified = self._compute_verification_status(
            signature_info=signature_info,
            sbom_info=sbom_info,
            provenance_info=provenance_info,
            errors=errors,
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = VerificationResult(
            image_reference=image_reference,
            image_digest=image_digest,
            verified=verified,
            signature=signature_info,
            sbom=sbom_info,
            provenance=provenance_info,
            trusted_registry=trusted_registry,
            verification_time_ms=processing_time,
            errors=errors,
            warnings=warnings,
        )

        # Cache result
        if self.config.cache_results:
            self._cache_result(image_reference, result)

        return result

    async def verify_image_signature(
        self,
        image_reference: str,
    ) -> bool:
        """Verify only the image signature.

        Args:
            image_reference: Container image reference

        Returns:
            True if signature is valid
        """
        result = await self._verify_signature(image_reference)
        return result.verified

    async def verify_sbom_attestation(
        self,
        image_reference: str,
    ) -> Optional[SBOMInfo]:
        """Verify SBOM attestation and extract SBOM data.

        Args:
            image_reference: Container image reference

        Returns:
            SBOMInfo with SBOM details, or None if not present
        """
        return await self._verify_sbom_attestation(image_reference)

    async def verify_provenance(
        self,
        image_reference: str,
    ) -> Optional[ProvenanceInfo]:
        """Verify SLSA provenance attestation.

        Args:
            image_reference: Container image reference

        Returns:
            ProvenanceInfo with provenance details, or None if not present
        """
        return await self._verify_provenance(image_reference)

    def _is_trusted_registry(self, image_reference: str) -> bool:
        """Check if image is from a trusted registry.

        Args:
            image_reference: Container image reference

        Returns:
            True if from trusted registry
        """
        for registry in self.config.trusted_registries:
            if image_reference.startswith(registry):
                return True
        return False

    async def _resolve_digest(self, image_reference: str) -> str:
        """Resolve image reference to digest.

        Args:
            image_reference: Container image reference

        Returns:
            Image digest (sha256:...)
        """
        if "@sha256:" in image_reference:
            return image_reference.split("@")[1]

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["cosign", "triangulate", image_reference],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ),
            )
            if result.returncode == 0:
                # Extract digest from output
                match = re.search(r"sha256:[a-f0-9]{64}", result.stdout)
                if match:
                    return match.group(0)
        except Exception as e:
            logger.debug(f"Could not resolve digest: {e}")

        return ""

    async def _verify_signature(self, image_reference: str) -> SignatureInfo:
        """Verify image signature using Cosign.

        Args:
            image_reference: Container image reference

        Returns:
            SignatureInfo with verification details
        """
        cmd = [
            "cosign",
            "verify",
            "--certificate-oidc-issuer",
            self.config.oidc_issuer,
            "--certificate-identity-regexp",
            self.config.certificate_identity_regexp,
            "--output",
            "json",
            image_reference,
        ]

        os.environ["COSIGN_EXPERIMENTAL"] = "true"

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
                # Parse signature details from JSON output
                try:
                    sig_data = json.loads(result.stdout)
                    if sig_data and len(sig_data) > 0:
                        sig = sig_data[0]
                        optional = sig.get("optional", {})
                        return SignatureInfo(
                            verified=True,
                            issuer=optional.get("Issuer", ""),
                            subject=optional.get("Subject", ""),
                            annotations=optional.get("Annotations", {}),
                            rekor_log_index=optional.get("Bundle", {})
                            .get("Payload", {})
                            .get("logIndex"),
                        )
                except json.JSONDecodeError:
                    pass

                return SignatureInfo(verified=True)
            else:
                logger.debug(f"Signature verification failed: {result.stderr}")
                return SignatureInfo(verified=False)

        except subprocess.TimeoutExpired:
            logger.error("Signature verification timed out")
            return SignatureInfo(verified=False)
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return SignatureInfo(verified=False)

    async def _verify_sbom_attestation(
        self,
        image_reference: str,
    ) -> SBOMInfo:
        """Verify SBOM attestation on image.

        Args:
            image_reference: Container image reference

        Returns:
            SBOMInfo with SBOM details
        """
        cmd = [
            "cosign",
            "verify-attestation",
            "--type",
            "cyclonedx",
            "--certificate-oidc-issuer",
            self.config.oidc_issuer,
            "--certificate-identity-regexp",
            self.config.certificate_identity_regexp,
            image_reference,
        ]

        os.environ["COSIGN_EXPERIMENTAL"] = "true"

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
                # Parse SBOM from attestation
                try:
                    attestations = [
                        json.loads(line) for line in result.stdout.strip().split("\n") if line
                    ]
                    if attestations:
                        payload = attestations[0].get("payload", "")
                        sbom = json.loads(base64.b64decode(payload)).get("predicate", {})

                        components = sbom.get("components", [])
                        top_components = [
                            {
                                "name": c.get("name", ""),
                                "version": c.get("version", ""),
                            }
                            for c in components[:20]
                        ]

                        return SBOMInfo(
                            present=True,
                            verified=True,
                            format=sbom.get("bomFormat", "CycloneDX"),
                            spec_version=sbom.get("specVersion", ""),
                            component_count=len(components),
                            components=top_components,
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Could not parse SBOM: {e}")
                    return SBOMInfo(present=True, verified=True)

                return SBOMInfo(present=True, verified=True)
            else:
                # Check if attestation exists but failed verification
                if "no matching attestations" in result.stderr.lower():
                    return SBOMInfo(present=False, verified=False)
                return SBOMInfo(present=True, verified=False)

        except subprocess.TimeoutExpired:
            logger.error("SBOM verification timed out")
            return SBOMInfo(present=False, verified=False)
        except Exception as e:
            logger.error(f"SBOM verification error: {e}")
            return SBOMInfo(present=False, verified=False)

    async def _verify_provenance(
        self,
        image_reference: str,
    ) -> ProvenanceInfo:
        """Verify SLSA provenance attestation.

        Args:
            image_reference: Container image reference

        Returns:
            ProvenanceInfo with provenance details
        """
        cmd = [
            "cosign",
            "verify-attestation",
            "--type",
            "slsaprovenance",
            "--certificate-oidc-issuer",
            self.config.oidc_issuer,
            "--certificate-identity-regexp",
            self.config.certificate_identity_regexp,
            image_reference,
        ]

        os.environ["COSIGN_EXPERIMENTAL"] = "true"

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
                try:
                    attestations = [
                        json.loads(line) for line in result.stdout.strip().split("\n") if line
                    ]
                    if attestations:
                        payload = attestations[0].get("payload", "")
                        provenance = json.loads(base64.b64decode(payload)).get(
                            "predicate", {}
                        )

                        # Determine SLSA level from completeness
                        completeness = provenance.get("metadata", {}).get(
                            "completeness", {}
                        )
                        slsa_level = self._determine_slsa_level(completeness, provenance)

                        materials = provenance.get("materials", [])
                        material_list = [
                            {
                                "uri": m.get("uri", ""),
                                "digest": str(m.get("digest", {})),
                            }
                            for m in materials[:10]
                        ]

                        source = provenance.get("invocation", {}).get(
                            "configSource", {}
                        )

                        return ProvenanceInfo(
                            present=True,
                            verified=True,
                            slsa_level=slsa_level,
                            builder_id=provenance.get("builder", {}).get("id", ""),
                            build_type=provenance.get("buildType", ""),
                            source_uri=source.get("uri", ""),
                            source_digest=str(source.get("digest", {})),
                            materials=material_list,
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Could not parse provenance: {e}")
                    return ProvenanceInfo(present=True, verified=True, slsa_level=1)

                return ProvenanceInfo(present=True, verified=True)
            else:
                if "no matching attestations" in result.stderr.lower():
                    return ProvenanceInfo(present=False, verified=False)
                return ProvenanceInfo(present=True, verified=False)

        except subprocess.TimeoutExpired:
            logger.error("Provenance verification timed out")
            return ProvenanceInfo(present=False, verified=False)
        except Exception as e:
            logger.error(f"Provenance verification error: {e}")
            return ProvenanceInfo(present=False, verified=False)

    def _determine_slsa_level(
        self,
        completeness: Dict[str, bool],
        provenance: Dict[str, Any],
    ) -> int:
        """Determine SLSA compliance level from provenance data.

        Args:
            completeness: Completeness indicators
            provenance: Full provenance data

        Returns:
            SLSA level (0-4)
        """
        # SLSA Level determination based on requirements
        # Level 1: Provenance exists
        # Level 2: Hosted build, signed provenance
        # Level 3: Hardened build platform, non-forgeable
        # Level 4: Two-party review, hermetic

        has_builder = bool(provenance.get("builder", {}).get("id"))
        has_source = bool(
            provenance.get("invocation", {}).get("configSource", {}).get("uri")
        )
        has_params = completeness.get("parameters", False)
        has_env = completeness.get("environment", False)
        has_materials = completeness.get("materials", False)

        if has_params and has_env and has_materials:
            return 2  # Full completeness = Level 2
        elif has_builder and has_source:
            return 1  # Basic provenance = Level 1
        else:
            return 0

    def _compute_verification_status(
        self,
        signature_info: SignatureInfo,
        sbom_info: Optional[SBOMInfo],
        provenance_info: Optional[ProvenanceInfo],
        errors: List[str],
    ) -> bool:
        """Compute overall verification status.

        Args:
            signature_info: Signature verification result
            sbom_info: SBOM verification result
            provenance_info: Provenance verification result
            errors: List of errors

        Returns:
            True if verification passed based on config
        """
        if errors:
            return False

        if self.config.verification_level == VerificationLevel.NONE:
            return True

        if not signature_info.verified:
            return self.config.trust_policy != TrustPolicy.REJECT_ALL

        if self.config.verification_level == VerificationLevel.SIGNATURE_ONLY:
            return True

        if self.config.verification_level == VerificationLevel.SBOM:
            return sbom_info is not None and sbom_info.verified

        if self.config.verification_level == VerificationLevel.FULL:
            sbom_ok = sbom_info is not None and sbom_info.verified
            prov_ok = provenance_info is not None and provenance_info.verified
            return sbom_ok and prov_ok

        return False

    def _get_cached_result(
        self,
        image_reference: str,
    ) -> Optional[VerificationResult]:
        """Get cached verification result if still valid.

        Args:
            image_reference: Container image reference

        Returns:
            Cached result or None
        """
        if image_reference in self._cache:
            result, cached_at = self._cache[image_reference]
            age = (datetime.utcnow() - cached_at).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return result
            else:
                del self._cache[image_reference]
        return None

    def _cache_result(
        self,
        image_reference: str,
        result: VerificationResult,
    ) -> None:
        """Cache verification result.

        Args:
            image_reference: Container image reference
            result: Verification result to cache
        """
        self._cache[image_reference] = (result, datetime.utcnow())

        # Prune old cache entries
        if len(self._cache) > 1000:
            now = datetime.utcnow()
            expired = [
                key
                for key, (_, cached_at) in self._cache.items()
                if (now - cached_at).total_seconds() > self.config.cache_ttl_seconds
            ]
            for key in expired:
                del self._cache[key]

    def clear_cache(self) -> None:
        """Clear all cached verification results."""
        self._cache.clear()
        logger.info("Verification cache cleared")
