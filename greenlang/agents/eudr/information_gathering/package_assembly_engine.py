# -*- coding: utf-8 -*-
"""
AGENT-EUDR-027: Information Gathering Agent - Package Assembly Engine

Assembles all gathered information (external database queries, certification
verification results, public data harvests, supplier profiles, normalization
logs, and completeness reports) into a unified ``InformationPackage`` for
due diligence statement submission. Each package contains a complete
provenance chain, evidence artifacts with content hashes, and a deterministic
package hash for integrity verification.

Production infrastructure includes:
    - Full provenance chain construction (collect -> normalize -> validate -> assemble)
    - Evidence artifact creation with SHA-256 content hashes
    - Deterministic package-level hash for integrity verification
    - Package version diffing for change tracking
    - Package validation (hash integrity, completeness, provenance chain)
    - S3 path generation for artifact storage
    - Retention period calculation (5 years per EUDR Article 31)
    - Assembly duration tracking via Prometheus histograms
    - SHA-256 provenance hash on every assembled package

Zero-Hallucination Guarantees:
    - Package hashes computed via deterministic SHA-256 over canonical JSON
    - Provenance chain verified via sequential hash chaining
    - Evidence content hashes computed from raw bytes/JSON
    - No LLM involvement in assembly, hashing, or validation
    - All timestamps generated from system clock (UTC)

Regulatory References:
    - EUDR Article 9: Complete information package for DDS
    - EUDR Article 10: Risk assessment based on assembled information
    - EUDR Article 12: Package submitted to competent authority on request
    - EUDR Article 31: 5-year record retention (1825 days default)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 7: Package Assembly)
Agent ID: GL-EUDR-IGA-027
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    get_config,
)
from greenlang.agents.eudr.information_gathering.models import (
    Article9ElementStatus,
    CertificateVerificationResult,
    CompletenessReport,
    EUDRCommodity,
    EvidenceArtifact,
    GapReport,
    InformationPackage,
    NormalizationRecord,
    PackageDiff,
    ProvenanceEntry,
    QueryResult,
    SupplierProfile,
)
from greenlang.agents.eudr.information_gathering.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)
from greenlang.agents.eudr.information_gathering.metrics import (
    record_package_assembled,
    observe_package_assembly_duration,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "PKG") -> str:
    """Generate a unique identifier with a prefix.

    Args:
        prefix: ID prefix (default: PKG).

    Returns:
        String ID in format PREFIX-UUID8.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class PackageAssemblyEngine:
    """Engine for assembling gathered information into EUDR information packages.

    Combines external database query results, certification verification
    results, public data harvests, supplier profiles, normalization logs,
    and completeness reports into a unified ``InformationPackage`` with
    full provenance chain and integrity hashing.

    Supports package versioning, integrity validation, and version diffing
    for tracking changes across successive gathering operations.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> engine = PackageAssemblyEngine()
        >>> package = engine.assemble_package(
        ...     operation_id="OP-001",
        ...     operator_id="EORI-DE-001",
        ...     commodity=EUDRCommodity.COFFEE,
        ...     article_9_elements={},
        ...     completeness_report=completeness_report,
        ...     supplier_profiles=[],
        ...     query_results={},
        ...     cert_results=[],
        ...     public_data={},
        ...     normalization_log=[],
        ... )
        >>> assert len(package.package_hash) == 64
    """

    def __init__(self, config: Optional[InformationGatheringConfig] = None) -> None:
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._assembly_history: List[InformationPackage] = []
        self._package_store: Dict[str, InformationPackage] = {}
        logger.info(
            "PackageAssemblyEngine initialized "
            "(s3_bucket=%s, s3_prefix=%s, retention_days=%d, "
            "max_size_mb=%d)",
            self._config.s3_bucket,
            self._config.s3_prefix,
            self._config.retention_days,
            self._config.max_package_size_mb,
        )

    def assemble_package(
        self,
        operation_id: str,
        operator_id: str,
        commodity: EUDRCommodity,
        article_9_elements: Dict[str, Article9ElementStatus],
        completeness_report: CompletenessReport,
        supplier_profiles: List[SupplierProfile],
        query_results: Dict[str, List[QueryResult]],
        cert_results: List[CertificateVerificationResult],
        public_data: Dict[str, Any],
        normalization_log: List[NormalizationRecord],
    ) -> InformationPackage:
        """Assemble all gathered data into a complete information package.

        Builds the package with full provenance chain (collect -> normalize
        -> validate -> assemble), creates evidence artifacts for each data
        source, and computes a deterministic package hash.

        Args:
            operation_id: Gathering operation identifier.
            operator_id: EORI or operator identifier.
            commodity: EUDR regulated commodity.
            article_9_elements: Dict of Article 9 element statuses.
            completeness_report: Completeness validation report.
            supplier_profiles: Aggregated supplier profiles.
            query_results: External database query results by source.
            cert_results: Certification verification results.
            public_data: Public data harvest results.
            normalization_log: Normalization audit records.

        Returns:
            Fully assembled InformationPackage with provenance chain
            and package hash.
        """
        start_time = time.monotonic()
        package_id = _generate_id("PKG")

        # Determine package version (increment if previous exists for operator)
        version = self._get_next_version(operator_id, commodity)

        # Build evidence artifacts
        evidence_artifacts = self._build_evidence_artifacts(
            package_id,
            article_9_elements,
            query_results,
            cert_results,
            supplier_profiles,
        )

        # Build provenance chain
        provenance_chain = self._build_provenance_chain(
            operation_id,
            query_results,
            normalization_log,
            completeness_report,
        )

        # Compute validity period
        assembled_at = _utcnow()
        valid_until = assembled_at + timedelta(days=self._config.retention_days)

        # Assemble the package
        package = InformationPackage(
            package_id=package_id,
            operator_id=operator_id,
            commodity=commodity,
            version=version,
            article_9_elements=article_9_elements,
            completeness_score=completeness_report.completeness_score,
            completeness_classification=completeness_report.completeness_classification.value,
            supplier_profiles=supplier_profiles,
            external_data=query_results,
            certification_results=cert_results,
            public_data=public_data,
            normalization_log=normalization_log,
            gap_report=completeness_report.gap_report,
            evidence_artifacts=evidence_artifacts,
            provenance_chain=[
                ProvenanceEntry(**entry) for entry in provenance_chain
            ],
            package_hash="",  # Computed below
            assembled_at=assembled_at,
            valid_until=valid_until,
        )

        # Compute package hash
        package.package_hash = self.compute_package_hash(package)

        # Store in local cache
        self._package_store[package_id] = package
        self._assembly_history.append(package)

        # Metrics
        elapsed = time.monotonic() - start_time
        record_package_assembled(commodity.value)
        observe_package_assembly_duration(commodity.value, elapsed)

        logger.info(
            "Assembled package %s for operator %s (%s v%d): "
            "completeness=%s%%, artifacts=%d, provenance_steps=%d, "
            "hash=%s (%.0fms)",
            package_id,
            operator_id,
            commodity.value,
            version,
            completeness_report.completeness_score,
            len(evidence_artifacts),
            len(provenance_chain),
            package.package_hash[:16] + "...",
            elapsed * 1000,
        )
        return package

    def create_evidence_artifact(
        self,
        article_9_element: str,
        source: str,
        data: Any,
        format: str = "json",
    ) -> EvidenceArtifact:
        """Create a single evidence artifact with content hash.

        Computes a SHA-256 content hash from the data and generates
        an S3 storage path for the artifact.

        Args:
            article_9_element: Article 9 element this artifact supports.
            source: Data source identifier.
            data: Artifact content (will be JSON-serialized for hashing).
            format: Content format (json, pdf, csv, xml).

        Returns:
            EvidenceArtifact with content hash and S3 path.
        """
        artifact_id = _generate_id("ART")

        # Compute content hash
        content_hash = _compute_hash(data) if isinstance(data, dict) else _compute_hash(
            {"content": str(data)}
        )

        # Generate S3 path
        now = _utcnow()
        s3_path = (
            f"s3://{self._config.s3_bucket}/{self._config.s3_prefix}"
            f"{now.strftime('%Y/%m/%d')}/{artifact_id}.{format}"
        )

        artifact = EvidenceArtifact(
            artifact_id=artifact_id,
            article_9_element=article_9_element,
            source=source,
            format=format,
            content_hash=content_hash,
            s3_path=s3_path,
            collected_at=now,
        )

        logger.debug(
            "Created evidence artifact %s for %s from %s (hash=%s)",
            artifact_id,
            article_9_element,
            source,
            content_hash[:16] + "...",
        )
        return artifact

    def _build_evidence_artifacts(
        self,
        package_id: str,
        article_9_elements: Dict[str, Article9ElementStatus],
        query_results: Dict[str, List[QueryResult]],
        cert_results: List[CertificateVerificationResult],
        supplier_profiles: List[SupplierProfile],
    ) -> List[EvidenceArtifact]:
        """Build evidence artifacts from all collected data sources.

        Creates one artifact per Article 9 element with status data,
        plus artifacts for external queries, certifications, and
        supplier profiles.

        Args:
            package_id: Parent package identifier.
            article_9_elements: Article 9 element statuses.
            query_results: External database query results.
            cert_results: Certification verification results.
            supplier_profiles: Supplier profiles.

        Returns:
            List of EvidenceArtifact objects.
        """
        artifacts: List[EvidenceArtifact] = []

        # Artifacts for each Article 9 element
        for elem_name, elem_status in article_9_elements.items():
            artifacts.append(
                self.create_evidence_artifact(
                    article_9_element=elem_name,
                    source=elem_status.source or "internal",
                    data={
                        "element_name": elem_name,
                        "status": elem_status.status.value,
                        "value_summary": elem_status.value_summary,
                        "confidence": str(elem_status.confidence),
                    },
                    format="json",
                )
            )

        # Artifacts for external query results
        for source_name, results in query_results.items():
            if results:
                artifacts.append(
                    self.create_evidence_artifact(
                        article_9_element="supply_chain_information",
                        source=source_name,
                        data={
                            "source": source_name,
                            "query_count": len(results),
                            "total_records": sum(r.record_count for r in results),
                        },
                        format="json",
                    )
                )

        # Artifacts for certification results
        for cert in cert_results:
            artifacts.append(
                self.create_evidence_artifact(
                    article_9_element="deforestation_free_evidence",
                    source=cert.certification_body.value,
                    data={
                        "certificate_id": cert.certificate_id,
                        "body": cert.certification_body.value,
                        "status": cert.verification_status.value,
                        "valid_until": str(cert.valid_until),
                    },
                    format="json",
                )
            )

        # Artifacts for supplier profiles
        for profile in supplier_profiles:
            artifacts.append(
                self.create_evidence_artifact(
                    article_9_element="supplier_identification",
                    source=",".join(profile.data_sources) if profile.data_sources else "aggregated",
                    data={
                        "supplier_id": profile.supplier_id,
                        "name": profile.name,
                        "country_code": profile.country_code,
                        "completeness": str(profile.completeness_score),
                    },
                    format="json",
                )
            )

        return artifacts

    def _build_provenance_chain(
        self,
        operation_id: str,
        query_results: Dict[str, List[QueryResult]],
        normalization_log: List[NormalizationRecord],
        completeness_report: CompletenessReport,
    ) -> List[Dict[str, Any]]:
        """Build the full provenance chain for the package.

        Chain steps: collect -> normalize -> validate -> assemble.
        Each step's output hash feeds into the next step's input hash.

        Args:
            operation_id: Gathering operation identifier.
            query_results: External query results for collect hash.
            normalization_log: Normalization records for normalize hash.
            completeness_report: Completeness report for validate hash.

        Returns:
            List of provenance entry dicts forming a valid chain.
        """
        tracker = ProvenanceTracker()

        # Step 1: Collect
        collect_data = {
            "operation_id": operation_id,
            "sources_queried": list(query_results.keys()),
            "total_queries": sum(len(v) for v in query_results.values()),
        }
        collect_hash = _compute_hash(collect_data)

        # Step 2: Normalize
        normalize_data = {
            "normalizations": len(normalization_log),
            "types": list({r.normalization_type.value for r in normalization_log}),
        }
        normalize_hash = _compute_hash(normalize_data)

        # Step 3: Validate
        validate_data = {
            "completeness_score": str(completeness_report.completeness_score),
            "classification": completeness_report.completeness_classification.value,
            "gaps": completeness_report.gap_report.total_gaps,
        }
        validate_hash = _compute_hash(validate_data)

        # Step 4: Assemble
        assemble_data = {
            "operation_id": operation_id,
            "collect_hash": collect_hash,
            "normalize_hash": normalize_hash,
            "validate_hash": validate_hash,
        }
        assemble_hash = _compute_hash(assemble_data)

        steps = [
            {"step": "collect", "source": "external_databases", "data": collect_data},
            {"step": "normalize", "source": "normalization_engine", "data": normalize_data},
            {"step": "validate", "source": "completeness_engine", "data": validate_data},
            {"step": "assemble", "source": "package_assembly", "data": assemble_data},
        ]

        chain = tracker.build_chain(steps, genesis_hash=GENESIS_HASH)
        return chain

    def compute_package_hash(self, package: InformationPackage) -> str:
        """Compute deterministic SHA-256 hash over the full package.

        Serializes key package fields into a canonical representation
        and computes the hash. The package_hash field is excluded from
        the input to avoid circular dependency.

        Args:
            package: InformationPackage to hash.

        Returns:
            64-character lowercase hex SHA-256 hash string.
        """
        hash_input = {
            "package_id": package.package_id,
            "operator_id": package.operator_id,
            "commodity": package.commodity.value,
            "version": package.version,
            "completeness_score": str(package.completeness_score),
            "completeness_classification": package.completeness_classification,
            "article_9_elements": {
                k: {
                    "status": v.status.value,
                    "confidence": str(v.confidence),
                }
                for k, v in package.article_9_elements.items()
            },
            "supplier_count": len(package.supplier_profiles),
            "cert_count": len(package.certification_results),
            "artifact_count": len(package.evidence_artifacts),
            "normalization_count": len(package.normalization_log),
            "gap_total": package.gap_report.total_gaps,
            "provenance_steps": len(package.provenance_chain),
            "assembled_at": str(package.assembled_at),
        }
        return _compute_hash(hash_input)

    def validate_package(
        self, package: InformationPackage
    ) -> Dict[str, Any]:
        """Verify package integrity: hash, completeness, and provenance chain.

        Performs three validation checks:
            1. Hash integrity: recompute and compare package_hash
            2. Completeness: verify Article 9 elements are present
            3. Provenance chain: verify hash chain continuity

        Args:
            package: InformationPackage to validate.

        Returns:
            Dict with is_valid (bool), checks (dict of check results),
            and errors (list of error messages).
        """
        errors: List[str] = []
        checks: Dict[str, bool] = {}

        # Check 1: Hash integrity
        recomputed_hash = self.compute_package_hash(package)
        hash_valid = recomputed_hash == package.package_hash
        checks["hash_integrity"] = hash_valid
        if not hash_valid:
            errors.append(
                f"Package hash mismatch: expected={package.package_hash[:16]}..., "
                f"computed={recomputed_hash[:16]}..."
            )

        # Check 2: Completeness - all 10 Article 9 elements should be present
        from greenlang.agents.eudr.information_gathering.models import Article9ElementName
        missing_elements: List[str] = []
        for elem in Article9ElementName:
            if elem.value not in package.article_9_elements:
                missing_elements.append(elem.value)
        elements_valid = len(missing_elements) == 0
        checks["elements_complete"] = elements_valid
        if not elements_valid:
            errors.append(
                f"Missing Article 9 elements: {missing_elements}"
            )

        # Check 3: Provenance chain continuity
        if package.provenance_chain:
            chain_entries = [
                {
                    "step": entry.step,
                    "source": entry.source,
                    "timestamp": str(entry.timestamp),
                    "actor": entry.actor,
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                }
                for entry in package.provenance_chain
            ]
            tracker = ProvenanceTracker()
            chain_valid = tracker.verify_chain(chain_entries)
        else:
            chain_valid = False
            errors.append("Provenance chain is empty")
        checks["provenance_chain"] = chain_valid
        if not chain_valid and package.provenance_chain:
            errors.append("Provenance chain integrity verification failed")

        # Check 4: Retention period
        if package.valid_until:
            retention_valid = package.valid_until > _utcnow()
            checks["retention_valid"] = retention_valid
            if not retention_valid:
                errors.append("Package has exceeded its retention period")
        else:
            checks["retention_valid"] = False
            errors.append("Package valid_until is not set")

        is_valid = all(checks.values())

        if is_valid:
            logger.info("Package %s validation PASSED (all %d checks)", package.package_id, len(checks))
        else:
            logger.warning(
                "Package %s validation FAILED: %d/%d checks passed, errors: %s",
                package.package_id,
                sum(1 for v in checks.values() if v),
                len(checks),
                errors,
            )

        return {
            "is_valid": is_valid,
            "checks": checks,
            "errors": errors,
            "package_id": package.package_id,
        }

    def diff_packages(
        self,
        package_a: InformationPackage,
        package_b: InformationPackage,
    ) -> PackageDiff:
        """Compare two information package versions and identify differences.

        Compares Article 9 elements, completeness scores, and evidence
        artifacts between two package versions.

        Args:
            package_a: First (typically older) package.
            package_b: Second (typically newer) package.

        Returns:
            PackageDiff with added, removed, and changed elements.
        """
        elements_a = set(package_a.article_9_elements.keys())
        elements_b = set(package_b.article_9_elements.keys())

        added = sorted(elements_b - elements_a)
        removed = sorted(elements_a - elements_b)

        # Changed: elements present in both but with different status or confidence
        changed: List[str] = []
        for elem_name in elements_a & elements_b:
            status_a = package_a.article_9_elements[elem_name]
            status_b = package_b.article_9_elements[elem_name]
            if (
                status_a.status != status_b.status
                or status_a.confidence != status_b.confidence
                or status_a.value_summary != status_b.value_summary
            ):
                changed.append(elem_name)
        changed.sort()

        score_delta = (
            package_b.completeness_score - package_a.completeness_score
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        diff = PackageDiff(
            package_a_id=package_a.package_id,
            package_b_id=package_b.package_id,
            added_elements=added,
            removed_elements=removed,
            changed_elements=changed,
            score_delta=score_delta,
            compared_at=_utcnow(),
        )

        logger.info(
            "Package diff %s vs %s: +%d -%d ~%d elements, score_delta=%s",
            package_a.package_id,
            package_b.package_id,
            len(added),
            len(removed),
            len(changed),
            score_delta,
        )
        return diff

    def _get_next_version(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
    ) -> int:
        """Determine the next version number for an operator+commodity.

        Scans assembly history for previous packages and increments
        the highest version number found.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity.

        Returns:
            Next version integer (starts at 1).
        """
        max_version = 0
        for pkg in self._assembly_history:
            if pkg.operator_id == operator_id and pkg.commodity == commodity:
                if pkg.version > max_version:
                    max_version = pkg.version
        return max_version + 1

    def get_package(self, package_id: str) -> Optional[InformationPackage]:
        """Retrieve a previously assembled package by ID.

        Args:
            package_id: Package identifier.

        Returns:
            InformationPackage or None if not found.
        """
        return self._package_store.get(package_id)

    def get_assembly_stats(self) -> Dict[str, Any]:
        """Return package assembly engine statistics.

        Returns:
            Dict with total_assembled, packages_stored,
            commodity_breakdown, and average_completeness keys.
        """
        commodity_counts: Dict[str, int] = {}
        total_completeness = Decimal("0")

        for pkg in self._assembly_history:
            c_key = pkg.commodity.value
            commodity_counts[c_key] = commodity_counts.get(c_key, 0) + 1
            total_completeness += pkg.completeness_score

        total = len(self._assembly_history)
        avg_completeness = (
            (total_completeness / Decimal(str(total))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if total > 0
            else Decimal("0")
        )

        return {
            "total_assembled": total,
            "packages_stored": len(self._package_store),
            "commodity_breakdown": commodity_counts,
            "average_completeness": float(avg_completeness),
        }

    def clear_store(self) -> None:
        """Clear package store and history (for testing)."""
        self._package_store.clear()
        self._assembly_history.clear()
        logger.info("Package store and assembly history cleared")
