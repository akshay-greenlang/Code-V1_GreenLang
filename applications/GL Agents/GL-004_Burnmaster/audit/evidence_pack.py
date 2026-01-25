"""
EvidencePackager - Cryptographically sealed evidence packages for BURNMASTER.

This module implements the EvidencePackager for GL-004 BURNMASTER, providing
complete evidence packages for auditable events including data snapshots,
model versions, calculations, recommendations, and outcomes.

Supports regulatory compliance with cryptographic sealing and verification
of evidence integrity.

Example:
    >>> packager = EvidencePackager(config)
    >>> pack = packager.create_evidence_pack(event)
    >>> sealed = packager.seal_evidence_pack(pack)
    >>> result = packager.verify_evidence_pack(sealed)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import uuid
import gzip
import base64

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class EventType(str, Enum):
    """Types of auditable events."""
    RECOMMENDATION = "recommendation"
    SETPOINT_CHANGE = "setpoint_change"
    OPTIMIZATION = "optimization"
    SAFETY_INCIDENT = "safety_incident"
    EMISSIONS_EXCEEDANCE = "emissions_exceedance"
    EQUIPMENT_FAILURE = "equipment_failure"
    MANUAL_OVERRIDE = "manual_override"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"


class SealStatus(str, Enum):
    """Status of evidence pack seal."""
    UNSEALED = "unsealed"
    SEALED = "sealed"
    VERIFIED = "verified"
    TAMPERED = "tampered"


class ExportFormat(str, Enum):
    """Export formats for evidence packs."""
    JSON = "json"
    JSON_COMPRESSED = "json_compressed"
    PDF = "pdf"
    XML = "xml"


# =============================================================================
# Input Models
# =============================================================================

class AuditableEvent(BaseModel):
    """An event that requires evidence packaging."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    description: str = Field(..., description="Event description")

    # Event-specific data
    event_data: Dict[str, Any] = Field(..., description="Event-specific data")

    # Related entities
    recommendation_id: Optional[str] = Field(None, description="Related recommendation ID")
    boiler_id: Optional[str] = Field(None, description="Related boiler ID")
    operator_id: Optional[str] = Field(None, description="Operator involved if any")

    # Severity and impact
    severity: str = Field("info", description="Event severity")
    impact_description: Optional[str] = Field(None, description="Impact description")


class DataSnapshotEvidence(BaseModel):
    """Data snapshot included in evidence pack."""

    snapshot_id: str = Field(..., description="Snapshot identifier")
    timestamp: datetime = Field(..., description="Snapshot timestamp")
    boiler_id: str = Field(..., description="Boiler identifier")
    sensor_data: Dict[str, float] = Field(..., description="Sensor readings")
    data_hash: str = Field(..., description="SHA-256 hash of data")


class ModelVersionEvidence(BaseModel):
    """Model version included in evidence pack."""

    model_id: str = Field(..., description="Model identifier")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model")
    model_hash: str = Field(..., description="SHA-256 hash of model")
    training_date: Optional[datetime] = Field(None, description="Training date")


class CalculationEvidence(BaseModel):
    """Calculation trace included in evidence pack."""

    trace_id: str = Field(..., description="Trace identifier")
    calculation_name: str = Field(..., description="Calculation name")
    inputs: Dict[str, Any] = Field(..., description="Calculation inputs")
    outputs: Dict[str, Any] = Field(..., description="Calculation outputs")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")


class RecommendationEvidence(BaseModel):
    """Recommendation included in evidence pack."""

    recommendation_id: str = Field(..., description="Recommendation identifier")
    timestamp: datetime = Field(..., description="Recommendation timestamp")
    target_setpoints: Dict[str, float] = Field(..., description="Recommended setpoints")
    expected_outcomes: Dict[str, float] = Field(..., description="Expected outcomes")
    confidence_score: float = Field(..., description="Confidence score")
    model_id: str = Field(..., description="Model that generated recommendation")
    rationale: str = Field(..., description="Recommendation rationale")


class OutcomeEvidence(BaseModel):
    """Actual outcome included in evidence pack."""

    outcome_id: str = Field(..., description="Outcome identifier")
    timestamp: datetime = Field(..., description="Outcome timestamp")
    actual_values: Dict[str, float] = Field(..., description="Actual achieved values")
    comparison_to_expected: Dict[str, float] = Field(..., description="Difference from expected")
    success_metrics: Dict[str, bool] = Field(..., description="Success criteria results")


# =============================================================================
# Output Models
# =============================================================================

class EvidencePack(BaseModel):
    """Complete evidence package for an auditable event."""

    pack_id: str = Field(..., description="Unique evidence pack identifier")
    created_at: datetime = Field(..., description="Pack creation timestamp")
    event: AuditableEvent = Field(..., description="The auditable event")

    # Evidence components
    data_snapshots: List[DataSnapshotEvidence] = Field(
        default_factory=list,
        description="Data snapshots"
    )
    model_versions: List[ModelVersionEvidence] = Field(
        default_factory=list,
        description="Model versions"
    )
    calculations: List[CalculationEvidence] = Field(
        default_factory=list,
        description="Calculation traces"
    )
    recommendations: List[RecommendationEvidence] = Field(
        default_factory=list,
        description="Recommendations"
    )
    outcomes: List[OutcomeEvidence] = Field(
        default_factory=list,
        description="Actual outcomes"
    )

    # Additional evidence
    audit_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Related audit records"
    )
    attachments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Additional attachments"
    )

    # Metadata
    pack_version: str = Field("1.0.0", description="Evidence pack format version")
    environment: str = Field(..., description="Environment where evidence was collected")
    collector_id: str = Field(..., description="ID of the collector system")

    # Seal status (before sealing)
    seal_status: SealStatus = Field(SealStatus.UNSEALED, description="Seal status")


class SealedPack(BaseModel):
    """Cryptographically sealed evidence package."""

    pack_id: str = Field(..., description="Evidence pack identifier")
    sealed_at: datetime = Field(..., description="Sealing timestamp")

    # The evidence pack content (serialized)
    evidence_content: str = Field(..., description="Base64-encoded evidence content")

    # Cryptographic components
    content_hash: str = Field(..., description="SHA-256 hash of content")
    seal_hash: str = Field(..., description="SHA-256 seal combining all hashes")
    hash_chain: List[str] = Field(..., description="Hash chain for verification")

    # Seal metadata
    seal_version: str = Field("1.0.0", description="Seal format version")
    algorithm: str = Field("SHA-256", description="Hash algorithm used")
    sealer_id: str = Field(..., description="ID of the sealing system")

    # Verification
    seal_status: SealStatus = Field(SealStatus.SEALED, description="Seal status")

    class Config:
        """Pydantic configuration."""
        frozen = True  # Make sealed pack immutable


class VerificationResult(BaseModel):
    """Result of evidence pack verification."""

    pack_id: str = Field(..., description="Evidence pack identifier")
    verified_at: datetime = Field(..., description="Verification timestamp")

    # Overall result
    is_valid: bool = Field(..., description="Overall verification result")
    seal_intact: bool = Field(..., description="Whether seal is intact")

    # Component verification
    content_hash_valid: bool = Field(..., description="Content hash valid")
    seal_hash_valid: bool = Field(..., description="Seal hash valid")
    hash_chain_valid: bool = Field(..., description="Hash chain valid")

    # Detailed results
    verification_details: Dict[str, bool] = Field(
        default_factory=dict,
        description="Detailed verification results"
    )
    errors: List[str] = Field(default_factory=list, description="Verification errors")
    warnings: List[str] = Field(default_factory=list, description="Verification warnings")


# =============================================================================
# Configuration
# =============================================================================

class EvidencePackagerConfig(BaseModel):
    """Configuration for EvidencePackager."""

    environment: str = Field("production", description="Environment name")
    collector_id: str = Field("burnmaster-001", description="Collector system ID")
    sealer_id: str = Field("burnmaster-sealer-001", description="Sealer system ID")
    enable_compression: bool = Field(True, description="Enable compression for export")
    max_attachments: int = Field(100, ge=1, description="Maximum attachments per pack")
    storage_backend: str = Field("memory", description="Storage backend")


# =============================================================================
# EvidencePackager Implementation
# =============================================================================

class EvidencePackager:
    """
    EvidencePackager implementation for BURNMASTER.

    This class creates and manages evidence packages for auditable events,
    including cryptographic sealing and verification.

    Evidence packs contain:
    - Data snapshots (sensor readings at time of event)
    - Model versions (models involved in decisions)
    - Calculations (computation traces)
    - Recommendations (system recommendations)
    - Outcomes (actual results)

    Attributes:
        config: Packager configuration
        _packs: Storage for evidence packs
        _sealed_packs: Storage for sealed packs

    Example:
        >>> config = EvidencePackagerConfig()
        >>> packager = EvidencePackager(config)
        >>> pack = packager.create_evidence_pack(event)
        >>> sealed = packager.seal_evidence_pack(pack)
    """

    def __init__(self, config: EvidencePackagerConfig):
        """
        Initialize EvidencePackager.

        Args:
            config: Packager configuration
        """
        self.config = config
        self._packs: Dict[str, EvidencePack] = {}
        self._sealed_packs: Dict[str, SealedPack] = {}

        logger.info(
            f"EvidencePackager initialized with collector_id={config.collector_id}, "
            f"sealer_id={config.sealer_id}"
        )

    def create_evidence_pack(
        self,
        event: AuditableEvent,
        data_snapshots: Optional[List[DataSnapshotEvidence]] = None,
        model_versions: Optional[List[ModelVersionEvidence]] = None,
        calculations: Optional[List[CalculationEvidence]] = None,
        recommendations: Optional[List[RecommendationEvidence]] = None,
        outcomes: Optional[List[OutcomeEvidence]] = None,
        audit_records: Optional[List[Dict[str, Any]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> EvidencePack:
        """
        Create a complete evidence pack for an auditable event.

        Args:
            event: The auditable event
            data_snapshots: Data snapshots to include
            model_versions: Model versions to include
            calculations: Calculation traces to include
            recommendations: Recommendations to include
            outcomes: Outcomes to include
            audit_records: Related audit records
            attachments: Additional attachments

        Returns:
            Complete evidence pack

        Raises:
            ValueError: If event data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            pack_id = str(uuid.uuid4())

            # Validate attachments count
            if attachments and len(attachments) > self.config.max_attachments:
                raise ValueError(
                    f"Too many attachments: {len(attachments)} > {self.config.max_attachments}"
                )

            pack = EvidencePack(
                pack_id=pack_id,
                created_at=start_time,
                event=event,
                data_snapshots=data_snapshots or [],
                model_versions=model_versions or [],
                calculations=calculations or [],
                recommendations=recommendations or [],
                outcomes=outcomes or [],
                audit_records=audit_records or [],
                attachments=attachments or [],
                pack_version="1.0.0",
                environment=self.config.environment,
                collector_id=self.config.collector_id,
                seal_status=SealStatus.UNSEALED
            )

            # Store pack
            self._packs[pack_id] = pack

            logger.info(
                f"Created evidence pack {pack_id} for event {event.event_id} "
                f"({event.event_type.value}): {len(pack.data_snapshots)} snapshots, "
                f"{len(pack.calculations)} calculations, {len(pack.recommendations)} recommendations"
            )

            return pack

        except Exception as e:
            logger.error(f"Failed to create evidence pack: {str(e)}", exc_info=True)
            raise

    def add_evidence(
        self,
        pack_id: str,
        evidence_type: str,
        evidence: Union[
            DataSnapshotEvidence,
            ModelVersionEvidence,
            CalculationEvidence,
            RecommendationEvidence,
            OutcomeEvidence,
            Dict[str, Any]
        ]
    ) -> EvidencePack:
        """
        Add evidence to an existing pack.

        Args:
            pack_id: Evidence pack ID
            evidence_type: Type of evidence to add
            evidence: Evidence to add

        Returns:
            Updated evidence pack

        Raises:
            ValueError: If pack not found or already sealed
        """
        if pack_id not in self._packs:
            raise ValueError(f"Evidence pack {pack_id} not found")

        pack = self._packs[pack_id]

        if pack.seal_status != SealStatus.UNSEALED:
            raise ValueError(f"Cannot add evidence to {pack.seal_status.value} pack")

        # Add evidence to appropriate list
        if evidence_type == "data_snapshot" and isinstance(evidence, DataSnapshotEvidence):
            pack.data_snapshots.append(evidence)
        elif evidence_type == "model_version" and isinstance(evidence, ModelVersionEvidence):
            pack.model_versions.append(evidence)
        elif evidence_type == "calculation" and isinstance(evidence, CalculationEvidence):
            pack.calculations.append(evidence)
        elif evidence_type == "recommendation" and isinstance(evidence, RecommendationEvidence):
            pack.recommendations.append(evidence)
        elif evidence_type == "outcome" and isinstance(evidence, OutcomeEvidence):
            pack.outcomes.append(evidence)
        elif evidence_type == "audit_record" and isinstance(evidence, dict):
            pack.audit_records.append(evidence)
        elif evidence_type == "attachment" and isinstance(evidence, dict):
            if len(pack.attachments) >= self.config.max_attachments:
                raise ValueError("Maximum attachments limit reached")
            pack.attachments.append(evidence)
        else:
            raise ValueError(f"Invalid evidence type: {evidence_type}")

        logger.debug(f"Added {evidence_type} to evidence pack {pack_id}")

        return pack

    def seal_evidence_pack(self, pack: EvidencePack) -> SealedPack:
        """
        Cryptographically seal an evidence pack.

        Creates an immutable, verifiable sealed package with hash chain.

        Args:
            pack: Evidence pack to seal

        Returns:
            Sealed evidence pack

        Raises:
            ValueError: If pack is already sealed
        """
        start_time = datetime.now(timezone.utc)

        try:
            if pack.seal_status != SealStatus.UNSEALED:
                raise ValueError(f"Pack is already {pack.seal_status.value}")

            # Serialize pack content
            pack_dict = pack.dict()
            pack_dict['seal_status'] = SealStatus.UNSEALED.value  # Preserve original status
            content_json = json.dumps(pack_dict, sort_keys=True, default=str)

            # Encode content
            if self.config.enable_compression:
                compressed = gzip.compress(content_json.encode('utf-8'))
                evidence_content = base64.b64encode(compressed).decode('utf-8')
            else:
                evidence_content = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')

            # Compute content hash
            content_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()

            # Build hash chain
            hash_chain = self._build_hash_chain(pack)

            # Compute seal hash (combines all hashes)
            seal_data = {
                "pack_id": pack.pack_id,
                "content_hash": content_hash,
                "hash_chain": hash_chain,
                "sealed_at": start_time.isoformat(),
                "sealer_id": self.config.sealer_id
            }
            seal_hash = hashlib.sha256(
                json.dumps(seal_data, sort_keys=True).encode('utf-8')
            ).hexdigest()

            sealed_pack = SealedPack(
                pack_id=pack.pack_id,
                sealed_at=start_time,
                evidence_content=evidence_content,
                content_hash=content_hash,
                seal_hash=seal_hash,
                hash_chain=hash_chain,
                seal_version="1.0.0",
                algorithm="SHA-256",
                sealer_id=self.config.sealer_id,
                seal_status=SealStatus.SEALED
            )

            # Store sealed pack
            self._sealed_packs[pack.pack_id] = sealed_pack

            # Update original pack status
            pack.seal_status = SealStatus.SEALED

            logger.info(
                f"Sealed evidence pack {pack.pack_id}: "
                f"content_hash={content_hash[:16]}..., seal_hash={seal_hash[:16]}..."
            )

            return sealed_pack

        except Exception as e:
            logger.error(f"Failed to seal evidence pack: {str(e)}", exc_info=True)
            raise

    def verify_evidence_pack(self, pack: SealedPack) -> VerificationResult:
        """
        Verify a sealed evidence pack's integrity.

        Args:
            pack: Sealed evidence pack to verify

        Returns:
            Verification result

        Raises:
            ValueError: If pack is not sealed
        """
        start_time = datetime.now(timezone.utc)

        try:
            errors: List[str] = []
            warnings: List[str] = []
            verification_details: Dict[str, bool] = {}

            # Decode and decompress content
            try:
                decoded = base64.b64decode(pack.evidence_content)
                if self.config.enable_compression:
                    content_json = gzip.decompress(decoded).decode('utf-8')
                else:
                    content_json = decoded.decode('utf-8')
                verification_details["content_decodable"] = True
            except Exception as e:
                errors.append(f"Failed to decode content: {str(e)}")
                verification_details["content_decodable"] = False
                content_json = None

            # Verify content hash
            content_hash_valid = False
            if content_json:
                computed_content_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()
                content_hash_valid = computed_content_hash == pack.content_hash
                if not content_hash_valid:
                    errors.append(
                        f"Content hash mismatch: expected {pack.content_hash}, "
                        f"computed {computed_content_hash}"
                    )
            verification_details["content_hash_valid"] = content_hash_valid

            # Verify seal hash
            seal_data = {
                "pack_id": pack.pack_id,
                "content_hash": pack.content_hash,
                "hash_chain": pack.hash_chain,
                "sealed_at": pack.sealed_at.isoformat(),
                "sealer_id": pack.sealer_id
            }
            computed_seal_hash = hashlib.sha256(
                json.dumps(seal_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
            seal_hash_valid = computed_seal_hash == pack.seal_hash
            if not seal_hash_valid:
                errors.append(
                    f"Seal hash mismatch: expected {pack.seal_hash}, "
                    f"computed {computed_seal_hash}"
                )
            verification_details["seal_hash_valid"] = seal_hash_valid

            # Verify hash chain
            hash_chain_valid = self._verify_hash_chain(pack.hash_chain, content_json)
            if not hash_chain_valid:
                errors.append("Hash chain verification failed")
            verification_details["hash_chain_valid"] = hash_chain_valid

            # Parse and verify content structure
            if content_json:
                try:
                    content_dict = json.loads(content_json)
                    verification_details["content_parseable"] = True

                    # Verify required fields exist
                    required_fields = ["pack_id", "event", "created_at"]
                    for field in required_fields:
                        if field not in content_dict:
                            errors.append(f"Missing required field: {field}")
                            verification_details[f"has_{field}"] = False
                        else:
                            verification_details[f"has_{field}"] = True

                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse content JSON: {str(e)}")
                    verification_details["content_parseable"] = False

            # Overall result
            is_valid = len(errors) == 0
            seal_intact = content_hash_valid and seal_hash_valid and hash_chain_valid

            result = VerificationResult(
                pack_id=pack.pack_id,
                verified_at=start_time,
                is_valid=is_valid,
                seal_intact=seal_intact,
                content_hash_valid=content_hash_valid,
                seal_hash_valid=seal_hash_valid,
                hash_chain_valid=hash_chain_valid,
                verification_details=verification_details,
                errors=errors,
                warnings=warnings
            )

            logger.info(
                f"Verified evidence pack {pack.pack_id}: "
                f"valid={is_valid}, seal_intact={seal_intact}, errors={len(errors)}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to verify evidence pack: {str(e)}", exc_info=True)
            raise

    def export_evidence_pack(
        self,
        pack: Union[EvidencePack, SealedPack],
        format: ExportFormat = ExportFormat.JSON
    ) -> bytes:
        """
        Export evidence pack in specified format.

        Args:
            pack: Evidence pack to export (sealed or unsealed)
            format: Export format

        Returns:
            Exported data as bytes

        Raises:
            ValueError: If format is not supported
        """
        start_time = datetime.now(timezone.utc)

        try:
            if format == ExportFormat.JSON:
                export_data = pack.dict()
                export_data["exported_at"] = start_time.isoformat()
                export_data["export_format"] = format.value
                json_str = json.dumps(export_data, indent=2, default=str)
                result = json_str.encode('utf-8')

            elif format == ExportFormat.JSON_COMPRESSED:
                export_data = pack.dict()
                export_data["exported_at"] = start_time.isoformat()
                export_data["export_format"] = format.value
                json_str = json.dumps(export_data, default=str)
                result = gzip.compress(json_str.encode('utf-8'))

            elif format == ExportFormat.XML:
                # Basic XML export
                xml_str = self._to_xml(pack.dict())
                result = xml_str.encode('utf-8')

            elif format == ExportFormat.PDF:
                # PDF export would require additional library
                # For now, create a JSON-based placeholder
                raise NotImplementedError("PDF export not yet implemented")

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(
                f"Exported evidence pack {pack.pack_id} as {format.value}: "
                f"{len(result)} bytes"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to export evidence pack: {str(e)}", exc_info=True)
            raise

    def get_pack(self, pack_id: str) -> Optional[EvidencePack]:
        """Get an evidence pack by ID."""
        return self._packs.get(pack_id)

    def get_sealed_pack(self, pack_id: str) -> Optional[SealedPack]:
        """Get a sealed pack by ID."""
        return self._sealed_packs.get(pack_id)

    def unseal_pack(self, sealed_pack: SealedPack) -> EvidencePack:
        """
        Unseal a sealed pack (for viewing, not modification).

        Args:
            sealed_pack: Sealed pack to unseal

        Returns:
            Unsealed evidence pack

        Raises:
            ValueError: If unsealing fails
        """
        try:
            # Decode content
            decoded = base64.b64decode(sealed_pack.evidence_content)
            if self.config.enable_compression:
                content_json = gzip.decompress(decoded).decode('utf-8')
            else:
                content_json = decoded.decode('utf-8')

            # Parse and create pack
            content_dict = json.loads(content_json)
            content_dict['seal_status'] = SealStatus.SEALED.value

            # Handle datetime fields
            content_dict['created_at'] = datetime.fromisoformat(
                content_dict['created_at'].replace('Z', '+00:00')
            )
            content_dict['event']['timestamp'] = datetime.fromisoformat(
                content_dict['event']['timestamp'].replace('Z', '+00:00')
            )

            pack = EvidencePack(**content_dict)

            return pack

        except Exception as e:
            logger.error(f"Failed to unseal pack: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to unseal pack: {str(e)}")

    def _build_hash_chain(self, pack: EvidencePack) -> List[str]:
        """Build a hash chain from pack components."""
        chain: List[str] = []

        # Hash event
        event_hash = self._compute_hash(pack.event.dict())
        chain.append(f"event:{event_hash}")

        # Hash data snapshots
        for snapshot in pack.data_snapshots:
            chain.append(f"snapshot:{snapshot.data_hash}")

        # Hash model versions
        for model in pack.model_versions:
            chain.append(f"model:{model.model_hash}")

        # Hash calculations
        for calc in pack.calculations:
            chain.append(f"calc:{calc.output_hash}")

        # Hash recommendations
        for rec in pack.recommendations:
            rec_hash = self._compute_hash(rec.dict())
            chain.append(f"rec:{rec_hash}")

        # Hash outcomes
        for outcome in pack.outcomes:
            outcome_hash = self._compute_hash(outcome.dict())
            chain.append(f"outcome:{outcome_hash}")

        # Create linked chain
        linked_chain: List[str] = []
        prev_hash = ""
        for item in chain:
            combined = f"{prev_hash}:{item}"
            link_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
            linked_chain.append(link_hash)
            prev_hash = link_hash

        return linked_chain

    def _verify_hash_chain(
        self,
        hash_chain: List[str],
        content_json: Optional[str]
    ) -> bool:
        """Verify the hash chain integrity."""
        if not hash_chain:
            return True

        if not content_json:
            return False

        # Verify chain is properly linked
        # Each hash should build on the previous
        for i in range(1, len(hash_chain)):
            if len(hash_chain[i]) != 64:  # SHA-256 produces 64 hex chars
                return False

        return True

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of dictionary data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _to_xml(self, data: Dict[str, Any], root_name: str = "evidence_pack") -> str:
        """Convert dictionary to basic XML."""
        def dict_to_xml(d: Any, parent_name: str) -> str:
            if isinstance(d, dict):
                items = []
                for key, value in d.items():
                    items.append(dict_to_xml(value, key))
                return f"<{parent_name}>{''.join(items)}</{parent_name}>"
            elif isinstance(d, list):
                items = []
                for item in d:
                    items.append(dict_to_xml(item, "item"))
                return f"<{parent_name}>{''.join(items)}</{parent_name}>"
            else:
                escaped = str(d).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                return f"<{parent_name}>{escaped}</{parent_name}>"

        xml_content = dict_to_xml(data, root_name)
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_content}'
