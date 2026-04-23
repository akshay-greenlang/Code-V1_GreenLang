"""
GL-007 FurnacePulse - Cryptographic Evidence Sealer

Implements cryptographic evidence sealing using Merkle trees for tamper-evident
audit trails of furnace monitoring events, compliance checks, and safety events.

Reference:
    - Merkle, R. "A Digital Signature Based on a Conventional Encryption Function", 1987
    - NFPA 86 Standard for Ovens and Furnaces - Record keeping requirements
    - ASME PTC 4.1 for fired steam generator data retention

Zero-Hallucination: All hash computations are deterministic SHA-256 operations
with no LLM involvement in cryptographic functions.

Features:
    - Merkle tree construction for batch evidence sealing
    - SHA-256 based leaf and node hashing
    - Merkle proof generation and verification
    - Timestamped evidence packages
    - Chain-of-custody tracking
    - Regulatory compliance metadata

Example:
    >>> sealer = EvidenceSealer()
    >>> evidence = SafetyEvidence(
    ...     event_id="SE-2024-001",
    ...     event_type="HIGH_TMT_ALARM",
    ...     timestamp=datetime.now(timezone.utc),
    ...     sensor_data={"TMT_MAX": 975.5}
    ... )
    >>> sealed = sealer.seal_evidence(evidence)
    >>> print(f"Evidence hash: {sealed.evidence_hash}")
    >>> print(f"Merkle root: {sealed.merkle_root}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import hashlib
import json
import logging
import base64
import uuid

from pydantic import BaseModel, Field, field_validator, computed_field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EvidenceType(str, Enum):
    """Type of evidence being sealed."""

    SAFETY_EVENT = "safety_event"
    COMPLIANCE_CHECK = "compliance_check"
    TMT_READING = "tmt_reading"
    FLAME_SUPERVISION = "flame_supervision"
    RUL_PREDICTION = "rul_prediction"
    MAINTENANCE_RECORD = "maintenance_record"
    CALIBRATION = "calibration"
    AUDIT_TRAIL = "audit_trail"
    CONFIGURATION_CHANGE = "configuration_change"
    ALARM_RESPONSE = "alarm_response"


class SealStatus(str, Enum):
    """Status of evidence seal."""

    SEALED = "sealed"
    VERIFIED = "verified"
    TAMPERED = "tampered"
    PENDING = "pending"
    EXPIRED = "expired"


class ComplianceFramework(str, Enum):
    """Regulatory compliance framework."""

    NFPA_86 = "nfpa_86"
    API_560 = "api_560"
    API_530 = "api_530"
    ASME_PTC_4_1 = "asme_ptc_4_1"
    OSHA = "osha"
    EPA = "epa"


# =============================================================================
# DATA MODELS
# =============================================================================

class EvidenceData(BaseModel):
    """
    Base evidence data model for sealing.

    Attributes:
        evidence_id: Unique evidence identifier
        evidence_type: Type of evidence
        timestamp: Evidence creation timestamp
        source_agent: Source agent ID (e.g., GL-007)
        furnace_id: Associated furnace identifier
        data: Evidence payload data
        metadata: Additional metadata
        compliance_refs: Relevant compliance references
    """

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evidence identifier"
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Evidence creation timestamp"
    )
    source_agent: str = Field(
        default="GL-007",
        description="Source agent ID"
    )
    furnace_id: str = Field(
        ...,
        description="Associated furnace identifier"
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Evidence payload data"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    compliance_refs: List[ComplianceFramework] = Field(
        default_factory=list,
        description="Relevant compliance frameworks"
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_canonical_string(self) -> str:
        """
        Convert to canonical string for hashing.

        Returns:
            Deterministic string representation
        """
        canonical = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "furnace_id": self.furnace_id,
            "data": self.data,
            "metadata": self.metadata,
            "compliance_refs": [c.value for c in self.compliance_refs],
        }
        return json.dumps(canonical, sort_keys=True, separators=(',', ':'))


@dataclass
class MerkleNode:
    """
    Node in the Merkle tree.

    Attributes:
        hash: SHA-256 hash of this node
        left: Left child node (None for leaves)
        right: Right child node (None for leaves)
        is_leaf: Whether this is a leaf node
        evidence_id: Evidence ID (for leaf nodes)
    """

    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    evidence_id: Optional[str] = None


@dataclass
class MerkleProof:
    """
    Merkle proof for evidence verification.

    Attributes:
        evidence_id: ID of evidence being proven
        leaf_hash: Hash of the evidence leaf
        proof_path: List of sibling hashes from leaf to root
        directions: List of directions (left/right) for each proof element
        merkle_root: Root hash of the Merkle tree
        tree_size: Number of leaves in the tree
        timestamp: Proof generation timestamp
    """

    evidence_id: str
    leaf_hash: str
    proof_path: List[str]
    directions: List[str]  # "left" or "right"
    merkle_root: str
    tree_size: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "leaf_hash": self.leaf_hash,
            "proof_path": self.proof_path,
            "directions": self.directions,
            "merkle_root": self.merkle_root,
            "tree_size": self.tree_size,
            "timestamp": self.timestamp.isoformat(),
        }


class SealedEvidence(BaseModel):
    """
    Cryptographically sealed evidence package.

    Attributes:
        evidence: Original evidence data
        evidence_hash: SHA-256 hash of evidence
        merkle_root: Root of Merkle tree (if part of batch)
        merkle_proof: Proof of inclusion in Merkle tree
        seal_timestamp: When evidence was sealed
        seal_status: Current seal status
        chain_of_custody: Chain of custody entries
        provenance_hash: Combined provenance hash
        signature: Optional digital signature
    """

    evidence: EvidenceData = Field(
        ...,
        description="Original evidence data"
    )
    evidence_hash: str = Field(
        ...,
        description="SHA-256 hash of evidence"
    )
    merkle_root: Optional[str] = Field(
        default=None,
        description="Merkle tree root hash"
    )
    merkle_proof: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Merkle proof of inclusion"
    )
    seal_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Seal creation timestamp"
    )
    seal_status: SealStatus = Field(
        default=SealStatus.SEALED,
        description="Current seal status"
    )
    chain_of_custody: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chain of custody entries"
    )
    provenance_hash: str = Field(
        default="",
        description="Combined provenance hash"
    )
    signature: Optional[str] = Field(
        default=None,
        description="Optional digital signature"
    )

    @computed_field
    @property
    def is_verified(self) -> bool:
        """Check if seal has been verified."""
        return self.seal_status == SealStatus.VERIFIED


class EvidenceBatch(BaseModel):
    """
    Batch of evidence sealed together in a Merkle tree.

    Attributes:
        batch_id: Unique batch identifier
        sealed_evidence: List of sealed evidence items
        merkle_root: Root hash of the batch Merkle tree
        tree_height: Height of the Merkle tree
        batch_timestamp: Batch creation timestamp
        batch_hash: Hash of entire batch
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch identifier"
    )
    sealed_evidence: List[SealedEvidence] = Field(
        default_factory=list,
        description="List of sealed evidence"
    )
    merkle_root: str = Field(
        default="",
        description="Merkle tree root hash"
    )
    tree_height: int = Field(
        default=0,
        description="Merkle tree height"
    )
    batch_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Batch creation timestamp"
    )
    batch_hash: str = Field(
        default="",
        description="Hash of entire batch"
    )


# =============================================================================
# MERKLE TREE IMPLEMENTATION
# =============================================================================

class MerkleTree:
    """
    Merkle tree implementation for evidence sealing.

    Constructs a binary hash tree from evidence leaves, enabling
    efficient proof of inclusion verification.

    Attributes:
        leaves: Leaf nodes of the tree
        root: Root node of the tree
        height: Height of the tree
    """

    def __init__(self) -> None:
        """Initialize empty Merkle tree."""
        self.leaves: List[MerkleNode] = []
        self.root: Optional[MerkleNode] = None
        self.height: int = 0
        self._leaf_map: Dict[str, int] = {}  # evidence_id -> leaf index

    def add_leaf(self, evidence: EvidenceData) -> str:
        """
        Add evidence as a leaf to the tree.

        Args:
            evidence: Evidence data to add

        Returns:
            SHA-256 hash of the leaf
        """
        # Compute leaf hash
        leaf_hash = self._hash_leaf(evidence)

        # Create leaf node
        leaf = MerkleNode(
            hash=leaf_hash,
            is_leaf=True,
            evidence_id=evidence.evidence_id,
        )

        # Store mapping
        self._leaf_map[evidence.evidence_id] = len(self.leaves)
        self.leaves.append(leaf)

        return leaf_hash

    def build(self) -> str:
        """
        Build the Merkle tree from leaves.

        Returns:
            Root hash of the tree
        """
        if not self.leaves:
            return ""

        # If odd number of leaves, duplicate last
        nodes = self.leaves.copy()
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])

        # Build tree bottom-up
        self.height = 0
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                # Compute parent hash
                parent_hash = self._hash_nodes(left.hash, right.hash)

                parent = MerkleNode(
                    hash=parent_hash,
                    left=left,
                    right=right,
                    is_leaf=False,
                )
                next_level.append(parent)

            nodes = next_level
            self.height += 1

        self.root = nodes[0]
        return self.root.hash

    def get_proof(self, evidence_id: str) -> Optional[MerkleProof]:
        """
        Generate Merkle proof for an evidence item.

        Args:
            evidence_id: ID of evidence to prove

        Returns:
            MerkleProof or None if evidence not found
        """
        if evidence_id not in self._leaf_map or self.root is None:
            return None

        leaf_index = self._leaf_map[evidence_id]
        leaf = self.leaves[leaf_index]

        proof_path = []
        directions = []

        # Navigate from leaf to root
        current_index = leaf_index
        level_nodes = self.leaves.copy()

        # Pad to power of 2
        while len(level_nodes) & (len(level_nodes) - 1) != 0:
            level_nodes.append(level_nodes[-1])

        while len(level_nodes) > 1:
            # Determine sibling
            if current_index % 2 == 0:
                # We're on left, sibling is on right
                sibling_index = current_index + 1
                directions.append("right")
            else:
                # We're on right, sibling is on left
                sibling_index = current_index - 1
                directions.append("left")

            if sibling_index < len(level_nodes):
                proof_path.append(level_nodes[sibling_index].hash)

            # Move to parent level
            next_level = []
            for i in range(0, len(level_nodes), 2):
                left = level_nodes[i]
                right = level_nodes[i + 1] if i + 1 < len(level_nodes) else left
                parent_hash = self._hash_nodes(left.hash, right.hash)
                next_level.append(MerkleNode(hash=parent_hash))

            level_nodes = next_level
            current_index = current_index // 2

        return MerkleProof(
            evidence_id=evidence_id,
            leaf_hash=leaf.hash,
            proof_path=proof_path,
            directions=directions,
            merkle_root=self.root.hash,
            tree_size=len(self.leaves),
        )

    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a Merkle proof.

        Args:
            proof: Merkle proof to verify

        Returns:
            True if proof is valid, False otherwise
        """
        current_hash = proof.leaf_hash

        for sibling_hash, direction in zip(proof.proof_path, proof.directions):
            if direction == "left":
                current_hash = self._hash_nodes(sibling_hash, current_hash)
            else:
                current_hash = self._hash_nodes(current_hash, sibling_hash)

        return current_hash == proof.merkle_root

    def _hash_leaf(self, evidence: EvidenceData) -> str:
        """
        Hash evidence data for leaf node.

        Uses SHA-256 with "leaf:" prefix to prevent second preimage attacks.
        """
        prefix = b"leaf:"
        data = evidence.to_canonical_string().encode('utf-8')
        return hashlib.sha256(prefix + data).hexdigest()

    def _hash_nodes(self, left: str, right: str) -> str:
        """
        Hash two child nodes for parent node.

        Uses SHA-256 with "node:" prefix to prevent second preimage attacks.
        """
        prefix = b"node:"
        data = (left + right).encode('utf-8')
        return hashlib.sha256(prefix + data).hexdigest()


# =============================================================================
# EVIDENCE SEALER
# =============================================================================

class EvidenceSealer:
    """
    Cryptographic evidence sealer for audit trails.

    Provides tamper-evident sealing of furnace monitoring evidence
    using SHA-256 hashing and Merkle trees.

    Features:
        - Individual evidence sealing
        - Batch sealing with Merkle trees
        - Proof generation and verification
        - Chain of custody tracking
        - Regulatory compliance metadata

    Reference:
        NFPA 86 Section 7.13 - Records
        ASME PTC 4.1 - Data retention requirements

    Example:
        >>> sealer = EvidenceSealer()
        >>> evidence = EvidenceData(
        ...     evidence_type=EvidenceType.SAFETY_EVENT,
        ...     furnace_id="FH-101",
        ...     data={"event": "HIGH_TMT_ALARM", "max_tmt": 975.5}
        ... )
        >>> sealed = sealer.seal_evidence(evidence)
        >>> print(f"Hash: {sealed.evidence_hash}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-007",
        include_timestamp: bool = True,
        auto_custody: bool = True,
    ) -> None:
        """
        Initialize evidence sealer.

        Args:
            agent_id: Agent identifier for custody entries
            include_timestamp: Include seal timestamp in hash
            auto_custody: Automatically add custody entry on seal
        """
        self.agent_id = agent_id
        self.include_timestamp = include_timestamp
        self.auto_custody = auto_custody
        self._sealed_count = 0
        self._batch_count = 0

        logger.info(f"Initialized EvidenceSealer v{self.VERSION}")

    def seal_evidence(
        self,
        evidence: EvidenceData,
        custody_entry: Optional[Dict[str, Any]] = None,
    ) -> SealedEvidence:
        """
        Seal individual evidence with SHA-256 hash.

        Args:
            evidence: Evidence data to seal
            custody_entry: Optional chain of custody entry

        Returns:
            SealedEvidence with hash and metadata
        """
        # Compute evidence hash
        evidence_hash = self._compute_evidence_hash(evidence)

        # Build chain of custody
        chain_of_custody = []
        if self.auto_custody:
            chain_of_custody.append({
                "action": "sealed",
                "agent": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hash": evidence_hash,
            })

        if custody_entry:
            chain_of_custody.append(custody_entry)

        # Compute provenance hash (includes custody)
        provenance_hash = self._compute_provenance_hash(
            evidence_hash, chain_of_custody
        )

        sealed = SealedEvidence(
            evidence=evidence,
            evidence_hash=evidence_hash,
            seal_timestamp=datetime.now(timezone.utc),
            seal_status=SealStatus.SEALED,
            chain_of_custody=chain_of_custody,
            provenance_hash=provenance_hash,
        )

        self._sealed_count += 1
        logger.info(f"Sealed evidence {evidence.evidence_id}: {evidence_hash[:16]}...")

        return sealed

    def seal_batch(
        self,
        evidence_list: List[EvidenceData],
    ) -> EvidenceBatch:
        """
        Seal batch of evidence using Merkle tree.

        Args:
            evidence_list: List of evidence to seal

        Returns:
            EvidenceBatch with Merkle root and proofs
        """
        if not evidence_list:
            raise ValueError("Cannot seal empty batch")

        # Build Merkle tree
        tree = MerkleTree()

        for evidence in evidence_list:
            tree.add_leaf(evidence)

        merkle_root = tree.build()

        # Seal each evidence with Merkle proof
        sealed_evidence = []
        for evidence in evidence_list:
            # Get individual hash
            evidence_hash = self._compute_evidence_hash(evidence)

            # Get Merkle proof
            proof = tree.get_proof(evidence.evidence_id)

            # Build custody chain
            chain_of_custody = [{
                "action": "batch_sealed",
                "agent": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hash": evidence_hash,
                "merkle_root": merkle_root,
                "batch_size": len(evidence_list),
            }]

            provenance_hash = self._compute_provenance_hash(
                evidence_hash, chain_of_custody
            )

            sealed = SealedEvidence(
                evidence=evidence,
                evidence_hash=evidence_hash,
                merkle_root=merkle_root,
                merkle_proof=proof.to_dict() if proof else None,
                seal_timestamp=datetime.now(timezone.utc),
                seal_status=SealStatus.SEALED,
                chain_of_custody=chain_of_custody,
                provenance_hash=provenance_hash,
            )
            sealed_evidence.append(sealed)

        # Compute batch hash
        batch_hash = self._compute_batch_hash(merkle_root, len(evidence_list))

        batch = EvidenceBatch(
            sealed_evidence=sealed_evidence,
            merkle_root=merkle_root,
            tree_height=tree.height,
            batch_timestamp=datetime.now(timezone.utc),
            batch_hash=batch_hash,
        )

        self._batch_count += 1
        logger.info(
            f"Sealed batch {batch.batch_id}: {len(evidence_list)} items, "
            f"root {merkle_root[:16]}..."
        )

        return batch

    def verify_evidence(
        self,
        sealed: SealedEvidence,
    ) -> Tuple[bool, str]:
        """
        Verify integrity of sealed evidence.

        Args:
            sealed: Sealed evidence to verify

        Returns:
            Tuple of (is_valid, message)
        """
        # Recompute evidence hash
        expected_hash = self._compute_evidence_hash(sealed.evidence)

        if expected_hash != sealed.evidence_hash:
            return False, "Evidence hash mismatch - data may be tampered"

        # Verify Merkle proof if present
        if sealed.merkle_proof and sealed.merkle_root:
            proof = MerkleProof(
                evidence_id=sealed.evidence.evidence_id,
                leaf_hash=sealed.merkle_proof["leaf_hash"],
                proof_path=sealed.merkle_proof["proof_path"],
                directions=sealed.merkle_proof["directions"],
                merkle_root=sealed.merkle_proof["merkle_root"],
                tree_size=sealed.merkle_proof["tree_size"],
            )

            tree = MerkleTree()
            if not tree.verify_proof(proof):
                return False, "Merkle proof verification failed"

            if proof.merkle_root != sealed.merkle_root:
                return False, "Merkle root mismatch"

        # Verify provenance hash
        expected_provenance = self._compute_provenance_hash(
            sealed.evidence_hash, sealed.chain_of_custody
        )

        if expected_provenance != sealed.provenance_hash:
            return False, "Provenance hash mismatch - custody chain may be tampered"

        return True, "Evidence verified successfully"

    def verify_batch(
        self,
        batch: EvidenceBatch,
    ) -> Tuple[bool, List[str]]:
        """
        Verify integrity of entire batch.

        Args:
            batch: Evidence batch to verify

        Returns:
            Tuple of (all_valid, list of messages)
        """
        messages = []
        all_valid = True

        # Verify each sealed evidence
        for sealed in batch.sealed_evidence:
            is_valid, msg = self.verify_evidence(sealed)
            if not is_valid:
                all_valid = False
                messages.append(f"{sealed.evidence.evidence_id}: {msg}")
            else:
                messages.append(f"{sealed.evidence.evidence_id}: OK")

        # Verify batch hash
        expected_batch_hash = self._compute_batch_hash(
            batch.merkle_root, len(batch.sealed_evidence)
        )

        if expected_batch_hash != batch.batch_hash:
            all_valid = False
            messages.append("Batch hash mismatch")

        return all_valid, messages

    def add_custody_entry(
        self,
        sealed: SealedEvidence,
        action: str,
        agent: str,
        notes: Optional[str] = None,
    ) -> SealedEvidence:
        """
        Add chain of custody entry to sealed evidence.

        Args:
            sealed: Sealed evidence to update
            action: Custody action (e.g., "accessed", "transferred")
            agent: Agent performing action
            notes: Optional notes

        Returns:
            Updated SealedEvidence
        """
        entry = {
            "action": action,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_hash": sealed.provenance_hash,
        }

        if notes:
            entry["notes"] = notes

        # Add entry
        sealed.chain_of_custody.append(entry)

        # Update provenance hash
        sealed.provenance_hash = self._compute_provenance_hash(
            sealed.evidence_hash, sealed.chain_of_custody
        )

        return sealed

    def export_for_audit(
        self,
        sealed: SealedEvidence,
    ) -> Dict[str, Any]:
        """
        Export sealed evidence for external audit.

        Args:
            sealed: Sealed evidence to export

        Returns:
            Dictionary with audit-ready format
        """
        return {
            "version": self.VERSION,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_id": sealed.evidence.evidence_id,
            "evidence_type": sealed.evidence.evidence_type.value,
            "furnace_id": sealed.evidence.furnace_id,
            "evidence_timestamp": sealed.evidence.timestamp.isoformat(),
            "evidence_hash": sealed.evidence_hash,
            "merkle_root": sealed.merkle_root,
            "merkle_proof": sealed.merkle_proof,
            "seal_timestamp": sealed.seal_timestamp.isoformat(),
            "seal_status": sealed.seal_status.value,
            "chain_of_custody": sealed.chain_of_custody,
            "provenance_hash": sealed.provenance_hash,
            "compliance_refs": [c.value for c in sealed.evidence.compliance_refs],
            "data": sealed.evidence.data,
            "metadata": sealed.evidence.metadata,
        }

    def _compute_evidence_hash(self, evidence: EvidenceData) -> str:
        """Compute SHA-256 hash of evidence data."""
        canonical = evidence.to_canonical_string()

        if self.include_timestamp:
            canonical = f"{canonical}:{evidence.timestamp.isoformat()}"

        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _compute_provenance_hash(
        self,
        evidence_hash: str,
        chain_of_custody: List[Dict[str, Any]],
    ) -> str:
        """Compute provenance hash including custody chain."""
        data = {
            "evidence_hash": evidence_hash,
            "chain_of_custody": chain_of_custody,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _compute_batch_hash(self, merkle_root: str, count: int) -> str:
        """Compute batch hash from root and count."""
        data = f"{merkle_root}:{count}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def seal_safety_event(
    event_id: str,
    event_type: str,
    furnace_id: str,
    sensor_data: Dict[str, float],
    nfpa_section: Optional[str] = None,
) -> SealedEvidence:
    """
    Convenience function to seal a safety event.

    Args:
        event_id: Event identifier
        event_type: Type of safety event
        furnace_id: Furnace identifier
        sensor_data: Sensor readings at event time
        nfpa_section: Relevant NFPA 86 section

    Returns:
        SealedEvidence package
    """
    evidence = EvidenceData(
        evidence_id=event_id,
        evidence_type=EvidenceType.SAFETY_EVENT,
        furnace_id=furnace_id,
        data={
            "event_type": event_type,
            "sensor_data": sensor_data,
            "nfpa_section": nfpa_section,
        },
        compliance_refs=[ComplianceFramework.NFPA_86],
    )

    sealer = EvidenceSealer()
    return sealer.seal_evidence(evidence)


def seal_compliance_check(
    check_id: str,
    furnace_id: str,
    check_results: Dict[str, Any],
    framework: ComplianceFramework = ComplianceFramework.NFPA_86,
) -> SealedEvidence:
    """
    Convenience function to seal a compliance check.

    Args:
        check_id: Check identifier
        furnace_id: Furnace identifier
        check_results: Results of compliance check
        framework: Compliance framework

    Returns:
        SealedEvidence package
    """
    evidence = EvidenceData(
        evidence_id=check_id,
        evidence_type=EvidenceType.COMPLIANCE_CHECK,
        furnace_id=furnace_id,
        data=check_results,
        compliance_refs=[framework],
    )

    sealer = EvidenceSealer()
    return sealer.seal_evidence(evidence)


def verify_evidence_chain(
    evidence_list: List[SealedEvidence],
) -> Tuple[bool, List[str]]:
    """
    Verify a chain of related evidence.

    Args:
        evidence_list: List of sealed evidence to verify

    Returns:
        Tuple of (all_valid, list of results)
    """
    sealer = EvidenceSealer()
    results = []
    all_valid = True

    for sealed in evidence_list:
        is_valid, msg = sealer.verify_evidence(sealed)
        results.append(f"{sealed.evidence.evidence_id}: {msg}")
        if not is_valid:
            all_valid = False

    return all_valid, results
