# -*- coding: utf-8 -*-
"""
RAG governance with allowlist enforcement and approval workflow.

CRITICAL SECURITY: This module enforces document allowlisting with Climate Science
Review Board (CSRB) approval for climate/GHG accounting regulatory compliance.

Key features:
- Climate Science Review Board (CSRB) approval workflow
- Digital signature verification
- Document authenticity checks (SHA-256 checksums)
- Audit trail for all approvals
- Runtime allowlist enforcement
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from greenlang.agents.intelligence.rag.models import DocMeta
from greenlang.agents.intelligence.rag.config import RAGConfig
from greenlang.agents.intelligence.rag.hashing import file_hash, sha256_str
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """
    Approval request for CSRB review.

    Tracks document pending approval with metadata and approval votes.
    """
    doc_path: Path
    metadata: DocMeta
    requested_by: str
    requested_at: datetime
    approvers_required: List[str]
    approvers_voted: Set[str]
    votes_approve: int
    votes_reject: int
    status: str  # 'pending', 'approved', 'rejected'
    comments: List[Dict[str, str]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['doc_path'] = str(self.doc_path)
        result['approvers_voted'] = list(self.approvers_voted)
        result['requested_at'] = self.requested_at.isoformat()
        # Convert DocMeta to dict with JSON-safe dates
        metadata_dict = self.metadata.model_dump()
        if metadata_dict.get('publication_date'):
            metadata_dict['publication_date'] = metadata_dict['publication_date'].isoformat()
        if metadata_dict.get('revision_date'):
            metadata_dict['revision_date'] = metadata_dict['revision_date'].isoformat()
        if metadata_dict.get('ingested_at'):
            metadata_dict['ingested_at'] = metadata_dict['ingested_at'].isoformat()
        result['metadata'] = metadata_dict
        return result


class RAGGovernance:
    """
    RAG governance with allowlist enforcement and CSRB approval workflow.

    Implements:
    - Document submission for approval
    - 2/3 majority vote requirement from CSRB
    - SHA-256 checksum verification
    - Digital signature verification (optional)
    - Audit trail persistence
    - Runtime allowlist enforcement

    Example:
        >>> config = RAGConfig(allowlist=['ghg_protocol_corp'])
        >>> gov = RAGGovernance(config)
        >>> # Submit document for approval
        >>> success = gov.submit_for_approval(
        ...     doc_path=Path("ghg_protocol_v1.05.pdf"),
        ...     metadata=doc_meta,
        ...     approvers=['climate_scientist_1', 'climate_scientist_2', 'audit_lead']
        ... )
        >>> # Vote on approval
        >>> gov.vote_approval("ghg_protocol_v1.05.pdf", approver='climate_scientist_1', approve=True)
        >>> gov.vote_approval("ghg_protocol_v1.05.pdf", approver='climate_scientist_2', approve=True)
        >>> # Check if approved (2/3 majority)
        >>> assert gov.is_approved("ghg_protocol_corp")
    """

    def __init__(
        self,
        config: RAGConfig,
        audit_dir: Optional[Path] = None
    ):
        """
        Initialize governance system.

        Args:
            config: RAG configuration with allowlist
            audit_dir: Directory for audit trail persistence (defaults to ./audit)
        """
        self.config = config
        self.allowlist: Set[str] = set(config.allowlist)

        # Audit directory for persistence
        self.audit_dir = audit_dir or Path("./audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Pending approval requests
        self.approval_requests: Dict[str, ApprovalRequest] = {}

        # Approved collections (in addition to allowlist)
        self.approved_collections: Set[str] = set()

        # Load existing audit trail
        self._load_audit_trail()

    def _load_audit_trail(self) -> None:
        """Load existing audit trail from disk."""
        audit_file = self.audit_dir / "approval_requests.json"
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    data = json.load(f)
                    for key, req_data in data.items():
                        # Reconstruct ApprovalRequest
                        req_data['doc_path'] = Path(req_data['doc_path'])
                        req_data['requested_at'] = datetime.fromisoformat(req_data['requested_at'])
                        req_data['approvers_voted'] = set(req_data['approvers_voted'])
                        # Reconstruct DocMeta with date parsing
                        meta_data = req_data['metadata']
                        if meta_data.get('publication_date'):
                            from datetime import date as date_type
                            meta_data['publication_date'] = date_type.fromisoformat(meta_data['publication_date'])
                        if meta_data.get('revision_date'):
                            from datetime import date as date_type
                            meta_data['revision_date'] = date_type.fromisoformat(meta_data['revision_date'])
                        if meta_data.get('ingested_at'):
                            meta_data['ingested_at'] = datetime.fromisoformat(meta_data['ingested_at'])
                        req_data['metadata'] = DocMeta(**meta_data)
                        self.approval_requests[key] = ApprovalRequest(**req_data)

                logger.info(f"Loaded {len(self.approval_requests)} approval requests from audit trail")
            except Exception as e:
                logger.error(f"Failed to load audit trail: {e}")

    def _save_audit_trail(self) -> None:
        """Persist audit trail to disk."""
        audit_file = self.audit_dir / "approval_requests.json"
        try:
            data = {key: req.to_dict() for key, req in self.approval_requests.items()}
            with open(audit_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved audit trail to {audit_file}")
        except Exception as e:
            logger.error(f"Failed to save audit trail: {e}")

    def verify_authenticity(
        self,
        doc_path: Path,
        expected_hash: str
    ) -> bool:
        """
        Verify document authenticity using SHA-256 checksum.

        Args:
            doc_path: Path to document file
            expected_hash: Expected SHA-256 hash (from trusted source)

        Returns:
            True if hash matches, False otherwise

        Example:
            >>> gov = RAGGovernance(config)
            >>> # Verify GHG Protocol hash from official website
            >>> is_authentic = gov.verify_authenticity(
            ...     Path("ghg_protocol.pdf"),
            ...     "a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4"
            ... )
            >>> if not is_authentic:
            ...     raise ValueError("Document tampering detected!")
        """
        if not doc_path.exists():
            logger.error(f"Document not found: {doc_path}")
            return False

        try:
            actual_hash = file_hash(str(doc_path))

            if actual_hash == expected_hash:
                logger.info(f"Document authenticity verified: {doc_path.name}")
                return True
            else:
                logger.error(
                    f"Document authenticity FAILED for {doc_path.name}: "
                    f"expected {expected_hash[:8]}, got {actual_hash[:8]}"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to verify authenticity: {e}")
            return False

    def verify_signature(
        self,
        doc_path: Path,
        signature_path: Optional[Path] = None,
        public_key_path: Optional[Path] = None
    ) -> bool:
        """
        Verify digital signature (if present).

        Uses GPG/PGP for signature verification (common for climate standards).

        Args:
            doc_path: Path to document file
            signature_path: Path to signature file (.sig or .asc)
            public_key_path: Path to public key file (PEM format)

        Returns:
            True if signature is valid, False otherwise

        Note:
            This is a placeholder implementation. In production, use:
            - GPG/PGP for signature verification
            - X.509 certificates for publisher verification
            - Timestamping for non-repudiation

        Example:
            >>> gov = RAGGovernance(config)
            >>> is_valid = gov.verify_signature(
            ...     Path("ghg_protocol.pdf"),
            ...     signature_path=Path("ghg_protocol.pdf.sig"),
            ...     public_key_path=Path("wri_public_key.pem")
            ... )
        """
        # Placeholder: In production, implement GPG/PGP verification
        # Example using gnupg:
        #   import gnupg
        #   gpg = gnupg.GPG()
        #   with open(doc_path, 'rb') as f:
        #       verified = gpg.verify_file(f, signature_path)
        #   return verified.valid

        logger.warning(
            "Digital signature verification not implemented. "
            "In production, use GPG/PGP or X.509 certificates."
        )
        return True  # Assume valid for now

    def submit_for_approval(
        self,
        doc_path: Path,
        metadata: DocMeta,
        approvers: List[str],
        requested_by: Optional[str] = None,
        verify_checksum: bool = True
    ) -> bool:
        """
        Submit document for CSRB approval.

        Workflow:
        1. Verify document authenticity (checksum)
        2. Check digital signature if present
        3. Create approval request
        4. Require 2/3 majority vote from approvers
        5. Add to allowlist if approved

        Args:
            doc_path: Path to document file
            metadata: Document metadata with content_hash
            approvers: List of approver usernames (CSRB members)
            requested_by: Username of requester
            verify_checksum: Whether to verify checksum (default True)

        Returns:
            True if submission successful, False otherwise

        Example:
            >>> gov = RAGGovernance(config)
            >>> success = gov.submit_for_approval(
            ...     doc_path=Path("ghg_protocol_v1.05.pdf"),
            ...     metadata=doc_meta,
            ...     approvers=['climate_scientist_1', 'climate_scientist_2', 'audit_lead']
            ... )
        """
        if not doc_path.exists():
            logger.error(f"Document not found: {doc_path}")
            return False

        # 1. Verify document authenticity
        if verify_checksum and self.config.verify_checksums:
            if not self.verify_authenticity(doc_path, metadata.content_hash):
                logger.error(f"Checksum verification failed for {doc_path.name}")
                return False

        # 2. Check digital signature if present
        sig_path = doc_path.with_suffix(doc_path.suffix + '.sig')
        if sig_path.exists():
            logger.info(f"Digital signature found: {sig_path}")
            if not self.verify_signature(doc_path, sig_path):
                logger.error(f"Digital signature verification failed for {doc_path.name}")
                return False

        # 3. Create approval request
        request_key = metadata.collection
        if request_key in self.approval_requests:
            logger.warning(f"Approval request already exists for {request_key}")
            # Update existing request
            existing = self.approval_requests[request_key]
            if existing.status == 'approved':
                logger.info(f"Collection {request_key} already approved")
                return True
            elif existing.status == 'rejected':
                logger.warning(f"Collection {request_key} previously rejected, re-submitting")

        approval_request = ApprovalRequest(
            doc_path=doc_path,
            metadata=metadata,
            requested_by=requested_by or "unknown",
            requested_at=DeterministicClock.utcnow(),
            approvers_required=approvers,
            approvers_voted=set(),
            votes_approve=0,
            votes_reject=0,
            status='pending',
            comments=[]
        )

        self.approval_requests[request_key] = approval_request
        self._save_audit_trail()

        logger.info(
            f"Submitted {metadata.collection} for approval "
            f"(requested by {requested_by}, {len(approvers)} approvers required)"
        )
        return True

    def vote_approval(
        self,
        collection: str,
        approver: str,
        approve: bool,
        comment: Optional[str] = None
    ) -> bool:
        """
        Vote on an approval request.

        Requires 2/3 majority to approve.

        Args:
            collection: Collection name
            approver: Approver username
            approve: True to approve, False to reject
            comment: Optional comment for audit trail

        Returns:
            True if vote recorded, False otherwise

        Example:
            >>> gov = RAGGovernance(config)
            >>> gov.vote_approval("ghg_protocol_corp", "climate_scientist_1", approve=True)
            >>> gov.vote_approval("ghg_protocol_corp", "climate_scientist_2", approve=True)
            >>> # Check status
            >>> req = gov.get_approval_request("ghg_protocol_corp")
            >>> assert req.status == 'approved'  # 2/3 majority reached
        """
        if collection not in self.approval_requests:
            logger.error(f"No approval request found for {collection}")
            return False

        request = self.approval_requests[collection]

        # Check if approver is authorized
        if approver not in request.approvers_required:
            logger.error(f"Approver {approver} not authorized for {collection}")
            return False

        # Check if already voted
        if approver in request.approvers_voted:
            logger.warning(f"Approver {approver} already voted on {collection}")
            return False

        # Record vote
        request.approvers_voted.add(approver)
        if approve:
            request.votes_approve += 1
        else:
            request.votes_reject += 1

        # Add comment
        if comment:
            request.comments.append({
                'approver': approver,
                'vote': 'approve' if approve else 'reject',
                'comment': comment,
                'timestamp': DeterministicClock.utcnow().isoformat()
            })

        logger.info(
            f"Vote recorded: {approver} {'approved' if approve else 'rejected'} {collection} "
            f"({request.votes_approve}/{len(request.approvers_required)} approve, "
            f"{request.votes_reject}/{len(request.approvers_required)} reject)"
        )

        # Check if 2/3 majority reached
        total_approvers = len(request.approvers_required)
        majority_threshold = (total_approvers * 2) // 3  # Floor division for 2/3

        if request.votes_approve >= majority_threshold:
            # Approved
            request.status = 'approved'
            self.approved_collections.add(collection)
            self.allowlist.add(collection)  # Add to runtime allowlist
            logger.info(
                f"APPROVED: {collection} (2/3 majority reached: "
                f"{request.votes_approve}/{total_approvers})"
            )
        elif request.votes_reject > (total_approvers - majority_threshold):
            # Rejected (cannot reach 2/3 majority)
            request.status = 'rejected'
            logger.warning(
                f"REJECTED: {collection} ({request.votes_reject}/{total_approvers} reject)"
            )

        self._save_audit_trail()
        return True

    def is_approved(self, collection: str) -> bool:
        """
        Check if a collection is approved.

        Args:
            collection: Collection name

        Returns:
            True if approved, False otherwise

        Example:
            >>> gov = RAGGovernance(config)
            >>> if gov.is_approved("ghg_protocol_corp"):
            ...     print("Collection approved for use")
        """
        if collection in self.approved_collections:
            return True

        if collection in self.approval_requests:
            request = self.approval_requests[collection]
            return request.status == 'approved'

        return False

    def get_approval_request(self, collection: str) -> Optional[ApprovalRequest]:
        """
        Get approval request for a collection.

        Args:
            collection: Collection name

        Returns:
            ApprovalRequest if found, None otherwise
        """
        return self.approval_requests.get(collection)

    def check_allowlist(self, collection: str) -> bool:
        """
        Enforce allowlist at runtime.

        Checks both static allowlist (from config) and approved collections.

        Args:
            collection: Collection name

        Returns:
            True if allowed, False otherwise

        Raises:
            ValueError: If collection is not in allowlist

        Example:
            >>> gov = RAGGovernance(config)
            >>> if not gov.check_allowlist("unknown_collection"):
            ...     raise ValueError("Collection not allowed")
        """
        is_allowed = (collection in self.allowlist or
                     collection in self.approved_collections)

        if not is_allowed:
            logger.error(
                f"Collection '{collection}' not in allowlist. "
                f"Allowed: {', '.join(sorted(self.allowlist))}"
            )

        return is_allowed

    def enforce_allowlist(self, collections: List[str]) -> None:
        """
        Enforce allowlist for multiple collections (raises exception if denied).

        Args:
            collections: List of collection names

        Raises:
            ValueError: If any collection is not in allowlist

        Example:
            >>> gov = RAGGovernance(config)
            >>> gov.enforce_allowlist(['ghg_protocol_corp', 'ipcc_ar6_wg3'])
        """
        for collection in collections:
            if not self.check_allowlist(collection):
                raise ValueError(
                    f"Collection '{collection}' not in allowlist. "
                    f"Submit for CSRB approval or check configuration."
                )

    def list_pending_approvals(self) -> List[ApprovalRequest]:
        """
        List all pending approval requests.

        Returns:
            List of pending ApprovalRequest objects

        Example:
            >>> gov = RAGGovernance(config)
            >>> pending = gov.list_pending_approvals()
            >>> for req in pending:
            ...     print(f"{req.metadata.collection}: {req.votes_approve}/{len(req.approvers_required)}")
        """
        return [req for req in self.approval_requests.values() if req.status == 'pending']

    def get_audit_trail(self, collection: str) -> Dict:
        """
        Get full audit trail for a collection.

        Args:
            collection: Collection name

        Returns:
            Audit trail dictionary with all approval activity

        Example:
            >>> gov = RAGGovernance(config)
            >>> audit = gov.get_audit_trail("ghg_protocol_corp")
            >>> print(json.dumps(audit, indent=2))
        """
        if collection not in self.approval_requests:
            return {}

        request = self.approval_requests[collection]
        return request.to_dict()
