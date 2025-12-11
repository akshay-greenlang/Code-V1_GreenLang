"""
Certification Registry

Tracks certified agents, version history, and certification lifecycle:
- Certified agent registry
- Version history tracking
- Expiration monitoring
- Re-certification triggers

Example:
    >>> from certification.registry import CertificationRegistry
    >>> registry = CertificationRegistry("./certs")
    >>> registry.register(certification_report)
    >>> status = registry.get_status("agent_id", "1.0.0")

"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CertificationStatus(Enum):
    """Certification status values."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_RENEWAL = "pending_renewal"
    NOT_CERTIFIED = "not_certified"


@dataclass
class CertifiedAgent:
    """Record of a certified agent."""
    agent_id: str
    agent_version: str
    certification_id: str
    certification_level: str
    overall_score: float
    certified_at: datetime
    valid_until: datetime
    status: CertificationStatus = CertificationStatus.ACTIVE
    pack_yaml_path: str = ""
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if certification is currently valid."""
        now = datetime.utcnow()
        return (
            self.status == CertificationStatus.ACTIVE and
            self.valid_until > now
        )

    @property
    def days_until_expiry(self) -> int:
        """Days until certification expires."""
        delta = self.valid_until - datetime.utcnow()
        return max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "certification_id": self.certification_id,
            "certification_level": self.certification_level,
            "overall_score": self.overall_score,
            "certified_at": self.certified_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "status": self.status.value,
            "pack_yaml_path": self.pack_yaml_path,
            "dimension_scores": self.dimension_scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CertifiedAgent":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_version=data["agent_version"],
            certification_id=data["certification_id"],
            certification_level=data["certification_level"],
            overall_score=data["overall_score"],
            certified_at=datetime.fromisoformat(data["certified_at"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            status=CertificationStatus(data.get("status", "active")),
            pack_yaml_path=data.get("pack_yaml_path", ""),
            dimension_scores=data.get("dimension_scores", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentVersionHistory:
    """Version history for an agent."""
    agent_id: str
    versions: List[CertifiedAgent] = field(default_factory=list)

    @property
    def latest_version(self) -> Optional[CertifiedAgent]:
        """Get latest certified version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.certified_at)

    @property
    def active_versions(self) -> List[CertifiedAgent]:
        """Get all active certifications."""
        return [v for v in self.versions if v.is_valid]

    def get_version(self, version: str) -> Optional[CertifiedAgent]:
        """Get specific version."""
        for v in self.versions:
            if v.agent_version == version:
                return v
        return None


class CertificationRegistry:
    """
    Central registry for tracking agent certifications.

    Features:
    - Store and retrieve certifications
    - Track version history
    - Monitor expirations
    - Manage re-certification triggers
    """

    # Default certification validity period (days)
    DEFAULT_VALIDITY_DAYS = 90

    # Warning threshold for expiration (days)
    EXPIRY_WARNING_DAYS = 30

    def __init__(
        self,
        storage_path: Union[str, Path] = "./certification_registry",
        validity_days: int = DEFAULT_VALIDITY_DAYS,
    ):
        """
        Initialize certification registry.

        Args:
            storage_path: Path to store certification records
            validity_days: Default validity period in days
        """
        self.storage_path = Path(storage_path)
        self.validity_days = validity_days
        self._agents: Dict[str, AgentVersionHistory] = {}

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing certifications
        self._load_registry()

        logger.info(f"CertificationRegistry initialized at {self.storage_path}")

    def _load_registry(self) -> None:
        """Load existing certifications from storage."""
        registry_file = self.storage_path / "registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for agent_id, versions in data.get("agents", {}).items():
                    history = AgentVersionHistory(agent_id=agent_id)
                    for version_data in versions:
                        cert = CertifiedAgent.from_dict(version_data)
                        history.versions.append(cert)
                    self._agents[agent_id] = history

                logger.info(f"Loaded {len(self._agents)} agents from registry")

            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to storage."""
        registry_file = self.storage_path / "registry.json"

        data = {
            "version": "1.0.0",
            "updated_at": datetime.utcnow().isoformat(),
            "agents": {
                agent_id: [v.to_dict() for v in history.versions]
                for agent_id, history in self._agents.items()
            },
        }

        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Registry saved")

    def register(
        self,
        certification_report: Any,  # CertificationReport
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CertifiedAgent:
        """
        Register a new certification.

        Args:
            certification_report: Certification report
            metadata: Optional additional metadata

        Returns:
            CertifiedAgent record
        """
        if not certification_report.is_certified:
            raise ValueError("Cannot register non-certified agent")

        # Extract dimension scores
        dimension_scores = {
            name: result.score
            for name, result in certification_report.dimension_results.items()
        }

        # Create certified agent record
        cert = CertifiedAgent(
            agent_id=certification_report.agent_id,
            agent_version=certification_report.agent_version,
            certification_id=certification_report.certification_id,
            certification_level=certification_report.certification_level.value,
            overall_score=certification_report.overall_score,
            certified_at=certification_report.timestamp,
            valid_until=certification_report.valid_until or (
                certification_report.timestamp + timedelta(days=self.validity_days)
            ),
            status=CertificationStatus.ACTIVE,
            pack_yaml_path=certification_report.pack_yaml_path,
            dimension_scores=dimension_scores,
            metadata=metadata or {},
        )

        # Add to registry
        if cert.agent_id not in self._agents:
            self._agents[cert.agent_id] = AgentVersionHistory(agent_id=cert.agent_id)

        self._agents[cert.agent_id].versions.append(cert)

        # Save to storage
        self._save_registry()

        # Save individual certificate
        self._save_certificate(cert)

        logger.info(
            f"Registered certification: {cert.certification_id} "
            f"for {cert.agent_id} v{cert.agent_version}"
        )

        return cert

    def _save_certificate(self, cert: CertifiedAgent) -> None:
        """Save individual certificate to storage."""
        cert_dir = self.storage_path / "certificates"
        cert_dir.mkdir(exist_ok=True)

        cert_file = cert_dir / f"{cert.certification_id}.json"

        with open(cert_file, "w", encoding="utf-8") as f:
            json.dump(cert.to_dict(), f, indent=2)

    def get_status(
        self,
        agent_id: str,
        version: Optional[str] = None,
    ) -> CertificationStatus:
        """
        Get certification status for an agent.

        Args:
            agent_id: Agent identifier
            version: Optional specific version

        Returns:
            CertificationStatus
        """
        if agent_id not in self._agents:
            return CertificationStatus.NOT_CERTIFIED

        history = self._agents[agent_id]

        if version:
            cert = history.get_version(version)
            if cert is None:
                return CertificationStatus.NOT_CERTIFIED
        else:
            cert = history.latest_version
            if cert is None:
                return CertificationStatus.NOT_CERTIFIED

        # Check expiration
        self._update_status(cert)

        return cert.status

    def _update_status(self, cert: CertifiedAgent) -> None:
        """Update certification status based on expiration."""
        now = datetime.utcnow()

        if cert.status == CertificationStatus.REVOKED:
            return  # Don't change revoked status

        if cert.valid_until < now:
            cert.status = CertificationStatus.EXPIRED
        elif cert.days_until_expiry <= self.EXPIRY_WARNING_DAYS:
            cert.status = CertificationStatus.PENDING_RENEWAL
        else:
            cert.status = CertificationStatus.ACTIVE

    def get_certificate(
        self,
        agent_id: str,
        version: Optional[str] = None,
    ) -> Optional[CertifiedAgent]:
        """
        Get certification record for an agent.

        Args:
            agent_id: Agent identifier
            version: Optional specific version

        Returns:
            CertifiedAgent or None
        """
        if agent_id not in self._agents:
            return None

        history = self._agents[agent_id]

        if version:
            cert = history.get_version(version)
        else:
            cert = history.latest_version

        if cert:
            self._update_status(cert)

        return cert

    def get_version_history(
        self,
        agent_id: str,
    ) -> Optional[AgentVersionHistory]:
        """
        Get full version history for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentVersionHistory or None
        """
        if agent_id not in self._agents:
            return None

        history = self._agents[agent_id]

        # Update statuses
        for cert in history.versions:
            self._update_status(cert)

        return history

    def revoke(
        self,
        agent_id: str,
        version: str,
        reason: str,
    ) -> bool:
        """
        Revoke a certification.

        Args:
            agent_id: Agent identifier
            version: Version to revoke
            reason: Revocation reason

        Returns:
            True if revoked successfully
        """
        if agent_id not in self._agents:
            return False

        history = self._agents[agent_id]
        cert = history.get_version(version)

        if cert is None:
            return False

        cert.status = CertificationStatus.REVOKED
        cert.metadata["revoked_at"] = datetime.utcnow().isoformat()
        cert.metadata["revocation_reason"] = reason

        self._save_registry()
        self._save_certificate(cert)

        logger.warning(
            f"Revoked certification: {cert.certification_id} "
            f"for {agent_id} v{version}. Reason: {reason}"
        )

        return True

    def get_expiring_soon(
        self,
        days: int = EXPIRY_WARNING_DAYS,
    ) -> List[CertifiedAgent]:
        """
        Get certifications expiring soon.

        Args:
            days: Number of days threshold

        Returns:
            List of expiring certifications
        """
        expiring = []

        for history in self._agents.values():
            for cert in history.versions:
                self._update_status(cert)
                if (
                    cert.status == CertificationStatus.ACTIVE or
                    cert.status == CertificationStatus.PENDING_RENEWAL
                ):
                    if cert.days_until_expiry <= days:
                        expiring.append(cert)

        return sorted(expiring, key=lambda c: c.valid_until)

    def get_all_active(self) -> List[CertifiedAgent]:
        """
        Get all active certifications.

        Returns:
            List of active certifications
        """
        active = []

        for history in self._agents.values():
            for cert in history.versions:
                self._update_status(cert)
                if cert.status == CertificationStatus.ACTIVE:
                    active.append(cert)

        return active

    def needs_recertification(
        self,
        agent_id: str,
        version: str,
        triggers: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if agent needs re-certification.

        Args:
            agent_id: Agent identifier
            version: Agent version
            triggers: Optional list of trigger conditions

        Returns:
            Tuple of (needs_recert, reasons)
        """
        triggers = triggers or []
        reasons = []

        cert = self.get_certificate(agent_id, version)

        if cert is None:
            return True, ["No existing certification"]

        # Check expiration
        if cert.status == CertificationStatus.EXPIRED:
            reasons.append("Certification expired")
        elif cert.status == CertificationStatus.PENDING_RENEWAL:
            reasons.append(f"Certification expires in {cert.days_until_expiry} days")
        elif cert.status == CertificationStatus.REVOKED:
            reasons.append("Certification was revoked")

        # Check triggers
        if "pack_changed" in triggers:
            reasons.append("Pack specification changed")
        if "dependency_updated" in triggers:
            reasons.append("Dependencies updated")
        if "regulation_changed" in triggers:
            reasons.append("Regulatory requirements changed")
        if "security_vulnerability" in triggers:
            reasons.append("Security vulnerability reported")

        return len(reasons) > 0, reasons

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate registry summary report.

        Returns:
            Dictionary with registry statistics
        """
        all_certs = []
        for history in self._agents.values():
            for cert in history.versions:
                self._update_status(cert)
                all_certs.append(cert)

        active = [c for c in all_certs if c.status == CertificationStatus.ACTIVE]
        expired = [c for c in all_certs if c.status == CertificationStatus.EXPIRED]
        pending = [c for c in all_certs if c.status == CertificationStatus.PENDING_RENEWAL]
        revoked = [c for c in all_certs if c.status == CertificationStatus.REVOKED]

        # Group by level
        by_level = {}
        for cert in active:
            level = cert.certification_level
            if level not in by_level:
                by_level[level] = 0
            by_level[level] += 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_agents": len(self._agents),
            "total_certifications": len(all_certs),
            "active": len(active),
            "expired": len(expired),
            "pending_renewal": len(pending),
            "revoked": len(revoked),
            "by_level": by_level,
            "expiring_in_30_days": len(self.get_expiring_soon(30)),
            "expiring_in_7_days": len(self.get_expiring_soon(7)),
        }

    def export_registry(
        self,
        output_path: str,
        format: str = "json",
    ) -> None:
        """
        Export registry to file.

        Args:
            output_path: Output file path
            format: Export format (json, csv)
        """
        if format == "json":
            data = {
                "registry": self.generate_report(),
                "agents": {
                    agent_id: {
                        "agent_id": history.agent_id,
                        "latest_version": (
                            history.latest_version.agent_version
                            if history.latest_version else None
                        ),
                        "active_count": len(history.active_versions),
                        "versions": [v.to_dict() for v in history.versions],
                    }
                    for agent_id, history in self._agents.items()
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "agent_id", "version", "certification_id", "level",
                    "score", "status", "certified_at", "valid_until"
                ])

                for history in self._agents.values():
                    for cert in history.versions:
                        self._update_status(cert)
                        writer.writerow([
                            cert.agent_id,
                            cert.agent_version,
                            cert.certification_id,
                            cert.certification_level,
                            cert.overall_score,
                            cert.status.value,
                            cert.certified_at.isoformat(),
                            cert.valid_until.isoformat(),
                        ])

        logger.info(f"Exported registry to {output_path}")
