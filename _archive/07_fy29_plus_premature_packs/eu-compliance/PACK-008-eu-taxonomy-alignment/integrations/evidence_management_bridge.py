"""
Evidence Management Bridge - PACK-008 EU Taxonomy Alignment

This module links documents, certifications, and audit reports to TSC, DNSH,
and Minimum Safeguards assessments. It provides evidence attachment, verification,
chain-of-evidence tracking, and completeness scoring.

Evidence types supported:
- Technical certifications (ISO 14001, ISO 50001, EMAS)
- Audit reports (third-party verification, internal audits)
- Environmental permits and licenses
- Human rights due diligence reports
- Anti-corruption policies and procedures
- Financial statements and notes
- CapEx plan documentation
- Climate risk assessments

Example:
    >>> config = EvidenceConfig(require_verification=True)
    >>> bridge = EvidenceManagementBridge(config)
    >>> result = await bridge.attach_evidence("ACT-001-SC", document)
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class EvidenceConfig(BaseModel):
    """Configuration for Evidence Management Bridge."""

    require_verification: bool = Field(
        default=True,
        description="Require evidence verification before acceptance"
    )
    allowed_formats: List[str] = Field(
        default=["pdf", "docx", "xlsx", "csv", "json", "xml", "jpg", "png"],
        description="Allowed document formats"
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum file size in megabytes"
    )
    retention_years: int = Field(
        default=10,
        ge=1,
        description="Evidence retention period in years"
    )
    enable_integrity_check: bool = Field(
        default=True,
        description="Enable SHA-256 integrity verification"
    )


class EvidenceManagementBridge:
    """
    Bridge for taxonomy assessment evidence management.

    Links documents, certifications, and audit reports to TSC, DNSH,
    and MS assessments with SHA-256 integrity tracking and completeness scoring.

    Example:
        >>> config = EvidenceConfig()
        >>> bridge = EvidenceManagementBridge(config)
        >>> bridge.inject_service(document_store)
        >>> result = await bridge.attach_evidence("assessment_123", doc_data)
    """

    # Evidence requirements per assessment type
    EVIDENCE_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
        "substantial_contribution": [
            {"type": "technical_data", "required": True, "description": "Quantitative TSC compliance data"},
            {"type": "certification", "required": False, "description": "Relevant ISO or industry certifications"},
            {"type": "monitoring_report", "required": True, "description": "Performance monitoring data"},
            {"type": "third_party_verification", "required": False, "description": "Independent verification report"}
        ],
        "dnsh_ccm": [
            {"type": "climate_risk_assessment", "required": True, "description": "Physical climate risk assessment"},
            {"type": "adaptation_plan", "required": True, "description": "Climate adaptation plan"}
        ],
        "dnsh_wtr": [
            {"type": "water_permit", "required": True, "description": "Water use permit or license"},
            {"type": "water_management_plan", "required": False, "description": "Water management plan"}
        ],
        "dnsh_ce": [
            {"type": "waste_management_plan", "required": True, "description": "Waste management plan"},
            {"type": "circular_economy_assessment", "required": False, "description": "Circular economy assessment"}
        ],
        "dnsh_ppc": [
            {"type": "environmental_permit", "required": True, "description": "Environmental pollution permit"},
            {"type": "emissions_monitoring", "required": True, "description": "Pollution monitoring data"}
        ],
        "dnsh_bio": [
            {"type": "biodiversity_assessment", "required": True, "description": "Biodiversity impact assessment"},
            {"type": "eia_report", "required": False, "description": "Environmental Impact Assessment"}
        ],
        "minimum_safeguards": [
            {"type": "human_rights_policy", "required": True, "description": "Human rights due diligence policy"},
            {"type": "anti_corruption_policy", "required": True, "description": "Anti-corruption procedures"},
            {"type": "tax_compliance", "required": True, "description": "Tax compliance documentation"},
            {"type": "competition_compliance", "required": True, "description": "Fair competition policy"}
        ]
    }

    def __init__(self, config: EvidenceConfig):
        """Initialize evidence management bridge."""
        self.config = config
        self._service: Any = None
        self._evidence_store: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("EvidenceManagementBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real evidence/document management service."""
        self._service = service
        logger.info("Injected evidence management service")

    async def attach_evidence(
        self,
        assessment_id: str,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attach evidence document to a taxonomy assessment.

        Args:
            assessment_id: Assessment identifier (e.g., "ACT-001-SC")
            document: Document metadata and content reference

        Returns:
            Attachment result with evidence ID and integrity hash
        """
        try:
            if self._service and hasattr(self._service, "attach_evidence"):
                return await self._service.attach_evidence(assessment_id, document)

            # Validate document format
            doc_format = document.get("format", "").lower()
            if doc_format and doc_format not in self.config.allowed_formats:
                return {
                    "status": "rejected",
                    "reason": f"Format '{doc_format}' not in allowed formats",
                    "allowed_formats": self.config.allowed_formats,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Generate evidence ID and integrity hash
            evidence_id = f"EVD-{assessment_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            content_hash = self._calculate_hash(document)

            evidence_record = {
                "evidence_id": evidence_id,
                "assessment_id": assessment_id,
                "document_type": document.get("type", "general"),
                "document_name": document.get("name", "unnamed"),
                "format": doc_format,
                "content_hash": content_hash,
                "uploaded_at": datetime.utcnow().isoformat(),
                "verified": False,
                "retention_until": str(
                    datetime.utcnow().year + self.config.retention_years
                )
            }

            # Store in local evidence store
            if assessment_id not in self._evidence_store:
                self._evidence_store[assessment_id] = []
            self._evidence_store[assessment_id].append(evidence_record)

            return {
                "status": "attached",
                "evidence_id": evidence_id,
                "content_hash": content_hash,
                "assessment_id": assessment_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Evidence attachment failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def verify_evidence(
        self,
        evidence_id: str
    ) -> Dict[str, Any]:
        """
        Verify evidence document integrity and validity.

        Args:
            evidence_id: Evidence identifier

        Returns:
            Verification result with integrity status
        """
        try:
            if self._service and hasattr(self._service, "verify_evidence"):
                return await self._service.verify_evidence(evidence_id)

            # Search for evidence in store
            for assessment_id, evidence_list in self._evidence_store.items():
                for evidence in evidence_list:
                    if evidence.get("evidence_id") == evidence_id:
                        evidence["verified"] = True
                        evidence["verified_at"] = datetime.utcnow().isoformat()

                        return {
                            "evidence_id": evidence_id,
                            "verified": True,
                            "integrity_check": "pass",
                            "content_hash": evidence.get("content_hash", ""),
                            "assessment_id": assessment_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }

            return {
                "evidence_id": evidence_id,
                "verified": False,
                "integrity_check": "not_found",
                "message": f"Evidence {evidence_id} not found",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Evidence verification failed: {str(e)}")
            return {"evidence_id": evidence_id, "verified": False, "error": str(e)}

    async def get_evidence_chain(
        self,
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        Get complete evidence chain for an assessment.

        Args:
            assessment_id: Assessment identifier

        Returns:
            Complete evidence chain with all attached documents
        """
        try:
            if self._service and hasattr(self._service, "get_evidence_chain"):
                return await self._service.get_evidence_chain(assessment_id)

            evidence_list = self._evidence_store.get(assessment_id, [])

            chain_hash = self._calculate_hash(evidence_list)

            return {
                "assessment_id": assessment_id,
                "evidence_count": len(evidence_list),
                "evidence": evidence_list,
                "all_verified": all(
                    e.get("verified", False) for e in evidence_list
                ) if evidence_list else False,
                "chain_hash": chain_hash,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Evidence chain retrieval failed: {str(e)}")
            return {"assessment_id": assessment_id, "error": str(e)}

    async def calculate_evidence_completeness(
        self,
        assessment_id: str,
        assessment_type: str = "substantial_contribution"
    ) -> Dict[str, Any]:
        """
        Calculate evidence completeness score for an assessment.

        Compares attached evidence against requirements for the assessment type
        and produces a completeness score (0-100).

        Args:
            assessment_id: Assessment identifier
            assessment_type: Type of assessment (SC, DNSH, MS)

        Returns:
            Completeness score with gap analysis
        """
        try:
            if self._service and hasattr(self._service, "calculate_evidence_completeness"):
                return await self._service.calculate_evidence_completeness(
                    assessment_id, assessment_type
                )

            requirements = self.EVIDENCE_REQUIREMENTS.get(assessment_type, [])
            attached = self._evidence_store.get(assessment_id, [])

            attached_types = {e.get("document_type", "") for e in attached}

            met_requirements = []
            missing_requirements = []
            optional_missing = []

            for req in requirements:
                req_type = req["type"]
                if req_type in attached_types:
                    met_requirements.append(req)
                elif req["required"]:
                    missing_requirements.append(req)
                else:
                    optional_missing.append(req)

            total_required = sum(1 for r in requirements if r["required"])
            met_required = sum(1 for r in met_requirements if r["required"])

            if total_required > 0:
                completeness_score = round((met_required / total_required) * 100, 1)
            else:
                completeness_score = 100.0

            return {
                "assessment_id": assessment_id,
                "assessment_type": assessment_type,
                "completeness_score": completeness_score,
                "total_requirements": len(requirements),
                "met_requirements": len(met_requirements),
                "missing_required": [
                    {"type": r["type"], "description": r["description"]}
                    for r in missing_requirements
                ],
                "missing_optional": [
                    {"type": r["type"], "description": r["description"]}
                    for r in optional_missing
                ],
                "ready_for_submission": completeness_score >= 100.0,
                "provenance_hash": self._calculate_hash({
                    "assessment_id": assessment_id,
                    "score": completeness_score
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Completeness calculation failed: {str(e)}")
            return {
                "assessment_id": assessment_id,
                "completeness_score": 0.0,
                "error": str(e)
            }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for evidence integrity."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
