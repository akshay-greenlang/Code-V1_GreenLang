"""
Consolidated Evidence Bridge - PACK-009 EU Climate Compliance Bundle

This module provides unified evidence management across all 4 constituent packs.
It handles evidence registration, retrieval, cross-pack reuse tracking, and
mapping evidence items to regulatory requirements across CSRD, CBAM, EUDR,
and EU Taxonomy.

The bridge handles:
- Evidence registration with unique identifiers
- Evidence retrieval by pack, type, or requirement
- Cross-pack evidence reuse tracking
- Mapping evidence to regulatory disclosure requirements
- Evidence completeness reporting

Example:
    >>> config = ConsolidatedEvidenceConfig()
    >>> bridge = ConsolidatedEvidenceBridge(config)
    >>> await bridge.register_evidence(evidence_item)
    >>> evidence = await bridge.get_evidence(evidence_id)
    >>> reuse_report = await bridge.get_reuse_report()
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConsolidatedEvidenceConfig(BaseModel):
    """Configuration for consolidated evidence bridge."""

    enable_reuse_tracking: bool = Field(
        default=True,
        description="Track evidence reuse across packs"
    )
    enable_completeness_check: bool = Field(
        default=True,
        description="Enable evidence completeness checking"
    )
    reporting_period_year: int = Field(
        default=2025,
        ge=2023,
        description="Reporting period fiscal year"
    )
    evidence_retention_years: int = Field(
        default=7,
        ge=1,
        description="Number of years to retain evidence"
    )
    auto_map_requirements: bool = Field(
        default=True,
        description="Automatically map evidence to requirements"
    )


# ---------------------------------------------------------------------------
# Evidence reuse map
# ---------------------------------------------------------------------------

EVIDENCE_REUSE_MAP: Dict[str, Dict[str, Any]] = {
    "ghg_verification_report": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "E1-6 GHG emissions verification",
        "taxonomy_requirement": "Climate DA TSC verification",
        "evidence_type": "third_party_report",
        "reuse_category": "emissions",
    },
    "energy_audit_report": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "E1-5 Energy consumption",
        "taxonomy_requirement": "Climate DA DNSH energy efficiency",
        "evidence_type": "audit_report",
        "reuse_category": "energy",
    },
    "financial_statements": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "ESRS 2 BP-2 Financial data",
        "taxonomy_requirement": "Art. 8 KPI denominator data",
        "evidence_type": "financial_document",
        "reuse_category": "financial",
    },
    "erp_data_export": {
        "applicable_packs": ["csrd", "cbam", "taxonomy"],
        "csrd_requirement": "Multiple ESRS datapoints",
        "cbam_requirement": "Art. 7 Quantity and value data",
        "taxonomy_requirement": "Art. 8 Activity-level data",
        "evidence_type": "system_export",
        "reuse_category": "operational",
    },
    "supplier_declarations": {
        "applicable_packs": ["csrd", "cbam", "eudr"],
        "csrd_requirement": "G1-2 Supply chain due diligence",
        "cbam_requirement": "Art. 10 Installation operator data",
        "eudr_requirement": "Art. 4 Supplier information",
        "evidence_type": "declaration",
        "reuse_category": "supply_chain",
    },
    "customs_declarations": {
        "applicable_packs": ["cbam", "eudr"],
        "cbam_requirement": "Art. 5 Import declarations",
        "eudr_requirement": "Art. 4 Trade documentation",
        "evidence_type": "customs_document",
        "reuse_category": "trade",
    },
    "iso14001_certificate": {
        "applicable_packs": ["csrd", "taxonomy", "eudr"],
        "csrd_requirement": "E1-2 Environmental management",
        "taxonomy_requirement": "DNSH pollution prevention",
        "eudr_requirement": "Art. 12 Management system evidence",
        "evidence_type": "certificate",
        "reuse_category": "management_system",
    },
    "human_rights_policy": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "S1-1 Human rights policy",
        "taxonomy_requirement": "Art. 18 Minimum Safeguards",
        "evidence_type": "policy_document",
        "reuse_category": "governance",
    },
    "anti_corruption_policy": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "G1-3 Anti-corruption measures",
        "taxonomy_requirement": "Art. 18 Minimum Safeguards",
        "evidence_type": "policy_document",
        "reuse_category": "governance",
    },
    "water_usage_report": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "E3-4 Water consumption",
        "taxonomy_requirement": "Env DA WTR water use",
        "evidence_type": "operational_report",
        "reuse_category": "environment",
    },
    "waste_management_report": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "E5-5 Waste generation",
        "taxonomy_requirement": "Env DA CE circular economy",
        "evidence_type": "operational_report",
        "reuse_category": "environment",
    },
    "biodiversity_assessment": {
        "applicable_packs": ["csrd", "taxonomy", "eudr"],
        "csrd_requirement": "E4-5 Biodiversity sites",
        "taxonomy_requirement": "Env DA BIO biodiversity",
        "eudr_requirement": "Art. 3 Deforestation-free verification",
        "evidence_type": "assessment_report",
        "reuse_category": "biodiversity",
    },
    "geolocation_data": {
        "applicable_packs": ["eudr"],
        "eudr_requirement": "Art. 4(2)(f) Geolocation",
        "evidence_type": "geospatial_data",
        "reuse_category": "traceability",
    },
    "satellite_imagery": {
        "applicable_packs": ["eudr"],
        "eudr_requirement": "Art. 10(2) Satellite monitoring",
        "evidence_type": "remote_sensing",
        "reuse_category": "traceability",
    },
    "installation_emissions_data": {
        "applicable_packs": ["cbam"],
        "cbam_requirement": "Art. 7 Embedded emissions calculation",
        "evidence_type": "emissions_data",
        "reuse_category": "emissions",
    },
    "verifier_accreditation": {
        "applicable_packs": ["cbam", "eudr"],
        "cbam_requirement": "Art. 8 Accredited verifier",
        "eudr_requirement": "Art. 10(6) Independent verification",
        "evidence_type": "accreditation",
        "reuse_category": "verification",
    },
    "transition_plan": {
        "applicable_packs": ["csrd", "taxonomy"],
        "csrd_requirement": "E1-1 Transition plan",
        "taxonomy_requirement": "Climate DA transitional activities",
        "evidence_type": "strategic_document",
        "reuse_category": "strategy",
    },
    "capex_plan": {
        "applicable_packs": ["taxonomy"],
        "taxonomy_requirement": "Art. 8(4) CapEx plan 5-year",
        "evidence_type": "financial_plan",
        "reuse_category": "financial",
    },
    "sbti_commitment_letter": {
        "applicable_packs": ["csrd"],
        "csrd_requirement": "E1-4 SBTi validation",
        "evidence_type": "commitment_letter",
        "reuse_category": "targets",
    },
    "tax_compliance_certificate": {
        "applicable_packs": ["taxonomy"],
        "taxonomy_requirement": "Art. 18 MS taxation",
        "evidence_type": "certificate",
        "reuse_category": "governance",
    },
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class ConsolidatedEvidenceBridge:
    """
    Consolidated Evidence Bridge for PACK-009 Bundle.

    Unified evidence management across all 4 constituent packs with
    registration, retrieval, cross-pack reuse tracking, and
    regulatory requirement mapping.

    Example:
        >>> config = ConsolidatedEvidenceConfig()
        >>> bridge = ConsolidatedEvidenceBridge(config)
        >>> reg = await bridge.register_evidence(item)
        >>> report = await bridge.get_reuse_report()
    """

    def __init__(self, config: ConsolidatedEvidenceConfig):
        """Initialize consolidated evidence bridge."""
        self.config = config
        self._evidence_store: Dict[str, Dict[str, Any]] = {}
        self._reuse_log: List[Dict[str, Any]] = []
        logger.info("ConsolidatedEvidenceBridge initialized")

    async def register_evidence(
        self, evidence_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register a new evidence item.

        Args:
            evidence_item: Evidence item with type, source, content reference.

        Returns:
            Registration result with evidence ID and reuse mappings.
        """
        try:
            evidence_id = evidence_item.get(
                "evidence_id", f"ev_{uuid.uuid4().hex[:12]}"
            )
            evidence_type = evidence_item.get("evidence_type", "unknown")
            source_pack = evidence_item.get("source_pack", "manual")
            description = evidence_item.get("description", "")
            content_ref = evidence_item.get("content_ref", "")

            record = {
                "evidence_id": evidence_id,
                "evidence_type": evidence_type,
                "source_pack": source_pack,
                "description": description,
                "content_ref": content_ref,
                "registered_at": datetime.utcnow().isoformat(),
                "reporting_year": self.config.reporting_period_year,
                "reuse_mappings": [],
                "provenance_hash": self._calculate_hash(evidence_item),
            }

            # Auto-map to requirements
            if self.config.auto_map_requirements:
                reuse_entries = self._find_reuse_mappings(evidence_type)
                record["reuse_mappings"] = reuse_entries
                if reuse_entries and self.config.enable_reuse_tracking:
                    self._reuse_log.append({
                        "evidence_id": evidence_id,
                        "evidence_type": evidence_type,
                        "reuse_count": len(reuse_entries),
                        "packs": list(
                            set(
                                pk
                                for entry in reuse_entries
                                for pk in entry.get("applicable_packs", [])
                            )
                        ),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            self._evidence_store[evidence_id] = record

            return {
                "status": "registered",
                "evidence_id": evidence_id,
                "evidence_type": evidence_type,
                "reuse_mappings_found": len(record["reuse_mappings"]),
                "provenance_hash": record["provenance_hash"],
                "timestamp": record["registered_at"],
            }

        except Exception as e:
            logger.error(f"Evidence registration failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_evidence(
        self, evidence_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve a registered evidence item.

        Args:
            evidence_id: Unique evidence identifier.

        Returns:
            Evidence record or not-found message.
        """
        record = self._evidence_store.get(evidence_id)
        if record:
            return {"status": "found", "evidence": record}
        return {
            "status": "not_found",
            "evidence_id": evidence_id,
            "message": f"Evidence '{evidence_id}' not found in store",
        }

    async def map_to_requirements(
        self, evidence_type: str
    ) -> Dict[str, Any]:
        """
        Map an evidence type to all applicable regulatory requirements.

        Args:
            evidence_type: Type of evidence (e.g. 'ghg_verification_report').

        Returns:
            Mapping result with per-pack requirements.
        """
        reuse_entries = self._find_reuse_mappings(evidence_type)

        if not reuse_entries:
            return {
                "evidence_type": evidence_type,
                "found": False,
                "message": f"No requirement mappings found for '{evidence_type}'",
                "requirements": {},
            }

        requirements: Dict[str, Any] = {}
        for entry in reuse_entries:
            for pk in entry.get("applicable_packs", []):
                req_key = f"{pk}_requirement"
                if req_key in entry:
                    if pk not in requirements:
                        requirements[pk] = []
                    requirements[pk].append({
                        "requirement": entry[req_key],
                        "evidence_category": entry.get("reuse_category", "general"),
                    })

        return {
            "evidence_type": evidence_type,
            "found": True,
            "requirements": requirements,
            "total_packs": len(requirements),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_reuse_report(self) -> Dict[str, Any]:
        """
        Generate a cross-pack evidence reuse report.

        Returns:
            Report with reuse statistics and recommendations.
        """
        try:
            total_evidence = len(self._evidence_store)
            total_reusable = sum(
                1
                for ev in self._evidence_store.values()
                if len(ev.get("reuse_mappings", [])) > 0
            )
            total_reuse_events = len(self._reuse_log)

            by_type: Dict[str, int] = {}
            by_pack: Dict[str, int] = {}
            by_category: Dict[str, int] = {}

            for ev in self._evidence_store.values():
                ev_type = ev.get("evidence_type", "unknown")
                by_type[ev_type] = by_type.get(ev_type, 0) + 1
                source = ev.get("source_pack", "unknown")
                by_pack[source] = by_pack.get(source, 0) + 1

            for reuse_key, reuse_info in EVIDENCE_REUSE_MAP.items():
                cat = reuse_info.get("reuse_category", "general")
                by_category[cat] = by_category.get(cat, 0) + 1

            # Calculate reuse potential from the reuse map
            multi_pack_evidence = sum(
                1
                for info in EVIDENCE_REUSE_MAP.values()
                if len(info.get("applicable_packs", [])) >= 2
            )
            total_map_entries = len(EVIDENCE_REUSE_MAP)

            return {
                "total_evidence_items": total_evidence,
                "reusable_items": total_reusable,
                "reuse_events_logged": total_reuse_events,
                "reuse_rate_pct": (
                    round(total_reusable / total_evidence * 100, 1)
                    if total_evidence > 0
                    else 0.0
                ),
                "by_type": by_type,
                "by_pack": by_pack,
                "evidence_categories": by_category,
                "reuse_potential": {
                    "multi_pack_evidence_types": multi_pack_evidence,
                    "single_pack_evidence_types": total_map_entries - multi_pack_evidence,
                    "cross_pack_reuse_pct": (
                        round(multi_pack_evidence / total_map_entries * 100, 1)
                        if total_map_entries > 0
                        else 0.0
                    ),
                },
                "reuse_log": self._reuse_log[-20:],
                "timestamp": datetime.utcnow().isoformat(),
                "provenance_hash": self._calculate_hash({
                    "total": total_evidence,
                    "reusable": total_reusable,
                }),
            }

        except Exception as e:
            logger.error(f"Reuse report generation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_completeness_report(self) -> Dict[str, Any]:
        """
        Generate evidence completeness report against all requirements.

        Returns:
            Completeness report per pack and overall.
        """
        try:
            if not self.config.enable_completeness_check:
                return {
                    "status": "disabled",
                    "message": "Completeness checking is disabled",
                }

            pack_completeness: Dict[str, Dict[str, Any]] = {}
            packs = ["csrd", "cbam", "eudr", "taxonomy"]

            for pk in packs:
                required_evidence: List[str] = []
                for ev_key, ev_info in EVIDENCE_REUSE_MAP.items():
                    if pk in ev_info.get("applicable_packs", []):
                        required_evidence.append(ev_key)

                covered = 0
                for req_ev in required_evidence:
                    ev_type = EVIDENCE_REUSE_MAP[req_ev].get(
                        "evidence_type", ""
                    )
                    if any(
                        stored.get("evidence_type") == ev_type
                        for stored in self._evidence_store.values()
                    ):
                        covered += 1

                total = len(required_evidence)
                pack_completeness[pk] = {
                    "required_evidence_types": total,
                    "covered": covered,
                    "missing": total - covered,
                    "completeness_pct": (
                        round(covered / total * 100, 1) if total > 0 else 0.0
                    ),
                }

            overall_required = len(EVIDENCE_REUSE_MAP)
            overall_covered = len(
                set(
                    ev.get("evidence_type", "")
                    for ev in self._evidence_store.values()
                )
            )

            return {
                "overall_completeness_pct": (
                    round(overall_covered / overall_required * 100, 1)
                    if overall_required > 0
                    else 0.0
                ),
                "overall_required": overall_required,
                "overall_covered": overall_covered,
                "per_pack": pack_completeness,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Completeness report failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_reuse_mappings(
        self, evidence_type: str
    ) -> List[Dict[str, Any]]:
        """Find reuse mappings for a given evidence type."""
        matches: List[Dict[str, Any]] = []
        for key, info in EVIDENCE_REUSE_MAP.items():
            if info.get("evidence_type") == evidence_type or key == evidence_type:
                matches.append({"reuse_key": key, **info})
        return matches

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
