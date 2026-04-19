# -*- coding: utf-8 -*-
"""
TaxonomyBridge - EU Taxonomy DNSH Validation Bridge for PACK-020
===================================================================

Validates EU Taxonomy Do No Significant Harm (DNSH) criteria for battery
manufacturing activities. Maps battery production to Taxonomy economic
activities, assesses alignment with the six environmental objectives,
and checks technical screening criteria (TSC) specific to battery
manufacturing under the Climate Delegated Act.

Battery manufacturing maps primarily to:
    - Activity 3.4: Manufacture of batteries (CCM NACE C27.2)
    - Activity 3.6: Manufacture of other low carbon technologies
    - Activity 6.5: Transport by motorbikes, passenger cars (EV assembly)

Methods:
    - check_dnsh_criteria()                 -- Check all 6 DNSH objectives
    - assess_taxonomy_alignment()           -- Full alignment assessment
    - get_battery_manufacturing_criteria()  -- Get TSC for battery manufacturing

Legal References:
    - Regulation (EU) 2020/852 (EU Taxonomy Regulation)
    - Commission Delegated Regulation (EU) 2021/2139 (Climate Delegated Act)
    - Commission Delegated Regulation (EU) 2021/2178 (Disclosure Delegated Act)
    - Activity 3.4 TSC for manufacture of batteries
    - Regulation (EU) 2023/1542, Art 7 (Carbon footprint) - linkage

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CLIMATE_MITIGATION = "climate_change_mitigation"
    CLIMATE_ADAPTATION = "climate_change_adaptation"
    WATER_MARINE = "water_and_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_and_ecosystems"

class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment status."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ASSESSED = "not_assessed"
    PARTIALLY_MET = "partially_met"

class AlignmentStatus(str, Enum):
    """Taxonomy alignment assessment status."""

    ALIGNED = "aligned"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"
    UNDER_REVIEW = "under_review"

class TaxonomyActivity(str, Enum):
    """Taxonomy economic activities relevant to battery manufacturing."""

    BATTERY_MANUFACTURE = "3.4"
    LOW_CARBON_TECH = "3.6"
    EV_TRANSPORT = "6.5"
    RECYCLING = "5.9"
    ENERGY_STORAGE = "4.10"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Bridge."""

    pack_id: str = Field(default="PACK-020")
    taxonomy_version: str = Field(default="2024")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    primary_activity: TaxonomyActivity = Field(
        default=TaxonomyActivity.BATTERY_MANUFACTURE
    )
    nace_code: str = Field(default="C27.2")
    include_minimum_safeguards: bool = Field(default=True)

class DNSHCriterion(BaseModel):
    """Single DNSH criterion assessment result."""

    objective: EnvironmentalObjective = Field(
        default=EnvironmentalObjective.CLIMATE_MITIGATION
    )
    criterion_id: str = Field(default="")
    criterion_description: str = Field(default="")
    status: DNSHStatus = Field(default=DNSHStatus.NOT_ASSESSED)
    evidence_provided: bool = Field(default=False)
    evidence_summary: str = Field(default="")
    battery_reg_reference: str = Field(default="")

class DNSHResult(BaseModel):
    """Complete DNSH assessment result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    activity: str = Field(default="3.4")
    criteria_assessed: int = Field(default=0)
    criteria_passed: int = Field(default=0)
    criteria_failed: int = Field(default=0)
    overall_dnsh: DNSHStatus = Field(default=DNSHStatus.NOT_ASSESSED)
    criteria: List[DNSHCriterion] = Field(default_factory=list)
    minimum_safeguards_met: bool = Field(default=False)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AlignmentResult(BaseModel):
    """Full Taxonomy alignment assessment result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    activity: str = Field(default="3.4")
    activity_name: str = Field(default="Manufacture of batteries")
    nace_code: str = Field(default="C27.2")
    substantial_contribution_objective: EnvironmentalObjective = Field(
        default=EnvironmentalObjective.CLIMATE_MITIGATION
    )
    substantial_contribution_met: bool = Field(default=False)
    dnsh_met: bool = Field(default=False)
    minimum_safeguards_met: bool = Field(default=False)
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.UNDER_REVIEW
    )
    capex_eur: float = Field(default=0.0, ge=0.0)
    opex_eur: float = Field(default=0.0, ge=0.0)
    revenue_eur: float = Field(default=0.0, ge=0.0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class BatteryManufacturingCriteria(BaseModel):
    """Technical screening criteria for battery manufacturing (Activity 3.4)."""

    activity_id: str = Field(default="3.4")
    activity_name: str = Field(default="Manufacture of batteries")
    nace_code: str = Field(default="C27.2")
    substantial_contribution_criteria: List[Dict[str, str]] = Field(
        default_factory=list
    )
    dnsh_criteria: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict
    )
    battery_reg_cross_references: List[Dict[str, str]] = Field(
        default_factory=list
    )
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# TSC Data for Activity 3.4 (Manufacture of batteries)
# ---------------------------------------------------------------------------

ACTIVITY_34_SC_CRITERIA: List[Dict[str, str]] = [
    {
        "criterion_id": "SC-3.4-1",
        "description": "Battery manufactured to reduce CO2 emissions in transport, "
                       "stationary energy storage, or other applications",
        "metric": "lifecycle_ghg_reduction",
    },
    {
        "criterion_id": "SC-3.4-2",
        "description": "Battery meets minimum carbon footprint threshold per EU 2024/1781",
        "metric": "carbon_footprint_kgco2e_per_kwh",
    },
]

ACTIVITY_34_DNSH_CRITERIA: Dict[str, List[Dict[str, str]]] = {
    EnvironmentalObjective.CLIMATE_ADAPTATION.value: [
        {
            "criterion_id": "DNSH-3.4-CCA-1",
            "description": "Physical climate risk assessment conducted (Appendix A)",
            "battery_reg_reference": "",
        },
    ],
    EnvironmentalObjective.WATER_MARINE.value: [
        {
            "criterion_id": "DNSH-3.4-WTR-1",
            "description": "Water use and discharge management per IED requirements",
            "battery_reg_reference": "",
        },
    ],
    EnvironmentalObjective.CIRCULAR_ECONOMY.value: [
        {
            "criterion_id": "DNSH-3.4-CE-1",
            "description": "Waste management plan for battery manufacturing waste",
            "battery_reg_reference": "Art 8 (recycled content)",
        },
        {
            "criterion_id": "DNSH-3.4-CE-2",
            "description": "Battery designed for durability, recyclability, and easy disassembly",
            "battery_reg_reference": "Art 11 (removability and replaceability)",
        },
    ],
    EnvironmentalObjective.POLLUTION_PREVENTION.value: [
        {
            "criterion_id": "DNSH-3.4-PP-1",
            "description": "Substance restrictions compliance (mercury, cadmium, lead)",
            "battery_reg_reference": "Art 6 (substance restrictions)",
        },
        {
            "criterion_id": "DNSH-3.4-PP-2",
            "description": "Emissions to air, water, soil within IED BAT-AEL limits",
            "battery_reg_reference": "",
        },
    ],
    EnvironmentalObjective.BIODIVERSITY.value: [
        {
            "criterion_id": "DNSH-3.4-BIO-1",
            "description": "EIA completed; no operations in biodiversity-sensitive areas",
            "battery_reg_reference": "Art 39 (supply chain DD)",
        },
    ],
}

BATTERY_REG_CROSS_REFERENCES: List[Dict[str, str]] = [
    {"taxonomy_criterion": "SC-3.4-2", "battery_reg_article": "Art 7", "overlap": "Carbon footprint"},
    {"taxonomy_criterion": "DNSH-3.4-CE-1", "battery_reg_article": "Art 8", "overlap": "Recycled content"},
    {"taxonomy_criterion": "DNSH-3.4-CE-2", "battery_reg_article": "Art 11", "overlap": "Removability"},
    {"taxonomy_criterion": "DNSH-3.4-PP-1", "battery_reg_article": "Art 6", "overlap": "Substance restrictions"},
    {"taxonomy_criterion": "DNSH-3.4-BIO-1", "battery_reg_article": "Art 39", "overlap": "Supply chain DD"},
]

# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------

class TaxonomyBridge:
    """EU Taxonomy DNSH validation bridge for PACK-020.

    Validates battery manufacturing activities against EU Taxonomy DNSH
    criteria, assesses alignment with environmental objectives, and
    cross-references with Battery Regulation requirements.

    Attributes:
        config: Bridge configuration.
        _dnsh_cache: Cached DNSH assessment results.

    Example:
        >>> bridge = TaxonomyBridge(TaxonomyBridgeConfig())
        >>> result = bridge.check_dnsh_criteria(context)
        >>> assert result.overall_dnsh in (DNSHStatus.PASSED, DNSHStatus.PARTIALLY_MET)
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize TaxonomyBridge."""
        self.config = config or TaxonomyBridgeConfig()
        self._dnsh_cache: Optional[DNSHResult] = None
        logger.info(
            "TaxonomyBridge initialized (activity=%s, nace=%s, version=%s)",
            self.config.primary_activity.value,
            self.config.nace_code,
            self.config.taxonomy_version,
        )

    def check_dnsh_criteria(
        self, context: Dict[str, Any]
    ) -> DNSHResult:
        """Check all DNSH criteria for battery manufacturing.

        Assesses the five DNSH objectives (excluding the SC objective,
        which is climate change mitigation for Activity 3.4).

        Args:
            context: Pipeline context with environmental data.

        Returns:
            DNSHResult with per-objective criterion assessments.
        """
        result = DNSHResult(
            started_at=utcnow(),
            activity=self.config.primary_activity.value,
        )

        try:
            criteria_list: List[DNSHCriterion] = []
            user_evidence = context.get("dnsh_evidence", {})

            for obj_value, obj_criteria in ACTIVITY_34_DNSH_CRITERIA.items():
                for criterion_data in obj_criteria:
                    evidence_key = criterion_data["criterion_id"].lower().replace("-", "_")
                    has_evidence = user_evidence.get(evidence_key, False)

                    status = DNSHStatus.PASSED if has_evidence else DNSHStatus.NOT_ASSESSED

                    criteria_list.append(DNSHCriterion(
                        objective=EnvironmentalObjective(obj_value),
                        criterion_id=criterion_data["criterion_id"],
                        criterion_description=criterion_data["description"],
                        status=status,
                        evidence_provided=has_evidence,
                        evidence_summary=user_evidence.get(
                            f"{evidence_key}_summary", ""
                        ),
                        battery_reg_reference=criterion_data.get(
                            "battery_reg_reference", ""
                        ),
                    ))

            result.criteria = criteria_list
            result.criteria_assessed = len(criteria_list)
            result.criteria_passed = sum(
                1 for c in criteria_list if c.status == DNSHStatus.PASSED
            )
            result.criteria_failed = sum(
                1 for c in criteria_list if c.status == DNSHStatus.FAILED
            )

            if result.criteria_assessed == 0:
                result.overall_dnsh = DNSHStatus.NOT_ASSESSED
            elif result.criteria_failed > 0:
                result.overall_dnsh = DNSHStatus.FAILED
            elif result.criteria_passed == result.criteria_assessed:
                result.overall_dnsh = DNSHStatus.PASSED
            else:
                result.overall_dnsh = DNSHStatus.PARTIALLY_MET

            if self.config.include_minimum_safeguards:
                result.minimum_safeguards_met = context.get(
                    "minimum_safeguards_met", False
                )

            result.status = "completed"
            self._dnsh_cache = result

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "assessed": result.criteria_assessed,
                    "passed": result.criteria_passed,
                    "overall": result.overall_dnsh.value,
                })

            logger.info(
                "DNSH check: %d assessed, %d passed, overall=%s",
                result.criteria_assessed,
                result.criteria_passed,
                result.overall_dnsh.value,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("DNSH check failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def assess_taxonomy_alignment(
        self, context: Dict[str, Any]
    ) -> AlignmentResult:
        """Assess full Taxonomy alignment for battery manufacturing.

        Checks substantial contribution, DNSH, and minimum safeguards
        to determine alignment status.

        Args:
            context: Pipeline context with alignment data.

        Returns:
            AlignmentResult with alignment determination.
        """
        result = AlignmentResult(
            activity=self.config.primary_activity.value,
            nace_code=self.config.nace_code,
        )

        try:
            # Step 1: Check substantial contribution
            sc_data = context.get("substantial_contribution", {})
            cf_per_kwh = sc_data.get("carbon_footprint_kgco2e_per_kwh", 0.0)
            result.substantial_contribution_met = cf_per_kwh > 0 and cf_per_kwh <= 100.0

            # Step 2: Check DNSH
            if self._dnsh_cache is None:
                self.check_dnsh_criteria(context)
            if self._dnsh_cache:
                result.dnsh_met = self._dnsh_cache.overall_dnsh == DNSHStatus.PASSED

            # Step 3: Check minimum safeguards
            result.minimum_safeguards_met = context.get(
                "minimum_safeguards_met", False
            )

            # Step 4: Financial KPIs
            result.capex_eur = context.get("taxonomy_capex_eur", 0.0)
            result.opex_eur = context.get("taxonomy_opex_eur", 0.0)
            result.revenue_eur = context.get("taxonomy_revenue_eur", 0.0)

            # Step 5: Determine alignment
            if (
                result.substantial_contribution_met
                and result.dnsh_met
                and result.minimum_safeguards_met
            ):
                result.alignment_status = AlignmentStatus.ALIGNED
            elif result.substantial_contribution_met:
                result.alignment_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED
            else:
                result.alignment_status = AlignmentStatus.NOT_ELIGIBLE

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "sc": result.substantial_contribution_met,
                    "dnsh": result.dnsh_met,
                    "safeguards": result.minimum_safeguards_met,
                    "alignment": result.alignment_status.value,
                })

            logger.info(
                "Taxonomy alignment: %s (SC=%s, DNSH=%s, safeguards=%s)",
                result.alignment_status.value,
                result.substantial_contribution_met,
                result.dnsh_met,
                result.minimum_safeguards_met,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Taxonomy alignment assessment failed: %s", str(exc))

        return result

    def get_battery_manufacturing_criteria(self) -> BatteryManufacturingCriteria:
        """Get technical screening criteria for battery manufacturing (Activity 3.4).

        Returns:
            BatteryManufacturingCriteria with SC, DNSH, and cross-references.
        """
        result = BatteryManufacturingCriteria(
            substantial_contribution_criteria=list(ACTIVITY_34_SC_CRITERIA),
            dnsh_criteria=dict(ACTIVITY_34_DNSH_CRITERIA),
            battery_reg_cross_references=list(BATTERY_REG_CROSS_REFERENCES),
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash({
                "sc_count": len(result.substantial_contribution_criteria),
                "dnsh_objectives": len(result.dnsh_criteria),
                "cross_refs": len(result.battery_reg_cross_references),
            })

        logger.info(
            "Battery manufacturing criteria: %d SC, %d DNSH objectives, %d cross-refs",
            len(result.substantial_contribution_criteria),
            len(result.dnsh_criteria),
            len(result.battery_reg_cross_references),
        )
        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "taxonomy_version": self.config.taxonomy_version,
            "primary_activity": self.config.primary_activity.value,
            "nace_code": self.config.nace_code,
            "dnsh_cached": self._dnsh_cache is not None,
            "objectives_count": len(EnvironmentalObjective),
            "cross_references": len(BATTERY_REG_CROSS_REFERENCES),
        }
