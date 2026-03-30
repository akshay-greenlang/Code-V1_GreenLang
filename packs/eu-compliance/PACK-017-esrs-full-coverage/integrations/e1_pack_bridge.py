# -*- coding: utf-8 -*-
"""
E1PackBridge - PACK-016 ESRS E1 Climate Integration Bridge for PACK-017
==========================================================================

Connects PACK-017 to PACK-016 ESRS E1 Climate Pack for importing E1
engine results, mapping E1 data to the PACK-017 common data model,
translating E1 compliance scores to the overall ESRS scorecard, and
forwarding E1 XBRL datapoints for report assembly.

Methods:
    - import_e1_results()      -- Import full E1 results from PACK-016
    - get_e1_compliance()      -- Get E1 compliance score for scorecard
    - get_e1_datapoints()      -- Get E1 XBRL-tagged datapoints
    - map_to_common_model()    -- Transform E1 data to PACK-017 CDM
    - export_e1_summary()      -- Export E1 summary for cross-standard checks

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-017 ESRS Full Coverage Pack
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

class E1ImportStatus(str, Enum):
    """E1 import operation status."""

    PENDING = "pending"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"

class E1DisclosureRequirement(str, Enum):
    """ESRS E1 Disclosure Requirements."""

    E1_1 = "E1-1"   # Transition plan for climate change mitigation
    E1_2 = "E1-2"   # Policies related to climate change mitigation/adaptation
    E1_3 = "E1-3"   # Actions and resources - climate change
    E1_4 = "E1-4"   # Targets related to climate change mitigation/adaptation
    E1_5 = "E1-5"   # Energy consumption and mix
    E1_6 = "E1-6"   # Gross Scopes 1, 2, 3 and Total GHG emissions
    E1_7 = "E1-7"   # GHG removals and carbon credits
    E1_8 = "E1-8"   # Internal carbon pricing
    E1_9 = "E1-9"   # Anticipated financial effects - climate-related risks

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class E1BridgeConfig(BaseModel):
    """Configuration for the E1 Pack Bridge."""

    source_pack_id: str = Field(default="PACK-016")
    source_pack_version: str = Field(default="1.0.0")
    target_pack_id: str = Field(default="PACK-017")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    auto_import: bool = Field(default=True)
    map_to_cdm: bool = Field(
        default=True,
        description="Automatically map E1 results to PACK-017 common data model",
    )

class E1ComplianceScore(BaseModel):
    """E1 compliance score for the overall ESRS scorecard."""

    standard: str = Field(default="ESRS E1")
    disclosure_requirement: str = Field(default="")
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_score: float = Field(default=0.0, ge=0.0, le=5.0)
    datapoints_required: int = Field(default=0)
    datapoints_populated: int = Field(default=0)
    mandatory_met: bool = Field(default=False)
    voluntary_met: bool = Field(default=False)

class E1DataPoint(BaseModel):
    """A single E1 XBRL datapoint."""

    datapoint_id: str = Field(default="")
    disclosure_requirement: str = Field(default="")
    xbrl_element: str = Field(default="")
    value: Any = Field(default=None)
    unit: str = Field(default="")
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    is_mandatory: bool = Field(default=False)

class E1ImportResult(BaseModel):
    """Result of an E1 import operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: E1ImportStatus = Field(default=E1ImportStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    disclosures_imported: int = Field(default=0)
    datapoints_imported: int = Field(default=0)
    compliance_scores: List[E1ComplianceScore] = Field(default_factory=list)
    e1_summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# E1 Datapoint Mapping (PACK-016 output keys -> PACK-017 CDM keys)
# ---------------------------------------------------------------------------

E1_CDM_FIELD_MAP: Dict[str, str] = {
    "scope1_total_tco2e": "cdm.emissions.scope1.total_tco2e",
    "scope2_location_tco2e": "cdm.emissions.scope2.location_based_tco2e",
    "scope2_market_tco2e": "cdm.emissions.scope2.market_based_tco2e",
    "scope3_total_tco2e": "cdm.emissions.scope3.total_tco2e",
    "total_energy_mwh": "cdm.energy.total_consumption_mwh",
    "renewable_share_pct": "cdm.energy.renewable_share_pct",
    "has_transition_plan": "cdm.transition.plan_adopted",
    "sbti_validated": "cdm.targets.sbti_validated",
    "total_credits_tco2e": "cdm.offsets.carbon_credits_tco2e",
}

E1_DISCLOSURE_DATAPOINT_COUNT: Dict[str, int] = {
    "E1-1": 12,
    "E1-2": 8,
    "E1-3": 14,
    "E1-4": 16,
    "E1-5": 18,
    "E1-6": 32,
    "E1-7": 10,
    "E1-8": 8,
    "E1-9": 14,
}

# ---------------------------------------------------------------------------
# E1PackBridge
# ---------------------------------------------------------------------------

class E1PackBridge:
    """PACK-016 ESRS E1 Climate integration bridge for PACK-017.

    Imports E1 engine results from PACK-016, maps them to the common
    data model, computes compliance scores for the overall scorecard,
    and provides E1 XBRL datapoints for the report assembly phase.

    Attributes:
        config: Bridge configuration.
        _import_cache: Cached E1 import results.

    Example:
        >>> bridge = E1PackBridge(E1BridgeConfig(reporting_year=2025))
        >>> result = bridge.import_e1_results(context)
        >>> assert result.status == E1ImportStatus.COMPLETED
    """

    def __init__(self, config: Optional[E1BridgeConfig] = None) -> None:
        """Initialize E1PackBridge."""
        self.config = config or E1BridgeConfig()
        self._import_cache: Optional[E1ImportResult] = None
        logger.info(
            "E1PackBridge initialized (source=%s, target=%s, year=%d)",
            self.config.source_pack_id,
            self.config.target_pack_id,
            self.config.reporting_year,
        )

    def import_e1_results(self, context: Dict[str, Any]) -> E1ImportResult:
        """Import full E1 results from PACK-016.

        Args:
            context: Pipeline context with E1 results or PACK-016 reference.

        Returns:
            E1ImportResult with imported disclosures and compliance scores.
        """
        result = E1ImportResult(started_at=utcnow())

        try:
            e1_data = context.get("e1_results", {})
            if not e1_data:
                result.status = E1ImportStatus.NOT_AVAILABLE
                result.warnings.append("No E1 results found in context; PACK-016 may not have run")
                self._finalize_result(result)
                return result

            result.status = E1ImportStatus.IMPORTING

            # Extract disclosure data
            result.e1_summary = {
                "scope1_tco2e": e1_data.get("scope1_tco2e", 0.0),
                "scope2_location_tco2e": e1_data.get("scope2_location_tco2e", 0.0),
                "scope2_market_tco2e": e1_data.get("scope2_market_tco2e", 0.0),
                "scope3_tco2e": e1_data.get("scope3_tco2e", 0.0),
                "total_energy_mwh": e1_data.get("total_energy_mwh", 0.0),
                "renewable_share_pct": e1_data.get("renewable_share_pct", 0.0),
                "has_transition_plan": e1_data.get("has_transition_plan", False),
                "sbti_validated": e1_data.get("sbti_validated", False),
                "source_pack": self.config.source_pack_id,
            }

            # Count imported disclosures and datapoints
            result.disclosures_imported = 9
            result.datapoints_imported = sum(E1_DISCLOSURE_DATAPOINT_COUNT.values())

            # Compute compliance scores per disclosure requirement
            result.compliance_scores = self._compute_compliance_scores(e1_data)

            # Map to CDM if enabled
            if self.config.map_to_cdm:
                cdm_data = self.map_to_common_model(e1_data)
                context["cdm_e1"] = cdm_data

            # Store in context
            context["e1_imported"] = result.e1_summary

            result.status = E1ImportStatus.COMPLETED
            self._import_cache = result

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.e1_summary)

            logger.info(
                "E1 import completed: %d disclosures, %d datapoints",
                result.disclosures_imported,
                result.datapoints_imported,
            )

        except Exception as exc:
            result.status = E1ImportStatus.FAILED
            result.errors.append(str(exc))
            logger.error("E1 import failed: %s", str(exc))

        self._finalize_result(result)
        return result

    def get_e1_compliance(self, context: Dict[str, Any]) -> List[E1ComplianceScore]:
        """Get E1 compliance scores for the overall ESRS scorecard.

        Args:
            context: Pipeline context with E1 data.

        Returns:
            List of E1ComplianceScore for each disclosure requirement.
        """
        e1_data = context.get("e1_results", context.get("e1_imported", {}))
        return self._compute_compliance_scores(e1_data)

    def get_e1_datapoints(self, context: Dict[str, Any]) -> List[E1DataPoint]:
        """Get E1 XBRL-tagged datapoints for report assembly.

        Args:
            context: Pipeline context with E1 data.

        Returns:
            List of E1DataPoint with XBRL element references.
        """
        e1_data = context.get("e1_results", {})
        datapoints: List[E1DataPoint] = []
        year = self.config.reporting_year

        # Scope 1 emissions datapoint
        datapoints.append(E1DataPoint(
            datapoint_id="E1-6-DP-01",
            disclosure_requirement="E1-6",
            xbrl_element="esrs-e1:GrossScope1GHGEmissions",
            value=e1_data.get("scope1_tco2e", 0.0),
            unit="tCO2e",
            period_start=f"{year}-01-01",
            period_end=f"{year}-12-31",
            is_mandatory=True,
        ))

        # Scope 2 location-based
        datapoints.append(E1DataPoint(
            datapoint_id="E1-6-DP-02",
            disclosure_requirement="E1-6",
            xbrl_element="esrs-e1:GrossScope2LocationBasedGHGEmissions",
            value=e1_data.get("scope2_location_tco2e", 0.0),
            unit="tCO2e",
            period_start=f"{year}-01-01",
            period_end=f"{year}-12-31",
            is_mandatory=True,
        ))

        # Scope 2 market-based
        datapoints.append(E1DataPoint(
            datapoint_id="E1-6-DP-03",
            disclosure_requirement="E1-6",
            xbrl_element="esrs-e1:GrossScope2MarketBasedGHGEmissions",
            value=e1_data.get("scope2_market_tco2e", 0.0),
            unit="tCO2e",
            period_start=f"{year}-01-01",
            period_end=f"{year}-12-31",
            is_mandatory=True,
        ))

        # Scope 3 total
        datapoints.append(E1DataPoint(
            datapoint_id="E1-6-DP-04",
            disclosure_requirement="E1-6",
            xbrl_element="esrs-e1:GrossScope3TotalGHGEmissions",
            value=e1_data.get("scope3_tco2e", 0.0),
            unit="tCO2e",
            period_start=f"{year}-01-01",
            period_end=f"{year}-12-31",
            is_mandatory=True,
        ))

        # Energy consumption
        datapoints.append(E1DataPoint(
            datapoint_id="E1-5-DP-01",
            disclosure_requirement="E1-5",
            xbrl_element="esrs-e1:TotalEnergyConsumption",
            value=e1_data.get("total_energy_mwh", 0.0),
            unit="MWh",
            period_start=f"{year}-01-01",
            period_end=f"{year}-12-31",
            is_mandatory=True,
        ))

        logger.info("Generated %d E1 XBRL datapoints", len(datapoints))
        return datapoints

    def map_to_common_model(self, e1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform E1 data to the PACK-017 common data model.

        Args:
            e1_data: Raw E1 results from PACK-016.

        Returns:
            Dict structured according to PACK-017 CDM schema.
        """
        cdm: Dict[str, Any] = {}
        for source_key, cdm_key in E1_CDM_FIELD_MAP.items():
            value = e1_data.get(source_key)
            if value is not None:
                cdm[cdm_key] = value

        cdm["cdm.metadata.source_pack"] = self.config.source_pack_id
        cdm["cdm.metadata.standard"] = "ESRS E1"
        cdm["cdm.metadata.mapped_at"] = utcnow().isoformat()

        logger.info("Mapped %d E1 fields to CDM", len(cdm))
        return cdm

    def export_e1_summary(self) -> Dict[str, Any]:
        """Export E1 summary for cross-standard consistency checks.

        Returns:
            Dict with E1 summary data or empty dict if not imported.
        """
        if self._import_cache and self._import_cache.status == E1ImportStatus.COMPLETED:
            return {
                "standard": "ESRS E1",
                "imported": True,
                "summary": self._import_cache.e1_summary,
                "compliance_scores": [
                    s.model_dump() for s in self._import_cache.compliance_scores
                ],
                "provenance_hash": self._import_cache.provenance_hash,
            }
        return {"standard": "ESRS E1", "imported": False, "summary": {}}

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "source_pack_id": self.config.source_pack_id,
            "target_pack_id": self.config.target_pack_id,
            "reporting_year": self.config.reporting_year,
            "auto_import": self.config.auto_import,
            "has_cached_import": self._import_cache is not None,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_compliance_scores(
        self, e1_data: Dict[str, Any]
    ) -> List[E1ComplianceScore]:
        """Compute compliance scores per disclosure requirement."""
        scores: List[E1ComplianceScore] = []
        for dr in E1DisclosureRequirement:
            dp_required = E1_DISCLOSURE_DATAPOINT_COUNT.get(dr.value, 0)
            dp_populated = e1_data.get(f"{dr.value.lower().replace('-', '_')}_datapoints", 0)
            completeness = (dp_populated / dp_required * 100) if dp_required > 0 else 0.0

            scores.append(E1ComplianceScore(
                standard="ESRS E1",
                disclosure_requirement=dr.value,
                completeness_pct=round(min(completeness, 100.0), 1),
                datapoints_required=dp_required,
                datapoints_populated=min(dp_populated, dp_required),
                mandatory_met=completeness >= 100.0,
            ))
        return scores

    def _finalize_result(self, result: E1ImportResult) -> None:
        """Set completed_at and duration_ms on a result."""
        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
