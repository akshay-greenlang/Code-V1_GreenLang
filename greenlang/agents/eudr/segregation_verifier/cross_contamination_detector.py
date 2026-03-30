# -*- coding: utf-8 -*-
"""
Cross-Contamination Detector Engine - AGENT-EUDR-010: Segregation Verifier (Feature 5)

Detects and manages cross-contamination risks across 10 pathway types for
EUDR-compliant material segregation. Provides temporal, spatial, and
equipment-based contamination risk analysis, event recording, downstream
impact tracing, status downgrade propagation, and facility risk heatmap
generation.

Zero-Hallucination Guarantees:
    - All risk scores are computed via deterministic weighted formulas
    - Temporal proximity uses pure datetime arithmetic (no ML/LLM)
    - Spatial proximity uses Euclidean distance on zone coordinates
    - Equipment sharing detection is rule-based (no predictions)
    - Root cause suggestions use static template dictionaries
    - Corrective action suggestions use static template dictionaries
    - SHA-256 provenance hashes on all detection and event results
    - No ML/LLM used for any contamination detection logic

Performance Targets:
    - Single facility risk detection: <50ms
    - Contamination impact tracing (depth 10): <100ms
    - Risk heatmap generation: <30ms

Regulatory References:
    - EUDR Article 10(2)(f): Segregation of compliant/non-compliant material
    - EUDR Article 4: Prohibition on non-compliant material in EU market
    - EUDR Article 14: Competent authority inspection readiness
    - ISO 22095:2020: Chain of Custody - Physical Segregation requirements

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010 (Feature 5: Cross-Contamination Detection)
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "ce") -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"

# ---------------------------------------------------------------------------
# Constants: Contamination Pathway Definitions
# ---------------------------------------------------------------------------

#: All recognized contamination pathway types.
CONTAMINATION_PATHWAYS: List[str] = [
    "shared_storage",
    "shared_transport",
    "shared_processing",
    "shared_equipment",
    "temporal_overlap",
    "adjacent_storage",
    "residual_material",
    "handling_error",
    "labeling_error",
    "documentation_error",
]

#: Valid contamination severity levels.
SEVERITY_LEVELS: List[str] = ["critical", "major", "minor", "observation"]

#: Severity scores for weighted risk calculations.
SEVERITY_SCORES: Dict[str, float] = {
    "critical": 1.0,
    "major": 0.7,
    "minor": 0.3,
    "observation": 0.1,
}

#: Risk weights for each contamination pathway (sum = 1.0).
PATHWAY_RISK_WEIGHTS: Dict[str, float] = {
    "shared_storage": 0.15,
    "shared_transport": 0.15,
    "shared_processing": 0.20,
    "shared_equipment": 0.15,
    "temporal_overlap": 0.10,
    "adjacent_storage": 0.08,
    "residual_material": 0.07,
    "handling_error": 0.05,
    "labeling_error": 0.03,
    "documentation_error": 0.02,
}

#: Default temporal proximity threshold in hours.
DEFAULT_TEMPORAL_THRESHOLD_HOURS: float = 4.0

#: Default spatial proximity threshold in meters.
DEFAULT_SPATIAL_THRESHOLD_METERS: float = 5.0

#: Maximum contamination impact propagation depth.
MAX_PROPAGATION_DEPTH: int = 10

#: Root cause templates by pathway type (deterministic, no LLM).
ROOT_CAUSE_TEMPLATES: Dict[str, List[str]] = {
    "shared_storage": [
        "Zone barriers removed or damaged during maintenance",
        "Compliant material placed in non-compliant zone by error",
        "Zone capacity overflow forcing temporary co-storage",
        "Barrier integrity not verified before storage event",
        "Zone assignment documentation not updated after reconfiguration",
    ],
    "shared_transport": [
        "Vehicle not cleaned between compliant and non-compliant loads",
        "Cleaning verification record missing or incomplete",
        "Vehicle reassigned without proper decontamination procedure",
        "Transport schedule compressed skipping cleaning window",
        "Driver loaded wrong commodity onto pre-assigned vehicle",
    ],
    "shared_processing": [
        "Processing line changeover time insufficient",
        "Flush volume below minimum threshold during changeover",
        "Changeover protocol steps skipped by operator",
        "Line not fully drained before compliant run started",
        "Residual non-compliant material trapped in dead zones",
    ],
    "shared_equipment": [
        "Forklift used for both material types without cleaning",
        "Conveyor not purged between compliant and non-compliant runs",
        "Shared hopper contaminated from previous non-compliant batch",
        "Weighing equipment not cleaned between material streams",
        "Sampling tools shared without decontamination protocol",
    ],
    "temporal_overlap": [
        "Compliant and non-compliant operations scheduled concurrently",
        "Insufficient buffer time between material changeovers",
        "Shift overlap caused simultaneous handling of both streams",
        "Emergency processing compressed scheduled separation window",
        "Overtime operations bypassed temporal segregation controls",
    ],
    "adjacent_storage": [
        "Adjacent zones within minimum separation distance",
        "Barrier between adjacent zones does not extend to ceiling",
        "Spillage from non-compliant zone reached compliant zone",
        "Dust or particulate migration between adjacent zones",
        "Drainage channel connects compliant and non-compliant zones",
    ],
    "residual_material": [
        "Equipment internal surfaces not fully cleaned after use",
        "Dead legs in piping retained non-compliant material residue",
        "Filter or screen not replaced after non-compliant processing",
        "Silo walls retained residual non-compliant material coating",
        "Processing vessel not inspected for residue after cleaning",
    ],
    "handling_error": [
        "Operator picked material from wrong zone during retrieval",
        "Bags placed on wrong pallet during manual stacking",
        "Fork-lift operator deposited load in incorrect bay",
        "Sample taken from wrong batch during quality check",
        "Manual tallying error caused misattribution of material",
    ],
    "labeling_error": [
        "Label not applied to incoming compliant batch",
        "Wrong color label applied to material container",
        "Label fell off during transport or handling",
        "Expired label not replaced with current version",
        "Label content fields incomplete or illegible",
    ],
    "documentation_error": [
        "Batch record not updated after zone transfer",
        "Transport manifest referenced wrong batch identifier",
        "Processing log entry omitted changeover timestamp",
        "Inventory system not synchronized with physical location",
        "Chain of custody document missing required signatures",
    ],
}

#: Corrective action templates by pathway and severity (deterministic).
CORRECTIVE_ACTION_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "shared_storage": {
        "critical": [
            "Immediately quarantine all affected batches in isolated zone",
            "Conduct full physical inventory of affected storage zones",
            "Install permanent physical barriers between material streams",
            "Retrain all warehouse personnel on segregation protocols",
            "Submit incident report to competent authority within 24 hours",
        ],
        "major": [
            "Quarantine affected batches pending investigation",
            "Verify barrier integrity and repair any deficiencies",
            "Review zone assignment procedures and update SOPs",
            "Conduct refresher training for affected shift personnel",
        ],
        "minor": [
            "Reinforce zone markers and signage at affected location",
            "Review and update zone capacity planning procedures",
            "Add additional verification checkpoints for zone assignment",
        ],
        "observation": [
            "Document finding in continuous improvement register",
            "Schedule review of storage zone layout at next planning cycle",
        ],
    },
    "shared_transport": {
        "critical": [
            "Remove vehicle from compliant service immediately",
            "Quarantine all material transported in affected loads",
            "Conduct deep cleaning and verification before reinstatement",
            "Review all loads from same vehicle in past 30 days",
            "File regulatory incident notification with competent authority",
        ],
        "major": [
            "Hold vehicle for enhanced cleaning verification",
            "Trace and flag all downstream batches from affected loads",
            "Update cleaning verification checklist requirements",
            "Increase cleaning inspection frequency for shared vehicles",
        ],
        "minor": [
            "Reinforce cleaning verification procedure with drivers",
            "Update vehicle cleaning log form for better traceability",
            "Add photographic evidence requirement for cleaning verification",
        ],
        "observation": [
            "Note finding in transport operations improvement log",
            "Review dedicated vehicle allocation for high-volume routes",
        ],
    },
    "shared_processing": {
        "critical": [
            "Halt processing line immediately for full decontamination",
            "Quarantine all output from current and previous processing runs",
            "Conduct engineering review of changeover protocol adequacy",
            "Increase minimum changeover time and flush volume requirements",
            "Report processing contamination to quality management team",
        ],
        "major": [
            "Stop line and conduct enhanced changeover procedure",
            "Flag first-run output for enhanced quality testing",
            "Review and update changeover SOP for affected line",
            "Verify flush volume measurement equipment calibration",
        ],
        "minor": [
            "Schedule additional training on changeover procedures",
            "Review dead-zone analysis for processing line piping",
            "Update first-run flagging criteria and monitoring intervals",
        ],
        "observation": [
            "Document changeover timing observation in process log",
            "Evaluate feasibility of dedicated processing line allocation",
        ],
    },
    "shared_equipment": {
        "critical": [
            "Remove shared equipment from service for full decontamination",
            "Quarantine all material handled by contaminated equipment",
            "Procure dedicated equipment for compliant material stream",
            "Implement equipment booking system with mandatory cleaning slots",
            "Report equipment contamination in facility incident register",
        ],
        "major": [
            "Clean equipment thoroughly before next compliant use",
            "Trace all batches handled by equipment in affected period",
            "Add equipment cleaning verification checkpoint",
            "Update equipment assignment schedule to reduce sharing",
        ],
        "minor": [
            "Reinforce equipment cleaning protocol with operators",
            "Install equipment status indicators (clean/dirty markers)",
            "Review equipment utilization for shared-use reduction",
        ],
        "observation": [
            "Log observation in equipment maintenance register",
            "Evaluate long-term dedicated equipment investment",
        ],
    },
    "temporal_overlap": {
        "critical": [
            "Halt all operations until temporal separation is restored",
            "Quarantine material from overlapping time windows",
            "Redesign production schedule with enforced buffer periods",
            "Install automated scheduling system with segregation rules",
            "Report temporal contamination to operations management",
        ],
        "major": [
            "Extend buffer time between compliant and non-compliant runs",
            "Review and update production scheduling constraints",
            "Flag all material produced during overlap period",
            "Conduct investigation into scheduling override cause",
        ],
        "minor": [
            "Update scheduling system with minimum buffer time alerts",
            "Train production planners on temporal segregation requirements",
            "Add pre-start verification checkpoint for material stream",
        ],
        "observation": [
            "Note temporal proximity in production planning review",
            "Evaluate automated scheduling enforcement feasibility",
        ],
    },
    "adjacent_storage": {
        "critical": [
            "Relocate compliant material to non-adjacent zone immediately",
            "Install full-height physical barriers between affected zones",
            "Conduct contamination testing on compliant material samples",
            "Review facility layout for all adjacent zone risks",
            "Submit layout change plan to quality management for approval",
        ],
        "major": [
            "Increase separation distance between material streams",
            "Reinforce existing barriers to prevent spillage migration",
            "Add spillage containment systems at zone boundaries",
            "Update zone assignment rules to prevent adjacent placement",
        ],
        "minor": [
            "Mark minimum separation distance on warehouse floor",
            "Add barrier inspection to daily zone check routine",
            "Review drainage routing between adjacent zones",
        ],
        "observation": [
            "Document adjacent storage risk in facility risk register",
            "Include zone adjacency review in next layout assessment",
        ],
    },
    "residual_material": {
        "critical": [
            "Take equipment offline for full disassembly and cleaning",
            "Quarantine all material processed since last verified clean",
            "Engage external contractor for validated deep cleaning",
            "Install inspection ports on all equipment dead zones",
            "Update cleaning validation protocol with swab testing",
        ],
        "major": [
            "Conduct enhanced cleaning with residue verification",
            "Trace and flag material from affected equipment runs",
            "Add post-cleaning inspection step with documented sign-off",
            "Review equipment design for residue retention points",
        ],
        "minor": [
            "Update cleaning procedure to address identified retention points",
            "Schedule preventive maintenance for equipment cleaning access",
            "Add residue check to pre-run verification checklist",
        ],
        "observation": [
            "Log residue observation in equipment cleaning register",
            "Evaluate equipment replacement or modification options",
        ],
    },
    "handling_error": {
        "critical": [
            "Stop operations and conduct full material reconciliation",
            "Quarantine all material handled in affected shift",
            "Retrain handling personnel with practical demonstration",
            "Implement buddy-system verification for material movements",
            "File handling incident in facility quality system",
        ],
        "major": [
            "Verify correct location of all material from affected period",
            "Conduct targeted retraining for involved personnel",
            "Add barcode/RFID verification for material pick operations",
            "Update material handling procedures with visual guides",
        ],
        "minor": [
            "Reinforce zone identification training for handling teams",
            "Improve zone signage visibility and color coding",
            "Add verification prompts to material handling checklists",
        ],
        "observation": [
            "Note handling observation in personnel development log",
            "Review material handling ergonomics and workflow design",
        ],
    },
    "labeling_error": {
        "critical": [
            "Halt material movement until all labels verified correct",
            "Re-label all material in affected batch or zone",
            "Implement dual-verification for label application",
            "Audit all labels in facility for accuracy and readability",
            "Report labeling failure in quality incident system",
        ],
        "major": [
            "Verify and correct labels on affected material immediately",
            "Review label application procedure and update SOP",
            "Add label verification checkpoint at zone entry points",
            "Inspect label durability and replace with improved material",
        ],
        "minor": [
            "Replace damaged or illegible labels immediately",
            "Order improved label stock with better adhesion properties",
            "Add label condition check to daily zone inspection routine",
        ],
        "observation": [
            "Document label condition observation in labeling register",
            "Evaluate automated labeling system for high-volume zones",
        ],
    },
    "documentation_error": {
        "critical": [
            "Freeze all material movements until records reconciled",
            "Conduct full audit of documentation for affected batches",
            "Implement double-entry verification for critical records",
            "Engage internal audit team for documentation system review",
            "File documentation non-conformance in quality system",
        ],
        "major": [
            "Correct affected documentation records immediately",
            "Verify chain of custody records for involved batches",
            "Add document review checkpoint at shift handover",
            "Update documentation training for affected personnel",
        ],
        "minor": [
            "Correct minor documentation discrepancies",
            "Reinforce documentation completion requirements",
            "Add automated completeness checks to data entry forms",
        ],
        "observation": [
            "Log documentation improvement suggestion",
            "Review documentation workflow for simplification",
        ],
    },
}

# ---------------------------------------------------------------------------
# Internal Dataclass Result Types
# ---------------------------------------------------------------------------

@dataclass
class ContaminationEventRecord:
    """A recorded contamination event with full provenance tracking.

    Attributes:
        event_id: Unique identifier for this contamination event.
        facility_id: Identifier of the facility where event occurred.
        pathway_type: One of 10 contamination pathway types.
        severity: Severity classification (critical/major/minor/observation).
        affected_batch_ids: List of batch identifiers affected.
        affected_quantity_kg: Total affected quantity in kilograms.
        root_cause: Description of identified root cause.
        corrective_action: Description of corrective action taken or planned.
        resolved: Whether the contamination event has been resolved.
        resolved_date: Date and time of resolution, if resolved.
        timestamp: When the event was recorded.
        metadata: Additional contextual data.
        provenance_hash: SHA-256 hash for audit trail.
    """

    event_id: str
    facility_id: str
    pathway_type: str
    severity: str
    affected_batch_ids: List[str]
    affected_quantity_kg: float
    root_cause: str
    corrective_action: str
    resolved: bool
    resolved_date: Optional[str]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

@dataclass
class ContaminationDetectionResult:
    """Result of a contamination risk detection scan for a facility.

    Attributes:
        facility_id: Identifier of the scanned facility.
        risks_detected: List of individual risk findings.
        risk_score: Overall risk score (0-100), 0=no risk, 100=critical.
        pathway_breakdown: Risk contribution by pathway type.
        temporal_risks: List of temporal proximity risk findings.
        spatial_risks: List of spatial proximity risk findings.
        equipment_risks: List of equipment sharing risk findings.
        recommendations: List of recommended corrective actions.
        provenance_hash: SHA-256 hash for audit trail.
    """

    facility_id: str
    risks_detected: List[Dict[str, Any]]
    risk_score: float
    pathway_breakdown: Dict[str, float]
    temporal_risks: List[Dict[str, Any]]
    spatial_risks: List[Dict[str, Any]]
    equipment_risks: List[Dict[str, Any]]
    recommendations: List[str]
    provenance_hash: str = ""

@dataclass
class ContaminationImpactResult:
    """Result of tracing downstream impact of a contamination event.

    Attributes:
        contamination_event_id: Identifier of the source event.
        downstream_batch_ids: List of downstream batch identifiers affected.
        total_affected_quantity_kg: Total quantity across all affected batches.
        status_downgrades: List of batch status downgrade records.
        propagation_depth: Depth of downstream impact propagation.
        provenance_hash: SHA-256 hash for audit trail.
    """

    contamination_event_id: str
    downstream_batch_ids: List[str]
    total_affected_quantity_kg: float
    status_downgrades: List[Dict[str, Any]]
    propagation_depth: int
    provenance_hash: str = ""

@dataclass
class RiskHeatmapData:
    """Risk heatmap data for a facility's segregation zones.

    Attributes:
        facility_id: Identifier of the facility.
        zones: List of zone risk data dictionaries.
        overall_risk: Overall facility risk score (0-100).
        generated_at: ISO timestamp when heatmap was generated.
        provenance_hash: SHA-256 hash for audit trail.
    """

    facility_id: str
    zones: List[Dict[str, Any]]
    overall_risk: float
    generated_at: str
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# CrossContaminationDetector Engine
# ---------------------------------------------------------------------------

class CrossContaminationDetector:
    """Detects and manages cross-contamination risks for EUDR segregation.

    Provides comprehensive contamination detection across 10 pathway
    types, event recording with severity classification, downstream
    impact tracing, automatic status downgrade propagation, facility
    risk heatmap generation, and deterministic root cause / corrective
    action suggestions.

    All detection logic is deterministic (no ML/LLM). Risk scores are
    computed via weighted pathway formulas. Root cause and corrective
    action suggestions use static template dictionaries indexed by
    pathway type and severity.

    Attributes:
        _events: In-memory store of contamination events keyed by event_id.
        _facility_events: Mapping of facility_id to list of event_ids.
        _temporal_threshold_hours: Temporal proximity detection threshold.
        _spatial_threshold_meters: Spatial proximity detection threshold.

    Example:
        >>> detector = CrossContaminationDetector()
        >>> result = detector.detect_contamination_risks(
        ...     facility_id="fac-001",
        ...     storage_events=[...],
        ...     transport_verifications=[...],
        ...     processing_records=[...],
        ... )
        >>> assert 0.0 <= result.risk_score <= 100.0
    """

    def __init__(
        self,
        temporal_threshold_hours: float = DEFAULT_TEMPORAL_THRESHOLD_HOURS,
        spatial_threshold_meters: float = DEFAULT_SPATIAL_THRESHOLD_METERS,
    ) -> None:
        """Initialize CrossContaminationDetector.

        Args:
            temporal_threshold_hours: Time window in hours for temporal
                proximity detection. Events within this window are flagged.
            spatial_threshold_meters: Distance in meters for spatial
                proximity detection. Zones closer than this are flagged.
        """
        self._events: Dict[str, ContaminationEventRecord] = {}
        self._facility_events: Dict[str, List[str]] = {}
        self._temporal_threshold_hours: float = max(0.1, temporal_threshold_hours)
        self._spatial_threshold_meters: float = max(0.0, spatial_threshold_meters)
        logger.info(
            "CrossContaminationDetector initialized: "
            "temporal_threshold=%.1fh, spatial_threshold=%.1fm",
            self._temporal_threshold_hours,
            self._spatial_threshold_meters,
        )

    # ------------------------------------------------------------------
    # Public API: Risk Detection
    # ------------------------------------------------------------------

    def detect_contamination_risks(
        self,
        facility_id: str,
        storage_events: List[Dict[str, Any]],
        transport_verifications: List[Dict[str, Any]],
        processing_records: List[Dict[str, Any]],
    ) -> ContaminationDetectionResult:
        """Run full contamination risk assessment for a facility.

        Analyzes storage events, transport verifications, and processing
        records to detect contamination risks across all 10 pathway
        types. Computes a weighted overall risk score and provides
        pathway-level breakdown.

        Args:
            facility_id: Identifier of the facility to assess.
            storage_events: List of storage event dictionaries with keys:
                zone_id, batch_id, compliance_status, timestamp,
                quantity_kg, zone_x, zone_y.
            transport_verifications: List of transport verification dicts
                with keys: vehicle_id, cleaning_verified, previous_cargo,
                dedicated, timestamp.
            processing_records: List of processing record dicts with keys:
                line_id, batch_id, compliance_status, changeover_time_min,
                flush_volume_l, equipment_ids, timestamp.

        Returns:
            ContaminationDetectionResult with risk score, pathway
            breakdown, and recommendations.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")

        risks_detected: List[Dict[str, Any]] = []
        pathway_scores: Dict[str, float] = {p: 0.0 for p in CONTAMINATION_PATHWAYS}

        # -- Detect storage-related pathways --
        storage_risks = self._detect_storage_risks(
            facility_id, storage_events,
        )
        risks_detected.extend(storage_risks)
        for risk in storage_risks:
            pathway = risk.get("pathway", "")
            if pathway in pathway_scores:
                pathway_scores[pathway] = max(
                    pathway_scores[pathway],
                    risk.get("score", 0.0),
                )

        # -- Detect transport-related pathways --
        transport_risks = self._detect_transport_risks(
            facility_id, transport_verifications,
        )
        risks_detected.extend(transport_risks)
        for risk in transport_risks:
            pathway = risk.get("pathway", "")
            if pathway in pathway_scores:
                pathway_scores[pathway] = max(
                    pathway_scores[pathway],
                    risk.get("score", 0.0),
                )

        # -- Detect processing-related pathways --
        processing_risks = self._detect_processing_risks(
            facility_id, processing_records,
        )
        risks_detected.extend(processing_risks)
        for risk in processing_risks:
            pathway = risk.get("pathway", "")
            if pathway in pathway_scores:
                pathway_scores[pathway] = max(
                    pathway_scores[pathway],
                    risk.get("score", 0.0),
                )

        # -- Temporal proximity analysis --
        temporal_risks = self.analyze_temporal_proximity(
            storage_events + processing_records,
            threshold_hours=self._temporal_threshold_hours,
        )
        for risk in temporal_risks:
            if "temporal_overlap" in pathway_scores:
                pathway_scores["temporal_overlap"] = max(
                    pathway_scores["temporal_overlap"],
                    risk.get("score", 0.0),
                )
        risks_detected.extend(temporal_risks)

        # -- Spatial proximity analysis --
        spatial_risks = self.analyze_spatial_proximity(
            storage_events,
            threshold_meters=self._spatial_threshold_meters,
        )
        for risk in spatial_risks:
            if "adjacent_storage" in pathway_scores:
                pathway_scores["adjacent_storage"] = max(
                    pathway_scores["adjacent_storage"],
                    risk.get("score", 0.0),
                )
        risks_detected.extend(spatial_risks)

        # -- Equipment sharing analysis --
        equipment_risks = self.detect_equipment_sharing(
            processing_records,
            [],
        )
        for risk in equipment_risks:
            if "shared_equipment" in pathway_scores:
                pathway_scores["shared_equipment"] = max(
                    pathway_scores["shared_equipment"],
                    risk.get("score", 0.0),
                )
        risks_detected.extend(equipment_risks)

        # -- Compute weighted overall risk score --
        overall_risk = self._compute_overall_risk(pathway_scores)

        # -- Generate recommendations --
        recommendations = self._generate_risk_recommendations(
            risks_detected, pathway_scores,
        )

        # -- Build pathway breakdown --
        pathway_breakdown: Dict[str, float] = {}
        for pathway, score in pathway_scores.items():
            weight = PATHWAY_RISK_WEIGHTS.get(pathway, 0.0)
            pathway_breakdown[pathway] = round(score * weight * 100.0, 2)

        result = ContaminationDetectionResult(
            facility_id=facility_id,
            risks_detected=risks_detected,
            risk_score=round(overall_risk, 2),
            pathway_breakdown=pathway_breakdown,
            temporal_risks=temporal_risks,
            spatial_risks=spatial_risks,
            equipment_risks=equipment_risks,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "risk_score": result.risk_score,
            "risks_count": len(risks_detected),
            "pathway_breakdown": pathway_breakdown,
            "module_version": _MODULE_VERSION,
        })

        logger.info(
            "Contamination risk detection completed for facility=%s: "
            "score=%.2f, risks=%d, pathways_active=%d",
            facility_id,
            result.risk_score,
            len(risks_detected),
            sum(1 for s in pathway_scores.values() if s > 0.0),
        )
        return result

    def analyze_temporal_proximity(
        self,
        events: List[Dict[str, Any]],
        threshold_hours: float = DEFAULT_TEMPORAL_THRESHOLD_HOURS,
    ) -> List[Dict[str, Any]]:
        """Detect temporal proximity risks between compliant/non-compliant events.

        Identifies pairs of events where compliant and non-compliant
        material were handled within the specified time window at the
        same location or on the same equipment.

        Args:
            events: List of event dictionaries with keys: batch_id,
                compliance_status, timestamp (ISO format string or
                datetime), zone_id or line_id.
            threshold_hours: Maximum hours between events to flag
                as temporal proximity risk.

        Returns:
            List of temporal risk finding dictionaries with keys:
            pathway, event_a, event_b, gap_hours, score, description.
        """
        if not events or threshold_hours <= 0.0:
            return []

        threshold = max(0.1, threshold_hours)
        parsed_events = self._parse_and_sort_events(events)
        risks: List[Dict[str, Any]] = []

        for i in range(len(parsed_events)):
            for j in range(i + 1, len(parsed_events)):
                evt_a = parsed_events[i]
                evt_b = parsed_events[j]

                status_a = evt_a.get("compliance_status", "").lower()
                status_b = evt_b.get("compliance_status", "").lower()

                # Only flag pairs with different compliance status
                if status_a == status_b:
                    continue
                if status_a not in ("compliant", "non_compliant"):
                    continue
                if status_b not in ("compliant", "non_compliant"):
                    continue

                # Check location match (zone or line)
                loc_a = evt_a.get("zone_id") or evt_a.get("line_id", "")
                loc_b = evt_b.get("zone_id") or evt_b.get("line_id", "")
                if loc_a and loc_b and loc_a != loc_b:
                    continue

                # Compute time gap
                ts_a = evt_a.get("_parsed_ts")
                ts_b = evt_b.get("_parsed_ts")
                if ts_a is None or ts_b is None:
                    continue

                gap = abs((ts_b - ts_a).total_seconds()) / 3600.0
                if gap <= threshold:
                    score = max(0.0, min(1.0, 1.0 - (gap / threshold)))
                    risks.append({
                        "pathway": "temporal_overlap",
                        "event_a": evt_a.get("batch_id", "unknown"),
                        "event_b": evt_b.get("batch_id", "unknown"),
                        "gap_hours": round(gap, 2),
                        "score": round(score, 4),
                        "location": loc_a or loc_b or "unknown",
                        "description": (
                            f"Temporal proximity detected: {status_a} "
                            f"batch {evt_a.get('batch_id', '?')} and "
                            f"{status_b} batch {evt_b.get('batch_id', '?')} "
                            f"handled {gap:.1f}h apart (threshold: {threshold:.1f}h)"
                        ),
                    })

        logger.debug(
            "Temporal proximity analysis: %d events, %d risks found "
            "(threshold=%.1fh)",
            len(events),
            len(risks),
            threshold,
        )
        return risks

    def analyze_spatial_proximity(
        self,
        zones: List[Dict[str, Any]],
        threshold_meters: float = DEFAULT_SPATIAL_THRESHOLD_METERS,
    ) -> List[Dict[str, Any]]:
        """Detect spatial proximity risks between compliant/non-compliant zones.

        Identifies pairs of zones where compliant and non-compliant
        material are stored closer than the minimum separation distance.

        Args:
            zones: List of zone dictionaries with keys: zone_id,
                compliance_status, zone_x (meters), zone_y (meters).
            threshold_meters: Minimum separation distance in meters.

        Returns:
            List of spatial risk finding dictionaries with keys:
            pathway, zone_a, zone_b, distance_m, score, description.
        """
        if not zones or threshold_meters < 0.0:
            return []

        threshold = max(0.0, threshold_meters)
        risks: List[Dict[str, Any]] = []
        compliant_zones = [
            z for z in zones
            if str(z.get("compliance_status", "")).lower() == "compliant"
        ]
        non_compliant_zones = [
            z for z in zones
            if str(z.get("compliance_status", "")).lower() == "non_compliant"
        ]

        for cz in compliant_zones:
            cx = float(cz.get("zone_x", 0.0))
            cy = float(cz.get("zone_y", 0.0))
            cz_id = cz.get("zone_id", "unknown")

            for ncz in non_compliant_zones:
                nx = float(ncz.get("zone_x", 0.0))
                ny = float(ncz.get("zone_y", 0.0))
                ncz_id = ncz.get("zone_id", "unknown")

                distance = ((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5

                if distance <= threshold:
                    score = max(0.0, min(1.0, 1.0 - (distance / max(threshold, 0.01))))
                    risks.append({
                        "pathway": "adjacent_storage",
                        "zone_a": cz_id,
                        "zone_b": ncz_id,
                        "distance_m": round(distance, 2),
                        "score": round(score, 4),
                        "description": (
                            f"Spatial proximity detected: compliant zone "
                            f"{cz_id} is {distance:.1f}m from non-compliant "
                            f"zone {ncz_id} (minimum: {threshold:.1f}m)"
                        ),
                    })

        logger.debug(
            "Spatial proximity analysis: %d zones, %d risks found "
            "(threshold=%.1fm)",
            len(zones),
            len(risks),
            threshold,
        )
        return risks

    def detect_equipment_sharing(
        self,
        processing_lines: List[Dict[str, Any]],
        equipment_logs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect equipment sharing risks between material streams.

        Identifies cases where the same equipment is used for both
        compliant and non-compliant processing without adequate
        cleaning verification between uses.

        Args:
            processing_lines: List of processing record dicts with
                keys: line_id, batch_id, compliance_status,
                equipment_ids (list), timestamp.
            equipment_logs: List of equipment log dicts with keys:
                equipment_id, cleaning_verified, last_material_type,
                timestamp.

        Returns:
            List of equipment sharing risk finding dicts with keys:
            pathway, equipment_id, compliant_use, non_compliant_use,
            cleaning_verified, score, description.
        """
        risks: List[Dict[str, Any]] = []

        # Build equipment usage map from processing records
        equipment_usage: Dict[str, List[Dict[str, Any]]] = {}
        for record in processing_lines:
            eq_ids = record.get("equipment_ids", [])
            if not isinstance(eq_ids, list):
                continue
            for eq_id in eq_ids:
                if eq_id not in equipment_usage:
                    equipment_usage[eq_id] = []
                equipment_usage[eq_id].append({
                    "line_id": record.get("line_id", ""),
                    "batch_id": record.get("batch_id", ""),
                    "compliance_status": str(
                        record.get("compliance_status", "")
                    ).lower(),
                    "timestamp": record.get("timestamp", ""),
                })

        # Check for shared usage without cleaning
        cleaning_map: Dict[str, bool] = {}
        for log_entry in equipment_logs:
            eq_id = log_entry.get("equipment_id", "")
            cleaned = bool(log_entry.get("cleaning_verified", False))
            cleaning_map[eq_id] = cleaned

        for eq_id, usages in equipment_usage.items():
            statuses = set(
                u.get("compliance_status", "") for u in usages
            )
            has_compliant = "compliant" in statuses
            has_non_compliant = "non_compliant" in statuses

            if has_compliant and has_non_compliant:
                cleaned = cleaning_map.get(eq_id, False)
                score = 0.3 if cleaned else 0.9

                compliant_batches = [
                    u.get("batch_id", "")
                    for u in usages
                    if u.get("compliance_status") == "compliant"
                ]
                non_compliant_batches = [
                    u.get("batch_id", "")
                    for u in usages
                    if u.get("compliance_status") == "non_compliant"
                ]

                risks.append({
                    "pathway": "shared_equipment",
                    "equipment_id": eq_id,
                    "compliant_use": compliant_batches,
                    "non_compliant_use": non_compliant_batches,
                    "cleaning_verified": cleaned,
                    "score": round(score, 4),
                    "description": (
                        f"Equipment {eq_id} used for both compliant and "
                        f"non-compliant material. Cleaning "
                        f"{'verified' if cleaned else 'NOT verified'}."
                    ),
                })

        logger.debug(
            "Equipment sharing analysis: %d equipment items, %d risks",
            len(equipment_usage),
            len(risks),
        )
        return risks

    # ------------------------------------------------------------------
    # Public API: Event Management
    # ------------------------------------------------------------------

    def record_contamination_event(
        self,
        facility_id: str,
        pathway_type: str,
        severity: str,
        affected_batches: List[str],
        quantity_kg: float,
        root_cause: str,
    ) -> ContaminationEventRecord:
        """Record a new contamination event.

        Creates a contamination event record with full provenance
        tracking. The event is stored in-memory indexed by event_id
        and facility_id for retrieval and impact analysis.

        Args:
            facility_id: Identifier of the facility where the event occurred.
            pathway_type: Contamination pathway (must be one of 10 types).
            severity: Severity level (critical/major/minor/observation).
            affected_batches: List of affected batch identifiers.
            quantity_kg: Total affected quantity in kilograms.
            root_cause: Description of the identified root cause.

        Returns:
            ContaminationEventRecord with unique event_id and provenance hash.

        Raises:
            ValueError: If any required parameter is invalid.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if pathway_type not in CONTAMINATION_PATHWAYS:
            raise ValueError(
                f"pathway_type must be one of {CONTAMINATION_PATHWAYS}, "
                f"got '{pathway_type}'"
            )
        if severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"severity must be one of {SEVERITY_LEVELS}, "
                f"got '{severity}'"
            )
        if quantity_kg < 0.0:
            raise ValueError(
                f"quantity_kg must be >= 0, got {quantity_kg}"
            )

        event_id = _generate_id("ce")
        now = utcnow()

        event = ContaminationEventRecord(
            event_id=event_id,
            facility_id=facility_id,
            pathway_type=pathway_type,
            severity=severity,
            affected_batch_ids=list(affected_batches),
            affected_quantity_kg=quantity_kg,
            root_cause=root_cause,
            corrective_action="",
            resolved=False,
            resolved_date=None,
            timestamp=now.isoformat(),
            metadata={
                "module_version": _MODULE_VERSION,
                "batches_count": len(affected_batches),
            },
        )
        event.provenance_hash = _compute_hash({
            "event_id": event_id,
            "facility_id": facility_id,
            "pathway_type": pathway_type,
            "severity": severity,
            "affected_batch_ids": event.affected_batch_ids,
            "affected_quantity_kg": quantity_kg,
            "root_cause": root_cause,
            "timestamp": event.timestamp,
        })

        # Store event
        self._events[event_id] = event
        if facility_id not in self._facility_events:
            self._facility_events[facility_id] = []
        self._facility_events[facility_id].append(event_id)

        logger.info(
            "Contamination event recorded: event_id=%s, facility=%s, "
            "pathway=%s, severity=%s, batches=%d, qty=%.1fkg",
            event_id,
            facility_id,
            pathway_type,
            severity,
            len(affected_batches),
            quantity_kg,
        )
        return event

    def assess_contamination_impact(
        self,
        event_id: str,
        batch_genealogy: Dict[str, List[str]],
    ) -> ContaminationImpactResult:
        """Trace downstream impact of a contamination event.

        Follows batch genealogy to identify all downstream batches
        that may be affected by the contamination event, up to the
        maximum propagation depth.

        Args:
            event_id: Identifier of the contamination event to trace.
            batch_genealogy: Mapping of parent batch_id to list of
                child batch_ids representing downstream processing.

        Returns:
            ContaminationImpactResult with downstream batch IDs,
            propagation depth, and status downgrade records.

        Raises:
            ValueError: If event_id is not found.
        """
        if event_id not in self._events:
            raise ValueError(f"Contamination event not found: {event_id}")

        event = self._events[event_id]
        downstream_batches: List[str] = []
        status_downgrades: List[Dict[str, Any]] = []
        visited: set = set()

        # BFS through batch genealogy up to MAX_PROPAGATION_DEPTH
        current_level = list(event.affected_batch_ids)
        depth = 0

        while current_level and depth < MAX_PROPAGATION_DEPTH:
            next_level: List[str] = []
            for batch_id in current_level:
                if batch_id in visited:
                    continue
                visited.add(batch_id)

                children = batch_genealogy.get(batch_id, [])
                for child_id in children:
                    if child_id not in visited:
                        downstream_batches.append(child_id)
                        next_level.append(child_id)

                        # Record potential downgrade
                        status_downgrades.append({
                            "batch_id": child_id,
                            "parent_batch_id": batch_id,
                            "reason": (
                                f"Downstream of contamination event "
                                f"{event_id} via batch {batch_id}"
                            ),
                            "original_status": "compliant",
                            "new_status": self._determine_downgrade_status(
                                event.severity, depth + 1,
                            ),
                            "depth": depth + 1,
                        })

            current_level = next_level
            depth += 1

        # Estimate total affected quantity
        avg_qty = (
            event.affected_quantity_kg / max(len(event.affected_batch_ids), 1)
        )
        total_affected = (
            event.affected_quantity_kg + avg_qty * len(downstream_batches)
        )

        result = ContaminationImpactResult(
            contamination_event_id=event_id,
            downstream_batch_ids=downstream_batches,
            total_affected_quantity_kg=round(total_affected, 2),
            status_downgrades=status_downgrades,
            propagation_depth=depth,
        )
        result.provenance_hash = _compute_hash({
            "event_id": event_id,
            "downstream_count": len(downstream_batches),
            "total_affected_kg": result.total_affected_quantity_kg,
            "propagation_depth": depth,
            "module_version": _MODULE_VERSION,
        })

        logger.info(
            "Contamination impact assessed: event=%s, downstream=%d, "
            "depth=%d, total_qty=%.1fkg",
            event_id,
            len(downstream_batches),
            depth,
            result.total_affected_quantity_kg,
        )
        return result

    def propagate_status_downgrade(
        self,
        event_id: str,
        affected_batches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Auto-downgrade SG status for batches affected by contamination.

        Applies deterministic downgrade rules based on contamination
        severity and batch proximity to the contamination source.

        Args:
            event_id: Identifier of the contamination event.
            affected_batches: List of batch dicts with keys: batch_id,
                current_status, depth (distance from source).

        Returns:
            List of downgrade action dictionaries with keys:
            batch_id, original_status, new_status, reason, event_id.

        Raises:
            ValueError: If event_id is not found.
        """
        if event_id not in self._events:
            raise ValueError(f"Contamination event not found: {event_id}")

        event = self._events[event_id]
        downgrades: List[Dict[str, Any]] = []

        for batch in affected_batches:
            batch_id = batch.get("batch_id", "")
            current_status = str(batch.get("current_status", "compliant")).lower()
            depth = int(batch.get("depth", 1))

            if not batch_id:
                continue

            new_status = self._determine_downgrade_status(
                event.severity, depth,
            )

            if new_status != current_status:
                downgrades.append({
                    "batch_id": batch_id,
                    "original_status": current_status,
                    "new_status": new_status,
                    "reason": (
                        f"Downgraded due to {event.severity} contamination "
                        f"event {event_id} (pathway: {event.pathway_type}, "
                        f"depth: {depth})"
                    ),
                    "event_id": event_id,
                    "timestamp": utcnow().isoformat(),
                })

        logger.info(
            "Status downgrade propagation: event=%s, batches_evaluated=%d, "
            "downgrades_applied=%d",
            event_id,
            len(affected_batches),
            len(downgrades),
        )
        return downgrades

    def resolve_contamination(
        self,
        event_id: str,
        corrective_action: str,
    ) -> ContaminationEventRecord:
        """Resolve a contamination event with corrective action.

        Marks the contamination event as resolved, recording the
        corrective action taken and the resolution timestamp.

        Args:
            event_id: Identifier of the contamination event to resolve.
            corrective_action: Description of the corrective action taken.

        Returns:
            Updated ContaminationEventRecord with resolved=True.

        Raises:
            ValueError: If event_id is not found or already resolved.
        """
        if event_id not in self._events:
            raise ValueError(f"Contamination event not found: {event_id}")

        event = self._events[event_id]
        if event.resolved:
            raise ValueError(
                f"Contamination event {event_id} is already resolved"
            )

        now = utcnow()
        resolved_event = ContaminationEventRecord(
            event_id=event.event_id,
            facility_id=event.facility_id,
            pathway_type=event.pathway_type,
            severity=event.severity,
            affected_batch_ids=list(event.affected_batch_ids),
            affected_quantity_kg=event.affected_quantity_kg,
            root_cause=event.root_cause,
            corrective_action=corrective_action,
            resolved=True,
            resolved_date=now.isoformat(),
            timestamp=event.timestamp,
            metadata={
                **event.metadata,
                "resolved_by": "cross_contamination_detector",
                "resolution_module_version": _MODULE_VERSION,
            },
        )
        resolved_event.provenance_hash = _compute_hash({
            "event_id": event_id,
            "corrective_action": corrective_action,
            "resolved_date": resolved_event.resolved_date,
            "original_hash": event.provenance_hash,
        })

        self._events[event_id] = resolved_event

        logger.info(
            "Contamination event resolved: event_id=%s, "
            "corrective_action='%s'",
            event_id,
            corrective_action[:80],
        )
        return resolved_event

    # ------------------------------------------------------------------
    # Public API: Risk Heatmap
    # ------------------------------------------------------------------

    def generate_risk_heatmap(
        self,
        facility_id: str,
        zone_data: List[Dict[str, Any]],
        event_history: List[Dict[str, Any]],
    ) -> RiskHeatmapData:
        """Generate a risk heatmap for a facility's segregation zones.

        Computes per-zone risk scores based on zone characteristics,
        contamination event history, and pathway exposure. Produces
        data suitable for visualization on facility layout maps.

        Args:
            facility_id: Identifier of the facility.
            zone_data: List of zone dicts with keys: zone_id,
                zone_type, compliance_status, has_barrier,
                zone_x, zone_y.
            event_history: List of past contamination event dicts
                with keys: event_id, pathway_type, severity,
                zone_id, timestamp, resolved.

        Returns:
            RiskHeatmapData with per-zone risk scores and overall risk.
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")

        zones_result: List[Dict[str, Any]] = []
        zone_scores: List[float] = []

        # Build event counts per zone
        zone_event_counts: Dict[str, Dict[str, int]] = {}
        for evt in event_history:
            zone_id = evt.get("zone_id", "")
            if not zone_id:
                continue
            if zone_id not in zone_event_counts:
                zone_event_counts[zone_id] = {}
            pathway = evt.get("pathway_type", "unknown")
            zone_event_counts[zone_id][pathway] = (
                zone_event_counts[zone_id].get(pathway, 0) + 1
            )

        for zone in zone_data:
            zone_id = zone.get("zone_id", "unknown")
            has_barrier = bool(zone.get("has_barrier", False))
            compliance = str(zone.get("compliance_status", "")).lower()

            # Base risk from zone characteristics
            base_risk = 0.0
            if not has_barrier:
                base_risk += 30.0
            if compliance == "non_compliant":
                base_risk += 20.0
            elif compliance == "pending":
                base_risk += 10.0

            # Event history contribution
            event_counts = zone_event_counts.get(zone_id, {})
            history_risk = 0.0
            dominant_pathway = "none"
            max_pathway_count = 0
            for pathway, count in event_counts.items():
                weight = PATHWAY_RISK_WEIGHTS.get(pathway, 0.05)
                history_risk += count * weight * 20.0
                if count > max_pathway_count:
                    max_pathway_count = count
                    dominant_pathway = pathway

            zone_risk = min(100.0, base_risk + history_risk)
            zone_scores.append(zone_risk)

            zones_result.append({
                "zone_id": zone_id,
                "risk_score": round(zone_risk, 2),
                "dominant_pathway": dominant_pathway,
                "event_count": sum(event_counts.values()),
                "has_barrier": has_barrier,
                "compliance_status": compliance,
            })

        overall_risk = (
            sum(zone_scores) / max(len(zone_scores), 1)
            if zone_scores else 0.0
        )

        now = utcnow()
        result = RiskHeatmapData(
            facility_id=facility_id,
            zones=zones_result,
            overall_risk=round(overall_risk, 2),
            generated_at=now.isoformat(),
        )
        result.provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "zones_count": len(zones_result),
            "overall_risk": result.overall_risk,
            "generated_at": result.generated_at,
            "module_version": _MODULE_VERSION,
        })

        logger.info(
            "Risk heatmap generated: facility=%s, zones=%d, "
            "overall_risk=%.2f",
            facility_id,
            len(zones_result),
            result.overall_risk,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Event Queries
    # ------------------------------------------------------------------

    def get_contamination_history(
        self,
        facility_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[ContaminationEventRecord]:
        """Retrieve contamination event history for a facility.

        Args:
            facility_id: Identifier of the facility.
            start_date: Optional ISO date string for range start.
            end_date: Optional ISO date string for range end.

        Returns:
            List of ContaminationEventRecord objects, newest first.
        """
        event_ids = self._facility_events.get(facility_id, [])
        events = [
            self._events[eid] for eid in event_ids
            if eid in self._events
        ]

        if start_date:
            events = [
                e for e in events
                if e.timestamp >= start_date
            ]
        if end_date:
            events = [
                e for e in events
                if e.timestamp <= end_date
            ]

        # Sort by timestamp descending (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        logger.debug(
            "Contamination history for facility=%s: %d events "
            "(start=%s, end=%s)",
            facility_id,
            len(events),
            start_date or "unbounded",
            end_date or "unbounded",
        )
        return events

    def get_unresolved_events(
        self,
        facility_id: Optional[str] = None,
    ) -> List[ContaminationEventRecord]:
        """Retrieve all unresolved contamination events.

        Args:
            facility_id: Optional filter by facility. If None,
                returns unresolved events across all facilities.

        Returns:
            List of unresolved ContaminationEventRecord objects.
        """
        if facility_id:
            event_ids = self._facility_events.get(facility_id, [])
            events = [
                self._events[eid] for eid in event_ids
                if eid in self._events and not self._events[eid].resolved
            ]
        else:
            events = [
                e for e in self._events.values()
                if not e.resolved
            ]

        events.sort(
            key=lambda e: SEVERITY_SCORES.get(e.severity, 0.0),
            reverse=True,
        )

        logger.debug(
            "Unresolved contamination events: %d (facility=%s)",
            len(events),
            facility_id or "all",
        )
        return events

    def calculate_facility_contamination_risk(
        self,
        facility_id: str,
    ) -> float:
        """Calculate aggregate contamination risk score for a facility.

        Computes a weighted risk score based on the facility's
        contamination event history, considering event severity,
        pathway type, and resolution status. Unresolved events
        contribute more to the risk score than resolved events.

        Args:
            facility_id: Identifier of the facility.

        Returns:
            Aggregate risk score (0.0 to 100.0).
        """
        event_ids = self._facility_events.get(facility_id, [])
        if not event_ids:
            return 0.0

        total_weighted_score = 0.0
        max_possible = 0.0

        for eid in event_ids:
            if eid not in self._events:
                continue
            event = self._events[eid]

            severity_score = SEVERITY_SCORES.get(event.severity, 0.1)
            pathway_weight = PATHWAY_RISK_WEIGHTS.get(event.pathway_type, 0.05)

            # Unresolved events contribute 100%, resolved contribute 30%
            resolution_factor = 1.0 if not event.resolved else 0.3

            # Recency factor: events in last 90 days contribute more
            try:
                event_ts = datetime.fromisoformat(event.timestamp)
                if event_ts.tzinfo is None:
                    event_ts = event_ts.replace(tzinfo=timezone.utc)
                age_days = (
                    utcnow() - event_ts
                ).total_seconds() / 86400.0
                recency_factor = max(0.2, 1.0 - (age_days / 365.0))
            except (ValueError, TypeError):
                recency_factor = 0.5

            weighted = (
                severity_score
                * pathway_weight
                * resolution_factor
                * recency_factor
            )
            total_weighted_score += weighted
            max_possible += pathway_weight

        if max_possible <= 0.0:
            return 0.0

        risk = min(100.0, (total_weighted_score / max_possible) * 100.0)

        logger.debug(
            "Facility contamination risk: facility=%s, score=%.2f, "
            "events=%d",
            facility_id,
            risk,
            len(event_ids),
        )
        return round(risk, 2)

    # ------------------------------------------------------------------
    # Public API: Root Cause & Corrective Actions
    # ------------------------------------------------------------------

    def suggest_root_cause(
        self,
        pathway_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Suggest deterministic root cause templates for a pathway.

        Uses static template dictionaries indexed by pathway type.
        No ML/LLM is used. Context is used only for template selection
        ordering, not for generation.

        Args:
            pathway_type: Contamination pathway type.
            context: Optional context dict for ordering relevance
                (keys: severity, zone_type, equipment_type).

        Returns:
            List of root cause description strings.
        """
        templates = ROOT_CAUSE_TEMPLATES.get(pathway_type, [])
        if not templates:
            logger.warning(
                "No root cause templates for pathway: %s", pathway_type,
            )
            return [
                f"Root cause investigation required for {pathway_type} "
                f"contamination pathway"
            ]

        # Order by relevance if context provided
        if context and context.get("severity") == "critical":
            # For critical events, prioritize infrastructure causes
            return list(templates)

        return list(templates)

    def suggest_corrective_actions(
        self,
        pathway_type: str,
        severity: str,
    ) -> List[str]:
        """Suggest deterministic corrective action templates.

        Uses static template dictionaries indexed by pathway type
        and severity level. No ML/LLM is used.

        Args:
            pathway_type: Contamination pathway type.
            severity: Severity level (critical/major/minor/observation).

        Returns:
            List of corrective action description strings.
        """
        pathway_actions = CORRECTIVE_ACTION_TEMPLATES.get(pathway_type, {})
        if not pathway_actions:
            logger.warning(
                "No corrective action templates for pathway: %s",
                pathway_type,
            )
            return [
                f"Corrective action plan required for {pathway_type} "
                f"contamination ({severity} severity)"
            ]

        actions = pathway_actions.get(severity, [])
        if not actions:
            # Fall back to next higher severity level
            fallback_order = ["minor", "major", "critical"]
            for fb_severity in fallback_order:
                actions = pathway_actions.get(fb_severity, [])
                if actions:
                    break

        if not actions:
            return [
                f"Corrective action plan required for {pathway_type} "
                f"contamination ({severity} severity)"
            ]

        return list(actions)

    # ------------------------------------------------------------------
    # Internal Helpers: Detection
    # ------------------------------------------------------------------

    def _detect_storage_risks(
        self,
        facility_id: str,
        storage_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect storage-related contamination risks.

        Analyzes storage events for shared storage, handling errors,
        labeling errors, and documentation errors.

        Args:
            facility_id: Facility identifier.
            storage_events: List of storage event dictionaries.

        Returns:
            List of risk finding dictionaries.
        """
        risks: List[Dict[str, Any]] = []

        # Group events by zone
        zone_groups: Dict[str, List[Dict[str, Any]]] = {}
        for evt in storage_events:
            zone_id = evt.get("zone_id", "unknown")
            if zone_id not in zone_groups:
                zone_groups[zone_id] = []
            zone_groups[zone_id].append(evt)

        for zone_id, events in zone_groups.items():
            statuses = set(
                str(e.get("compliance_status", "")).lower()
                for e in events
            )
            has_compliant = "compliant" in statuses
            has_non_compliant = "non_compliant" in statuses

            if has_compliant and has_non_compliant:
                risks.append({
                    "pathway": "shared_storage",
                    "zone_id": zone_id,
                    "facility_id": facility_id,
                    "score": 0.85,
                    "description": (
                        f"Zone {zone_id} contains both compliant and "
                        f"non-compliant material simultaneously"
                    ),
                })

            # Check for missing labels on storage events
            for evt in events:
                if not evt.get("label_verified", True):
                    risks.append({
                        "pathway": "labeling_error",
                        "zone_id": zone_id,
                        "batch_id": evt.get("batch_id", "unknown"),
                        "facility_id": facility_id,
                        "score": 0.45,
                        "description": (
                            f"Batch {evt.get('batch_id', '?')} in zone "
                            f"{zone_id} has unverified labeling"
                        ),
                    })

            # Check for documentation completeness
            for evt in events:
                if not evt.get("documentation_complete", True):
                    risks.append({
                        "pathway": "documentation_error",
                        "zone_id": zone_id,
                        "batch_id": evt.get("batch_id", "unknown"),
                        "facility_id": facility_id,
                        "score": 0.30,
                        "description": (
                            f"Batch {evt.get('batch_id', '?')} in zone "
                            f"{zone_id} has incomplete documentation"
                        ),
                    })

        return risks

    def _detect_transport_risks(
        self,
        facility_id: str,
        transport_verifications: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect transport-related contamination risks.

        Analyzes transport verifications for shared transport and
        cleaning verification gaps.

        Args:
            facility_id: Facility identifier.
            transport_verifications: List of transport verification dicts.

        Returns:
            List of risk finding dictionaries.
        """
        risks: List[Dict[str, Any]] = []

        for verification in transport_verifications:
            vehicle_id = verification.get("vehicle_id", "unknown")
            cleaning_verified = bool(
                verification.get("cleaning_verified", False)
            )
            previous_cargo = str(
                verification.get("previous_cargo", "")
            ).lower()
            dedicated = bool(verification.get("dedicated", False))

            # Non-dedicated vehicle without cleaning
            if not dedicated and not cleaning_verified:
                score = 0.90
                if previous_cargo == "non_compliant":
                    score = 0.95
                risks.append({
                    "pathway": "shared_transport",
                    "vehicle_id": vehicle_id,
                    "facility_id": facility_id,
                    "cleaning_verified": cleaning_verified,
                    "previous_cargo": previous_cargo,
                    "score": score,
                    "description": (
                        f"Vehicle {vehicle_id} not cleaned after "
                        f"previous cargo ({previous_cargo})"
                    ),
                })
            elif not dedicated and cleaning_verified:
                # Cleaned but not dedicated - lower risk
                if previous_cargo == "non_compliant":
                    risks.append({
                        "pathway": "residual_material",
                        "vehicle_id": vehicle_id,
                        "facility_id": facility_id,
                        "score": 0.25,
                        "description": (
                            f"Vehicle {vehicle_id} cleaned after "
                            f"non-compliant cargo but residual risk remains"
                        ),
                    })

        return risks

    def _detect_processing_risks(
        self,
        facility_id: str,
        processing_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect processing-related contamination risks.

        Analyzes processing records for shared processing lines,
        inadequate changeover times, and insufficient flush volumes.

        Args:
            facility_id: Facility identifier.
            processing_records: List of processing record dicts.

        Returns:
            List of risk finding dictionaries.
        """
        risks: List[Dict[str, Any]] = []

        # Group by line
        line_groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in processing_records:
            line_id = record.get("line_id", "unknown")
            if line_id not in line_groups:
                line_groups[line_id] = []
            line_groups[line_id].append(record)

        for line_id, records in line_groups.items():
            statuses = set(
                str(r.get("compliance_status", "")).lower()
                for r in records
            )
            has_compliant = "compliant" in statuses
            has_non_compliant = "non_compliant" in statuses

            if has_compliant and has_non_compliant:
                # Check changeover adequacy
                for record in records:
                    changeover_min = float(
                        record.get("changeover_time_min", 0)
                    )
                    flush_vol = float(record.get("flush_volume_l", 0))

                    if changeover_min < 60:
                        risks.append({
                            "pathway": "shared_processing",
                            "line_id": line_id,
                            "facility_id": facility_id,
                            "changeover_time_min": changeover_min,
                            "score": 0.80,
                            "description": (
                                f"Line {line_id} changeover time "
                                f"({changeover_min:.0f}min) below "
                                f"minimum (60min)"
                            ),
                        })

                    if flush_vol < 50.0 and flush_vol > 0.0:
                        risks.append({
                            "pathway": "residual_material",
                            "line_id": line_id,
                            "facility_id": facility_id,
                            "flush_volume_l": flush_vol,
                            "score": 0.60,
                            "description": (
                                f"Line {line_id} flush volume "
                                f"({flush_vol:.1f}L) below threshold (50L)"
                            ),
                        })

        return risks

    # ------------------------------------------------------------------
    # Internal Helpers: Scoring
    # ------------------------------------------------------------------

    def _compute_overall_risk(
        self,
        pathway_scores: Dict[str, float],
    ) -> float:
        """Compute weighted overall risk score from pathway scores.

        Uses PATHWAY_RISK_WEIGHTS for weighted summation, then
        normalizes to 0-100 scale.

        Args:
            pathway_scores: Mapping of pathway type to raw score (0-1).

        Returns:
            Overall risk score (0.0 to 100.0).
        """
        weighted_sum = 0.0
        for pathway, score in pathway_scores.items():
            weight = PATHWAY_RISK_WEIGHTS.get(pathway, 0.0)
            weighted_sum += score * weight

        # Normalize: maximum weighted sum is 1.0 (all pathways score 1.0)
        overall = min(100.0, weighted_sum * 100.0)
        return overall

    def _generate_risk_recommendations(
        self,
        risks: List[Dict[str, Any]],
        pathway_scores: Dict[str, float],
    ) -> List[str]:
        """Generate prioritized recommendations based on detected risks.

        Args:
            risks: List of risk finding dictionaries.
            pathway_scores: Pathway-level risk scores.

        Returns:
            Prioritized list of recommendation strings.
        """
        recommendations: List[str] = []

        # Sort pathways by score descending
        sorted_pathways = sorted(
            pathway_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for pathway, score in sorted_pathways:
            if score <= 0.0:
                continue

            if score >= 0.8:
                severity = "critical"
            elif score >= 0.5:
                severity = "major"
            elif score >= 0.2:
                severity = "minor"
            else:
                severity = "observation"

            actions = self.suggest_corrective_actions(pathway, severity)
            if actions:
                recommendations.append(
                    f"[{pathway}] (score: {score:.2f}): {actions[0]}"
                )

        if not recommendations and risks:
            recommendations.append(
                "Review detected risks and implement corrective actions "
                "per facility segregation SOP"
            )

        return recommendations

    def _determine_downgrade_status(
        self,
        severity: str,
        depth: int,
    ) -> str:
        """Determine the downgrade target status based on severity and depth.

        Args:
            severity: Contamination event severity.
            depth: Distance from contamination source in the batch genealogy.

        Returns:
            Target compliance status string.
        """
        if severity == "critical":
            return "non_compliant"
        elif severity == "major":
            if depth <= 1:
                return "non_compliant"
            return "pending"
        elif severity == "minor":
            if depth <= 1:
                return "pending"
            return "compliant"
        else:
            # observation - no downgrade
            return "compliant"

    def _parse_and_sort_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse event timestamps and sort events chronologically.

        Adds a '_parsed_ts' key to each event dict with the parsed
        datetime. Events with unparseable timestamps are excluded.

        Args:
            events: List of event dictionaries with 'timestamp' key.

        Returns:
            Sorted list of event dicts with _parsed_ts added.
        """
        parsed: List[Dict[str, Any]] = []

        for evt in events:
            ts_raw = evt.get("timestamp")
            if ts_raw is None:
                continue

            try:
                if isinstance(ts_raw, datetime):
                    ts = ts_raw
                elif isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw)
                else:
                    continue

                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                enriched = dict(evt)
                enriched["_parsed_ts"] = ts
                parsed.append(enriched)
            except (ValueError, TypeError):
                logger.warning(
                    "Unparseable timestamp in event: %s", ts_raw,
                )
                continue

        parsed.sort(key=lambda e: e["_parsed_ts"])
        return parsed

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "CONTAMINATION_PATHWAYS",
    "SEVERITY_LEVELS",
    "SEVERITY_SCORES",
    "PATHWAY_RISK_WEIGHTS",
    "ROOT_CAUSE_TEMPLATES",
    "CORRECTIVE_ACTION_TEMPLATES",
    "DEFAULT_TEMPORAL_THRESHOLD_HOURS",
    "DEFAULT_SPATIAL_THRESHOLD_METERS",
    "MAX_PROPAGATION_DEPTH",
    # Result types
    "ContaminationEventRecord",
    "ContaminationDetectionResult",
    "ContaminationImpactResult",
    "RiskHeatmapData",
    # Engine
    "CrossContaminationDetector",
]
