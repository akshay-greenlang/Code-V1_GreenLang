# -*- coding: utf-8 -*-
"""
Natural-Language Explanation Generator for GL-005 COMBUSENSE

Implements operator-ready natural-language explanations as specified in
GL-005 Playbook Section 10.4.

Two approaches are supported:
    1. Template-first: Deterministic templating with controlled vocabulary
       for safety-critical contexts (primary approach)
    2. LLM-assisted: Uses LLM to produce narratives from constrained
       evidence bundles (optional, with safety guardrails)

Key Features:
    - Template-based narrative generation
    - Controlled vocabulary for safety-critical messaging
    - Evidence bundle processing
    - Multi-language support (extensible)
    - Audit trail with provenance tracking

Reference: GL-005 Playbook Section 10.4 (Natural-Language Explanations)

Author: GreenLang GL-005 Team
Version: 1.0.0
Performance Target: <10ms per narrative
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - NARRATIVE TEMPLATES
# =============================================================================

# CQI update templates
CQI_TEMPLATES = {
    "excellent": (
        "Combustion quality is excellent (CQI: $cqi). "
        "All parameters are within optimal operating bands. "
        "$additional_context"
    ),
    "good": (
        "Combustion quality is good (CQI: $cqi). "
        "$primary_driver. "
        "$additional_context"
    ),
    "acceptable": (
        "Combustion quality is acceptable (CQI: $cqi). "
        "Primary driver: $primary_driver. "
        "Recommended action: $recommendation. "
        "$additional_context"
    ),
    "poor": (
        "ATTENTION: Combustion quality is poor (CQI: $cqi). "
        "$primary_driver. "
        "Immediate action recommended: $recommendation. "
        "Safety note: $safety_note"
    ),
    "critical": (
        "WARNING: Combustion quality is critical (CQI: $cqi). "
        "Critical issue: $primary_driver. "
        "IMMEDIATE ACTION REQUIRED: $recommendation. "
        "SAFETY REMINDER: $safety_note"
    ),
}

# Incident templates by type
INCIDENT_TEMPLATES = {
    "CO_SPIKE": (
        "CO EXCURSION DETECTED at $timestamp. "
        "CO increased from $co_before ppm to $co_after ppm (+$co_delta ppm) "
        "over $duration. "
        "Likely cause: $root_cause. "
        "CQI impact: $cqi_delta points. "
        "Recommended checks: $checks. "
        "$safety_reminder"
    ),
    "NOX_SPIKE": (
        "NOx EXCURSION DETECTED at $timestamp. "
        "NOx increased from $nox_before ppm to $nox_after ppm (+$nox_delta ppm). "
        "Likely cause: $root_cause. "
        "This may indicate $interpretation. "
        "CQI impact: $cqi_delta points. "
        "Recommended checks: $checks. "
        "$safety_reminder"
    ),
    "COMBUSTION_RICH": (
        "RICH COMBUSTION CONDITION at $timestamp. "
        "O2 dropped to $o2_after% (target: $o2_target%). "
        "CO elevated at $co_after ppm. "
        "This indicates $interpretation. "
        "Potential cause: $root_cause. "
        "CQI impact: $cqi_delta points. "
        "Check: $checks. "
        "$safety_reminder"
    ),
    "COMBUSTION_LEAN": (
        "LEAN COMBUSTION CONDITION at $timestamp. "
        "O2 increased to $o2_after% (target: $o2_target%). "
        "Excess air estimated at $excess_air%. "
        "This indicates $interpretation. "
        "Efficiency impact: Approximately $efficiency_loss% loss. "
        "CQI impact: $cqi_delta points. "
        "Check: $checks."
    ),
    "FLAME_INSTABILITY": (
        "FLAME INSTABILITY DETECTED at $timestamp. "
        "Flame intensity variance increased to $flame_variance (baseline: $flame_baseline). "
        "$interpretation. "
        "Potential causes: $root_cause. "
        "CQI impact: $cqi_delta points. "
        "Immediate action: $checks. "
        "$safety_reminder"
    ),
    "SENSOR_DRIFT": (
        "SENSOR DRIFT DETECTED: $sensor_name at $timestamp. "
        "Reading deviated by $drift_amount from expected value. "
        "Drift rate: $drift_rate per hour. "
        "Recommendation: $checks. "
        "Note: CQI calculations may have reduced accuracy until corrected."
    ),
    "INTERLOCK_BYPASS": (
        "SAFETY INTERLOCK BYPASS ACTIVE: $interlock_name since $bypass_start. "
        "Duration: $bypass_duration. "
        "Bypass reason: $bypass_reason. "
        "CQI CAPPED at $cqi_cap due to active bypass. "
        "IMPORTANT: $safety_reminder"
    ),
    "ANALYZER_INVALID": (
        "ANALYZER STATUS: $analyzer_name is $status at $timestamp. "
        "Reason: $reason. "
        "$impact. "
        "Action: $checks."
    ),
    "DEFAULT": (
        "DIAGNOSTIC EVENT at $timestamp: $event_type. "
        "Details: $details. "
        "CQI impact: $cqi_delta points. "
        "Recommended action: $checks."
    ),
}

# Safety reminder templates
SAFETY_REMINDERS = {
    "default": (
        "SIS/BMS remains authoritative for all safety functions. "
        "GL-005 provides advisory diagnostics only."
    ),
    "bypass": (
        "Prolonged bypass may affect system safety. "
        "Ensure bypass is properly documented and time-limited per site procedures. "
        "SIS/BMS remains authoritative for all safety functions."
    ),
    "emissions": (
        "Elevated emissions may require notification per site environmental procedures."
    ),
    "flame": (
        "Flame stability issues may precede flameout. "
        "Monitor flame scanner readings closely. "
        "BMS will initiate safety shutdown if flame loss is confirmed."
    ),
    "rich_combustion": (
        "Rich combustion increases CO emissions and potential safety risks. "
        "BMS interlocks will activate if limits are exceeded."
    ),
}

# Signal change descriptions
SIGNAL_CHANGE_TEMPLATES = {
    "o2_increase": "O2 increased from $before% to $after% over $duration",
    "o2_decrease": "O2 decreased from $before% to $after% over $duration",
    "co_increase": "CO increased from $before ppm to $after ppm (+$delta ppm)",
    "co_decrease": "CO decreased from $before ppm to $after ppm",
    "nox_increase": "NOx increased from $before ppm to $after ppm (+$delta ppm)",
    "nox_decrease": "NOx decreased from $before ppm to $after ppm",
    "flame_increase": "Flame intensity increased from $before% to $after%",
    "flame_decrease": "Flame intensity decreased from $before% to $after%",
}

# Controlled vocabulary for safety-critical messaging
CONTROLLED_VOCABULARY = {
    "actions": [
        "Check", "Verify", "Inspect", "Review", "Monitor", "Confirm",
        "Investigate", "Examine", "Evaluate", "Assess"
    ],
    "severities": [
        "Informational", "Warning", "Attention Required",
        "Immediate Action Required", "Critical"
    ],
    "avoid_phrases": [
        "immediately do", "must do", "shut down", "turn off",
        "override", "bypass", "ignore"
    ],
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class NarrativeStyle(str, Enum):
    """Narrative style options"""
    OPERATOR = "operator"  # Concise, action-oriented
    ENGINEER = "engineer"  # Technical detail
    EXECUTIVE = "executive"  # High-level summary
    AUDIT = "audit"  # Full detail with timestamps


class NarrativeSeverity(str, Enum):
    """Narrative severity levels"""
    INFO = "info"
    WARNING = "warning"
    ATTENTION = "attention"
    IMMEDIATE = "immediate"
    CRITICAL = "critical"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class NarrativeResult:
    """Generated narrative result"""
    narrative_id: str
    timestamp: datetime
    narrative_text: str
    summary: str  # One-line summary
    severity: NarrativeSeverity
    style: NarrativeStyle

    # Components
    event_description: str
    signal_changes: str
    root_cause_statement: str
    recommendations: str
    safety_reminder: str

    # Metadata
    source_event_type: str
    source_incident_id: Optional[str]
    confidence: float
    template_used: str
    provenance_hash: str


@dataclass(frozen=True)
class EvidenceBundle:
    """
    Evidence bundle for narrative generation

    Contains the minimum required content per Playbook Section 10.4.
    """
    operating_mode: str
    load_context: str
    signal_deltas: Dict[str, Dict[str, float]]
    event_type: str
    severity: str
    confidence: float
    top_attributions: List[Dict[str, Any]]
    time_segments: List[Dict[str, Any]]
    safety_status: Dict[str, Any]
    recommended_checks: List[str]


# =============================================================================
# INPUT MODELS
# =============================================================================

class NarrativeInput(BaseModel):
    """Input for narrative generation"""
    event_type: str
    asset_id: str
    incident_id: Optional[str] = None

    # CQI context
    cqi_current: float = 0.0
    cqi_previous: float = 0.0
    cqi_grade: str = "acceptable"

    # Signal values
    o2_before: float = 3.0
    o2_after: float = 3.0
    co_before: float = 50.0
    co_after: float = 50.0
    nox_before: float = 30.0
    nox_after: float = 30.0
    flame_before: float = 80.0
    flame_after: float = 80.0

    # Targets
    o2_target: float = 3.0
    co_target: float = 50.0
    nox_target: float = 30.0

    # Context
    operating_mode: str = "RUN"
    load_percent: float = 75.0
    duration_seconds: float = 60.0

    # Attribution info
    primary_driver: str = ""
    root_cause: str = ""
    top_drivers: List[Dict[str, Any]] = Field(default_factory=list)

    # Safety
    bypass_active: bool = False
    bypass_name: Optional[str] = None
    bypass_duration_s: float = 0.0

    # Recommendations
    recommended_checks: List[str] = Field(default_factory=list)

    # Style
    style: NarrativeStyle = NarrativeStyle.OPERATOR


# =============================================================================
# NARRATIVE GENERATOR
# =============================================================================

class NarrativeGenerator:
    """
    Template-based natural-language narrative generator

    Generates operator-ready explanations using controlled vocabulary
    and safety-approved templates.
    """

    def __init__(
        self,
        default_style: NarrativeStyle = NarrativeStyle.OPERATOR,
        include_timestamps: bool = True
    ):
        """
        Initialize narrative generator

        Args:
            default_style: Default narrative style
            include_timestamps: Whether to include timestamps in narratives
        """
        self.default_style = default_style
        self.include_timestamps = include_timestamps

    def generate(self, input_data: NarrativeInput) -> NarrativeResult:
        """
        Generate narrative from input data

        Args:
            input_data: Narrative input with event and signal data

        Returns:
            NarrativeResult with formatted narrative text
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)

        # Determine severity
        severity = self._determine_severity(input_data)

        # Select template based on event type
        template_key, template = self._select_template(
            input_data.event_type, input_data.cqi_grade
        )

        # Build template variables
        variables = self._build_template_variables(input_data, timestamp)

        # Generate narrative text
        try:
            narrative_text = Template(template).safe_substitute(variables)
        except Exception as e:
            logger.error(f"Template substitution failed: {e}")
            narrative_text = self._generate_fallback_narrative(input_data, timestamp)

        # Clean up narrative (remove double spaces, etc.)
        narrative_text = self._clean_narrative(narrative_text)

        # Generate summary (first sentence or truncated)
        summary = self._generate_summary(narrative_text, input_data.event_type)

        # Generate component narratives
        event_description = self._generate_event_description(input_data)
        signal_changes = self._generate_signal_changes(input_data)
        root_cause_statement = self._generate_root_cause_statement(input_data)
        recommendations = self._generate_recommendations(input_data)
        safety_reminder = self._get_safety_reminder(input_data.event_type)

        # Calculate provenance hash
        calc_time = time.perf_counter() - start_time
        provenance_hash = self._calculate_hash(narrative_text, timestamp)

        return NarrativeResult(
            narrative_id=f"NAR-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            narrative_text=narrative_text,
            summary=summary,
            severity=severity,
            style=input_data.style,
            event_description=event_description,
            signal_changes=signal_changes,
            root_cause_statement=root_cause_statement,
            recommendations=recommendations,
            safety_reminder=safety_reminder,
            source_event_type=input_data.event_type,
            source_incident_id=input_data.incident_id,
            confidence=0.9,  # Template-based is high confidence
            template_used=template_key,
            provenance_hash=provenance_hash
        )

    def generate_from_evidence_bundle(
        self,
        bundle: EvidenceBundle,
        asset_id: str,
        incident_id: Optional[str] = None
    ) -> NarrativeResult:
        """
        Generate narrative from evidence bundle

        This is the primary interface for generating narratives from
        explainability outputs (Playbook Section 10.4).

        Args:
            bundle: Evidence bundle from explainability engine
            asset_id: Asset identifier
            incident_id: Optional incident identifier

        Returns:
            NarrativeResult
        """
        # Convert bundle to NarrativeInput
        signal_deltas = bundle.signal_deltas

        # Extract before/after values
        o2_delta = signal_deltas.get("o2_percent", {})
        co_delta = signal_deltas.get("co_ppm", {})
        nox_delta = signal_deltas.get("nox_ppm", {})
        flame_delta = signal_deltas.get("flame_intensity", {})

        # Get primary driver
        primary_driver = ""
        if bundle.top_attributions:
            top = bundle.top_attributions[0]
            primary_driver = f"{top.get('feature', 'Unknown')}: {top.get('interpretation', '')}"

        # Build input
        input_data = NarrativeInput(
            event_type=bundle.event_type,
            asset_id=asset_id,
            incident_id=incident_id,
            cqi_grade=self._severity_to_grade(bundle.severity),
            o2_before=o2_delta.get("before", 3.0),
            o2_after=o2_delta.get("after", 3.0),
            co_before=co_delta.get("before", 50.0),
            co_after=co_delta.get("after", 50.0),
            nox_before=nox_delta.get("before", 30.0),
            nox_after=nox_delta.get("after", 30.0),
            flame_before=flame_delta.get("before", 80.0),
            flame_after=flame_delta.get("after", 80.0),
            operating_mode=bundle.operating_mode,
            load_percent=float(bundle.load_context.replace("Load: ", "").replace("%", "")),
            primary_driver=primary_driver,
            top_drivers=bundle.top_attributions,
            recommended_checks=bundle.recommended_checks,
            bypass_active=bundle.safety_status.get("bypass_active", False),
        )

        return self.generate(input_data)

    def generate_cqi_narrative(
        self,
        cqi: float,
        grade: str,
        top_driver: Optional[str] = None,
        recommendation: Optional[str] = None
    ) -> str:
        """
        Generate simple CQI status narrative

        Args:
            cqi: Current CQI value
            grade: CQI grade (excellent, good, acceptable, poor, critical)
            top_driver: Primary driver description
            recommendation: Recommended action

        Returns:
            Formatted narrative string
        """
        template = CQI_TEMPLATES.get(grade.lower(), CQI_TEMPLATES["acceptable"])

        variables = {
            "cqi": f"{cqi:.1f}",
            "primary_driver": top_driver or "No significant deviations",
            "recommendation": recommendation or "Continue monitoring",
            "safety_note": SAFETY_REMINDERS["default"],
            "additional_context": "",
        }

        return Template(template).safe_substitute(variables)

    def _determine_severity(self, input_data: NarrativeInput) -> NarrativeSeverity:
        """Determine narrative severity from input"""
        cqi_delta = input_data.cqi_current - input_data.cqi_previous

        if input_data.cqi_grade == "critical" or cqi_delta < -30:
            return NarrativeSeverity.CRITICAL
        elif input_data.cqi_grade == "poor" or cqi_delta < -20:
            return NarrativeSeverity.IMMEDIATE
        elif input_data.cqi_grade == "acceptable" or cqi_delta < -10:
            return NarrativeSeverity.ATTENTION
        elif cqi_delta < -5:
            return NarrativeSeverity.WARNING
        else:
            return NarrativeSeverity.INFO

    def _select_template(
        self, event_type: str, cqi_grade: str
    ) -> Tuple[str, str]:
        """Select appropriate template for event type"""
        # First try incident templates
        if event_type in INCIDENT_TEMPLATES:
            return event_type, INCIDENT_TEMPLATES[event_type]

        # Map CQI events to CQI templates
        if event_type in ("cqi_update", "cqi_degradation"):
            grade = cqi_grade.lower()
            if grade in CQI_TEMPLATES:
                return f"cqi_{grade}", CQI_TEMPLATES[grade]

        # Default template
        return "default", INCIDENT_TEMPLATES["DEFAULT"]

    def _build_template_variables(
        self, input_data: NarrativeInput, timestamp: datetime
    ) -> Dict[str, str]:
        """Build template substitution variables"""
        cqi_delta = input_data.cqi_current - input_data.cqi_previous

        variables = {
            # Timestamp
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),

            # CQI
            "cqi": f"{input_data.cqi_current:.1f}",
            "cqi_before": f"{input_data.cqi_previous:.1f}",
            "cqi_delta": f"{cqi_delta:+.1f}",

            # O2
            "o2_before": f"{input_data.o2_before:.1f}",
            "o2_after": f"{input_data.o2_after:.1f}",
            "o2_target": f"{input_data.o2_target:.1f}",
            "o2_delta": f"{input_data.o2_after - input_data.o2_before:+.1f}",

            # CO
            "co_before": f"{input_data.co_before:.0f}",
            "co_after": f"{input_data.co_after:.0f}",
            "co_delta": f"{input_data.co_after - input_data.co_before:+.0f}",

            # NOx
            "nox_before": f"{input_data.nox_before:.0f}",
            "nox_after": f"{input_data.nox_after:.0f}",
            "nox_delta": f"{input_data.nox_after - input_data.nox_before:+.0f}",

            # Flame
            "flame_before": f"{input_data.flame_before:.0f}",
            "flame_after": f"{input_data.flame_after:.0f}",
            "flame_variance": f"{abs(input_data.flame_after - input_data.flame_before):.1f}",
            "flame_baseline": "5.0",

            # Context
            "duration": self._format_duration(input_data.duration_seconds),
            "operating_mode": input_data.operating_mode,
            "load_percent": f"{input_data.load_percent:.0f}",

            # Drivers and causes
            "primary_driver": input_data.primary_driver or "Multiple factors",
            "root_cause": input_data.root_cause or "Under investigation",

            # Derived values
            "excess_air": self._calculate_excess_air_str(input_data.o2_after),
            "efficiency_loss": self._estimate_efficiency_loss_str(input_data.o2_after),

            # Interpretation
            "interpretation": self._get_interpretation(input_data),

            # Recommendations
            "checks": "; ".join(input_data.recommended_checks[:3]) if input_data.recommended_checks else "Review recent changes",
            "recommendation": input_data.recommended_checks[0] if input_data.recommended_checks else "Monitor closely",

            # Safety
            "safety_reminder": self._get_safety_reminder(input_data.event_type),
            "safety_note": SAFETY_REMINDERS["default"],

            # Bypass info
            "bypass_duration": self._format_duration(input_data.bypass_duration_s),
            "bypass_reason": "Manual override" if input_data.bypass_active else "N/A",
            "cqi_cap": "30",

            # Additional
            "additional_context": "",
            "details": input_data.primary_driver or "See diagnostic details",
            "event_type": input_data.event_type,
            "interlock_name": input_data.bypass_name or "Unknown",
            "bypass_start": "See event log",
            "sensor_name": "Analyzer",
            "analyzer_name": "Combustion analyzer",
            "status": "Invalid",
            "reason": "Calibration in progress",
            "impact": "CQI data quality score reduced",
            "drift_amount": "N/A",
            "drift_rate": "N/A",
        }

        return variables

    def _generate_event_description(self, input_data: NarrativeInput) -> str:
        """Generate event description component"""
        event_map = {
            "CO_SPIKE": "Carbon monoxide excursion detected",
            "NOX_SPIKE": "Nitrogen oxide excursion detected",
            "COMBUSTION_RICH": "Rich combustion condition",
            "COMBUSTION_LEAN": "Lean combustion condition",
            "FLAME_INSTABILITY": "Flame stability degradation",
            "SENSOR_DRIFT": "Sensor drift detected",
            "INTERLOCK_BYPASS": "Safety interlock bypass active",
        }
        return event_map.get(input_data.event_type, f"Diagnostic event: {input_data.event_type}")

    def _generate_signal_changes(self, input_data: NarrativeInput) -> str:
        """Generate signal changes description"""
        changes = []

        o2_delta = input_data.o2_after - input_data.o2_before
        if abs(o2_delta) > 0.2:
            direction = "increased" if o2_delta > 0 else "decreased"
            changes.append(f"O2 {direction} from {input_data.o2_before:.1f}% to {input_data.o2_after:.1f}%")

        co_delta = input_data.co_after - input_data.co_before
        if abs(co_delta) > 10:
            direction = "increased" if co_delta > 0 else "decreased"
            changes.append(f"CO {direction} from {input_data.co_before:.0f} to {input_data.co_after:.0f} ppm")

        nox_delta = input_data.nox_after - input_data.nox_before
        if abs(nox_delta) > 5:
            direction = "increased" if nox_delta > 0 else "decreased"
            changes.append(f"NOx {direction} from {input_data.nox_before:.0f} to {input_data.nox_after:.0f} ppm")

        return "; ".join(changes) if changes else "No significant signal changes"

    def _generate_root_cause_statement(self, input_data: NarrativeInput) -> str:
        """Generate root cause statement"""
        if input_data.root_cause:
            return f"Likely cause: {input_data.root_cause}"

        if input_data.primary_driver:
            return f"Primary driver: {input_data.primary_driver}"

        return "Root cause under investigation"

    def _generate_recommendations(self, input_data: NarrativeInput) -> str:
        """Generate recommendations text"""
        if not input_data.recommended_checks:
            return "Continue monitoring; review if condition persists"

        # Use controlled vocabulary
        recommendations = []
        for check in input_data.recommended_checks[:5]:
            # Ensure action words are from controlled vocabulary
            clean_check = self._apply_controlled_vocabulary(check)
            recommendations.append(clean_check)

        return "; ".join(recommendations)

    def _apply_controlled_vocabulary(self, text: str) -> str:
        """Ensure text uses controlled vocabulary"""
        # Replace unsafe phrases
        for phrase in CONTROLLED_VOCABULARY["avoid_phrases"]:
            if phrase.lower() in text.lower():
                text = re.sub(
                    re.escape(phrase),
                    "Verify",
                    text,
                    flags=re.IGNORECASE
                )
        return text

    def _get_safety_reminder(self, event_type: str) -> str:
        """Get appropriate safety reminder for event type"""
        reminder_map = {
            "INTERLOCK_BYPASS": "bypass",
            "CO_SPIKE": "rich_combustion",
            "COMBUSTION_RICH": "rich_combustion",
            "FLAME_INSTABILITY": "flame",
            "NOX_SPIKE": "emissions",
        }
        key = reminder_map.get(event_type, "default")
        return SAFETY_REMINDERS.get(key, SAFETY_REMINDERS["default"])

    def _get_interpretation(self, input_data: NarrativeInput) -> str:
        """Get engineering interpretation of the event"""
        interpretations = {
            "CO_SPIKE": "incomplete combustion, possibly due to air shortage or fuel quality issue",
            "NOX_SPIKE": "high flame temperature or excess air imbalance causing thermal NOx formation",
            "COMBUSTION_RICH": "fuel-rich operation with insufficient combustion air",
            "COMBUSTION_LEAN": "excess air operation reducing thermal efficiency",
            "FLAME_INSTABILITY": "unstable flame pattern that may indicate fuel or air flow issues",
        }
        return interpretations.get(input_data.event_type, "abnormal combustion condition")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form"""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def _calculate_excess_air_str(self, o2: float) -> str:
        """Calculate excess air percentage from O2"""
        if o2 >= 21:
            return "100+"
        excess_air = (o2 / (21 - o2)) * 100
        return f"{excess_air:.0f}"

    def _estimate_efficiency_loss_str(self, o2: float) -> str:
        """Estimate efficiency loss from excess air"""
        # Rough estimate: ~0.5% efficiency loss per 1% excess air above optimal
        if o2 <= 3:
            return "0"
        excess = o2 - 3.0
        loss = excess * 0.5
        return f"{loss:.1f}"

    def _severity_to_grade(self, severity: str) -> str:
        """Convert severity code to CQI grade"""
        severity_map = {
            "S1": "excellent",
            "S2": "good",
            "S3": "poor",
            "S4": "critical",
        }
        return severity_map.get(severity, "acceptable")

    def _generate_summary(self, narrative: str, event_type: str) -> str:
        """Generate one-line summary"""
        # Take first sentence or truncate
        if "." in narrative:
            summary = narrative.split(".")[0] + "."
        else:
            summary = narrative[:100] + "..." if len(narrative) > 100 else narrative

        return summary

    def _generate_fallback_narrative(
        self, input_data: NarrativeInput, timestamp: datetime
    ) -> str:
        """Generate fallback narrative if template fails"""
        cqi_delta = input_data.cqi_current - input_data.cqi_previous
        return (
            f"Diagnostic event: {input_data.event_type} at {timestamp.isoformat()}. "
            f"CQI changed by {cqi_delta:+.1f} points to {input_data.cqi_current:.1f}. "
            f"Review diagnostic details for more information."
        )

    def _clean_narrative(self, text: str) -> str:
        """Clean up narrative text"""
        # Remove double spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        # Ensure single space after punctuation
        text = re.sub(r'([.,;:])([A-Z])', r'\1 \2', text)
        return text.strip()

    def _calculate_hash(self, narrative: str, timestamp: datetime) -> str:
        """Calculate provenance hash"""
        data = {
            "narrative_checksum": hash(narrative),
            "timestamp": timestamp.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]


# =============================================================================
# LLM-ASSISTED GENERATOR (OPTIONAL)
# =============================================================================

class LLMAssistedGenerator:
    """
    LLM-assisted narrative generator (optional)

    Uses LLM to produce narratives from constrained evidence bundles
    with safety guardrails per Playbook Section 10.4.

    Safety Rules:
    - No direct actuation instructions
    - Must highlight uncertainty
    - Must include safety reminders
    - Constrained to evidence bundle content
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize LLM-assisted generator

        Args:
            llm_client: Optional LLM client (Claude, GPT-4, etc.)
        """
        self.llm_client = llm_client
        self.template_generator = NarrativeGenerator()

        # Safety guardrail prompt
        self.system_prompt = """You are a combustion diagnostics assistant generating operator-ready explanations.

STRICT RULES:
1. NEVER instruct direct actuation (don't say "turn off", "shut down", "override")
2. Use action words: Check, Verify, Inspect, Review, Monitor, Investigate
3. ALWAYS include safety reminder about SIS/BMS being authoritative
4. Highlight uncertainty with phrases like "may indicate", "suggests", "possible"
5. ONLY use information from the provided evidence bundle
6. Keep explanations concise and operator-focused

Generate a clear, professional diagnostic narrative from the evidence bundle."""

    def generate(self, bundle: EvidenceBundle, asset_id: str) -> NarrativeResult:
        """
        Generate LLM-assisted narrative

        Falls back to template-based if LLM is unavailable.

        Args:
            bundle: Evidence bundle
            asset_id: Asset identifier

        Returns:
            NarrativeResult
        """
        if not self.llm_client:
            logger.info("LLM client not available, using template-based generation")
            return self.template_generator.generate_from_evidence_bundle(
                bundle, asset_id
            )

        try:
            # Build LLM prompt from evidence bundle
            prompt = self._build_llm_prompt(bundle)

            # Call LLM (implementation depends on client)
            llm_response = self._call_llm(prompt)

            # Validate response against safety rules
            validated_response = self._validate_response(llm_response)

            # Create result
            timestamp = datetime.now(timezone.utc)
            return NarrativeResult(
                narrative_id=f"NAR-LLM-{timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp=timestamp,
                narrative_text=validated_response,
                summary=validated_response.split(".")[0] + ".",
                severity=NarrativeSeverity.INFO,
                style=NarrativeStyle.OPERATOR,
                event_description=bundle.event_type,
                signal_changes="",
                root_cause_statement="",
                recommendations="; ".join(bundle.recommended_checks[:3]),
                safety_reminder=SAFETY_REMINDERS["default"],
                source_event_type=bundle.event_type,
                source_incident_id=None,
                confidence=0.75,  # LLM is lower confidence than templates
                template_used="llm_assisted",
                provenance_hash=hashlib.sha256(validated_response.encode()).hexdigest()[:16]
            )

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to templates")
            return self.template_generator.generate_from_evidence_bundle(
                bundle, asset_id
            )

    def _build_llm_prompt(self, bundle: EvidenceBundle) -> str:
        """Build prompt for LLM from evidence bundle"""
        return f"""Evidence Bundle:
- Operating Mode: {bundle.operating_mode}
- Load: {bundle.load_context}
- Event Type: {bundle.event_type}
- Severity: {bundle.severity}
- Confidence: {bundle.confidence:.0%}

Signal Changes:
{json.dumps(bundle.signal_deltas, indent=2)}

Top Contributing Factors:
{json.dumps(bundle.top_attributions, indent=2)}

Recommended Checks:
{json.dumps(bundle.recommended_checks, indent=2)}

Safety Status:
{json.dumps(bundle.safety_status, indent=2)}

Generate a concise operator-ready explanation (2-4 sentences)."""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API (placeholder for actual implementation)"""
        # This would be implemented with actual LLM client
        # For now, return a placeholder
        return "LLM response placeholder"

    def _validate_response(self, response: str) -> str:
        """Validate LLM response against safety rules"""
        # Check for forbidden phrases
        for phrase in CONTROLLED_VOCABULARY["avoid_phrases"]:
            if phrase.lower() in response.lower():
                response = re.sub(
                    re.escape(phrase),
                    "Verify",
                    response,
                    flags=re.IGNORECASE
                )

        # Ensure safety reminder is included
        if "SIS" not in response and "BMS" not in response:
            response += f" {SAFETY_REMINDERS['default']}"

        return response


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def create_default_generator() -> NarrativeGenerator:
    """Create narrative generator with default settings"""
    return NarrativeGenerator()


def generate_cqi_narrative_quick(
    cqi: float,
    grade: str = "good",
    driver: str = "",
    recommendation: str = ""
) -> str:
    """Quick narrative generation for testing"""
    generator = NarrativeGenerator()
    return generator.generate_cqi_narrative(cqi, grade, driver, recommendation)


def generate_incident_narrative_quick(
    event_type: str,
    o2: float,
    co: float,
    nox: float,
    cqi_delta: float = -10.0
) -> NarrativeResult:
    """Quick incident narrative for testing"""
    generator = NarrativeGenerator()
    input_data = NarrativeInput(
        event_type=event_type,
        asset_id="test",
        cqi_current=75.0 + cqi_delta,
        cqi_previous=75.0,
        o2_after=o2,
        co_after=co,
        nox_after=nox,
        primary_driver="O2 deviation",
    )
    return generator.generate(input_data)
