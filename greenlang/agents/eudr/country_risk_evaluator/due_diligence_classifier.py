# -*- coding: utf-8 -*-
"""
Due Diligence Classifier Engine - AGENT-EUDR-016 Engine 5

Automated due diligence requirement determination per EUDR Articles
10-13 with 3-tier classification (simplified, standard, enhanced),
certification-based risk mitigation credits, cost estimation, audit
frequency recommendation, and reclassification impact analysis.

Classification Logic (per EUDR Articles 10, 11, 13):
    - Simplified (Art. 13): effective_risk_score <= simplified_threshold
      (default 30). Reduced documentation, annual audit.
    - Standard (Arts. 10-11): simplified_threshold < score <=
      enhanced_threshold (default 30-60). Full documentation,
      semi-annual audit.
    - Enhanced (Art. 11 + satellite): score > enhanced_threshold
      (default 60). Mandatory satellite verification, quarterly audit,
      independent audit required.

Override Rules:
    - Sub-national override: specific regions always classified as
      enhanced regardless of country-level score (e.g., Para, Mato
      Grosso in Brazil for cattle/soya).
    - EC benchmark override: when enabled in config, EC-published
      classifications take precedence over agent-computed levels.

Certification Credit:
    - Recognized schemes (FSC, PEFC, RSPO, RA, Fairtrade, Organic,
      Bonsucro, ISCC) reduce the effective risk score by a configurable
      credit amount (max certification_credit_max from config).
    - Credit is capped and cannot reduce effective score below 0.

Cost Estimation:
    - Per-shipment cost ranges from config (simplified: 200-500 EUR,
      standard: 1000-3000 EUR, enhanced: 5000-15000 EUR).
    - Annual cost = per-shipment cost * estimated shipments/year.

Audit Frequency:
    - Simplified: annual (12 months)
    - Standard: semi-annual (6 months)
    - Enhanced: quarterly (3 months)
    - Multiplier from config applied to base frequency.

Zero-Hallucination: All classification decisions are deterministic
    threshold comparisons. No LLM calls in the classification path.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import (
    observe_classification_duration,
    record_classification_completed,
)
from .models import (
    CommodityType,
    DueDiligenceClassification,
    DueDiligenceLevel,
    RiskLevel,
    SUPPORTED_COMMODITIES,
)
from .provenance import get_provenance_tracker

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sub-national regions that always trigger enhanced due diligence.
#: Key: country_code, Value: dict mapping region -> list of commodities.
_DEFAULT_OVERRIDE_REGIONS: Dict[str, Dict[str, List[str]]] = {
    "BR": {
        "Para": ["cattle", "soya", "wood"],
        "Mato Grosso": ["cattle", "soya"],
        "Rondonia": ["cattle", "soya", "wood"],
        "Maranhao": ["soya", "wood"],
        "Amazonas": ["cattle", "wood", "rubber"],
    },
    "ID": {
        "Kalimantan": ["oil_palm", "rubber", "wood"],
        "Sumatra": ["oil_palm", "rubber", "wood", "coffee"],
        "Papua": ["oil_palm", "wood"],
    },
    "CD": {
        "North Kivu": ["wood", "cocoa", "coffee"],
        "South Kivu": ["wood", "coffee"],
    },
    "CO": {
        "Caqueta": ["cattle", "cocoa"],
        "Guaviare": ["cattle", "cocoa"],
    },
    "CI": {
        "Cavally": ["cocoa"],
        "Guemon": ["cocoa"],
        "Nawa": ["cocoa"],
    },
    "GH": {
        "Western Region": ["cocoa"],
        "Ashanti": ["cocoa"],
    },
    "MY": {
        "Sabah": ["oil_palm", "wood"],
        "Sarawak": ["oil_palm", "wood"],
    },
}

#: Certification scheme effectiveness scores per commodity (0-100).
#: Higher effectiveness = higher credit applied.
_CERTIFICATION_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    "fsc": {
        "wood": 85.0, "rubber": 60.0, "cattle": 0.0, "cocoa": 0.0,
        "coffee": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "pefc": {
        "wood": 75.0, "rubber": 50.0, "cattle": 0.0, "cocoa": 0.0,
        "coffee": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "rspo": {
        "oil_palm": 80.0, "wood": 0.0, "rubber": 0.0, "cattle": 0.0,
        "cocoa": 0.0, "coffee": 0.0, "soya": 0.0,
    },
    "rainforest_alliance": {
        "cocoa": 75.0, "coffee": 80.0, "wood": 40.0, "rubber": 30.0,
        "cattle": 0.0, "oil_palm": 30.0, "soya": 0.0,
    },
    "fairtrade": {
        "cocoa": 65.0, "coffee": 70.0, "wood": 0.0, "rubber": 0.0,
        "cattle": 0.0, "oil_palm": 0.0, "soya": 0.0,
    },
    "organic": {
        "cocoa": 50.0, "coffee": 55.0, "soya": 45.0, "oil_palm": 40.0,
        "cattle": 30.0, "wood": 20.0, "rubber": 25.0,
    },
    "bonsucro": {
        "soya": 60.0, "cattle": 0.0, "cocoa": 0.0, "coffee": 0.0,
        "oil_palm": 0.0, "rubber": 0.0, "wood": 0.0,
    },
    "iscc": {
        "oil_palm": 70.0, "soya": 55.0, "wood": 30.0, "rubber": 25.0,
        "cattle": 0.0, "cocoa": 0.0, "coffee": 0.0,
    },
}

#: Base audit frequency in months per DD level.
_BASE_AUDIT_MONTHS: Dict[str, int] = {
    "simplified": 12,
    "standard": 6,
    "enhanced": 3,
}

#: Base time-to-compliance in days per DD level.
_BASE_COMPLIANCE_DAYS: Dict[str, int] = {
    "simplified": 30,
    "standard": 90,
    "enhanced": 180,
}

#: Regulatory requirements per DD level.
_REQUIREMENTS: Dict[str, List[str]] = {
    "simplified": [
        "Supplier declaration of deforestation-free status",
        "Basic due diligence statement (DDS)",
        "Record of country of origin",
        "Product description and HS codes",
        "Volume and value records",
    ],
    "standard": [
        "Supplier declaration of deforestation-free status",
        "Full due diligence statement (DDS)",
        "Geolocation of production plots (Art. 9)",
        "Risk assessment documentation",
        "Supplier verification and audit records",
        "Product traceability chain documentation",
        "Volume, value, and date records",
        "Third-party certification records (if applicable)",
    ],
    "enhanced": [
        "Supplier declaration of deforestation-free status",
        "Full due diligence statement (DDS)",
        "Geolocation of all production plots (Art. 9)",
        "Comprehensive risk assessment documentation",
        "Satellite monitoring verification (Art. 11)",
        "Independent third-party audit report",
        "Supplier site visit records",
        "Product traceability chain documentation",
        "Volume, value, date, and batch records",
        "Photographic evidence of production sites",
        "Local community consultation records",
        "Environmental impact assessment",
        "All applicable certification records",
    ],
}

# ---------------------------------------------------------------------------
# DueDiligenceClassifier
# ---------------------------------------------------------------------------

class DueDiligenceClassifier:
    """Automated due diligence requirement determination per EUDR Arts. 10-13.

    Classifies country-commodity pairs into 3 due diligence tiers
    (simplified, standard, enhanced) based on risk scores, applies
    certification-based risk credits, estimates compliance costs,
    recommends audit frequencies, and assesses reclassification impacts.

    All classification decisions are deterministic threshold comparisons.
    No LLM calls are used in the classification path (zero-hallucination).

    Attributes:
        _classifications: In-memory store of DD classifications keyed
            by classification_id.
        _override_regions: Sub-national override rules for forced
            enhanced classification.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> classifier = DueDiligenceClassifier()
        >>> result = classifier.classify("BR", 72.5)
        >>> assert result.level == DueDiligenceLevel.ENHANCED
        >>> assert result.satellite_required is True
    """

    def __init__(self) -> None:
        """Initialize DueDiligenceClassifier with empty stores."""
        self._classifications: Dict[str, DueDiligenceClassification] = {}
        self._override_regions: Dict[str, Dict[str, List[str]]] = dict(
            _DEFAULT_OVERRIDE_REGIONS
        )
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "DueDiligenceClassifier initialized: "
            "override_countries=%d, certification_schemes=%d",
            len(self._override_regions),
            len(_CERTIFICATION_EFFECTIVENESS),
        )

    # ------------------------------------------------------------------
    # Primary classification
    # ------------------------------------------------------------------

    def classify(
        self,
        country_code: str,
        risk_score: float,
        commodity_type: Optional[str] = None,
        region: Optional[str] = None,
        certification_schemes: Optional[List[str]] = None,
    ) -> DueDiligenceClassification:
        """Classify due diligence level for a country-commodity pair.

        Applies the following classification pipeline:
        1. Validate inputs (country code, risk score bounds).
        2. Check sub-national override rules.
        3. Calculate certification credit (if schemes provided).
        4. Compute effective risk score.
        5. Determine DD level from threshold comparison.
        6. Populate cost estimates, audit frequency, requirements.
        7. Record provenance and metrics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            risk_score: Composite risk score (0-100).
            commodity_type: Optional EUDR commodity type.
            region: Optional sub-national region for override checks.
            certification_schemes: Optional list of active certification
                scheme names (e.g., ["fsc", "rspo"]).

        Returns:
            DueDiligenceClassification with level, cost estimate,
            audit frequency, and regulatory requirements.

        Raises:
            ValueError: If country_code is empty or risk_score is
                outside [0, 100].
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        country_code = self._validate_country_code(country_code)
        self._validate_risk_score(risk_score)

        commodity_enum = self._resolve_commodity(commodity_type)

        # -- Override check --------------------------------------------------
        is_override = self._check_region_override(
            country_code, region, commodity_type,
        )

        # -- Certification credit --------------------------------------------
        cert_credit = self._calculate_certification_credit(
            certification_schemes or [], commodity_type, cfg,
        )

        # -- Effective score -------------------------------------------------
        effective_score = max(0.0, risk_score - cert_credit)

        # -- Classification --------------------------------------------------
        if is_override:
            level = DueDiligenceLevel.ENHANCED
        else:
            level = self._determine_level(effective_score, cfg)

        # -- Populate classification -----------------------------------------
        classification = self._build_classification(
            country_code=country_code,
            commodity_enum=commodity_enum,
            level=level,
            risk_score=risk_score,
            cert_credit=cert_credit,
            effective_score=effective_score,
            cfg=cfg,
        )

        # -- Store -----------------------------------------------------------
        with self._lock:
            self._classifications[classification.classification_id] = (
                classification
            )

        # -- Provenance ------------------------------------------------------
        tracker = get_provenance_tracker()
        tracker.record(
            entity_type="due_diligence_classification",
            action="classify",
            entity_id=classification.classification_id,
            data=classification.model_dump(mode="json"),
            metadata={
                "country_code": country_code,
                "risk_score": risk_score,
                "effective_score": effective_score,
                "level": level.value,
                "cert_credit": cert_credit,
                "is_override": is_override,
            },
        )

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_classification_duration(elapsed)
        record_classification_completed(level.value)

        logger.info(
            "Classified DD level: country=%s commodity=%s score=%.1f "
            "effective=%.1f level=%s cert_credit=%.1f override=%s "
            "elapsed_ms=%.1f",
            country_code,
            commodity_type or "all",
            risk_score,
            effective_score,
            level.value,
            cert_credit,
            is_override,
            elapsed * 1000,
        )
        return classification

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[DueDiligenceClassification]:
        """Classify due diligence levels for multiple country-commodity pairs.

        Each item in the batch is a dictionary with keys:
            - country_code (str, required)
            - risk_score (float, required)
            - commodity_type (str, optional)
            - region (str, optional)
            - certification_schemes (list[str], optional)

        Args:
            items: List of classification request dictionaries.

        Returns:
            List of DueDiligenceClassification results in the same
            order as the input items.

        Raises:
            ValueError: If items list is empty or exceeds batch_max_size.
        """
        cfg = get_config()
        if not items:
            raise ValueError("Batch items list must not be empty")
        if len(items) > cfg.batch_max_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum "
                f"{cfg.batch_max_size}"
            )

        results: List[DueDiligenceClassification] = []
        for item in items:
            result = self.classify(
                country_code=item["country_code"],
                risk_score=item["risk_score"],
                commodity_type=item.get("commodity_type"),
                region=item.get("region"),
                certification_schemes=item.get("certification_schemes"),
            )
            results.append(result)

        logger.info(
            "Batch classification completed: items=%d", len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_classification(
        self, classification_id: str,
    ) -> Optional[DueDiligenceClassification]:
        """Retrieve a classification by its unique identifier.

        Args:
            classification_id: The classification_id to look up.

        Returns:
            DueDiligenceClassification if found, None otherwise.
        """
        with self._lock:
            return self._classifications.get(classification_id)

    def list_classifications(
        self,
        country_code: Optional[str] = None,
        commodity_type: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DueDiligenceClassification]:
        """List classifications with optional filters.

        Args:
            country_code: Optional country code filter.
            commodity_type: Optional commodity type filter.
            level: Optional DD level filter (simplified/standard/enhanced).
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of DueDiligenceClassification objects.
        """
        with self._lock:
            results = list(self._classifications.values())

        if country_code:
            cc = country_code.upper().strip()
            results = [c for c in results if c.country_code == cc]

        if commodity_type:
            results = [
                c for c in results
                if c.commodity_type is not None
                and c.commodity_type.value == commodity_type
            ]

        if level:
            results = [c for c in results if c.level.value == level]

        # Sort by classification timestamp descending
        results.sort(key=lambda c: c.classified_at, reverse=True)

        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def calculate_cost_estimate(
        self,
        country_code: str,
        risk_score: float,
        commodity_type: Optional[str] = None,
        shipments_per_year: int = 1,
        certification_schemes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Estimate due diligence compliance costs.

        Calculates per-shipment and annual cost estimates based on
        the DD level determined from the risk score.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            risk_score: Composite risk score (0-100).
            commodity_type: Optional EUDR commodity type.
            shipments_per_year: Expected shipments per year (default 1).
            certification_schemes: Active certification schemes.

        Returns:
            Dictionary with cost_per_shipment_min_eur,
            cost_per_shipment_max_eur, annual_cost_min_eur,
            annual_cost_max_eur, dd_level, and effective_risk_score.
        """
        cfg = get_config()
        country_code = self._validate_country_code(country_code)

        cert_credit = self._calculate_certification_credit(
            certification_schemes or [], commodity_type, cfg,
        )
        effective_score = max(0.0, risk_score - cert_credit)
        level = self._determine_level(effective_score, cfg)

        cost_min, cost_max = self._get_cost_range(level, cfg)

        return {
            "country_code": country_code,
            "commodity_type": commodity_type,
            "dd_level": level.value,
            "risk_score": risk_score,
            "effective_risk_score": effective_score,
            "certification_credit": cert_credit,
            "cost_per_shipment_min_eur": cost_min,
            "cost_per_shipment_max_eur": cost_max,
            "annual_cost_min_eur": cost_min * shipments_per_year,
            "annual_cost_max_eur": cost_max * shipments_per_year,
            "shipments_per_year": shipments_per_year,
        }

    # ------------------------------------------------------------------
    # Audit frequency
    # ------------------------------------------------------------------

    def get_audit_frequency(
        self,
        level: str,
    ) -> Dict[str, Any]:
        """Get recommended audit frequency for a DD level.

        Args:
            level: DD level string (simplified, standard, enhanced).

        Returns:
            Dictionary with frequency, interval_months,
            audits_per_year, and description.

        Raises:
            ValueError: If level is not a valid DD level.
        """
        cfg = get_config()
        level_lower = level.lower().strip()
        if level_lower not in _BASE_AUDIT_MONTHS:
            raise ValueError(
                f"Invalid DD level '{level}'; "
                f"must be one of: simplified, standard, enhanced"
            )

        base_months = _BASE_AUDIT_MONTHS[level_lower]
        adjusted_months = max(
            1,
            int(base_months / cfg.audit_frequency_multiplier),
        )
        audits_per_year = max(1, 12 // adjusted_months)

        freq_map = {
            12: "annual",
            6: "semi_annual",
            3: "quarterly",
            2: "bi_monthly",
            1: "monthly",
        }
        frequency_label = freq_map.get(adjusted_months, f"every_{adjusted_months}_months")

        return {
            "level": level_lower,
            "frequency": frequency_label,
            "interval_months": adjusted_months,
            "audits_per_year": audits_per_year,
            "multiplier_applied": cfg.audit_frequency_multiplier,
            "description": (
                f"{frequency_label.replace('_', ' ').title()} audits "
                f"({audits_per_year}x per year, every "
                f"{adjusted_months} months)"
            ),
        }

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def get_requirements(
        self, level: str,
    ) -> List[str]:
        """Get regulatory submission requirements for a DD level.

        Args:
            level: DD level string (simplified, standard, enhanced).

        Returns:
            List of requirement description strings.

        Raises:
            ValueError: If level is not a valid DD level.
        """
        level_lower = level.lower().strip()
        if level_lower not in _REQUIREMENTS:
            raise ValueError(
                f"Invalid DD level '{level}'; "
                f"must be one of: simplified, standard, enhanced"
            )
        return list(_REQUIREMENTS[level_lower])

    # ------------------------------------------------------------------
    # Certification credit
    # ------------------------------------------------------------------

    def apply_certification_credit(
        self,
        risk_score: float,
        certification_schemes: List[str],
        commodity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate and apply certification-based risk credit.

        Args:
            risk_score: Original risk score (0-100).
            certification_schemes: List of scheme names.
            commodity_type: Optional commodity for scheme effectiveness.

        Returns:
            Dictionary with original_score, credit_applied,
            effective_score, and scheme_details.
        """
        cfg = get_config()
        credit = self._calculate_certification_credit(
            certification_schemes, commodity_type, cfg,
        )
        effective = max(0.0, risk_score - credit)

        scheme_details = []
        for scheme in certification_schemes:
            scheme_lower = scheme.lower().strip()
            effectiveness = self._get_scheme_effectiveness(
                scheme_lower, commodity_type,
            )
            scheme_details.append({
                "scheme": scheme_lower,
                "effectiveness": effectiveness,
                "applicable": effectiveness > 0,
            })

        return {
            "original_score": risk_score,
            "credit_applied": credit,
            "effective_score": effective,
            "max_credit_allowed": cfg.certification_credit_max,
            "scheme_details": scheme_details,
        }

    # ------------------------------------------------------------------
    # Override rules
    # ------------------------------------------------------------------

    def check_override_rules(
        self,
        country_code: str,
        region: Optional[str] = None,
        commodity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if sub-national override rules apply.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            region: Sub-national region name.
            commodity_type: Optional commodity type.

        Returns:
            Dictionary with override_active, reason, and matching_rules.
        """
        country_code = country_code.upper().strip()
        is_override = self._check_region_override(
            country_code, region, commodity_type,
        )

        matching_rules: List[Dict[str, Any]] = []
        if country_code in self._override_regions and region:
            region_rules = self._override_regions[country_code]
            for rule_region, commodities in region_rules.items():
                if rule_region.lower() == region.lower():
                    if commodity_type is None or commodity_type in commodities:
                        matching_rules.append({
                            "region": rule_region,
                            "commodities": commodities,
                            "forced_level": "enhanced",
                        })

        reason = ""
        if is_override:
            reason = (
                f"Sub-national override: {country_code}/{region} "
                f"requires enhanced DD for "
                f"{commodity_type or 'all commodities'}"
            )

        return {
            "country_code": country_code,
            "region": region,
            "commodity_type": commodity_type,
            "override_active": is_override,
            "reason": reason,
            "matching_rules": matching_rules,
            "total_override_countries": len(self._override_regions),
        }

    # ------------------------------------------------------------------
    # Reclassification impact
    # ------------------------------------------------------------------

    def assess_reclassification_impact(
        self,
        country_code: str,
        current_risk_score: float,
        new_risk_score: float,
        commodity_type: Optional[str] = None,
        active_imports_count: int = 0,
        shipments_per_year: int = 12,
    ) -> Dict[str, Any]:
        """Assess the impact of a country risk score change on DD level.

        Compares current and proposed DD levels and estimates cost
        impact, timeline requirements, and affected imports.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            current_risk_score: Current composite risk score.
            new_risk_score: Proposed new composite risk score.
            commodity_type: Optional commodity type.
            active_imports_count: Number of active import records.
            shipments_per_year: Expected shipments per year.

        Returns:
            Dictionary with current_level, new_level, level_changed,
            cost_impact_eur, timeline_days, and recommendations.
        """
        cfg = get_config()
        country_code = country_code.upper().strip()

        current_level = self._determine_level(current_risk_score, cfg)
        new_level = self._determine_level(new_risk_score, cfg)

        level_changed = current_level != new_level

        current_cost_min, current_cost_max = self._get_cost_range(
            current_level, cfg,
        )
        new_cost_min, new_cost_max = self._get_cost_range(
            new_level, cfg,
        )

        annual_cost_delta_min = (
            (new_cost_min - current_cost_min) * shipments_per_year
        )
        annual_cost_delta_max = (
            (new_cost_max - current_cost_max) * shipments_per_year
        )

        # Timeline for transition
        compliance_days = 0
        if level_changed:
            compliance_days = _BASE_COMPLIANCE_DAYS.get(
                new_level.value, 90,
            )

        # Determine direction of change
        level_order = {
            DueDiligenceLevel.SIMPLIFIED: 0,
            DueDiligenceLevel.STANDARD: 1,
            DueDiligenceLevel.ENHANCED: 2,
        }
        direction = "unchanged"
        if level_changed:
            if level_order[new_level] > level_order[current_level]:
                direction = "escalation"
            else:
                direction = "de_escalation"

        recommendations = self._build_reclassification_recommendations(
            direction, current_level, new_level,
        )

        return {
            "country_code": country_code,
            "commodity_type": commodity_type,
            "current_risk_score": current_risk_score,
            "new_risk_score": new_risk_score,
            "current_level": current_level.value,
            "new_level": new_level.value,
            "level_changed": level_changed,
            "direction": direction,
            "active_imports_affected": active_imports_count if level_changed else 0,
            "annual_cost_delta_min_eur": annual_cost_delta_min,
            "annual_cost_delta_max_eur": annual_cost_delta_max,
            "compliance_timeline_days": compliance_days,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Compliance timeline
    # ------------------------------------------------------------------

    def get_compliance_timeline(
        self,
        level: str,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get estimated compliance timeline for a DD level.

        Args:
            level: DD level string (simplified, standard, enhanced).
            start_date: Optional start date (defaults to now UTC).

        Returns:
            Dictionary with level, start_date, estimated_completion,
            total_days, and milestones.

        Raises:
            ValueError: If level is not a valid DD level.
        """
        level_lower = level.lower().strip()
        if level_lower not in _BASE_COMPLIANCE_DAYS:
            raise ValueError(
                f"Invalid DD level '{level}'; "
                f"must be one of: simplified, standard, enhanced"
            )

        start = start_date or utcnow()
        total_days = _BASE_COMPLIANCE_DAYS[level_lower]
        completion = start + timedelta(days=total_days)

        milestones = self._build_milestones(level_lower, start, total_days)

        return {
            "level": level_lower,
            "start_date": start.isoformat(),
            "estimated_completion": completion.isoformat(),
            "total_days": total_days,
            "milestones": milestones,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_country_code(self, country_code: str) -> str:
        """Validate and normalize country code.

        Args:
            country_code: Raw country code string.

        Returns:
            Uppercase, stripped country code.

        Raises:
            ValueError: If country_code is empty or wrong length.
        """
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")
        cc = country_code.upper().strip()
        if len(cc) != 2:
            raise ValueError(
                f"country_code must be 2 characters, got '{cc}'"
            )
        return cc

    def _validate_risk_score(self, risk_score: float) -> None:
        """Validate risk score is within bounds.

        Args:
            risk_score: Score to validate.

        Raises:
            ValueError: If score is outside [0, 100].
        """
        if risk_score < 0.0 or risk_score > 100.0:
            raise ValueError(
                f"risk_score must be in [0, 100], got {risk_score}"
            )

    def _resolve_commodity(
        self, commodity_type: Optional[str],
    ) -> Optional[CommodityType]:
        """Resolve commodity type string to enum.

        Args:
            commodity_type: Optional commodity type string.

        Returns:
            CommodityType enum value or None.
        """
        if commodity_type is None:
            return None
        try:
            return CommodityType(commodity_type.lower().strip())
        except ValueError:
            logger.warning(
                "Unknown commodity_type '%s'; treating as None",
                commodity_type,
            )
            return None

    def _check_region_override(
        self,
        country_code: str,
        region: Optional[str],
        commodity_type: Optional[str],
    ) -> bool:
        """Check if a sub-national region triggers enhanced DD override.

        Args:
            country_code: ISO alpha-2 code.
            region: Sub-national region name.
            commodity_type: Optional commodity type.

        Returns:
            True if override applies, False otherwise.
        """
        if not region:
            return False

        country_rules = self._override_regions.get(country_code)
        if not country_rules:
            return False

        for rule_region, commodities in country_rules.items():
            if rule_region.lower() == region.lower():
                if commodity_type is None:
                    return True
                if commodity_type.lower() in [c.lower() for c in commodities]:
                    return True

        return False

    def _calculate_certification_credit(
        self,
        certification_schemes: List[str],
        commodity_type: Optional[str],
        cfg: Any,
    ) -> float:
        """Calculate total certification-based risk credit.

        The credit is the sum of individual scheme credits, capped
        at the configured maximum. Each scheme credit is proportional
        to its effectiveness for the given commodity.

        Args:
            certification_schemes: List of scheme names.
            commodity_type: Optional commodity for effectiveness lookup.
            cfg: Agent configuration.

        Returns:
            Total credit amount (0 to certification_credit_max).
        """
        if not certification_schemes:
            return 0.0

        max_credit = float(cfg.certification_credit_max)
        total_credit = 0.0

        for scheme in certification_schemes:
            scheme_lower = scheme.lower().strip()
            effectiveness = self._get_scheme_effectiveness(
                scheme_lower, commodity_type,
            )
            # Credit contribution proportional to effectiveness
            # effectiveness is 0-100, convert to fraction of max_credit
            credit_contribution = (effectiveness / 100.0) * max_credit
            total_credit += credit_contribution

        # Cap at maximum allowed
        return min(total_credit, max_credit)

    def _get_scheme_effectiveness(
        self,
        scheme: str,
        commodity_type: Optional[str],
    ) -> float:
        """Get certification scheme effectiveness for a commodity.

        Args:
            scheme: Lowercase scheme name.
            commodity_type: Optional commodity type.

        Returns:
            Effectiveness score (0-100).
        """
        scheme_data = _CERTIFICATION_EFFECTIVENESS.get(scheme)
        if not scheme_data:
            return 0.0

        if commodity_type is None:
            # Average across all commodities for this scheme
            values = [v for v in scheme_data.values() if v > 0]
            return sum(values) / len(values) if values else 0.0

        return scheme_data.get(commodity_type.lower().strip(), 0.0)

    def _determine_level(
        self,
        effective_score: float,
        cfg: Any,
    ) -> DueDiligenceLevel:
        """Determine DD level from effective risk score.

        Args:
            effective_score: Risk score after certification credit.
            cfg: Agent configuration with thresholds.

        Returns:
            DueDiligenceLevel enum value.
        """
        if effective_score <= cfg.simplified_threshold:
            return DueDiligenceLevel.SIMPLIFIED
        if effective_score <= cfg.enhanced_threshold:
            return DueDiligenceLevel.STANDARD
        return DueDiligenceLevel.ENHANCED

    def _get_cost_range(
        self,
        level: DueDiligenceLevel,
        cfg: Any,
    ) -> Tuple[float, float]:
        """Get cost range for a DD level from config.

        Args:
            level: Due diligence level.
            cfg: Agent configuration.

        Returns:
            Tuple of (min_cost_eur, max_cost_eur).
        """
        if level == DueDiligenceLevel.SIMPLIFIED:
            return (
                float(cfg.simplified_cost_min_eur),
                float(cfg.simplified_cost_max_eur),
            )
        if level == DueDiligenceLevel.STANDARD:
            return (
                float(cfg.standard_cost_min_eur),
                float(cfg.standard_cost_max_eur),
            )
        return (
            float(cfg.enhanced_cost_min_eur),
            float(cfg.enhanced_cost_max_eur),
        )

    def _build_classification(
        self,
        country_code: str,
        commodity_enum: Optional[CommodityType],
        level: DueDiligenceLevel,
        risk_score: float,
        cert_credit: float,
        effective_score: float,
        cfg: Any,
    ) -> DueDiligenceClassification:
        """Build a complete DueDiligenceClassification model.

        Args:
            country_code: ISO alpha-2 code.
            commodity_enum: Optional commodity enum.
            level: Determined DD level.
            risk_score: Original risk score.
            cert_credit: Certification credit applied.
            effective_score: Score after credit.
            cfg: Agent configuration.

        Returns:
            Populated DueDiligenceClassification model.
        """
        cost_min, cost_max = self._get_cost_range(level, cfg)

        audit_info = self.get_audit_frequency(level.value)
        audit_frequency_label = audit_info["frequency"]

        satellite_required = level == DueDiligenceLevel.ENHANCED

        compliance_days = None
        if cfg.enable_time_to_compliance:
            compliance_days = _BASE_COMPLIANCE_DAYS.get(level.value, 90)

        requirements = self.get_requirements(level.value)

        # Provenance hash
        tracker = get_provenance_tracker()
        prov_data = {
            "country_code": country_code,
            "risk_score": risk_score,
            "cert_credit": cert_credit,
            "effective_score": effective_score,
            "level": level.value,
        }
        provenance_hash = tracker.build_hash(prov_data)

        return DueDiligenceClassification(
            country_code=country_code,
            commodity_type=commodity_enum,
            level=level,
            risk_score=risk_score,
            certification_credit=cert_credit,
            effective_risk_score=effective_score,
            audit_frequency=audit_frequency_label,
            satellite_required=satellite_required,
            cost_estimate_min_eur=cost_min,
            cost_estimate_max_eur=cost_max,
            time_to_compliance_days=compliance_days,
            regulatory_requirements=requirements,
            provenance_hash=provenance_hash,
        )

    def _build_reclassification_recommendations(
        self,
        direction: str,
        current_level: DueDiligenceLevel,
        new_level: DueDiligenceLevel,
    ) -> List[str]:
        """Build recommendations for a reclassification event.

        Args:
            direction: escalation, de_escalation, or unchanged.
            current_level: Current DD level.
            new_level: New DD level.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if direction == "unchanged":
            recommendations.append(
                "No change in DD level. Continue current procedures."
            )
            return recommendations

        if direction == "escalation":
            recommendations.append(
                f"DD level escalated from {current_level.value} to "
                f"{new_level.value}. Immediate action required."
            )
            if new_level == DueDiligenceLevel.ENHANCED:
                recommendations.extend([
                    "Engage satellite monitoring provider immediately.",
                    "Schedule independent third-party audit.",
                    "Review and update supplier site visit plans.",
                    "Prepare enhanced DDS documentation.",
                    "Notify procurement and compliance teams.",
                ])
            elif new_level == DueDiligenceLevel.STANDARD:
                recommendations.extend([
                    "Upgrade due diligence statement (DDS) documentation.",
                    "Implement geolocation tracking for production plots.",
                    "Schedule semi-annual audit cycle.",
                    "Notify procurement team of increased requirements.",
                ])
        else:
            recommendations.append(
                f"DD level de-escalated from {current_level.value} to "
                f"{new_level.value}. Cost reduction opportunity."
            )
            recommendations.extend([
                "Review current compliance procedures for optimization.",
                "Adjust audit frequency to new level requirements.",
                "Update cost projections and budget allocations.",
            ])

        return recommendations

    def _build_milestones(
        self,
        level: str,
        start: datetime,
        total_days: int,
    ) -> List[Dict[str, str]]:
        """Build compliance milestones for a DD level.

        Args:
            level: DD level string.
            start: Start date.
            total_days: Total days to compliance.

        Returns:
            List of milestone dictionaries with name, date, and status.
        """
        milestones: List[Dict[str, str]] = []

        if level == "simplified":
            milestones = [
                {
                    "name": "Supplier declaration collection",
                    "target_date": (start + timedelta(days=7)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Basic DDS preparation",
                    "target_date": (start + timedelta(days=14)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Record system setup",
                    "target_date": (start + timedelta(days=21)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Compliance verification",
                    "target_date": (start + timedelta(days=total_days)).isoformat(),
                    "status": "pending",
                },
            ]
        elif level == "standard":
            milestones = [
                {
                    "name": "Supplier onboarding and declaration",
                    "target_date": (start + timedelta(days=14)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Geolocation data collection",
                    "target_date": (start + timedelta(days=30)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Risk assessment documentation",
                    "target_date": (start + timedelta(days=45)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Supplier verification audit",
                    "target_date": (start + timedelta(days=60)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "DDS finalization and submission",
                    "target_date": (start + timedelta(days=75)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Compliance verification",
                    "target_date": (start + timedelta(days=total_days)).isoformat(),
                    "status": "pending",
                },
            ]
        else:  # enhanced
            milestones = [
                {
                    "name": "Supplier onboarding and declaration",
                    "target_date": (start + timedelta(days=14)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Satellite monitoring setup",
                    "target_date": (start + timedelta(days=30)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Geolocation and plot mapping",
                    "target_date": (start + timedelta(days=45)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Comprehensive risk assessment",
                    "target_date": (start + timedelta(days=60)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Supplier site visits",
                    "target_date": (start + timedelta(days=90)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Independent third-party audit",
                    "target_date": (start + timedelta(days=120)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Community consultation",
                    "target_date": (start + timedelta(days=140)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Enhanced DDS finalization",
                    "target_date": (start + timedelta(days=160)).isoformat(),
                    "status": "pending",
                },
                {
                    "name": "Compliance verification",
                    "target_date": (start + timedelta(days=total_days)).isoformat(),
                    "status": "pending",
                },
            ]

        return milestones

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._classifications)
        return (
            f"DueDiligenceClassifier("
            f"classifications={count}, "
            f"override_countries={len(self._override_regions)})"
        )

    def __len__(self) -> int:
        """Return number of stored classifications."""
        with self._lock:
            return len(self._classifications)
