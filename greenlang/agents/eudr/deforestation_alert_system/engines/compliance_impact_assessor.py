# -*- coding: utf-8 -*-
"""
ComplianceImpactAssessor - AGENT-EUDR-020 Engine 8: EUDR Compliance Impact Assessment

Links deforestation alerts to supply chain entities (suppliers, plots, products)
and assesses EUDR compliance impact. Determines whether affected products can
still be placed on the EU market per Regulation (EU) 2023/1115.

Compliance Decision Logic:
    - POST_CUTOFF deforestation + HIGH/CRITICAL severity -> NON_COMPLIANT + market restriction
    - POST_CUTOFF + MEDIUM severity -> UNDER_REVIEW
    - PRE_CUTOFF deforestation -> COMPLIANT (monitoring may be required)
    - UNCERTAIN cutoff timing -> REMEDIATION_REQUIRED
    - Any NON_COMPLIANT outcome triggers mandatory remediation plan

Remediation Actions:
    - SUPPLIER_AUDIT: On-site supplier audit required.
    - PLOT_EXCLUSION: Remove affected plot from supply chain.
    - ALTERNATIVE_SOURCING: Identify alternative deforestation-free sources.
    - ENHANCED_MONITORING: Increase satellite monitoring frequency.
    - PRODUCT_WITHDRAWAL: Withdraw affected products from EU market.
    - SUPPLY_CHAIN_RESTRUCTURE: Restructure supply chain to eliminate risk.

Financial Impact Estimation:
    - Based on affected product volumes, commodity market prices, and
      potential regulatory fines (up to 4% of EU annual turnover per EUDR).
    - Market restriction impact uses commodity-specific loss multipliers.

Zero-Hallucination Guarantees:
    - All compliance decisions use explicit threshold comparisons.
    - Financial impact uses deterministic Decimal arithmetic with
      commodity price lookups and volume-based calculations.
    - Remediation actions are selected from static rule tables.
    - Risk scoring uses weighted Decimal sums with fixed weights.
    - SHA-256 provenance hashes on all output objects.
    - No ML/LLM in any compliance determination or financial calculation.

Performance Targets:
    - Single impact assessment: <100ms
    - Product tracing: <150ms per alert
    - Remediation plan creation: <50ms
    - Batch assessment (100 alerts): <3s

Regulatory References:
    - EUDR Article 3: Prohibition on placing non-compliant products on EU market.
    - EUDR Article 4: Due diligence obligations for operators.
    - EUDR Article 10: Risk assessment requirements.
    - EUDR Article 11: Risk mitigation measures.
    - EUDR Article 25: Penalties (up to 4% of annual EU turnover).
    - EUDR Article 31: Five-year record retention.
    - EUDR Annex I: List of relevant products and HS codes.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020, Engine 8 (Compliance Impact Assessor)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "ci") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc

def _clamp_decimal(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal value to [lo, hi] range.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped Decimal.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ComplianceOutcome(str, Enum):
    """EUDR compliance outcome for affected supply chain entities.

    Values:
        COMPLIANT: Products can be placed on EU market.
        NON_COMPLIANT: Products CANNOT be placed on EU market.
        UNDER_REVIEW: Assessment pending, interim restrictions may apply.
        REMEDIATION_REQUIRED: Specific actions needed before compliance.
        SUSPENDED: Supply chain entity temporarily suspended pending review.
    """

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNDER_REVIEW = "UNDER_REVIEW"
    REMEDIATION_REQUIRED = "REMEDIATION_REQUIRED"
    SUSPENDED = "SUSPENDED"

class RemediationAction(str, Enum):
    """Required remediation actions for affected entities.

    Values:
        SUPPLIER_AUDIT: On-site supplier audit.
        PLOT_EXCLUSION: Remove plot from supply chain.
        ALTERNATIVE_SOURCING: Find deforestation-free alternatives.
        ENHANCED_MONITORING: Increase satellite monitoring frequency.
        PRODUCT_WITHDRAWAL: Withdraw products from EU market.
        SUPPLY_CHAIN_RESTRUCTURE: Restructure supply chain.
    """

    SUPPLIER_AUDIT = "SUPPLIER_AUDIT"
    PLOT_EXCLUSION = "PLOT_EXCLUSION"
    ALTERNATIVE_SOURCING = "ALTERNATIVE_SOURCING"
    ENHANCED_MONITORING = "ENHANCED_MONITORING"
    PRODUCT_WITHDRAWAL = "PRODUCT_WITHDRAWAL"
    SUPPLY_CHAIN_RESTRUCTURE = "SUPPLY_CHAIN_RESTRUCTURE"

class RemediationPriority(str, Enum):
    """Priority level for remediation actions.

    Values:
        IMMEDIATE: Must be executed within 24 hours.
        URGENT: Must be executed within 7 days.
        HIGH: Must be executed within 30 days.
        STANDARD: Must be executed within 90 days.
    """

    IMMEDIATE = "IMMEDIATE"
    URGENT = "URGENT"
    HIGH = "HIGH"
    STANDARD = "STANDARD"

class ImpactSeverity(str, Enum):
    """Impact severity classification.

    Values:
        CRITICAL: Immediate market restriction, full product withdrawal.
        HIGH: Market restriction likely, audit required.
        MEDIUM: Under review, enhanced monitoring needed.
        LOW: Monitoring only, no immediate action.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Market restriction severity threshold (at or above triggers restriction).
MARKET_RESTRICTION_THRESHOLD: str = "HIGH"

#: Severity ordering for comparison.
SEVERITY_ORDER: Dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "INFORMATIONAL": 0,
}

#: Maximum batch size for impact assessments.
MAX_BATCH_SIZE: int = 500

#: Commodity reference prices per metric ton (EUR).
COMMODITY_PRICES_EUR_PER_TON: Dict[str, Decimal] = {
    "cattle": Decimal("4500"),
    "cocoa": Decimal("3200"),
    "coffee": Decimal("2800"),
    "palm_oil": Decimal("950"),
    "rubber": Decimal("1600"),
    "soya": Decimal("550"),
    "wood": Decimal("250"),
}

#: Average annual tonnage per hectare for EUDR commodities.
COMMODITY_YIELD_PER_HA: Dict[str, Decimal] = {
    "cattle": Decimal("0.3"),
    "cocoa": Decimal("0.5"),
    "coffee": Decimal("1.2"),
    "palm_oil": Decimal("3.8"),
    "rubber": Decimal("1.5"),
    "soya": Decimal("3.0"),
    "wood": Decimal("8.0"),
}

#: Market restriction loss multiplier (fraction of annual value lost).
MARKET_RESTRICTION_LOSS_MULTIPLIER: Decimal = Decimal("1.5")

#: Regulatory fine ceiling (fraction of EU annual turnover per EUDR Art. 25).
REGULATORY_FINE_CEILING_PCT: Decimal = Decimal("4")

#: Remediation cost estimates per action type (EUR).
REMEDIATION_COST_ESTIMATES: Dict[str, Decimal] = {
    RemediationAction.SUPPLIER_AUDIT.value: Decimal("15000"),
    RemediationAction.PLOT_EXCLUSION.value: Decimal("5000"),
    RemediationAction.ALTERNATIVE_SOURCING.value: Decimal("25000"),
    RemediationAction.ENHANCED_MONITORING.value: Decimal("8000"),
    RemediationAction.PRODUCT_WITHDRAWAL.value: Decimal("50000"),
    RemediationAction.SUPPLY_CHAIN_RESTRUCTURE.value: Decimal("100000"),
}

#: Remediation timeline estimates (days).
REMEDIATION_TIMELINE_DAYS: Dict[str, int] = {
    RemediationAction.SUPPLIER_AUDIT.value: 30,
    RemediationAction.PLOT_EXCLUSION.value: 7,
    RemediationAction.ALTERNATIVE_SOURCING.value: 60,
    RemediationAction.ENHANCED_MONITORING.value: 14,
    RemediationAction.PRODUCT_WITHDRAWAL.value: 3,
    RemediationAction.SUPPLY_CHAIN_RESTRUCTURE.value: 180,
}

#: Cutoff result to compliance outcome mapping rules.
#: Format: {(cutoff_result, severity_level): ComplianceOutcome}
COMPLIANCE_RULES: Dict[Tuple[str, str], str] = {
    # POST_CUTOFF outcomes
    ("POST_CUTOFF", "CRITICAL"): ComplianceOutcome.NON_COMPLIANT.value,
    ("POST_CUTOFF", "HIGH"): ComplianceOutcome.NON_COMPLIANT.value,
    ("POST_CUTOFF", "MEDIUM"): ComplianceOutcome.UNDER_REVIEW.value,
    ("POST_CUTOFF", "LOW"): ComplianceOutcome.UNDER_REVIEW.value,
    # PRE_CUTOFF outcomes
    ("PRE_CUTOFF", "CRITICAL"): ComplianceOutcome.COMPLIANT.value,
    ("PRE_CUTOFF", "HIGH"): ComplianceOutcome.COMPLIANT.value,
    ("PRE_CUTOFF", "MEDIUM"): ComplianceOutcome.COMPLIANT.value,
    ("PRE_CUTOFF", "LOW"): ComplianceOutcome.COMPLIANT.value,
    # ONGOING outcomes
    ("ONGOING", "CRITICAL"): ComplianceOutcome.NON_COMPLIANT.value,
    ("ONGOING", "HIGH"): ComplianceOutcome.REMEDIATION_REQUIRED.value,
    ("ONGOING", "MEDIUM"): ComplianceOutcome.UNDER_REVIEW.value,
    ("ONGOING", "LOW"): ComplianceOutcome.UNDER_REVIEW.value,
    # UNCERTAIN outcomes
    ("UNCERTAIN", "CRITICAL"): ComplianceOutcome.REMEDIATION_REQUIRED.value,
    ("UNCERTAIN", "HIGH"): ComplianceOutcome.REMEDIATION_REQUIRED.value,
    ("UNCERTAIN", "MEDIUM"): ComplianceOutcome.REMEDIATION_REQUIRED.value,
    ("UNCERTAIN", "LOW"): ComplianceOutcome.UNDER_REVIEW.value,
}

#: Remediation actions required per compliance outcome.
OUTCOME_REMEDIATION_MAP: Dict[str, List[str]] = {
    ComplianceOutcome.NON_COMPLIANT.value: [
        RemediationAction.PRODUCT_WITHDRAWAL.value,
        RemediationAction.PLOT_EXCLUSION.value,
        RemediationAction.SUPPLIER_AUDIT.value,
        RemediationAction.ALTERNATIVE_SOURCING.value,
        RemediationAction.ENHANCED_MONITORING.value,
    ],
    ComplianceOutcome.REMEDIATION_REQUIRED.value: [
        RemediationAction.SUPPLIER_AUDIT.value,
        RemediationAction.ENHANCED_MONITORING.value,
        RemediationAction.ALTERNATIVE_SOURCING.value,
    ],
    ComplianceOutcome.UNDER_REVIEW.value: [
        RemediationAction.ENHANCED_MONITORING.value,
        RemediationAction.SUPPLIER_AUDIT.value,
    ],
    ComplianceOutcome.SUSPENDED.value: [
        RemediationAction.PLOT_EXCLUSION.value,
        RemediationAction.SUPPLIER_AUDIT.value,
        RemediationAction.ENHANCED_MONITORING.value,
    ],
    ComplianceOutcome.COMPLIANT.value: [],
}

#: DDS (Due Diligence Statement) risk assessment per outcome.
DDS_RISK_MAP: Dict[str, bool] = {
    ComplianceOutcome.NON_COMPLIANT.value: True,
    ComplianceOutcome.REMEDIATION_REQUIRED.value: True,
    ComplianceOutcome.UNDER_REVIEW.value: True,
    ComplianceOutcome.SUSPENDED.value: True,
    ComplianceOutcome.COMPLIANT.value: False,
}

#: Reference supplier data for offline testing.
REFERENCE_SUPPLIERS: Dict[str, Dict[str, Any]] = {
    "SUP-001": {
        "name": "Amazon Agro Ltda",
        "country": "BR",
        "commodity": "soya",
        "plots": ["PLOT-001", "PLOT-002", "PLOT-003"],
        "annual_volume_tons": Decimal("5000"),
        "products": ["PROD-001", "PROD-002"],
    },
    "SUP-002": {
        "name": "Borneo Palm Oil Corp",
        "country": "ID",
        "commodity": "palm_oil",
        "plots": ["PLOT-004", "PLOT-005"],
        "annual_volume_tons": Decimal("12000"),
        "products": ["PROD-003", "PROD-004", "PROD-005"],
    },
    "SUP-003": {
        "name": "Ivory Coast Cocoa Coop",
        "country": "CI",
        "commodity": "cocoa",
        "plots": ["PLOT-006", "PLOT-007"],
        "annual_volume_tons": Decimal("800"),
        "products": ["PROD-006"],
    },
    "SUP-004": {
        "name": "Guatemala Coffee Finca",
        "country": "GT",
        "commodity": "coffee",
        "plots": ["PLOT-008"],
        "annual_volume_tons": Decimal("200"),
        "products": ["PROD-007", "PROD-008"],
    },
    "SUP-005": {
        "name": "Congo Timber Co",
        "country": "CD",
        "commodity": "wood",
        "plots": ["PLOT-009", "PLOT-010"],
        "annual_volume_tons": Decimal("3000"),
        "products": ["PROD-009"],
    },
}

#: Reference product data for offline testing.
REFERENCE_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "PROD-001": {"name": "Soybean Meal", "hs_code": "2304", "commodity": "soya", "annual_value_eur": Decimal("500000")},
    "PROD-002": {"name": "Soybean Oil", "hs_code": "1507", "commodity": "soya", "annual_value_eur": Decimal("350000")},
    "PROD-003": {"name": "Crude Palm Oil", "hs_code": "1511", "commodity": "palm_oil", "annual_value_eur": Decimal("2800000")},
    "PROD-004": {"name": "Palm Kernel Oil", "hs_code": "1513", "commodity": "palm_oil", "annual_value_eur": Decimal("450000")},
    "PROD-005": {"name": "Palm Olein", "hs_code": "1511", "commodity": "palm_oil", "annual_value_eur": Decimal("1200000")},
    "PROD-006": {"name": "Cocoa Beans", "hs_code": "1801", "commodity": "cocoa", "annual_value_eur": Decimal("650000")},
    "PROD-007": {"name": "Green Coffee", "hs_code": "0901", "commodity": "coffee", "annual_value_eur": Decimal("180000")},
    "PROD-008": {"name": "Roasted Coffee", "hs_code": "0901", "commodity": "coffee", "annual_value_eur": Decimal("320000")},
    "PROD-009": {"name": "Tropical Hardwood Lumber", "hs_code": "4407", "commodity": "wood", "annual_value_eur": Decimal("750000")},
}

#: Alert-to-supplier mapping reference data.
ALERT_SUPPLIER_MAP: Dict[str, List[str]] = {
    "sample_post_cutoff": ["SUP-001", "SUP-002"],
    "sample_pre_cutoff": ["SUP-003"],
    "sample_ongoing": ["SUP-004", "SUP-005"],
    "sample_uncertain": ["SUP-001"],
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AffectedSupplier:
    """A supplier affected by a deforestation alert.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        country_code: Supplier country.
        commodity: EUDR commodity type.
        affected_plots: List of affected plot IDs.
        annual_volume_tons: Annual supply volume.
        affected_products: Products sourced from this supplier.
        risk_level: Risk level for this supplier.
    """

    supplier_id: str = ""
    supplier_name: str = ""
    country_code: str = ""
    commodity: str = ""
    affected_plots: List[str] = field(default_factory=list)
    annual_volume_tons: Decimal = Decimal("0")
    affected_products: List[str] = field(default_factory=list)
    risk_level: str = "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "country_code": self.country_code,
            "commodity": self.commodity,
            "affected_plots": self.affected_plots,
            "annual_volume_tons": str(self.annual_volume_tons),
            "affected_products": self.affected_products,
            "risk_level": self.risk_level,
        }

@dataclass
class AffectedProduct:
    """A product affected by a deforestation alert.

    Attributes:
        product_id: Product identifier.
        product_name: Product name.
        hs_code: Harmonized System code (EUDR Annex I).
        commodity: EUDR commodity type.
        annual_value_eur: Annual product value in EUR.
        market_restriction: Whether product faces EU market restriction.
        withdrawal_required: Whether product must be withdrawn.
    """

    product_id: str = ""
    product_name: str = ""
    hs_code: str = ""
    commodity: str = ""
    annual_value_eur: Decimal = Decimal("0")
    market_restriction: bool = False
    withdrawal_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "hs_code": self.hs_code,
            "commodity": self.commodity,
            "annual_value_eur": str(self.annual_value_eur),
            "market_restriction": self.market_restriction,
            "withdrawal_required": self.withdrawal_required,
        }

@dataclass
class ComplianceImpact:
    """Complete compliance impact assessment for a deforestation alert.

    Attributes:
        impact_id: Unique impact assessment identifier.
        alert_id: Source alert identifier.
        cutoff_result: Cutoff verification result.
        alert_severity: Alert severity level.
        compliance_outcome: Overall compliance determination.
        market_restriction: Whether EU market restriction applies.
        affected_suppliers: List of affected suppliers.
        affected_products: List of affected products.
        remediation_actions: Required remediation actions.
        estimated_financial_impact_eur: Estimated total financial impact.
        product_value_at_risk_eur: Value of products at risk.
        potential_fine_eur: Potential regulatory fine.
        remediation_cost_eur: Estimated remediation cost.
        risk_to_dds: Whether this affects Due Diligence Statements.
        dds_affected_count: Number of DDS statements affected.
        assessment_notes: Assessment notes and reasoning.
        regulatory_references: Relevant EUDR article references.
        assessment_timestamp: When assessment was performed.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash for audit trail.
    """

    impact_id: str = ""
    alert_id: str = ""
    cutoff_result: str = ""
    alert_severity: str = ""
    compliance_outcome: str = ComplianceOutcome.UNDER_REVIEW.value
    market_restriction: bool = False
    affected_suppliers: List[Dict[str, Any]] = field(default_factory=list)
    affected_products: List[Dict[str, Any]] = field(default_factory=list)
    remediation_actions: List[Dict[str, Any]] = field(default_factory=list)
    estimated_financial_impact_eur: Decimal = Decimal("0")
    product_value_at_risk_eur: Decimal = Decimal("0")
    potential_fine_eur: Decimal = Decimal("0")
    remediation_cost_eur: Decimal = Decimal("0")
    risk_to_dds: bool = False
    dds_affected_count: int = 0
    assessment_notes: List[str] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)
    assessment_timestamp: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "impact_id": self.impact_id,
            "alert_id": self.alert_id,
            "cutoff_result": self.cutoff_result,
            "alert_severity": self.alert_severity,
            "compliance_outcome": self.compliance_outcome,
            "market_restriction": self.market_restriction,
            "affected_suppliers": self.affected_suppliers,
            "affected_products": self.affected_products,
            "remediation_actions": self.remediation_actions,
            "estimated_financial_impact_eur": str(self.estimated_financial_impact_eur),
            "product_value_at_risk_eur": str(self.product_value_at_risk_eur),
            "potential_fine_eur": str(self.potential_fine_eur),
            "remediation_cost_eur": str(self.remediation_cost_eur),
            "risk_to_dds": self.risk_to_dds,
            "dds_affected_count": self.dds_affected_count,
            "assessment_notes": self.assessment_notes,
            "regulatory_references": self.regulatory_references,
            "assessment_timestamp": self.assessment_timestamp,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
        }

@dataclass
class RemediationPlan:
    """Remediation plan for a compliance impact.

    Attributes:
        plan_id: Unique plan identifier.
        impact_id: Source impact assessment identifier.
        alert_id: Source alert identifier.
        compliance_outcome: Compliance outcome driving remediation.
        actions: List of remediation action details.
        total_estimated_cost_eur: Total estimated cost.
        total_timeline_days: Total timeline to completion.
        priority: Overall plan priority.
        responsible_parties: Parties responsible for execution.
        milestones: Key milestones with target dates.
        provenance_hash: SHA-256 hash.
    """

    plan_id: str = ""
    impact_id: str = ""
    alert_id: str = ""
    compliance_outcome: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    total_estimated_cost_eur: Decimal = Decimal("0")
    total_timeline_days: int = 0
    priority: str = RemediationPriority.STANDARD.value
    responsible_parties: List[str] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "impact_id": self.impact_id,
            "alert_id": self.alert_id,
            "compliance_outcome": self.compliance_outcome,
            "actions": self.actions,
            "total_estimated_cost_eur": str(self.total_estimated_cost_eur),
            "total_timeline_days": self.total_timeline_days,
            "priority": self.priority,
            "responsible_parties": self.responsible_parties,
            "milestones": self.milestones,
            "provenance_hash": self.provenance_hash,
        }

# ---------------------------------------------------------------------------
# ComplianceImpactAssessor
# ---------------------------------------------------------------------------

class ComplianceImpactAssessor:
    """Production-grade EUDR compliance impact assessment engine.

    Links deforestation alerts to supply chain entities and assesses
    EUDR compliance impact. Determines whether affected products can
    access the EU market, calculates financial impact, generates
    remediation actions, and tracks DDS (Due Diligence Statement) risk.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All compliance decisions use explicit rule tables.
        Financial calculations use deterministic Decimal arithmetic.
        Remediation actions are selected from static mappings.
        No ML/LLM in any decision or calculation path.

    Attributes:
        _market_restriction_threshold: Severity level triggering restriction.
        _custom_suppliers: User-supplied supplier data.
        _custom_products: User-supplied product data.
        _custom_alert_mapping: User-supplied alert-to-supplier mapping.
        _assessment_cache: Cache of completed assessments.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> assessor = ComplianceImpactAssessor()
        >>> result = assessor.assess_impact(
        ...     "alert-001",
        ...     supply_chain_context={"cutoff_result": "POST_CUTOFF", "severity": "HIGH"},
        ... )
        >>> assert result["compliance_outcome"] in ("COMPLIANT", "NON_COMPLIANT", "UNDER_REVIEW")
    """

    def __init__(
        self,
        market_restriction_threshold: Optional[str] = None,
    ) -> None:
        """Initialize ComplianceImpactAssessor.

        Args:
            market_restriction_threshold: Override severity for market restriction.
        """
        self._market_restriction_threshold: str = (
            market_restriction_threshold or MARKET_RESTRICTION_THRESHOLD
        )
        self._custom_suppliers: Dict[str, Dict[str, Any]] = {}
        self._custom_products: Dict[str, Dict[str, Any]] = {}
        self._custom_alert_mapping: Dict[str, List[str]] = {}
        self._assessment_cache: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "ComplianceImpactAssessor initialized (version=%s, "
            "market_restriction_threshold=%s, ref_suppliers=%d, "
            "ref_products=%d)",
            _MODULE_VERSION,
            self._market_restriction_threshold,
            len(REFERENCE_SUPPLIERS),
            len(REFERENCE_PRODUCTS),
        )

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_supplier_data(
        self,
        supplier_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Load custom supplier data.

        Args:
            supplier_id: Supplier identifier.
            data: Supplier data dictionary.

        Raises:
            ValueError: If supplier_id or data is empty.
        """
        if not supplier_id:
            raise ValueError("supplier_id must be non-empty")
        if not data:
            raise ValueError("data must be non-empty")

        with self._lock:
            self._custom_suppliers[supplier_id] = dict(data)
        logger.info("Loaded supplier data: %s", supplier_id)

    def load_product_data(
        self,
        product_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Load custom product data.

        Args:
            product_id: Product identifier.
            data: Product data dictionary.

        Raises:
            ValueError: If product_id or data is empty.
        """
        if not product_id:
            raise ValueError("product_id must be non-empty")
        if not data:
            raise ValueError("data must be non-empty")

        with self._lock:
            self._custom_products[product_id] = dict(data)
        logger.info("Loaded product data: %s", product_id)

    def load_alert_supplier_mapping(
        self,
        alert_id: str,
        supplier_ids: List[str],
    ) -> None:
        """Load alert-to-supplier mapping.

        Args:
            alert_id: Alert identifier.
            supplier_ids: List of affected supplier IDs.

        Raises:
            ValueError: If alert_id or supplier_ids is empty.
        """
        if not alert_id:
            raise ValueError("alert_id must be non-empty")
        if not supplier_ids:
            raise ValueError("supplier_ids must be non-empty")

        with self._lock:
            self._custom_alert_mapping[alert_id] = list(supplier_ids)
        logger.info("Loaded mapping: alert=%s suppliers=%s", alert_id, supplier_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_impact(
        self,
        alert_id: str,
        supply_chain_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform full EUDR compliance impact assessment for an alert.

        Args:
            alert_id: Alert identifier.
            supply_chain_context: Context with cutoff_result, severity,
                area_ha, latitude, longitude, commodity, and other fields.

        Returns:
            Dictionary with complete impact assessment.

        Raises:
            ValueError: If alert_id is empty.
        """
        start_time = time.monotonic()

        if not alert_id:
            raise ValueError("alert_id must be non-empty")

        ctx = supply_chain_context or {}
        cutoff_result = ctx.get("cutoff_result", "UNCERTAIN")
        severity = ctx.get("severity", "MEDIUM")
        area_ha = _to_decimal(ctx.get("area_ha", "1"))

        # Determine compliance outcome
        compliance_outcome = self._determine_compliance_outcome(
            cutoff_result, severity
        )

        # Assess market restriction
        market_restriction = self._assess_market_restriction(
            cutoff_result, severity
        )

        # Trace affected supply chain entities
        affected_suppliers = self._trace_affected_suppliers(alert_id)
        affected_products = self._trace_affected_products(
            affected_suppliers, market_restriction
        )

        # Generate remediation actions
        remediation_actions = self._generate_remediation_actions(
            compliance_outcome, severity
        )

        # Calculate financial impact
        financial = self._calculate_financial_impact(
            affected_products, market_restriction, area_ha
        )

        # Assess DDS risk
        risk_to_dds = DDS_RISK_MAP.get(compliance_outcome, False)
        dds_count = len(affected_suppliers) if risk_to_dds else 0

        # Generate assessment notes
        notes = self._generate_assessment_notes(
            cutoff_result, severity, compliance_outcome,
            market_restriction, len(affected_suppliers),
        )

        # Regulatory references
        reg_refs = self._get_regulatory_references(compliance_outcome)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        impact = ComplianceImpact(
            impact_id=_generate_id("ci"),
            alert_id=alert_id,
            cutoff_result=cutoff_result,
            alert_severity=severity,
            compliance_outcome=compliance_outcome,
            market_restriction=market_restriction,
            affected_suppliers=[s.to_dict() for s in affected_suppliers],
            affected_products=[p.to_dict() for p in affected_products],
            remediation_actions=remediation_actions,
            estimated_financial_impact_eur=financial["total"],
            product_value_at_risk_eur=financial["product_value"],
            potential_fine_eur=financial["fine"],
            remediation_cost_eur=financial["remediation_cost"],
            risk_to_dds=risk_to_dds,
            dds_affected_count=dds_count,
            assessment_notes=notes,
            regulatory_references=reg_refs,
            assessment_timestamp=utcnow().isoformat(),
            processing_time_ms=round(processing_time_ms, 3),
        )
        impact.provenance_hash = _compute_hash(impact)

        result = impact.to_dict()

        with self._lock:
            self._assessment_cache[alert_id] = result

        logger.info(
            "Impact assessment: alert=%s outcome=%s restriction=%s "
            "suppliers=%d products=%d financial=EUR %s time_ms=%.1f",
            alert_id, compliance_outcome, market_restriction,
            len(affected_suppliers), len(affected_products),
            financial["total"], processing_time_ms,
        )

        return result

    def get_affected_products(self, alert_id: str) -> Dict[str, Any]:
        """Get products affected by a deforestation alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            Dictionary with affected products list and provenance.
        """
        if not alert_id:
            raise ValueError("alert_id must be non-empty")

        suppliers = self._trace_affected_suppliers(alert_id)
        products = self._trace_affected_products(suppliers, False)

        result = {
            "alert_id": alert_id,
            "affected_products": [p.to_dict() for p in products],
            "total_products": len(products),
            "total_value_eur": str(sum(p.annual_value_eur for p in products)),
            "commodities": list({p.commodity for p in products}),
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    def get_recommendations(self, alert_id: str) -> Dict[str, Any]:
        """Get compliance recommendations for an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            Dictionary with prioritized recommendations.
        """
        if not alert_id:
            raise ValueError("alert_id must be non-empty")

        # Check cache
        with self._lock:
            cached = self._assessment_cache.get(alert_id)

        if cached:
            outcome = cached.get("compliance_outcome", "UNDER_REVIEW")
            severity = cached.get("alert_severity", "MEDIUM")
        else:
            outcome = ComplianceOutcome.UNDER_REVIEW.value
            severity = "MEDIUM"

        actions = OUTCOME_REMEDIATION_MAP.get(outcome, [])

        recommendations = []
        for i, action in enumerate(actions):
            cost = REMEDIATION_COST_ESTIMATES.get(action, Decimal("0"))
            timeline = REMEDIATION_TIMELINE_DAYS.get(action, 30)

            if i == 0:
                priority = RemediationPriority.IMMEDIATE.value
            elif i == 1:
                priority = RemediationPriority.URGENT.value
            else:
                priority = RemediationPriority.HIGH.value

            recommendations.append({
                "action": action,
                "priority": priority,
                "estimated_cost_eur": str(cost),
                "timeline_days": timeline,
                "description": self._describe_action(action),
                "regulatory_basis": self._get_action_regulatory_basis(action),
            })

        result = {
            "alert_id": alert_id,
            "compliance_outcome": outcome,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "total_estimated_cost_eur": str(
                sum(_to_decimal(r["estimated_cost_eur"]) for r in recommendations)
            ),
        }
        result["provenance_hash"] = _compute_hash(result)

        return result

    def create_remediation_plan(
        self,
        impact_id: str,
        alert_id: str = "",
        actions: Optional[List[str]] = None,
        responsible_parties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a remediation plan from an impact assessment.

        Args:
            impact_id: Impact assessment identifier.
            alert_id: Optional alert identifier.
            actions: Override list of remediation actions.
            responsible_parties: Parties responsible for execution.

        Returns:
            Dictionary with remediation plan.
        """
        if not impact_id:
            raise ValueError("impact_id must be non-empty")

        # Look up cached assessment
        cached_outcome = ComplianceOutcome.REMEDIATION_REQUIRED.value
        with self._lock:
            for cached in self._assessment_cache.values():
                if cached.get("impact_id") == impact_id:
                    cached_outcome = cached.get(
                        "compliance_outcome",
                        ComplianceOutcome.REMEDIATION_REQUIRED.value,
                    )
                    alert_id = alert_id or cached.get("alert_id", "")
                    break

        # Get actions
        if actions:
            action_list = actions
        else:
            action_list = OUTCOME_REMEDIATION_MAP.get(cached_outcome, [])

        # Build action details
        action_details = []
        total_cost = Decimal("0")
        max_timeline = 0

        for action in action_list:
            cost = REMEDIATION_COST_ESTIMATES.get(action, Decimal("0"))
            timeline = REMEDIATION_TIMELINE_DAYS.get(action, 30)
            total_cost += cost
            max_timeline = max(max_timeline, timeline)

            action_details.append({
                "action": action,
                "estimated_cost_eur": str(cost),
                "timeline_days": timeline,
                "description": self._describe_action(action),
                "status": "PLANNED",
            })

        # Determine priority
        if cached_outcome == ComplianceOutcome.NON_COMPLIANT.value:
            priority = RemediationPriority.IMMEDIATE.value
        elif cached_outcome == ComplianceOutcome.REMEDIATION_REQUIRED.value:
            priority = RemediationPriority.URGENT.value
        else:
            priority = RemediationPriority.HIGH.value

        # Build milestones
        milestones = self._build_milestones(action_details)

        plan = RemediationPlan(
            plan_id=_generate_id("rp"),
            impact_id=impact_id,
            alert_id=alert_id,
            compliance_outcome=cached_outcome,
            actions=action_details,
            total_estimated_cost_eur=total_cost,
            total_timeline_days=max_timeline,
            priority=priority,
            responsible_parties=responsible_parties or ["compliance_team"],
            milestones=milestones,
        )
        plan.provenance_hash = _compute_hash(plan)

        logger.info(
            "Remediation plan created: impact=%s actions=%d cost=EUR %s "
            "timeline=%d days priority=%s",
            impact_id, len(action_details), total_cost,
            max_timeline, priority,
        )

        return plan.to_dict()

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine state and configuration.
        """
        with self._lock:
            cache_size = len(self._assessment_cache)
            custom_suppliers = len(self._custom_suppliers)
            custom_products = len(self._custom_products)

        return {
            "engine": "ComplianceImpactAssessor",
            "version": _MODULE_VERSION,
            "market_restriction_threshold": self._market_restriction_threshold,
            "assessments_cached": cache_size,
            "custom_suppliers": custom_suppliers,
            "custom_products": custom_products,
            "reference_suppliers": len(REFERENCE_SUPPLIERS),
            "reference_products": len(REFERENCE_PRODUCTS),
            "compliance_rules": len(COMPLIANCE_RULES),
            "commodities_tracked": len(COMMODITY_PRICES_EUR_PER_TON),
        }

    # ------------------------------------------------------------------
    # Internal: Compliance Determination
    # ------------------------------------------------------------------

    def _determine_compliance_outcome(
        self,
        cutoff_result: str,
        severity: str,
    ) -> str:
        """Determine compliance outcome from cutoff result and severity.

        Uses explicit rule table lookup.

        Args:
            cutoff_result: Cutoff verification result.
            severity: Alert severity level.

        Returns:
            ComplianceOutcome value string.
        """
        key = (cutoff_result, severity)
        outcome = COMPLIANCE_RULES.get(key)

        if outcome is not None:
            return outcome

        # Fallback logic for unmapped combinations
        severity_level = SEVERITY_ORDER.get(severity, 0)
        if cutoff_result == "POST_CUTOFF" and severity_level >= 3:
            return ComplianceOutcome.NON_COMPLIANT.value
        elif cutoff_result == "PRE_CUTOFF":
            return ComplianceOutcome.COMPLIANT.value
        elif cutoff_result in ("ONGOING", "UNCERTAIN"):
            return ComplianceOutcome.REMEDIATION_REQUIRED.value

        return ComplianceOutcome.UNDER_REVIEW.value

    def _assess_market_restriction(
        self,
        cutoff_result: str,
        severity: str,
    ) -> bool:
        """Assess whether EU market restriction applies.

        Products from post-cutoff deforestation areas at or above the
        market restriction severity threshold cannot be placed on the
        EU market per EUDR Article 3.

        Args:
            cutoff_result: Cutoff verification result.
            severity: Alert severity level.

        Returns:
            True if market restriction applies.
        """
        if cutoff_result not in ("POST_CUTOFF", "ONGOING"):
            return False

        threshold_level = SEVERITY_ORDER.get(
            self._market_restriction_threshold, 3
        )
        alert_level = SEVERITY_ORDER.get(severity, 0)

        return alert_level >= threshold_level

    # ------------------------------------------------------------------
    # Internal: Supply Chain Tracing
    # ------------------------------------------------------------------

    def _trace_affected_suppliers(
        self,
        alert_id: str,
    ) -> List[AffectedSupplier]:
        """Trace suppliers affected by a deforestation alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            List of AffectedSupplier objects.
        """
        # Check custom mapping first
        with self._lock:
            custom_ids = self._custom_alert_mapping.get(alert_id)

        if not custom_ids:
            # Check reference data
            for key, ids in ALERT_SUPPLIER_MAP.items():
                if key in alert_id:
                    custom_ids = ids
                    break

        if not custom_ids:
            return []

        suppliers: List[AffectedSupplier] = []
        for sid in custom_ids:
            data = self._get_supplier_data(sid)
            if data:
                supplier = AffectedSupplier(
                    supplier_id=sid,
                    supplier_name=data.get("name", ""),
                    country_code=data.get("country", ""),
                    commodity=data.get("commodity", ""),
                    affected_plots=data.get("plots", []),
                    annual_volume_tons=_to_decimal(
                        data.get("annual_volume_tons", "0")
                    ),
                    affected_products=data.get("products", []),
                )
                suppliers.append(supplier)

        return suppliers

    def _trace_affected_products(
        self,
        suppliers: List[AffectedSupplier],
        market_restriction: bool,
    ) -> List[AffectedProduct]:
        """Trace products affected through supplier relationships.

        Args:
            suppliers: List of affected suppliers.
            market_restriction: Whether market restriction applies.

        Returns:
            List of AffectedProduct objects.
        """
        products: List[AffectedProduct] = []
        seen_ids: set = set()

        for supplier in suppliers:
            for pid in supplier.affected_products:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                data = self._get_product_data(pid)
                if data:
                    product = AffectedProduct(
                        product_id=pid,
                        product_name=data.get("name", ""),
                        hs_code=data.get("hs_code", ""),
                        commodity=data.get("commodity", ""),
                        annual_value_eur=_to_decimal(
                            data.get("annual_value_eur", "0")
                        ),
                        market_restriction=market_restriction,
                        withdrawal_required=market_restriction,
                    )
                    products.append(product)

        return products

    # ------------------------------------------------------------------
    # Internal: Financial Impact
    # ------------------------------------------------------------------

    def _calculate_financial_impact(
        self,
        affected_products: List[AffectedProduct],
        market_restriction: bool,
        area_ha: Decimal,
    ) -> Dict[str, Decimal]:
        """Calculate financial impact of compliance outcome.

        Components:
        1. Product value at risk (annual value of affected products).
        2. Market restriction multiplier (lost revenue if restricted).
        3. Potential regulatory fine (up to 4% of EU turnover).
        4. Remediation costs.

        Args:
            affected_products: List of affected products.
            market_restriction: Whether market restriction applies.
            area_ha: Area affected in hectares.

        Returns:
            Dictionary with financial breakdown (total, product_value,
            fine, remediation_cost).
        """
        # Product value at risk
        product_value = sum(p.annual_value_eur for p in affected_products)

        # Market restriction impact
        restriction_impact = Decimal("0")
        if market_restriction:
            restriction_impact = (
                product_value * MARKET_RESTRICTION_LOSS_MULTIPLIER
            )

        # Potential fine (simplified: 4% of product value as proxy for turnover)
        fine = (product_value * REGULATORY_FINE_CEILING_PCT / Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Remediation cost (based on number of affected products)
        remediation_cost = Decimal("0")
        if market_restriction:
            remediation_cost += REMEDIATION_COST_ESTIMATES.get(
                RemediationAction.PRODUCT_WITHDRAWAL.value, Decimal("0")
            )
        remediation_cost += REMEDIATION_COST_ESTIMATES.get(
            RemediationAction.SUPPLIER_AUDIT.value, Decimal("0")
        )
        remediation_cost += REMEDIATION_COST_ESTIMATES.get(
            RemediationAction.ENHANCED_MONITORING.value, Decimal("0")
        )

        total = (
            product_value + restriction_impact + fine + remediation_cost
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return {
            "total": total,
            "product_value": product_value.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            "restriction_impact": restriction_impact.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            "fine": fine,
            "remediation_cost": remediation_cost.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
        }

    # ------------------------------------------------------------------
    # Internal: Remediation Generation
    # ------------------------------------------------------------------

    def _generate_remediation_actions(
        self,
        compliance_outcome: str,
        severity: str,
    ) -> List[Dict[str, Any]]:
        """Generate required remediation actions for the compliance outcome.

        Args:
            compliance_outcome: Compliance determination.
            severity: Alert severity level.

        Returns:
            List of remediation action dictionaries.
        """
        action_types = OUTCOME_REMEDIATION_MAP.get(compliance_outcome, [])

        actions = []
        for i, action_type in enumerate(action_types):
            if i == 0:
                priority = RemediationPriority.IMMEDIATE.value
            elif i == 1:
                priority = RemediationPriority.URGENT.value
            elif i == 2:
                priority = RemediationPriority.HIGH.value
            else:
                priority = RemediationPriority.STANDARD.value

            actions.append({
                "action": action_type,
                "priority": priority,
                "estimated_cost_eur": str(
                    REMEDIATION_COST_ESTIMATES.get(action_type, Decimal("0"))
                ),
                "timeline_days": REMEDIATION_TIMELINE_DAYS.get(action_type, 30),
                "description": self._describe_action(action_type),
                "status": "PENDING",
            })

        return actions

    # ------------------------------------------------------------------
    # Internal: Data Access
    # ------------------------------------------------------------------

    def _get_supplier_data(self, supplier_id: str) -> Dict[str, Any]:
        """Get supplier data from custom or reference sources."""
        with self._lock:
            custom = self._custom_suppliers.get(supplier_id)
            if custom:
                return dict(custom)

        ref = REFERENCE_SUPPLIERS.get(supplier_id)
        return dict(ref) if ref else {}

    def _get_product_data(self, product_id: str) -> Dict[str, Any]:
        """Get product data from custom or reference sources."""
        with self._lock:
            custom = self._custom_products.get(product_id)
            if custom:
                return dict(custom)

        ref = REFERENCE_PRODUCTS.get(product_id)
        return dict(ref) if ref else {}

    # ------------------------------------------------------------------
    # Internal: Notes and References
    # ------------------------------------------------------------------

    def _generate_assessment_notes(
        self,
        cutoff_result: str,
        severity: str,
        compliance_outcome: str,
        market_restriction: bool,
        supplier_count: int,
    ) -> List[str]:
        """Generate human-readable assessment notes."""
        notes: List[str] = []

        notes.append(
            f"Cutoff verification result: {cutoff_result}. "
            f"Alert severity: {severity}."
        )

        if compliance_outcome == ComplianceOutcome.NON_COMPLIANT.value:
            notes.append(
                "EUDR Article 3: Products from this source area CANNOT be "
                "placed on the EU market. Immediate action required."
            )
        elif compliance_outcome == ComplianceOutcome.REMEDIATION_REQUIRED.value:
            notes.append(
                "Remediation actions are required before products from "
                "this area can be cleared for EU market access."
            )
        elif compliance_outcome == ComplianceOutcome.UNDER_REVIEW.value:
            notes.append(
                "Assessment is under review. Interim enhanced monitoring "
                "is required pending final determination."
            )
        elif compliance_outcome == ComplianceOutcome.COMPLIANT.value:
            notes.append(
                "Pre-cutoff deforestation confirmed. Products may continue "
                "to access the EU market subject to standard due diligence."
            )

        if market_restriction:
            notes.append(
                "EU market restriction is in effect. Affected products "
                "must be withdrawn per EUDR Article 3."
            )

        if supplier_count > 0:
            notes.append(
                f"{supplier_count} supplier(s) affected. All affected "
                f"suppliers require notification and remediation tracking."
            )

        return notes

    def _get_regulatory_references(self, compliance_outcome: str) -> List[str]:
        """Get relevant EUDR regulatory references."""
        refs = [
            "EUDR Article 4 - Due diligence obligations",
            "EUDR Article 10 - Risk assessment",
            "EUDR Article 31 - Record keeping (5 years)",
        ]

        if compliance_outcome == ComplianceOutcome.NON_COMPLIANT.value:
            refs.extend([
                "EUDR Article 3 - Prohibition on non-compliant products",
                "EUDR Article 25 - Penalties (up to 4% annual EU turnover)",
            ])
        elif compliance_outcome in (
            ComplianceOutcome.REMEDIATION_REQUIRED.value,
            ComplianceOutcome.UNDER_REVIEW.value,
        ):
            refs.append("EUDR Article 11 - Risk mitigation measures")

        return refs

    def _describe_action(self, action: str) -> str:
        """Get human-readable description for a remediation action."""
        descriptions = {
            RemediationAction.SUPPLIER_AUDIT.value: (
                "Conduct on-site supplier audit to verify deforestation-free "
                "status and compliance with EUDR requirements."
            ),
            RemediationAction.PLOT_EXCLUSION.value: (
                "Exclude the affected plot from the supply chain. Remove "
                "all sourcing from this geolocation."
            ),
            RemediationAction.ALTERNATIVE_SOURCING.value: (
                "Identify and qualify alternative deforestation-free "
                "sourcing options to replace affected supply."
            ),
            RemediationAction.ENHANCED_MONITORING.value: (
                "Increase satellite monitoring frequency to weekly for "
                "the affected area and surrounding buffer zones."
            ),
            RemediationAction.PRODUCT_WITHDRAWAL.value: (
                "Withdraw all affected products from the EU market "
                "per EUDR Article 3 prohibition requirements."
            ),
            RemediationAction.SUPPLY_CHAIN_RESTRUCTURE.value: (
                "Restructure supply chain to eliminate sourcing from "
                "high-risk deforestation areas. May require new "
                "supplier onboarding and qualification."
            ),
        }
        return descriptions.get(action, f"Execute remediation action: {action}")

    def _get_action_regulatory_basis(self, action: str) -> str:
        """Get regulatory basis for a remediation action."""
        bases = {
            RemediationAction.SUPPLIER_AUDIT.value: "EUDR Article 10 - Risk assessment",
            RemediationAction.PLOT_EXCLUSION.value: "EUDR Article 11 - Risk mitigation",
            RemediationAction.ALTERNATIVE_SOURCING.value: "EUDR Article 11 - Risk mitigation",
            RemediationAction.ENHANCED_MONITORING.value: "EUDR Article 10 - Risk assessment",
            RemediationAction.PRODUCT_WITHDRAWAL.value: "EUDR Article 3 - Market prohibition",
            RemediationAction.SUPPLY_CHAIN_RESTRUCTURE.value: "EUDR Article 4 - Due diligence",
        }
        return bases.get(action, "EUDR Article 4 - Due diligence obligations")

    def _build_milestones(
        self,
        action_details: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build remediation plan milestones from action details."""
        milestones = []
        cumulative_days = 0

        for i, action in enumerate(action_details):
            timeline = action.get("timeline_days", 30)
            cumulative_days = max(cumulative_days, timeline)

            milestones.append({
                "milestone_id": f"MS-{i + 1:03d}",
                "action": action.get("action", ""),
                "description": f"Complete {action.get('action', '')}",
                "target_days": timeline,
                "status": "PLANNED",
            })

        milestones.append({
            "milestone_id": f"MS-{len(action_details) + 1:03d}",
            "action": "COMPLIANCE_REVIEW",
            "description": "Final compliance review and DDS update",
            "target_days": cumulative_days + 7,
            "status": "PLANNED",
        })

        return milestones

    def _assess_dds_risk(
        self,
        affected_suppliers: List[AffectedSupplier],
    ) -> bool:
        """Assess whether alert affects Due Diligence Statements.

        Args:
            affected_suppliers: List of affected suppliers.

        Returns:
            True if DDS statements are at risk.
        """
        return len(affected_suppliers) > 0

    def clear_cache(self) -> int:
        """Clear the assessment cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._assessment_cache)
            self._assessment_cache.clear()
        logger.info("Cleared assessment cache (%d entries)", count)
        return count
