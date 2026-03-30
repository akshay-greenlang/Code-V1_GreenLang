# -*- coding: utf-8 -*-
"""
LabellingComplianceEngine - PACK-020 Battery Passport Engine 6
================================================================

Validates labelling and marking requirements for batteries placed
on the EU market per Articles 13 and 14 of the EU Battery
Regulation (2023/1542).

Articles 13 and 14 of the EU Battery Regulation set out the
labelling, marking, and information requirements that must be met
before a battery can be placed on the internal market.  The
requirements vary by battery category and include CE marking,
QR codes, collection symbols, capacity labels, hazardous substance
declarations, chemistry identifiers, carbon footprint classes,
and manufacturer information.

Battery Categories Covered:
    - Portable Batteries (including button cells)
    - LMT Batteries (Light Means of Transport)
    - SLI Batteries (Starting, Lighting, Ignition)
    - EV Batteries (Electric Vehicle)
    - Industrial Batteries (including stationary storage)

Key Labelling Requirements:
    - Art 13(1): CE marking per Regulation (EC) 768/2008
    - Art 13(2): QR code with battery passport link
    - Art 13(3): Crossed-out wheeled bin (separate collection)
    - Art 13(4): Chemical symbols for Cd, Pb, Hg content
    - Art 13(5): Capacity in Ah or Wh
    - Art 13(6): Battery chemistry label
    - Art 14(1): Carbon footprint declaration (EV/industrial)
    - Art 14(2): Carbon footprint performance class

Regulatory References:
    - EU Regulation 2023/1542 (EU Battery Regulation), Art 13-14
    - Regulation (EC) No 765/2008 (CE marking)
    - Directive 2006/66/EC (legacy battery labelling)
    - ISO 1043 (plastics symbols for battery enclosures)
    - IEC 62902 (secondary lithium battery marking)

Zero-Hallucination:
    - Compliance percentages use deterministic ratio arithmetic
    - Element checks are binary present/absent/incorrect
    - Category-to-requirement mapping uses static lookup tables
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BatteryCategory(str, Enum):
    """Battery category per EU Battery Regulation classification.

    Determines which labelling requirements apply to a given
    battery type.
    """
    PORTABLE = "portable"
    LMT = "lmt"
    SLI = "sli"
    EV = "ev"
    INDUSTRIAL = "industrial"

class LabelElement(str, Enum):
    """Individual labelling element required under Art 13-14.

    Each element represents a specific piece of information or
    marking that may be required on the battery or its packaging.
    """
    CE_MARKING = "ce_marking"
    QR_CODE = "qr_code"
    COLLECTION_SYMBOL = "collection_symbol"
    CAPACITY_LABEL = "capacity_label"
    HAZARDOUS_SUBSTANCE = "hazardous_substance"
    BATTERY_CHEMISTRY = "battery_chemistry"
    CARBON_FOOTPRINT = "carbon_footprint"
    SEPARATE_COLLECTION = "separate_collection"
    MANUFACTURER_INFO = "manufacturer_info"
    COUNTRY_OF_ORIGIN = "country_of_origin"

class LabelStatus(str, Enum):
    """Status of a labelling element on a battery.

    Indicates whether a required label element is correctly
    present, missing, incorrect, or not required for the
    battery category.
    """
    PRESENT = "present"
    MISSING = "missing"
    INCORRECT = "incorrect"
    NOT_REQUIRED = "not_required"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label element descriptions with regulatory references.
LABEL_ELEMENT_DESCRIPTIONS: Dict[str, str] = {
    LabelElement.CE_MARKING.value: (
        "CE conformity marking per Regulation (EC) 765/2008, "
        "indicating conformity with applicable EU harmonised legislation. "
        "Must be at least 5mm in height. Art 13(1)."
    ),
    LabelElement.QR_CODE.value: (
        "QR code providing access to the battery passport data "
        "including manufacturer, battery model, batch, capacity, "
        "and compliance information. Art 13(2)."
    ),
    LabelElement.COLLECTION_SYMBOL.value: (
        "Crossed-out wheeled bin symbol per Annex IX indicating "
        "the battery must be collected separately from mixed "
        "municipal waste. Art 13(3)."
    ),
    LabelElement.CAPACITY_LABEL.value: (
        "Rated capacity expressed in ampere-hours (Ah) or "
        "watt-hours (Wh) as applicable. Art 13(5)."
    ),
    LabelElement.HAZARDOUS_SUBSTANCE.value: (
        "Chemical symbols for cadmium (Cd), lead (Pb), or "
        "mercury (Hg) if concentrations exceed thresholds. "
        "Art 13(4)."
    ),
    LabelElement.BATTERY_CHEMISTRY.value: (
        "Identification of battery chemistry type (e.g., NMC, "
        "LFP, NCA, lead-acid). Art 13(6)."
    ),
    LabelElement.CARBON_FOOTPRINT.value: (
        "Carbon footprint declaration per lifecycle assessment, "
        "expressed as kg CO2e per kWh. Required for EV and "
        "industrial batteries. Art 14(1)."
    ),
    LabelElement.SEPARATE_COLLECTION.value: (
        "Information on separate collection requirements and "
        "available collection schemes. Art 13(3)."
    ),
    LabelElement.MANUFACTURER_INFO.value: (
        "Name, registered trade name or trademark, postal and "
        "email address of the manufacturer and, where applicable, "
        "the importer or distributor. Art 13(7)."
    ),
    LabelElement.COUNTRY_OF_ORIGIN.value: (
        "Country of origin of the battery. Art 13(8)."
    ),
}

# Required label elements by battery category.
# True means required, False means not required for that category.
CATEGORY_REQUIREMENTS: Dict[str, Dict[str, bool]] = {
    BatteryCategory.PORTABLE.value: {
        LabelElement.CE_MARKING.value: True,
        LabelElement.QR_CODE.value: True,
        LabelElement.COLLECTION_SYMBOL.value: True,
        LabelElement.CAPACITY_LABEL.value: True,
        LabelElement.HAZARDOUS_SUBSTANCE.value: True,
        LabelElement.BATTERY_CHEMISTRY.value: True,
        LabelElement.CARBON_FOOTPRINT.value: False,
        LabelElement.SEPARATE_COLLECTION.value: True,
        LabelElement.MANUFACTURER_INFO.value: True,
        LabelElement.COUNTRY_OF_ORIGIN.value: False,
    },
    BatteryCategory.LMT.value: {
        LabelElement.CE_MARKING.value: True,
        LabelElement.QR_CODE.value: True,
        LabelElement.COLLECTION_SYMBOL.value: True,
        LabelElement.CAPACITY_LABEL.value: True,
        LabelElement.HAZARDOUS_SUBSTANCE.value: True,
        LabelElement.BATTERY_CHEMISTRY.value: True,
        LabelElement.CARBON_FOOTPRINT.value: False,
        LabelElement.SEPARATE_COLLECTION.value: True,
        LabelElement.MANUFACTURER_INFO.value: True,
        LabelElement.COUNTRY_OF_ORIGIN.value: True,
    },
    BatteryCategory.SLI.value: {
        LabelElement.CE_MARKING.value: True,
        LabelElement.QR_CODE.value: True,
        LabelElement.COLLECTION_SYMBOL.value: True,
        LabelElement.CAPACITY_LABEL.value: True,
        LabelElement.HAZARDOUS_SUBSTANCE.value: True,
        LabelElement.BATTERY_CHEMISTRY.value: True,
        LabelElement.CARBON_FOOTPRINT.value: False,
        LabelElement.SEPARATE_COLLECTION.value: True,
        LabelElement.MANUFACTURER_INFO.value: True,
        LabelElement.COUNTRY_OF_ORIGIN.value: False,
    },
    BatteryCategory.EV.value: {
        LabelElement.CE_MARKING.value: True,
        LabelElement.QR_CODE.value: True,
        LabelElement.COLLECTION_SYMBOL.value: False,
        LabelElement.CAPACITY_LABEL.value: True,
        LabelElement.HAZARDOUS_SUBSTANCE.value: True,
        LabelElement.BATTERY_CHEMISTRY.value: True,
        LabelElement.CARBON_FOOTPRINT.value: True,
        LabelElement.SEPARATE_COLLECTION.value: False,
        LabelElement.MANUFACTURER_INFO.value: True,
        LabelElement.COUNTRY_OF_ORIGIN.value: True,
    },
    BatteryCategory.INDUSTRIAL.value: {
        LabelElement.CE_MARKING.value: True,
        LabelElement.QR_CODE.value: True,
        LabelElement.COLLECTION_SYMBOL.value: False,
        LabelElement.CAPACITY_LABEL.value: True,
        LabelElement.HAZARDOUS_SUBSTANCE.value: True,
        LabelElement.BATTERY_CHEMISTRY.value: True,
        LabelElement.CARBON_FOOTPRINT.value: True,
        LabelElement.SEPARATE_COLLECTION.value: False,
        LabelElement.MANUFACTURER_INFO.value: True,
        LabelElement.COUNTRY_OF_ORIGIN.value: True,
    },
}

# CE marking minimum size requirements (mm).
CE_MARKING_MIN_HEIGHT_MM: int = 5

# QR code minimum module size (mm) for readability.
QR_CODE_MIN_MODULE_MM: float = 0.5

# Corrective action templates by label element.
CORRECTIVE_ACTIONS: Dict[str, str] = {
    LabelElement.CE_MARKING.value: (
        "Apply CE marking to the battery or packaging at minimum "
        "5mm height per Regulation (EC) 765/2008. Ensure marking "
        "is visible, legible, and indelible."
    ),
    LabelElement.QR_CODE.value: (
        "Generate and apply a QR code linking to the battery passport "
        "data. QR code must be machine-readable and durable for the "
        "battery's expected lifetime."
    ),
    LabelElement.COLLECTION_SYMBOL.value: (
        "Apply the crossed-out wheeled bin symbol per Annex IX. "
        "Symbol must be printed on the battery or packaging and "
        "cover at least 3% of the largest side area."
    ),
    LabelElement.CAPACITY_LABEL.value: (
        "Label the rated capacity in Ah (ampere-hours) or Wh "
        "(watt-hours) as applicable. Capacity must be determined "
        "per the relevant IEC standard."
    ),
    LabelElement.HAZARDOUS_SUBSTANCE.value: (
        "Add chemical symbols (Cd, Pb, Hg) beneath the collection "
        "symbol if the battery contains cadmium >0.002%, lead >0.004%, "
        "or mercury >0.0005% by weight."
    ),
    LabelElement.BATTERY_CHEMISTRY.value: (
        "Add the battery chemistry identification to the label. "
        "Use standard nomenclature (e.g., Li-ion NMC, LFP, NCA, "
        "Pb-acid). Refer to IEC 62902."
    ),
    LabelElement.CARBON_FOOTPRINT.value: (
        "Calculate and declare the carbon footprint per kWh of "
        "energy provided over the battery lifecycle. Follow the "
        "EU Battery Regulation delegated act methodology."
    ),
    LabelElement.SEPARATE_COLLECTION.value: (
        "Include separate collection information indicating "
        "available collection points and return obligations "
        "per Articles 59 and 60."
    ),
    LabelElement.MANUFACTURER_INFO.value: (
        "Include the manufacturer's name, registered trade name, "
        "postal address, and email on the battery or accompanying "
        "documentation."
    ),
    LabelElement.COUNTRY_OF_ORIGIN.value: (
        "Indicate the country of origin of the battery. Use the "
        "ISO 3166-1 alpha-2 country code or full country name."
    ),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class LabelRequirement(BaseModel):
    """A single labelling requirement for a battery category.

    Describes whether a specific label element is required for
    a given battery category and its regulatory basis.
    """
    element: LabelElement = Field(
        ...,
        description="Label element identifier",
    )
    required: bool = Field(
        default=True,
        description="Whether this element is required for the battery category",
    )
    category_applicable: str = Field(
        default="",
        description="Battery category this requirement applies to",
    )
    description: str = Field(
        default="",
        description="Regulatory description of the requirement",
        max_length=2000,
    )

class LabelElementCheck(BaseModel):
    """Result of checking a single label element on a battery.

    Records the status of a label element, whether it meets
    requirements, and any corrective actions needed.
    """
    element: LabelElement = Field(
        ...,
        description="Label element checked",
    )
    required: bool = Field(
        default=True,
        description="Whether this element is required",
    )
    status: LabelStatus = Field(
        default=LabelStatus.MISSING,
        description="Status of the element on the battery",
    )
    compliant: bool = Field(
        default=False,
        description="Whether the element meets compliance requirements",
    )
    description: str = Field(
        default="",
        description="Description of the requirement",
    )
    corrective_action: str = Field(
        default="",
        description="Required corrective action if non-compliant",
        max_length=2000,
    )

class LabelCheckResult(BaseModel):
    """Result of a complete labelling compliance check for a battery.

    Contains the per-element check results, aggregate compliance
    statistics, and actionable recommendations.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this check",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of check (UTC)",
    )
    battery_id: str = Field(
        default="",
        description="Battery identifier checked",
    )
    category: BatteryCategory = Field(
        default=BatteryCategory.PORTABLE,
        description="Battery category",
    )
    elements_checked: int = Field(
        default=0,
        description="Total number of elements checked",
    )
    required_count: int = Field(
        default=0,
        description="Number of required elements for this category",
    )
    compliant_count: int = Field(
        default=0,
        description="Number of compliant elements",
    )
    non_compliant_count: int = Field(
        default=0,
        description="Number of non-compliant required elements",
    )
    compliance_pct: float = Field(
        default=0.0,
        description="Compliance percentage for required elements (0-100)",
    )
    element_checks: List[LabelElementCheck] = Field(
        default_factory=list,
        description="Per-element check results",
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of missing required element names",
    )
    incorrect_elements: List[str] = Field(
        default_factory=list,
        description="List of incorrect element names",
    )
    not_required_elements: List[str] = Field(
        default_factory=list,
        description="Elements not required for this category",
    )
    overall_compliant: bool = Field(
        default=False,
        description="Whether the battery is fully compliant with labelling",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for compliance",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LabellingComplianceEngine:
    """Labelling and marking compliance engine per Art 13-14.

    Provides deterministic, zero-hallucination validation of:
    - Required label elements per battery category
    - Individual element presence and correctness
    - Compliance rate calculation
    - Missing/incorrect element identification
    - Corrective action generation
    - Category-specific requirement lookup

    All checks are deterministic lookup-based.  No LLM is used
    in any compliance determination path.

    Usage::

        engine = LabellingComplianceEngine()
        labels = {
            "ce_marking": "present",
            "qr_code": "present",
            "capacity_label": "missing",
        }
        result = engine.check_labelling(
            battery_id="BAT-001",
            category=BatteryCategory.EV,
            labels=labels,
        )
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise LabellingComplianceEngine."""
        logger.info(
            "LabellingComplianceEngine v%s initialised",
            self.engine_version,
        )

    # ------------------------------------------------------------------ #
    # Full Labelling Compliance Check                                      #
    # ------------------------------------------------------------------ #

    def check_labelling(
        self,
        battery_id: str,
        category: BatteryCategory,
        labels: Dict[str, str],
    ) -> LabelCheckResult:
        """Perform a complete labelling compliance check.

        Validates all required label elements for the given battery
        category against the provided label statuses.

        Args:
            battery_id: Unique battery identifier.
            category: Battery category (determines requirements).
            labels: Dict mapping element name to status string
                    ("present", "missing", "incorrect").

        Returns:
            LabelCheckResult with per-element checks, compliance
            statistics, and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Checking labelling compliance for battery %s, category=%s",
            battery_id,
            category.value,
        )

        required_elements = self.get_required_elements(category)
        element_checks: List[LabelElementCheck] = []
        missing: List[str] = []
        incorrect: List[str] = []
        not_required: List[str] = []
        compliant_count = 0
        non_compliant_count = 0

        for element in LabelElement:
            is_required = element.value in [
                r.element.value for r in required_elements
                if r.required
            ]

            check = self.validate_element(
                element=element,
                category=category,
                label_status=labels.get(element.value, "missing"),
                is_required=is_required,
            )
            element_checks.append(check)

            if not is_required:
                not_required.append(element.value)
            elif check.compliant:
                compliant_count += 1
            else:
                non_compliant_count += 1
                if check.status == LabelStatus.MISSING:
                    missing.append(element.value)
                elif check.status == LabelStatus.INCORRECT:
                    incorrect.append(element.value)

        required_count = compliant_count + non_compliant_count
        compliance_pct = _round2(
            _safe_divide(
                float(compliant_count),
                float(required_count),
                0.0,
            ) * 100.0
        )
        overall_compliant = non_compliant_count == 0

        recommendations = self.generate_corrective_actions(
            missing_elements=missing,
            incorrect_elements=incorrect,
            category=category,
            compliance_pct=compliance_pct,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = LabelCheckResult(
            battery_id=battery_id,
            category=category,
            elements_checked=len(element_checks),
            required_count=required_count,
            compliant_count=compliant_count,
            non_compliant_count=non_compliant_count,
            compliance_pct=compliance_pct,
            element_checks=element_checks,
            missing_elements=missing,
            incorrect_elements=incorrect,
            not_required_elements=not_required,
            overall_compliant=overall_compliant,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Labelling check for %s: %d/%d compliant (%.1f%%), "
            "overall=%s in %.3f ms",
            battery_id,
            compliant_count,
            required_count,
            compliance_pct,
            overall_compliant,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Required Elements per Category                                       #
    # ------------------------------------------------------------------ #

    def get_required_elements(
        self, category: BatteryCategory
    ) -> List[LabelRequirement]:
        """Get the list of required label elements for a battery category.

        Looks up the static category requirements table and returns
        a list of LabelRequirement objects with regulatory descriptions.

        Args:
            category: Battery category to look up.

        Returns:
            List of LabelRequirement objects for the category.
        """
        requirements = CATEGORY_REQUIREMENTS.get(category.value, {})
        result: List[LabelRequirement] = []

        for element in LabelElement:
            is_required = requirements.get(element.value, False)
            description = LABEL_ELEMENT_DESCRIPTIONS.get(
                element.value, ""
            )

            result.append(LabelRequirement(
                element=element,
                required=is_required,
                category_applicable=category.value,
                description=description,
            ))

        return result

    def get_required_element_names(
        self, category: BatteryCategory
    ) -> List[str]:
        """Get the names of required elements for a category.

        Convenience method returning just the element names.

        Args:
            category: Battery category.

        Returns:
            List of required element name strings.
        """
        requirements = CATEGORY_REQUIREMENTS.get(category.value, {})
        return sorted([
            element_name
            for element_name, is_required in requirements.items()
            if is_required
        ])

    # ------------------------------------------------------------------ #
    # Individual Element Validation                                        #
    # ------------------------------------------------------------------ #

    def validate_element(
        self,
        element: LabelElement,
        category: BatteryCategory,
        label_status: str,
        is_required: bool = True,
    ) -> LabelElementCheck:
        """Validate a single label element against requirements.

        Args:
            element: Label element to validate.
            category: Battery category for context.
            label_status: Status string ("present", "missing", "incorrect").
            is_required: Whether this element is required.

        Returns:
            LabelElementCheck with compliance determination.
        """
        # Parse status
        status = self._parse_label_status(label_status)

        # Determine compliance
        if not is_required:
            compliant = True
            corrective = ""
        elif status == LabelStatus.PRESENT:
            compliant = True
            corrective = ""
        elif status == LabelStatus.MISSING:
            compliant = False
            corrective = CORRECTIVE_ACTIONS.get(element.value, "")
        elif status == LabelStatus.INCORRECT:
            compliant = False
            corrective = CORRECTIVE_ACTIONS.get(element.value, "")
        else:
            compliant = True
            corrective = ""

        description = LABEL_ELEMENT_DESCRIPTIONS.get(element.value, "")

        return LabelElementCheck(
            element=element,
            required=is_required,
            status=status if is_required else LabelStatus.NOT_REQUIRED,
            compliant=compliant,
            description=description,
            corrective_action=corrective,
        )

    # ------------------------------------------------------------------ #
    # Corrective Action Generation                                         #
    # ------------------------------------------------------------------ #

    def generate_corrective_actions(
        self,
        missing_elements: List[str],
        incorrect_elements: List[str],
        category: BatteryCategory,
        compliance_pct: float,
    ) -> List[str]:
        """Generate corrective action recommendations.

        Provides prioritised, actionable recommendations based on
        identified labelling non-conformities.

        Args:
            missing_elements: List of missing required element names.
            incorrect_elements: List of incorrect element names.
            category: Battery category.
            compliance_pct: Current compliance percentage.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        total_issues = len(missing_elements) + len(incorrect_elements)

        if total_issues == 0:
            recommendations.append(
                f"All required labelling elements for {category.value} "
                "batteries are present and correct. No corrective actions "
                "required."
            )
            return recommendations

        # Overall compliance warning
        if compliance_pct < 50.0:
            recommendations.append(
                f"CRITICAL: Labelling compliance is {compliance_pct}%. "
                "Battery cannot be placed on the EU market until all "
                "required labelling elements are present and correct."
            )
        elif compliance_pct < 100.0:
            recommendations.append(
                f"Labelling compliance is {compliance_pct}%. "
                f"{total_issues} element(s) require corrective action "
                "before EU market placement."
            )

        # Missing element actions
        for element_name in missing_elements:
            action = CORRECTIVE_ACTIONS.get(element_name, "")
            if action:
                recommendations.append(
                    f"MISSING [{element_name}]: {action}"
                )

        # Incorrect element actions
        for element_name in incorrect_elements:
            action = CORRECTIVE_ACTIONS.get(element_name, "")
            if action:
                recommendations.append(
                    f"INCORRECT [{element_name}]: Review and correct. "
                    f"{action}"
                )

        # CE marking priority
        if LabelElement.CE_MARKING.value in missing_elements:
            recommendations.append(
                "PRIORITY: CE marking is a mandatory legal requirement. "
                "Battery is non-compliant with EU market access rules "
                "without CE marking."
            )

        # QR code priority
        if LabelElement.QR_CODE.value in missing_elements:
            recommendations.append(
                "PRIORITY: QR code with battery passport link is "
                "mandatory. Implement digital product passport "
                "infrastructure."
            )

        # Carbon footprint for EV/industrial
        if (
            LabelElement.CARBON_FOOTPRINT.value in missing_elements
            and category in (BatteryCategory.EV, BatteryCategory.INDUSTRIAL)
        ):
            recommendations.append(
                "PRIORITY: Carbon footprint declaration is mandatory "
                f"for {category.value} batteries per Art 14. Complete "
                "lifecycle carbon footprint assessment using PEFCR methodology."
            )

        return recommendations

    # ------------------------------------------------------------------ #
    # Category Comparison                                                  #
    # ------------------------------------------------------------------ #

    def compare_category_requirements(self) -> Dict[str, Any]:
        """Compare labelling requirements across all battery categories.

        Useful for manufacturers producing multiple battery types
        to understand the full set of labelling obligations.

        Returns:
            Dict with per-category and cross-category comparison.
        """
        comparison: Dict[str, Any] = {
            "categories": {},
            "universal_elements": [],
            "category_specific_elements": {},
        }

        # Per-category requirements
        for category in BatteryCategory:
            required = self.get_required_element_names(category)
            comparison["categories"][category.value] = {
                "required_count": len(required),
                "required_elements": required,
            }

        # Find universal elements (required in all categories)
        all_elements = set(LabelElement)
        universal: List[str] = []
        for element in LabelElement:
            required_in_all = all(
                CATEGORY_REQUIREMENTS.get(cat.value, {}).get(
                    element.value, False
                )
                for cat in BatteryCategory
            )
            if required_in_all:
                universal.append(element.value)

        comparison["universal_elements"] = sorted(universal)

        # Category-specific elements
        for category in BatteryCategory:
            specific: List[str] = []
            reqs = CATEGORY_REQUIREMENTS.get(category.value, {})
            for element_name, is_required in reqs.items():
                if is_required and element_name not in universal:
                    specific.append(element_name)
            comparison["category_specific_elements"][
                category.value
            ] = sorted(specific)

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Element Detail Lookup                                                #
    # ------------------------------------------------------------------ #

    def get_element_description(
        self, element: LabelElement
    ) -> Dict[str, Any]:
        """Get detailed information about a label element.

        Args:
            element: Label element to look up.

        Returns:
            Dict with element details and regulatory references.
        """
        description = LABEL_ELEMENT_DESCRIPTIONS.get(
            element.value, ""
        )
        corrective = CORRECTIVE_ACTIONS.get(element.value, "")

        # Determine which categories require this element
        applicable_categories: List[str] = []
        for category in BatteryCategory:
            reqs = CATEGORY_REQUIREMENTS.get(category.value, {})
            if reqs.get(element.value, False):
                applicable_categories.append(category.value)

        return {
            "element": element.value,
            "description": description,
            "corrective_action_template": corrective,
            "applicable_categories": applicable_categories,
            "total_categories_applicable": len(applicable_categories),
            "provenance_hash": _compute_hash({
                "element": element.value,
                "categories": applicable_categories,
            }),
        }

    def get_all_element_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get descriptions for all label elements.

        Returns:
            Dict mapping element name to detail dict.
        """
        return {
            element.value: self.get_element_description(element)
            for element in LabelElement
        }

    # ------------------------------------------------------------------ #
    # Batch Checking                                                       #
    # ------------------------------------------------------------------ #

    def check_labelling_batch(
        self,
        batteries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check labelling compliance for a batch of batteries.

        Args:
            batteries: List of dicts, each with "battery_id", "category",
                       and "labels" keys.

        Returns:
            Dict with per-battery results and batch summary.
        """
        t0 = time.perf_counter()
        results: List[LabelCheckResult] = []

        for battery in batteries:
            battery_id = battery.get("battery_id", _new_uuid())
            category_str = battery.get("category", "portable")
            labels = battery.get("labels", {})

            try:
                category = BatteryCategory(category_str.lower())
            except ValueError:
                category = BatteryCategory.PORTABLE

            result = self.check_labelling(
                battery_id=battery_id,
                category=category,
                labels=labels,
            )
            results.append(result)

        # Aggregate statistics
        total = len(results)
        fully_compliant = sum(1 for r in results if r.overall_compliant)
        avg_compliance = _round2(
            _safe_divide(
                sum(r.compliance_pct for r in results),
                float(total),
                0.0,
            )
        )

        # Most common missing elements
        missing_counts: Dict[str, int] = {}
        for r in results:
            for elem in r.missing_elements:
                missing_counts[elem] = missing_counts.get(elem, 0) + 1

        sorted_missing = sorted(
            missing_counts.items(), key=lambda x: x[1], reverse=True
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        batch_result: Dict[str, Any] = {
            "total_batteries": total,
            "fully_compliant": fully_compliant,
            "non_compliant": total - fully_compliant,
            "batch_compliance_rate": _round2(
                _safe_divide(float(fully_compliant), float(total), 0.0)
                * 100.0
            ),
            "average_element_compliance": avg_compliance,
            "most_common_missing": [
                {"element": name, "count": count}
                for name, count in sorted_missing[:5]
            ],
            "results": [
                {
                    "battery_id": r.battery_id,
                    "category": r.category.value,
                    "compliance_pct": r.compliance_pct,
                    "overall_compliant": r.overall_compliant,
                    "missing_count": len(r.missing_elements),
                    "incorrect_count": len(r.incorrect_elements),
                }
                for r in results
            ],
            "processing_time_ms": elapsed_ms,
        }

        batch_result["provenance_hash"] = _compute_hash(batch_result)

        logger.info(
            "Batch labelling check: %d batteries, %d/%d compliant "
            "(%.1f%%) in %.3f ms",
            total,
            fully_compliant,
            total,
            batch_result["batch_compliance_rate"],
            elapsed_ms,
        )
        return batch_result

    # ------------------------------------------------------------------ #
    # Summary Utilities                                                    #
    # ------------------------------------------------------------------ #

    def get_compliance_summary(
        self, result: LabelCheckResult
    ) -> Dict[str, Any]:
        """Return a structured summary of a labelling check result.

        Args:
            result: LabelCheckResult to summarise.

        Returns:
            Dict with compliance summary.
        """
        return {
            "battery_id": result.battery_id,
            "category": result.category.value,
            "overall_compliant": result.overall_compliant,
            "compliance_pct": result.compliance_pct,
            "required_count": result.required_count,
            "compliant_count": result.compliant_count,
            "non_compliant_count": result.non_compliant_count,
            "missing_elements": result.missing_elements,
            "incorrect_elements": result.incorrect_elements,
            "recommendation_count": len(result.recommendations),
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _parse_label_status(self, status_str: str) -> LabelStatus:
        """Parse a status string to LabelStatus enum.

        Handles case-insensitive matching and common aliases.

        Args:
            status_str: Status string to parse.

        Returns:
            LabelStatus enum value.
        """
        normalised = status_str.strip().lower()

        status_map: Dict[str, LabelStatus] = {
            "present": LabelStatus.PRESENT,
            "yes": LabelStatus.PRESENT,
            "true": LabelStatus.PRESENT,
            "ok": LabelStatus.PRESENT,
            "compliant": LabelStatus.PRESENT,
            "missing": LabelStatus.MISSING,
            "no": LabelStatus.MISSING,
            "false": LabelStatus.MISSING,
            "absent": LabelStatus.MISSING,
            "incorrect": LabelStatus.INCORRECT,
            "wrong": LabelStatus.INCORRECT,
            "error": LabelStatus.INCORRECT,
            "invalid": LabelStatus.INCORRECT,
            "not_required": LabelStatus.NOT_REQUIRED,
            "n/a": LabelStatus.NOT_REQUIRED,
            "na": LabelStatus.NOT_REQUIRED,
        }

        return status_map.get(normalised, LabelStatus.MISSING)
