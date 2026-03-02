# -*- coding: utf-8 -*-
"""
TaxonomyMapper - EFRAG ESRS XBRL Taxonomy Mapper

This module implements the XBRL taxonomy mapper for CSRD/ESRS reporting.
It loads the EFRAG ESRS taxonomy from JSON and provides lookup, mapping,
context generation, unit resolution, and filing indicator management.

The mapper follows a thread-safe singleton pattern using RLock so that
a single taxonomy instance is shared across all agents and threads.

Key Capabilities:
    - Map ESRS data point IDs to XBRL element QNames
    - Support all 12 ESRS standards (ESRS-1, ESRS-2, E1-E5, S1-S4, G1)
    - Handle typed dimensions (country, region) and explicit dimensions
    - Generate context references for dimensional data
    - Map units (ISO 4217 currencies, tCO2e, MWh, m3, ha, %)
    - Filing indicator management per standard
    - Multi-language label resolution (en, de, fr, es)

Version: 1.1.0
Author: GreenLang CSRD Team
License: MIT
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAXONOMY_DATA_DIR = Path(__file__).parent / "taxonomy_data"
TAXONOMY_FILE = TAXONOMY_DATA_DIR / "efrag_esrs_taxonomy.json"
FILING_INDICATORS_FILE = TAXONOMY_DATA_DIR / "filing_indicators.json"
CALCULATION_LINKBASE_FILE = TAXONOMY_DATA_DIR / "calculation_linkbase.json"

SUPPORTED_LANGUAGES = ("en", "de", "fr", "es")

SUPPORTED_STANDARDS = (
    "ESRS-1", "ESRS-2",
    "ESRS-E1", "ESRS-E2", "ESRS-E3", "ESRS-E4", "ESRS-E5",
    "ESRS-S1", "ESRS-S2", "ESRS-S3", "ESRS-S4",
    "ESRS-G1",
)

# ISO 4217 currency codes commonly used in ESEF
ISO_4217_CURRENCIES: FrozenSet[str] = frozenset({
    "EUR", "USD", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK",
    "HUF", "RON", "BGN", "HRK", "JPY", "CNY", "AUD", "CAD", "BRL",
    "INR", "KRW", "MXN", "SGD", "HKD", "TWD", "ZAR", "TRY", "RUB",
    "NZD", "THB", "IDR", "MYR", "PHP", "AED", "SAR", "ILS", "CLP",
    "COP", "PEN", "ARS",
})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PeriodType(str, Enum):
    """XBRL context period type."""
    INSTANT = "instant"
    DURATION = "duration"


class BalanceType(str, Enum):
    """XBRL balance type for monetary / numeric items."""
    DEBIT = "debit"
    CREDIT = "credit"


class DimensionType(str, Enum):
    """Explicit vs. typed dimension."""
    EXPLICIT = "explicit"
    TYPED = "typed"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DimensionMember(BaseModel):
    """Single member of an explicit dimension domain."""

    member_id: str = Field(..., description="XBRL member QName, e.g. esrs:Scope1Member")
    label: str = Field(..., description="Human-readable label")


class DimensionInfo(BaseModel):
    """Describes an XBRL dimension (explicit or typed)."""

    dimension_id: str = Field(..., description="Dimension QName, e.g. esrs:GHGScopeDimension")
    dimension_type: DimensionType = Field(..., description="explicit or typed")
    is_typed: bool = Field(False, description="True for typed dimensions")
    domain: Optional[str] = Field(None, description="Domain QName for explicit dims")
    members: Optional[Dict[str, DimensionMember]] = Field(
        None, description="Member map for explicit dims"
    )
    typed_domain_ref: Optional[str] = Field(None, description="Typed domain ref for typed dims")
    data_type: Optional[str] = Field(None, description="Data type for typed dimension values")
    description: Optional[str] = Field(None, description="Human-readable description")


class TaxonomyElement(BaseModel):
    """Represents a single XBRL taxonomy element."""

    element_id: str = Field(..., description="Fully-qualified XBRL element QName")
    name: str = Field(..., description="Local element name")
    namespace: str = Field(..., description="Namespace prefix (e.g. esrs-e1)")
    data_type: str = Field(..., description="XBRL data type (e.g. xbrli:monetaryItemType)")
    period_type: PeriodType = Field(..., description="instant or duration")
    abstract: bool = Field(False, description="True if element is abstract")
    nillable: bool = Field(True, description="True if element may be nil")
    substitution_group: str = Field(
        "xbrli:item", description="Substitution group QName"
    )
    balance_type: Optional[BalanceType] = Field(
        None, description="debit or credit for monetary items"
    )
    standard: str = Field(..., description="Owning ESRS standard (e.g. ESRS-E1)")
    disclosure_requirement: str = Field(
        ..., description="Disclosure requirement ID (e.g. E1-6)"
    )
    dimensions: Optional[List[str]] = Field(
        None, description="Applicable dimension keys"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Multi-language labels {lang: text}"
    )

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Ensure at least an English label exists."""
        if v and "en" not in v:
            logger.warning("Taxonomy element missing English label")
        return v

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_label(self, lang: str = "en") -> str:
        """Return the label in the requested language, falling back to English."""
        return self.labels.get(lang, self.labels.get("en", self.name))

    @property
    def qname(self) -> str:
        """Return the fully-qualified QName (namespace:name)."""
        return f"{self.namespace}:{self.name}"

    @property
    def is_numeric(self) -> bool:
        """Return True if the element holds a numeric value."""
        numeric_types = {
            "xbrli:monetaryItemType",
            "xbrli:decimalItemType",
            "xbrli:integerItemType",
            "xbrli:pureItemType",
            "esrs:tCO2eItemType",
            "esrs:MWhItemType",
            "esrs:cubicMetreItemType",
            "esrs:hectareItemType",
            "esrs:tonnesItemType",
            "esrs:tCO2ePerRevenueItemType",
            "esrs:MWhPerRevenueItemType",
        }
        return self.data_type in numeric_types

    @property
    def is_monetary(self) -> bool:
        """Return True if the element is a monetary item."""
        return self.data_type == "xbrli:monetaryItemType"

    @property
    def is_text(self) -> bool:
        """Return True if the element is a text (string) item."""
        return self.data_type == "xbrli:stringItemType"

    @property
    def is_boolean(self) -> bool:
        """Return True if the element is boolean."""
        return self.data_type == "xbrli:booleanItemType"

    @property
    def is_percentage(self) -> bool:
        """Return True if the element represents a pure / percentage value."""
        return self.data_type == "xbrli:pureItemType"


class UnitDefinition(BaseModel):
    """XBRL unit definition."""

    unit_id: str = Field(..., description="Unit reference ID")
    measure: Optional[str] = Field(None, description="Single measure QName")
    label: str = Field(..., description="Human-readable label")
    item_type: str = Field(..., description="Matching item type")
    numerator: Optional[str] = Field(None, description="Numerator for divide units")
    denominator: Optional[str] = Field(None, description="Denominator for divide units")
    is_divide: bool = Field(False, description="True if unit is a divide (ratio)")


class FilingIndicator(BaseModel):
    """Filing indicator for a single ESRS standard."""

    standard_id: str = Field(..., description="Standard ID (e.g. ESRS-E1)")
    full_name: str = Field(..., description="Full standard name")
    mandatory: bool = Field(..., description="True if always required")
    materiality_subject: bool = Field(..., description="True if subject to materiality")
    description: str = Field(..., description="Short description")
    filing_code: str = Field(..., description="Filing code (e.g. FI_E1)")
    minimum_disclosures: List[str] = Field(
        default_factory=list, description="Minimum disclosure requirement IDs"
    )
    mandatory_if_material: Optional[List[str]] = Field(
        None, description="Mandatory DRs when standard is material"
    )
    omission_explanation_required: bool = Field(
        False, description="True if omission needs explanation"
    )
    notes: Optional[str] = Field(None, description="Additional notes")


class CalculationRelationship(BaseModel):
    """A calculation linkbase relationship (parent -> children with weights)."""

    parent: str = Field(..., description="Parent data point ID")
    parent_element: str = Field(..., description="Parent XBRL element QName")
    standard: str = Field(..., description="ESRS standard")
    disclosure_requirement: str = Field(..., description="Disclosure requirement")
    children: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of {child, element, weight} dicts"
    )
    calculation_type: Optional[str] = Field(
        None, description="sum (default) or ratio"
    )
    variant: Optional[str] = Field(None, description="Variant label, e.g. market_based")
    numerator: Optional[str] = Field(None, description="For ratio: numerator element ID")
    denominator: Optional[str] = Field(None, description="For ratio: denominator element ID")
    description: str = Field("", description="Human-readable description")


class ContextDimension(BaseModel):
    """A single dimension/member pair for an XBRL context."""

    dimension_id: str = Field(..., description="Dimension QName")
    member_id: Optional[str] = Field(None, description="Member QName (explicit)")
    typed_value: Optional[str] = Field(None, description="Typed dimension value")
    is_typed: bool = Field(False, description="True for typed dimension")


class ContextSpec(BaseModel):
    """Specification for building an XBRL context element."""

    entity_scheme: str = Field(
        "http://standards.iso.org/iso/17442",
        description="Entity identifier scheme (LEI default)",
    )
    entity_identifier: str = Field(..., description="Entity identifier (LEI)")
    period_type: PeriodType = Field(..., description="instant or duration")
    instant_date: Optional[str] = Field(None, description="YYYY-MM-DD for instant")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD for duration start")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD for duration end")
    dimensions: List[ContextDimension] = Field(
        default_factory=list, description="Scenario dimensions"
    )


# ---------------------------------------------------------------------------
# TaxonomyMapper (Thread-Safe Singleton)
# ---------------------------------------------------------------------------

class TaxonomyMapper:
    """
    EFRAG ESRS XBRL Taxonomy Mapper.

    Thread-safe singleton that loads the EFRAG ESRS taxonomy once and provides
    fast lookups for elements, dimensions, units, filing indicators, and
    calculation relationships.

    Usage::

        mapper = TaxonomyMapper.get_instance()
        element = mapper.get_element("E1_E16_001")
        ctx_id = mapper.get_context_ref({"ghg_scope": "scope_1"})
        unit = mapper.get_unit_ref("tCO2e")

    Attributes:
        _elements: Dict of data_point_id -> TaxonomyElement
        _dimensions: Dict of dimension_key -> DimensionInfo
        _units: Dict of unit_key -> UnitDefinition
        _filing_indicators: Dict of standard_id -> FilingIndicator
        _calculation_relationships: Dict of rel_key -> CalculationRelationship
        _namespaces: Dict of prefix -> URI
    """

    _instance: Optional["TaxonomyMapper"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "TaxonomyMapper":
        """Enforce singleton via __new__."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize taxonomy data (only once)."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._elements: Dict[str, TaxonomyElement] = {}
            self._elements_by_qname: Dict[str, TaxonomyElement] = {}
            self._elements_by_standard: Dict[str, List[str]] = {}
            self._dimensions: Dict[str, DimensionInfo] = {}
            self._units: Dict[str, UnitDefinition] = {}
            self._filing_indicators: Dict[str, FilingIndicator] = {}
            self._calculation_relationships: Dict[str, CalculationRelationship] = {}
            self._namespaces: Dict[str, str] = {}
            self._presentation_tree: Dict[str, Any] = {}
            self._custom_types: Dict[str, Any] = {}
            self._context_cache: Dict[str, str] = {}
            self._load_taxonomy()
            self._load_filing_indicators()
            self._load_calculation_linkbase()
            self._initialized = True
            logger.info(
                "TaxonomyMapper initialized: %d elements, %d dimensions, "
                "%d units, %d filing indicators, %d calculation relationships",
                len(self._elements),
                len(self._dimensions),
                len(self._units),
                len(self._filing_indicators),
                len(self._calculation_relationships),
            )

    # ------------------------------------------------------------------
    # Class-level accessors
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "TaxonomyMapper":
        """Return the singleton instance, creating it if necessary."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Data loading (private)
    # ------------------------------------------------------------------

    def _load_taxonomy(self) -> None:
        """Load the EFRAG ESRS taxonomy from JSON."""
        start = datetime.now()
        try:
            with open(TAXONOMY_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            logger.error("Taxonomy file not found: %s", TAXONOMY_FILE)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Invalid taxonomy JSON: %s", exc)
            raise

        self._namespaces = data.get("namespaces", {})
        self._custom_types = data.get("custom_types", {})
        self._presentation_tree = data.get("presentation_tree", {})

        # Load dimensions
        for dim_key, dim_data in data.get("dimensions", {}).items():
            members = None
            if dim_data.get("members"):
                members = {
                    mk: DimensionMember(**mv)
                    for mk, mv in dim_data["members"].items()
                }
            self._dimensions[dim_key] = DimensionInfo(
                dimension_id=dim_data["dimension_id"],
                dimension_type=DimensionType(dim_data["dimension_type"]),
                is_typed=dim_data.get("is_typed", False),
                domain=dim_data.get("domain"),
                members=members,
                typed_domain_ref=dim_data.get("typed_domain_ref"),
                data_type=dim_data.get("data_type"),
                description=dim_data.get("description"),
            )

        # Load elements
        for dp_id, elem_data in data.get("elements", {}).items():
            balance = None
            if elem_data.get("balance_type"):
                balance = BalanceType(elem_data["balance_type"])

            element = TaxonomyElement(
                element_id=elem_data["element_id"],
                name=elem_data["name"],
                namespace=elem_data["namespace"],
                data_type=elem_data["data_type"],
                period_type=PeriodType(elem_data["period_type"]),
                abstract=elem_data.get("abstract", False),
                nillable=elem_data.get("nillable", True),
                substitution_group=elem_data.get("substitution_group", "xbrli:item"),
                balance_type=balance,
                standard=elem_data["standard"],
                disclosure_requirement=elem_data.get("disclosure_requirement", ""),
                dimensions=elem_data.get("dimensions"),
                labels=elem_data.get("labels", {}),
            )
            self._elements[dp_id] = element
            self._elements_by_qname[element.qname] = element

            std = element.standard
            self._elements_by_standard.setdefault(std, []).append(dp_id)

        # Load units
        for unit_key, unit_data in data.get("units", {}).items():
            self._units[unit_key] = UnitDefinition(
                unit_id=unit_data["unit_id"],
                measure=unit_data.get("measure"),
                label=unit_data["label"],
                item_type=unit_data["item_type"],
                numerator=unit_data.get("numerator"),
                denominator=unit_data.get("denominator"),
                is_divide=unit_data.get("is_divide", False),
            )

        elapsed = (datetime.now() - start).total_seconds() * 1000
        logger.info("Taxonomy loaded in %.1f ms", elapsed)

    def _load_filing_indicators(self) -> None:
        """Load filing indicators from JSON."""
        try:
            with open(FILING_INDICATORS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            logger.warning("Filing indicators file not found: %s", FILING_INDICATORS_FILE)
            return
        except json.JSONDecodeError as exc:
            logger.error("Invalid filing indicators JSON: %s", exc)
            return

        for std_id, fi_data in data.get("filing_indicators", {}).items():
            self._filing_indicators[std_id] = FilingIndicator(
                standard_id=fi_data["standard_id"],
                full_name=fi_data["full_name"],
                mandatory=fi_data["mandatory"],
                materiality_subject=fi_data["materiality_subject"],
                description=fi_data["description"],
                filing_code=fi_data["filing_code"],
                minimum_disclosures=fi_data.get("minimum_disclosures", []),
                mandatory_if_material=fi_data.get("mandatory_if_material"),
                omission_explanation_required=fi_data.get(
                    "omission_explanation_required", False
                ),
                notes=fi_data.get("notes"),
            )

    def _load_calculation_linkbase(self) -> None:
        """Load calculation linkbase from JSON."""
        try:
            with open(CALCULATION_LINKBASE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            logger.warning(
                "Calculation linkbase file not found: %s", CALCULATION_LINKBASE_FILE
            )
            return
        except json.JSONDecodeError as exc:
            logger.error("Invalid calculation linkbase JSON: %s", exc)
            return

        for rel_key, rel_data in data.get("calculation_relationships", {}).items():
            self._calculation_relationships[rel_key] = CalculationRelationship(
                parent=rel_data["parent"],
                parent_element=rel_data["parent_element"],
                standard=rel_data["standard"],
                disclosure_requirement=rel_data["disclosure_requirement"],
                children=rel_data.get("children"),
                calculation_type=rel_data.get("calculation_type"),
                variant=rel_data.get("variant"),
                numerator=rel_data.get("numerator"),
                denominator=rel_data.get("denominator"),
                description=rel_data.get("description", ""),
            )

    # ------------------------------------------------------------------
    # Element lookups
    # ------------------------------------------------------------------

    def get_element(self, data_point_id: str) -> Optional[TaxonomyElement]:
        """
        Look up a taxonomy element by its data point ID.

        Args:
            data_point_id: ESRS data point identifier (e.g. 'E1_E16_001').

        Returns:
            TaxonomyElement or None if not found.
        """
        element = self._elements.get(data_point_id)
        if element is None:
            logger.debug("Element not found for data point ID: %s", data_point_id)
        return element

    def get_element_by_qname(self, qname: str) -> Optional[TaxonomyElement]:
        """
        Look up a taxonomy element by its QName.

        Args:
            qname: Qualified name (e.g. 'esrs-e1:GrossScope1GHGEmissions').

        Returns:
            TaxonomyElement or None if not found.
        """
        return self._elements_by_qname.get(qname)

    def get_elements_by_standard(self, standard: str) -> List[TaxonomyElement]:
        """
        Return all elements belonging to a given ESRS standard.

        Args:
            standard: Standard identifier (e.g. 'ESRS-E1').

        Returns:
            List of TaxonomyElement objects (may be empty).
        """
        dp_ids = self._elements_by_standard.get(standard, [])
        return [self._elements[dp] for dp in dp_ids if dp in self._elements]

    def get_all_element_ids(self) -> List[str]:
        """Return all registered data point IDs."""
        return list(self._elements.keys())

    def get_all_elements(self) -> Dict[str, TaxonomyElement]:
        """Return the complete element dictionary (read-only copy)."""
        return dict(self._elements)

    def element_exists(self, data_point_id: str) -> bool:
        """Return True if the data point ID exists in the taxonomy."""
        return data_point_id in self._elements

    def get_element_label(
        self, data_point_id: str, lang: str = "en"
    ) -> Optional[str]:
        """
        Return the label for a data point in the requested language.

        Args:
            data_point_id: ESRS data point identifier.
            lang: ISO 639-1 language code (default 'en').

        Returns:
            Label string or None if element not found.
        """
        element = self._elements.get(data_point_id)
        if element is None:
            return None
        return element.get_label(lang)

    # ------------------------------------------------------------------
    # Dimension handling
    # ------------------------------------------------------------------

    def get_dimension(self, dimension_key: str) -> Optional[DimensionInfo]:
        """
        Look up a dimension definition by its key.

        Args:
            dimension_key: Dimension key (e.g. 'ghg_scope', 'gender').

        Returns:
            DimensionInfo or None.
        """
        return self._dimensions.get(dimension_key)

    def get_all_dimensions(self) -> Dict[str, DimensionInfo]:
        """Return all dimension definitions."""
        return dict(self._dimensions)

    def get_dimension_members(
        self, dimension_key: str
    ) -> Optional[Dict[str, DimensionMember]]:
        """
        Return the members of an explicit dimension.

        Args:
            dimension_key: Dimension key.

        Returns:
            Dict of member_key -> DimensionMember, or None.
        """
        dim = self._dimensions.get(dimension_key)
        if dim is None or dim.is_typed:
            return None
        return dim.members

    def get_member_id(
        self, dimension_key: str, member_key: str
    ) -> Optional[str]:
        """
        Resolve a member key to its XBRL member QName.

        Args:
            dimension_key: Dimension key (e.g. 'ghg_scope').
            member_key: Member key (e.g. 'scope_1').

        Returns:
            QName string (e.g. 'esrs:Scope1Member') or None.
        """
        dim = self._dimensions.get(dimension_key)
        if dim is None or dim.members is None:
            return None
        member = dim.members.get(member_key)
        return member.member_id if member else None

    def validate_dimension_member(
        self, dimension_key: str, member_key: str
    ) -> bool:
        """
        Check whether a member key is valid for a given dimension.

        Args:
            dimension_key: Dimension key.
            member_key: Member key to validate.

        Returns:
            True if the member is valid.
        """
        dim = self._dimensions.get(dimension_key)
        if dim is None:
            return False
        if dim.is_typed:
            return True  # typed dimensions accept any conformant value
        if dim.members is None:
            return False
        return member_key in dim.members

    def get_applicable_dimensions(
        self, data_point_id: str
    ) -> List[DimensionInfo]:
        """
        Return the dimensions applicable to a given data point.

        Args:
            data_point_id: ESRS data point identifier.

        Returns:
            List of DimensionInfo objects (may be empty).
        """
        element = self._elements.get(data_point_id)
        if element is None or element.dimensions is None:
            return []
        result: List[DimensionInfo] = []
        for dim_key in element.dimensions:
            dim = self._dimensions.get(dim_key)
            if dim is not None:
                result.append(dim)
        return result

    # ------------------------------------------------------------------
    # Context reference generation
    # ------------------------------------------------------------------

    def get_context_ref(
        self,
        dimensions: Optional[Dict[str, str]] = None,
        period_type: PeriodType = PeriodType.DURATION,
        instant_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Generate a deterministic context reference ID from dimensions and period.

        The ID is a stable hash so that identical dimension combinations always
        produce the same context ID. This avoids duplicate context elements in
        the iXBRL output.

        Args:
            dimensions: Dict of dimension_key -> member_key (or typed value).
            period_type: instant or duration.
            instant_date: Date string for instant periods.
            start_date: Start date for duration periods.
            end_date: End date for duration periods.

        Returns:
            Context reference ID string (e.g. 'ctx_abc123').
        """
        parts: List[str] = [period_type.value]

        if period_type == PeriodType.INSTANT and instant_date:
            parts.append(f"i:{instant_date}")
        elif period_type == PeriodType.DURATION:
            if start_date:
                parts.append(f"s:{start_date}")
            if end_date:
                parts.append(f"e:{end_date}")

        if dimensions:
            for dim_key in sorted(dimensions.keys()):
                parts.append(f"{dim_key}={dimensions[dim_key]}")

        cache_key = "|".join(parts)
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
        ctx_id = f"ctx_{digest}"
        self._context_cache[cache_key] = ctx_id
        return ctx_id

    def build_context_spec(
        self,
        entity_identifier: str,
        period_type: PeriodType,
        instant_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> ContextSpec:
        """
        Build a full ContextSpec object for XBRL context creation.

        Args:
            entity_identifier: LEI or other entity ID.
            period_type: instant or duration.
            instant_date: Date for instant periods.
            start_date: Start for duration periods.
            end_date: End for duration periods.
            dimensions: Dict of dimension_key -> member_key / typed value.

        Returns:
            ContextSpec ready for iXBRL generation.
        """
        ctx_dims: List[ContextDimension] = []
        if dimensions:
            for dim_key, member_or_value in dimensions.items():
                dim_info = self._dimensions.get(dim_key)
                if dim_info is None:
                    logger.warning("Unknown dimension key: %s", dim_key)
                    continue

                if dim_info.is_typed:
                    ctx_dims.append(
                        ContextDimension(
                            dimension_id=dim_info.dimension_id,
                            typed_value=member_or_value,
                            is_typed=True,
                        )
                    )
                else:
                    member_id = self.get_member_id(dim_key, member_or_value)
                    if member_id is None:
                        logger.warning(
                            "Unknown member '%s' for dimension '%s'",
                            member_or_value,
                            dim_key,
                        )
                        continue
                    ctx_dims.append(
                        ContextDimension(
                            dimension_id=dim_info.dimension_id,
                            member_id=member_id,
                            is_typed=False,
                        )
                    )

        return ContextSpec(
            entity_identifier=entity_identifier,
            period_type=period_type,
            instant_date=instant_date,
            start_date=start_date,
            end_date=end_date,
            dimensions=ctx_dims,
        )

    # ------------------------------------------------------------------
    # Unit handling
    # ------------------------------------------------------------------

    def get_unit_ref(self, unit_key: str) -> Optional[str]:
        """
        Return the unit reference ID for a given unit key.

        Supports standard keys (e.g. 'tCO2e', 'MWh', 'EUR') as well as
        ISO 4217 currency codes.

        Args:
            unit_key: Unit key from the taxonomy or an ISO 4217 code.

        Returns:
            Unit reference ID string or None.
        """
        unit = self._units.get(unit_key)
        if unit is not None:
            return unit.unit_id

        # Try as ISO 4217 currency code
        upper = unit_key.upper()
        if upper in ISO_4217_CURRENCIES:
            return f"iso4217:{upper}"

        logger.debug("Unit not found for key: %s", unit_key)
        return None

    def get_unit_definition(self, unit_key: str) -> Optional[UnitDefinition]:
        """
        Return the full unit definition for a unit key.

        Args:
            unit_key: Unit key.

        Returns:
            UnitDefinition or None.
        """
        return self._units.get(unit_key)

    def get_all_units(self) -> Dict[str, UnitDefinition]:
        """Return all registered unit definitions."""
        return dict(self._units)

    def get_unit_for_element(self, data_point_id: str) -> Optional[str]:
        """
        Determine the appropriate unit reference for a given data point.

        Args:
            data_point_id: ESRS data point identifier.

        Returns:
            Unit reference ID or None for non-numeric elements.
        """
        element = self._elements.get(data_point_id)
        if element is None:
            return None
        if not element.is_numeric:
            return None

        # Map data_type -> unit key
        type_to_unit: Dict[str, str] = {
            "esrs:tCO2eItemType": "tCO2e",
            "esrs:MWhItemType": "MWh",
            "esrs:cubicMetreItemType": "m3",
            "esrs:hectareItemType": "ha",
            "esrs:tonnesItemType": "tonnes",
            "esrs:tCO2ePerRevenueItemType": "tCO2e_per_revenue",
            "esrs:MWhPerRevenueItemType": "MWh_per_revenue",
            "xbrli:pureItemType": "pure",
        }

        unit_key = type_to_unit.get(element.data_type)
        if unit_key:
            return self.get_unit_ref(unit_key)

        # Monetary items need a currency (default EUR)
        if element.is_monetary:
            return "iso4217:EUR"

        # Integer / decimal without specific unit
        if element.data_type in ("xbrli:integerItemType", "xbrli:decimalItemType"):
            return "xbrli:pure"

        return None

    # ------------------------------------------------------------------
    # Filing indicator management
    # ------------------------------------------------------------------

    def get_filing_indicator(self, standard_id: str) -> Optional[FilingIndicator]:
        """
        Return the filing indicator for a given standard.

        Args:
            standard_id: Standard identifier (e.g. 'ESRS-E1').

        Returns:
            FilingIndicator or None.
        """
        return self._filing_indicators.get(standard_id)

    def get_all_filing_indicators(self) -> Dict[str, FilingIndicator]:
        """Return all filing indicators."""
        return dict(self._filing_indicators)

    def get_mandatory_standards(self) -> List[str]:
        """Return IDs of standards that are always mandatory."""
        return [
            fi.standard_id
            for fi in self._filing_indicators.values()
            if fi.mandatory
        ]

    def get_material_standards(
        self, material_topics: Set[str]
    ) -> List[str]:
        """
        Given a set of material topic IDs, return which standards must be filed.

        The mandatory standards (ESRS-1, ESRS-2) are always included.

        Args:
            material_topics: Set of standard IDs determined to be material
                             (e.g. {'ESRS-E1', 'ESRS-S1', 'ESRS-G1'}).

        Returns:
            Sorted list of standard IDs to file.
        """
        result: Set[str] = set()
        for fi in self._filing_indicators.values():
            if fi.mandatory:
                result.add(fi.standard_id)
            elif fi.standard_id in material_topics:
                result.add(fi.standard_id)
        return sorted(result)

    def get_required_disclosures(
        self, standard_id: str, is_material: bool = True
    ) -> List[str]:
        """
        Return the required disclosure requirement IDs for a standard.

        Args:
            standard_id: Standard identifier.
            is_material: True if the topic is material.

        Returns:
            List of disclosure requirement IDs.
        """
        fi = self._filing_indicators.get(standard_id)
        if fi is None:
            return []
        if fi.mandatory:
            return fi.minimum_disclosures
        if is_material:
            if fi.mandatory_if_material:
                return fi.mandatory_if_material
            return fi.minimum_disclosures
        return []

    def is_first_year_exempt(
        self, standard_id: str, disclosure_id: str, reporting_year: int = 1
    ) -> bool:
        """
        Check whether a disclosure is exempt under phase-in rules.

        Args:
            standard_id: Standard identifier.
            disclosure_id: Disclosure requirement ID.
            reporting_year: Which year of CSRD reporting (1, 2, 3).

        Returns:
            True if the disclosure can be omitted in the given year.
        """
        fi = self._filing_indicators.get(standard_id)
        if fi is None:
            return False

        # Read phase-in from the raw filing indicator data
        fi_file = FILING_INDICATORS_FILE
        try:
            with open(fi_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        fi_raw = raw.get("filing_indicators", {}).get(standard_id, {})
        phase_in = fi_raw.get("phase_in")
        if phase_in is None:
            return False

        if reporting_year > 1 and standard_id not in ("ESRS-E4",):
            return False

        exemptions = phase_in.get("first_year_exemptions", [])
        for exemption in exemptions:
            if disclosure_id in exemption or exemption.endswith("-full"):
                return True

        return False

    # ------------------------------------------------------------------
    # Calculation linkbase
    # ------------------------------------------------------------------

    def get_calculation_relationship(
        self, rel_key: str
    ) -> Optional[CalculationRelationship]:
        """
        Return a calculation relationship by its key.

        Args:
            rel_key: Relationship key (e.g. 'total_ghg_emissions').

        Returns:
            CalculationRelationship or None.
        """
        return self._calculation_relationships.get(rel_key)

    def get_all_calculation_relationships(
        self,
    ) -> Dict[str, CalculationRelationship]:
        """Return all calculation relationships."""
        return dict(self._calculation_relationships)

    def get_calculation_children(
        self, parent_dp_id: str
    ) -> List[Tuple[str, float]]:
        """
        Return (child_dp_id, weight) pairs for a given parent.

        Args:
            parent_dp_id: Parent data point ID.

        Returns:
            List of (child_id, weight) tuples.
        """
        result: List[Tuple[str, float]] = []
        for rel in self._calculation_relationships.values():
            if rel.parent == parent_dp_id and rel.children:
                for child in rel.children:
                    result.append((child["child"], child["weight"]))
        return result

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    def get_namespace_uri(self, prefix: str) -> Optional[str]:
        """
        Return the namespace URI for a given prefix.

        Args:
            prefix: Namespace prefix (e.g. 'esrs-e1').

        Returns:
            URI string or None.
        """
        return self._namespaces.get(prefix)

    def get_all_namespaces(self) -> Dict[str, str]:
        """Return all registered namespace prefix -> URI mappings."""
        return dict(self._namespaces)

    # ------------------------------------------------------------------
    # Presentation tree
    # ------------------------------------------------------------------

    def get_presentation_tree(
        self, standard: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return the presentation tree structure.

        Args:
            standard: If given, return only the tree for that standard.

        Returns:
            Dict representing the presentation hierarchy.
        """
        if standard:
            return self._presentation_tree.get(standard, {})
        return dict(self._presentation_tree)

    def get_presentation_order(self, standard: str) -> int:
        """
        Return the presentation order number for a standard.

        Args:
            standard: Standard identifier.

        Returns:
            Order number (1-based) or 999 if not found.
        """
        tree = self._presentation_tree.get(standard, {})
        return tree.get("order", 999)

    def get_sections_for_standard(
        self, standard: str
    ) -> List[Dict[str, Any]]:
        """
        Return the disclosure sections for a standard.

        Args:
            standard: Standard identifier.

        Returns:
            List of section dicts with 'id', 'label', 'elements'.
        """
        tree = self._presentation_tree.get(standard, {})
        return tree.get("sections", [])

    # ------------------------------------------------------------------
    # Provenance / hashing
    # ------------------------------------------------------------------

    def compute_taxonomy_hash(self) -> str:
        """
        Compute a SHA-256 hash of the loaded taxonomy for audit provenance.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        hasher = hashlib.sha256()
        for dp_id in sorted(self._elements.keys()):
            elem = self._elements[dp_id]
            hasher.update(f"{dp_id}|{elem.element_id}|{elem.data_type}".encode("utf-8"))
        for dim_key in sorted(self._dimensions.keys()):
            dim = self._dimensions[dim_key]
            hasher.update(f"{dim_key}|{dim.dimension_id}".encode("utf-8"))
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of loaded taxonomy data for diagnostics.

        Returns:
            Dict with counts and metadata.
        """
        return {
            "total_elements": len(self._elements),
            "total_dimensions": len(self._dimensions),
            "total_units": len(self._units),
            "total_filing_indicators": len(self._filing_indicators),
            "total_calculation_relationships": len(self._calculation_relationships),
            "standards_with_elements": list(self._elements_by_standard.keys()),
            "element_counts_by_standard": {
                std: len(ids) for std, ids in self._elements_by_standard.items()
            },
            "taxonomy_hash": self.compute_taxonomy_hash(),
            "supported_languages": list(SUPPORTED_LANGUAGES),
        }
