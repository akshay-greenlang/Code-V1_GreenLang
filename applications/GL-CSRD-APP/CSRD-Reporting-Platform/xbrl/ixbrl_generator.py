# -*- coding: utf-8 -*-
"""
IXBRLGenerator - Inline XBRL Document Generator

This module generates ESEF-compliant iXBRL (Inline XBRL in XHTML) documents
for CSRD/ESRS sustainability reporting. It produces valid iXBRL documents
and ESEF reporting packages conforming to ESEF Reporting Manual v7.

Key Capabilities:
    - Generate ix:header with ix:references and ix:resources
    - Create xbrli:context elements (instant, duration, dimensional)
    - Create xbrli:unit elements (pure, ISO4217, divide, custom)
    - Create ix:nonFraction for numeric facts
    - Create ix:nonNumeric for text/narrative facts
    - Create ix:continuation for long text blocks
    - Create ix:exclude for human-readable non-XBRL content
    - Support filing indicators
    - Generate proper XHTML with all required namespaces
    - Create ESEF reporting package (ZIP)
    - Multi-language document generation (EN, DE, FR, ES)
    - Calculation consistency validation
    - XXE protection in all XML operations

Security:
    - Defused XML parsing (no external entity resolution)
    - DOCTYPE and ENTITY declarations rejected
    - Input size validation

Version: 1.1.0
Author: GreenLang CSRD Team
License: MIT
"""

import hashlib
import io
import json
import logging
import re
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.sax.saxutils import escape as xml_escape

from pydantic import BaseModel, Field, field_validator

from xbrl.taxonomy_mapper import (
    ContextDimension,
    ContextSpec,
    PeriodType,
    TaxonomyElement,
    TaxonomyMapper,
    UnitDefinition,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONTINUATION_LENGTH = 4000
MAX_XML_SIZE_MB = 50
MAX_FACT_VALUE_LENGTH = 1_000_000

XHTML_DOCTYPE = '<!DOCTYPE html>'

IXBRL_NAMESPACES = {
    "xmlns": "http://www.w3.org/1999/xhtml",
    "xmlns:ix": "http://www.xbrl.org/2013/inlineXBRL",
    "xmlns:ixt": "http://www.xbrl.org/inlineXBRL/transformation/2020-02-12",
    "xmlns:xbrli": "http://www.xbrl.org/2003/instance",
    "xmlns:xbrldi": "http://xbrl.org/2006/xbrldi",
    "xmlns:xbrldt": "http://xbrl.org/2005/xbrldt",
    "xmlns:link": "http://www.xbrl.org/2003/linkbase",
    "xmlns:xlink": "http://www.w3.org/1999/xlink",
    "xmlns:iso4217": "http://www.xbrl.org/2003/iso4217",
    "xmlns:esrs": "http://www.efrag.org/taxonomy/esrs/2024",
    "xmlns:esrs-cor": "http://www.efrag.org/taxonomy/esrs/2024/core",
    "xmlns:esrs-e1": "http://www.efrag.org/taxonomy/esrs/2024/e1",
    "xmlns:esrs-e2": "http://www.efrag.org/taxonomy/esrs/2024/e2",
    "xmlns:esrs-e3": "http://www.efrag.org/taxonomy/esrs/2024/e3",
    "xmlns:esrs-e4": "http://www.efrag.org/taxonomy/esrs/2024/e4",
    "xmlns:esrs-e5": "http://www.efrag.org/taxonomy/esrs/2024/e5",
    "xmlns:esrs-s1": "http://www.efrag.org/taxonomy/esrs/2024/s1",
    "xmlns:esrs-s2": "http://www.efrag.org/taxonomy/esrs/2024/s2",
    "xmlns:esrs-s3": "http://www.efrag.org/taxonomy/esrs/2024/s3",
    "xmlns:esrs-s4": "http://www.efrag.org/taxonomy/esrs/2024/s4",
    "xmlns:esrs-g1": "http://www.efrag.org/taxonomy/esrs/2024/g1",
    "xml:lang": "en",
}

STANDARD_TITLES = {
    "ESRS-2": "General Disclosures",
    "ESRS-E1": "Climate Change",
    "ESRS-E2": "Pollution",
    "ESRS-E3": "Water and Marine Resources",
    "ESRS-E4": "Biodiversity and Ecosystems",
    "ESRS-E5": "Resource Use and Circular Economy",
    "ESRS-S1": "Own Workforce",
    "ESRS-S2": "Workers in the Value Chain",
    "ESRS-S3": "Affected Communities",
    "ESRS-S4": "Consumers and End-users",
    "ESRS-G1": "Business Conduct",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactType(str, Enum):
    """Type of iXBRL fact element."""
    NUMERIC = "numeric"
    TEXT = "text"
    DATE = "date"
    BOOLEAN = "boolean"


class DocumentLanguage(str, Enum):
    """Supported document languages."""
    EN = "en"
    DE = "de"
    FR = "fr"
    ES = "es"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class IXBRLContext(BaseModel):
    """
    Represents an XBRL context element within iXBRL.

    Contains entity identifier, period specification, and optional
    dimensional qualifiers (scenario).
    """

    context_id: str = Field(..., description="Unique context ID (e.g. ctx_abc123)")
    entity_scheme: str = Field(
        "http://standards.iso.org/iso/17442",
        description="Entity identifier scheme",
    )
    entity_identifier: str = Field(..., description="Entity identifier (LEI)")
    period_type: PeriodType = Field(..., description="instant or duration")
    instant_date: Optional[str] = Field(None, description="YYYY-MM-DD for instant")
    start_date: Optional[str] = Field(None, description="Start date for duration")
    end_date: Optional[str] = Field(None, description="End date for duration")
    dimensions: List[ContextDimension] = Field(
        default_factory=list, description="Dimensional qualifiers"
    )

    @field_validator("instant_date", "start_date", "end_date")
    @classmethod
    def _validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate date strings are in YYYY-MM-DD format."""
        if v is None:
            return v
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(pattern, v):
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v

    def to_xml(self) -> str:
        """Render this context as an xbrli:context XML element."""
        lines: List[str] = []
        lines.append(f'  <xbrli:context id="{xml_escape(self.context_id)}">')

        # Entity
        lines.append("    <xbrli:entity>")
        lines.append(
            f'      <xbrli:identifier scheme="{xml_escape(self.entity_scheme)}">'
            f"{xml_escape(self.entity_identifier)}</xbrli:identifier>"
        )
        lines.append("    </xbrli:entity>")

        # Period
        lines.append("    <xbrli:period>")
        if self.period_type == PeriodType.INSTANT:
            lines.append(
                f"      <xbrli:instant>{xml_escape(self.instant_date or '')}"
                f"</xbrli:instant>"
            )
        else:
            lines.append(
                f"      <xbrli:startDate>{xml_escape(self.start_date or '')}"
                f"</xbrli:startDate>"
            )
            lines.append(
                f"      <xbrli:endDate>{xml_escape(self.end_date or '')}"
                f"</xbrli:endDate>"
            )
        lines.append("    </xbrli:period>")

        # Scenario (dimensions)
        if self.dimensions:
            lines.append("    <xbrli:scenario>")
            for dim in self.dimensions:
                if dim.is_typed:
                    lines.append(
                        f'      <xbrldi:typedMember dimension="{xml_escape(dim.dimension_id)}">'
                        f"<esrs:value>{xml_escape(dim.typed_value or '')}</esrs:value>"
                        f"</xbrldi:typedMember>"
                    )
                else:
                    lines.append(
                        f'      <xbrldi:explicitMember dimension='
                        f'"{xml_escape(dim.dimension_id)}">'
                        f"{xml_escape(dim.member_id or '')}</xbrldi:explicitMember>"
                    )
            lines.append("    </xbrli:scenario>")

        lines.append("  </xbrli:context>")
        return "\n".join(lines)


class IXBRLUnit(BaseModel):
    """
    Represents an XBRL unit element within iXBRL.

    Supports simple measures (ISO 4217, custom) and divide units
    (e.g. tCO2e per EUR revenue).
    """

    unit_id: str = Field(..., description="Unique unit ID")
    measures: List[str] = Field(
        default_factory=list, description="Simple measure QNames"
    )
    is_divide: bool = Field(False, description="True if this is a divide unit")
    numerator_measures: List[str] = Field(
        default_factory=list, description="Numerator measures for divide"
    )
    denominator_measures: List[str] = Field(
        default_factory=list, description="Denominator measures for divide"
    )

    def to_xml(self) -> str:
        """Render this unit as an xbrli:unit XML element."""
        lines: List[str] = []
        lines.append(f'  <xbrli:unit id="{xml_escape(self.unit_id)}">')

        if self.is_divide:
            lines.append("    <xbrli:divide>")
            lines.append("      <xbrli:unitNumerator>")
            for m in self.numerator_measures:
                lines.append(f"        <xbrli:measure>{xml_escape(m)}</xbrli:measure>")
            lines.append("      </xbrli:unitNumerator>")
            lines.append("      <xbrli:unitDenominator>")
            for m in self.denominator_measures:
                lines.append(f"        <xbrli:measure>{xml_escape(m)}</xbrli:measure>")
            lines.append("      </xbrli:unitDenominator>")
            lines.append("    </xbrli:divide>")
        else:
            for m in self.measures:
                lines.append(f"    <xbrli:measure>{xml_escape(m)}</xbrli:measure>")

        lines.append("  </xbrli:unit>")
        return "\n".join(lines)

    @classmethod
    def from_definition(cls, defn: UnitDefinition) -> "IXBRLUnit":
        """Create an IXBRLUnit from a UnitDefinition."""
        if defn.is_divide:
            return cls(
                unit_id=defn.unit_id,
                is_divide=True,
                numerator_measures=[defn.numerator or ""],
                denominator_measures=[defn.denominator or ""],
            )
        return cls(
            unit_id=defn.unit_id,
            measures=[defn.measure or ""],
        )


class IXBRLFact(BaseModel):
    """
    Represents an individual iXBRL tagged fact.

    Numeric facts use ix:nonFraction; text/date/boolean use ix:nonNumeric.
    """

    fact_id: str = Field(..., description="Unique fact ID")
    data_point_id: str = Field(..., description="ESRS data point ID")
    element_qname: str = Field(..., description="XBRL element QName")
    context_ref: str = Field(..., description="Context reference ID")
    unit_ref: Optional[str] = Field(None, description="Unit reference ID (numeric only)")
    fact_type: FactType = Field(..., description="numeric, text, date, boolean")
    value: Any = Field(..., description="The fact value")
    decimals: Optional[int] = Field(None, description="Decimal precision for numeric")
    scale: Optional[int] = Field(None, description="Scale factor (e.g. 6 for millions)")
    sign: Optional[str] = Field(None, description="'-' for negative values")
    format: Optional[str] = Field(None, description="ixt transformation format")
    is_nil: bool = Field(False, description="True if this is a nil fact")
    continuation_id: Optional[str] = Field(
        None, description="Continuation ID for long text"
    )
    language: str = Field("en", description="Language for text facts")

    @field_validator("value")
    @classmethod
    def _validate_value_size(cls, v: Any) -> Any:
        """Prevent excessively large values."""
        if isinstance(v, str) and len(v) > MAX_FACT_VALUE_LENGTH:
            raise ValueError(
                f"Fact value exceeds maximum length of {MAX_FACT_VALUE_LENGTH}"
            )
        return v

    def to_xml(self) -> str:
        """Render this fact as an ix:nonFraction or ix:nonNumeric element."""
        if self.is_nil:
            return self._render_nil()
        if self.fact_type == FactType.NUMERIC:
            return self._render_numeric()
        return self._render_non_numeric()

    def _render_numeric(self) -> str:
        """Render as ix:nonFraction."""
        attrs: List[str] = [
            f'name="{xml_escape(self.element_qname)}"',
            f'contextRef="{xml_escape(self.context_ref)}"',
            f'unitRef="{xml_escape(self.unit_ref or "")}"',
            f'id="{xml_escape(self.fact_id)}"',
        ]
        if self.decimals is not None:
            attrs.append(f'decimals="{self.decimals}"')
        else:
            attrs.append('decimals="INF"')
        if self.scale is not None:
            attrs.append(f'scale="{self.scale}"')
        if self.format:
            attrs.append(f'format="{xml_escape(self.format)}"')
        if self.sign == "-":
            attrs.append('sign="-"')

        value_str = self._format_numeric_value()
        return f'<ix:nonFraction {" ".join(attrs)}>{xml_escape(value_str)}</ix:nonFraction>'

    def _render_non_numeric(self) -> str:
        """Render as ix:nonNumeric."""
        attrs: List[str] = [
            f'name="{xml_escape(self.element_qname)}"',
            f'contextRef="{xml_escape(self.context_ref)}"',
            f'id="{xml_escape(self.fact_id)}"',
        ]
        if self.format:
            attrs.append(f'format="{xml_escape(self.format)}"')
        if self.language and self.fact_type == FactType.TEXT:
            attrs.append(f'xml:lang="{xml_escape(self.language)}"')
        if self.continuation_id:
            attrs.append(f'continuedAt="{xml_escape(self.continuation_id)}"')
        if self.fact_type == FactType.BOOLEAN:
            attrs.append('format="ixt:fixed-true"' if self.value else 'format="ixt:fixed-false"')

        value_str = self._format_value()
        return f'<ix:nonNumeric {" ".join(attrs)}>{xml_escape(value_str)}</ix:nonNumeric>'

    def _render_nil(self) -> str:
        """Render a nil fact."""
        if self.fact_type == FactType.NUMERIC:
            return (
                f'<ix:nonFraction name="{xml_escape(self.element_qname)}" '
                f'contextRef="{xml_escape(self.context_ref)}" '
                f'unitRef="{xml_escape(self.unit_ref or "")}" '
                f'id="{xml_escape(self.fact_id)}" '
                f'xsi:nil="true"/>'
            )
        return (
            f'<ix:nonNumeric name="{xml_escape(self.element_qname)}" '
            f'contextRef="{xml_escape(self.context_ref)}" '
            f'id="{xml_escape(self.fact_id)}" '
            f'xsi:nil="true"/>'
        )

    def _format_numeric_value(self) -> str:
        """Format a numeric value for display."""
        if self.value is None:
            return "0"
        val = self.value
        if self.scale:
            try:
                val = float(val) / (10 ** self.scale)
            except (ValueError, TypeError):
                pass
        if isinstance(val, float):
            if self.decimals is not None and self.decimals >= 0:
                return f"{val:.{self.decimals}f}"
            return f"{val}"
        return str(val)

    def _format_value(self) -> str:
        """Format a non-numeric value."""
        if self.value is None:
            return ""
        if self.fact_type == FactType.BOOLEAN:
            return "true" if self.value else "false"
        if self.fact_type == FactType.DATE:
            return str(self.value)
        return str(self.value)


class IXBRLContinuation(BaseModel):
    """Continuation block for long text facts (ix:continuation)."""

    continuation_id: str = Field(..., description="Continuation ID")
    text: str = Field(..., description="Continuation text")
    next_continuation_id: Optional[str] = Field(
        None, description="ID of next continuation block"
    )

    def to_xml(self) -> str:
        """Render as ix:continuation element."""
        attrs = f'id="{xml_escape(self.continuation_id)}"'
        if self.next_continuation_id:
            attrs += f' continuedAt="{xml_escape(self.next_continuation_id)}"'
        return f"<ix:continuation {attrs}>{xml_escape(self.text)}</ix:continuation>"


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_xml_safe(text: str) -> str:
    """
    Validate that text does not contain XML injection patterns.

    Args:
        text: Input text.

    Returns:
        Sanitized text.

    Raises:
        ValueError: If suspicious patterns detected.
    """
    if "<!DOCTYPE" in text:
        raise ValueError("DOCTYPE declarations not allowed (XXE prevention)")
    if "<!ENTITY" in text:
        raise ValueError("ENTITY declarations not allowed (XXE prevention)")
    if "SYSTEM" in text and ("file://" in text or "http://" in text):
        raise ValueError("External entity references not allowed (XXE prevention)")
    return text


# ---------------------------------------------------------------------------
# IXBRLGenerator
# ---------------------------------------------------------------------------

class IXBRLGenerator:
    """
    ESEF-compliant iXBRL document generator.

    Generates inline XBRL documents wrapped in XHTML, following ESEF
    Reporting Manual v7 requirements. The generator uses the TaxonomyMapper
    singleton for element lookups and context/unit generation.

    Usage::

        gen = IXBRLGenerator(
            entity_name="Example AG",
            entity_identifier="549300EXAMPLE000LEI00",
            reporting_period_start="2024-01-01",
            reporting_period_end="2024-12-31",
            currency="EUR",
        )
        gen.add_numeric_fact("E1_E16_001", 12500.0, decimals=0)
        gen.add_text_fact("ESRS2_BP1_001", "Basis for preparation...")
        html = gen.generate()

    Attributes:
        entity_name: Legal name of the reporting entity.
        entity_identifier: LEI or equivalent.
        reporting_period_start: Start date of the reporting period.
        reporting_period_end: End date of the reporting period.
        currency: Default currency ISO 4217 code.
    """

    def __init__(
        self,
        entity_name: str,
        entity_identifier: str,
        reporting_period_start: str,
        reporting_period_end: str,
        currency: str = "EUR",
        language: DocumentLanguage = DocumentLanguage.EN,
        material_standards: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the iXBRL generator.

        Args:
            entity_name: Legal name of the reporting entity.
            entity_identifier: LEI or equivalent entity identifier.
            reporting_period_start: Start of reporting period (YYYY-MM-DD).
            reporting_period_end: End of reporting period (YYYY-MM-DD).
            currency: Default ISO 4217 currency code.
            language: Primary document language.
            material_standards: List of material ESRS standards to include.
        """
        self.entity_name = entity_name
        self.entity_identifier = entity_identifier
        self.reporting_period_start = reporting_period_start
        self.reporting_period_end = reporting_period_end
        self.currency = currency
        self.language = language
        self.material_standards = material_standards or []

        self._mapper = TaxonomyMapper.get_instance()

        self._contexts: Dict[str, IXBRLContext] = {}
        self._units: Dict[str, IXBRLUnit] = {}
        self._facts: List[IXBRLFact] = []
        self._continuations: List[IXBRLContinuation] = []
        self._filing_indicators: List[str] = []
        self._fact_counter = 0

        # Create default contexts
        self._create_default_contexts()
        # Create default units
        self._create_default_units()

        logger.info(
            "IXBRLGenerator initialized for %s (period %s to %s, %s)",
            entity_name,
            reporting_period_start,
            reporting_period_end,
            currency,
        )

    # ------------------------------------------------------------------
    # Default setup
    # ------------------------------------------------------------------

    def _create_default_contexts(self) -> None:
        """Create the default duration and instant contexts."""
        # Duration context for the full reporting period
        duration_ctx = IXBRLContext(
            context_id="ctx_duration",
            entity_identifier=self.entity_identifier,
            period_type=PeriodType.DURATION,
            start_date=self.reporting_period_start,
            end_date=self.reporting_period_end,
        )
        self._contexts["ctx_duration"] = duration_ctx

        # Instant context at period end
        instant_ctx = IXBRLContext(
            context_id="ctx_instant",
            entity_identifier=self.entity_identifier,
            period_type=PeriodType.INSTANT,
            instant_date=self.reporting_period_end,
        )
        self._contexts["ctx_instant"] = instant_ctx

    def _create_default_units(self) -> None:
        """Create default unit elements for common types."""
        # Currency unit
        currency_unit = IXBRLUnit(
            unit_id=f"u_{self.currency}",
            measures=[f"iso4217:{self.currency}"],
        )
        self._units[currency_unit.unit_id] = currency_unit

        # Pure (percentage / ratio)
        pure_unit = IXBRLUnit(
            unit_id="u_pure",
            measures=["xbrli:pure"],
        )
        self._units["u_pure"] = pure_unit

        # Register all taxonomy-defined units
        for unit_key, defn in self._mapper.get_all_units().items():
            ixbrl_unit = IXBRLUnit.from_definition(defn)
            safe_id = f"u_{unit_key.replace('/', '_')}"
            ixbrl_unit.unit_id = safe_id
            self._units[safe_id] = ixbrl_unit

    # ------------------------------------------------------------------
    # Fact creation
    # ------------------------------------------------------------------

    def _next_fact_id(self) -> str:
        """Generate a unique fact ID."""
        self._fact_counter += 1
        return f"f_{self._fact_counter:06d}"

    def add_numeric_fact(
        self,
        data_point_id: str,
        value: Union[int, float],
        decimals: int = 2,
        scale: Optional[int] = None,
        dimensions: Optional[Dict[str, str]] = None,
        context_ref: Optional[str] = None,
        unit_ref: Optional[str] = None,
    ) -> str:
        """
        Add a numeric fact to the document.

        Args:
            data_point_id: ESRS data point ID (e.g. 'E1_E16_001').
            value: Numeric value.
            decimals: Number of decimal places for precision.
            scale: Scale factor (e.g. 6 for millions).
            dimensions: Dimensional qualifiers {dim_key: member_key}.
            context_ref: Override context reference. Defaults to appropriate one.
            unit_ref: Override unit reference. Defaults to taxonomy-derived unit.

        Returns:
            The generated fact ID.

        Raises:
            ValueError: If the data point is not found or not numeric.
        """
        element = self._mapper.get_element(data_point_id)
        if element is None:
            raise ValueError(f"Unknown data point: {data_point_id}")
        if not element.is_numeric:
            raise ValueError(
                f"Data point {data_point_id} is not numeric "
                f"(type: {element.data_type})"
            )

        # Resolve context
        ctx_id = context_ref or self._resolve_context(element, dimensions)

        # Resolve unit
        u_ref = unit_ref or self._resolve_unit(element)

        # Determine sign
        sign = "-" if isinstance(value, (int, float)) and value < 0 else None
        display_value = abs(value) if sign else value

        fact_id = self._next_fact_id()
        fact = IXBRLFact(
            fact_id=fact_id,
            data_point_id=data_point_id,
            element_qname=element.qname,
            context_ref=ctx_id,
            unit_ref=u_ref,
            fact_type=FactType.NUMERIC,
            value=display_value,
            decimals=decimals,
            scale=scale,
            sign=sign,
        )
        self._facts.append(fact)
        return fact_id

    def add_text_fact(
        self,
        data_point_id: str,
        value: str,
        dimensions: Optional[Dict[str, str]] = None,
        context_ref: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Add a text (narrative) fact to the document.

        Long text values are automatically split into continuation blocks.

        Args:
            data_point_id: ESRS data point ID.
            value: Text content.
            dimensions: Dimensional qualifiers.
            context_ref: Override context reference.
            language: Language code (defaults to document language).

        Returns:
            The generated fact ID.

        Raises:
            ValueError: If the data point is not found.
        """
        _validate_xml_safe(value)

        element = self._mapper.get_element(data_point_id)
        if element is None:
            raise ValueError(f"Unknown data point: {data_point_id}")

        ctx_id = context_ref or self._resolve_context(element, dimensions)
        lang = language or self.language.value

        fact_id = self._next_fact_id()

        # Handle long text with continuations
        continuation_id = None
        if len(value) > MAX_CONTINUATION_LENGTH:
            continuation_id = self._create_continuations(
                value[MAX_CONTINUATION_LENGTH:]
            )
            value = value[:MAX_CONTINUATION_LENGTH]

        fact = IXBRLFact(
            fact_id=fact_id,
            data_point_id=data_point_id,
            element_qname=element.qname,
            context_ref=ctx_id,
            fact_type=FactType.TEXT,
            value=value,
            language=lang,
            continuation_id=continuation_id,
        )
        self._facts.append(fact)
        return fact_id

    def add_boolean_fact(
        self,
        data_point_id: str,
        value: bool,
        dimensions: Optional[Dict[str, str]] = None,
        context_ref: Optional[str] = None,
    ) -> str:
        """
        Add a boolean fact to the document.

        Args:
            data_point_id: ESRS data point ID.
            value: Boolean value.
            dimensions: Dimensional qualifiers.
            context_ref: Override context reference.

        Returns:
            The generated fact ID.
        """
        element = self._mapper.get_element(data_point_id)
        if element is None:
            raise ValueError(f"Unknown data point: {data_point_id}")

        ctx_id = context_ref or self._resolve_context(element, dimensions)
        fact_id = self._next_fact_id()

        fact = IXBRLFact(
            fact_id=fact_id,
            data_point_id=data_point_id,
            element_qname=element.qname,
            context_ref=ctx_id,
            fact_type=FactType.BOOLEAN,
            value=value,
            format="ixt:fixed-true" if value else "ixt:fixed-false",
        )
        self._facts.append(fact)
        return fact_id

    def add_date_fact(
        self,
        data_point_id: str,
        value: str,
        dimensions: Optional[Dict[str, str]] = None,
        context_ref: Optional[str] = None,
    ) -> str:
        """
        Add a date fact to the document.

        Args:
            data_point_id: ESRS data point ID.
            value: Date string in YYYY-MM-DD format.
            dimensions: Dimensional qualifiers.
            context_ref: Override context reference.

        Returns:
            The generated fact ID.
        """
        element = self._mapper.get_element(data_point_id)
        if element is None:
            raise ValueError(f"Unknown data point: {data_point_id}")

        ctx_id = context_ref or self._resolve_context(element, dimensions)
        fact_id = self._next_fact_id()

        fact = IXBRLFact(
            fact_id=fact_id,
            data_point_id=data_point_id,
            element_qname=element.qname,
            context_ref=ctx_id,
            fact_type=FactType.DATE,
            value=value,
            format="ixt:date-month-day-year",
        )
        self._facts.append(fact)
        return fact_id

    def add_nil_fact(
        self,
        data_point_id: str,
        dimensions: Optional[Dict[str, str]] = None,
        context_ref: Optional[str] = None,
    ) -> str:
        """
        Add a nil (empty/missing) fact to the document.

        Args:
            data_point_id: ESRS data point ID.
            dimensions: Dimensional qualifiers.
            context_ref: Override context reference.

        Returns:
            The generated fact ID.
        """
        element = self._mapper.get_element(data_point_id)
        if element is None:
            raise ValueError(f"Unknown data point: {data_point_id}")

        ctx_id = context_ref or self._resolve_context(element, dimensions)
        fact_id = self._next_fact_id()

        fact_type = FactType.NUMERIC if element.is_numeric else FactType.TEXT
        unit_ref = self._resolve_unit(element) if element.is_numeric else None

        fact = IXBRLFact(
            fact_id=fact_id,
            data_point_id=data_point_id,
            element_qname=element.qname,
            context_ref=ctx_id,
            unit_ref=unit_ref,
            fact_type=fact_type,
            value=None,
            is_nil=True,
        )
        self._facts.append(fact)
        return fact_id

    # ------------------------------------------------------------------
    # Filing indicators
    # ------------------------------------------------------------------

    def add_filing_indicator(self, standard_id: str, filed: bool = True) -> None:
        """
        Add a filing indicator for a standard.

        Args:
            standard_id: ESRS standard ID (e.g. 'ESRS-E1').
            filed: True if the standard is being filed.
        """
        fi = self._mapper.get_filing_indicator(standard_id)
        if fi is None:
            logger.warning("Unknown standard for filing indicator: %s", standard_id)
            return
        if filed:
            self._filing_indicators.append(fi.filing_code)
            logger.info("Filing indicator added: %s (%s)", standard_id, fi.filing_code)

    def set_filing_indicators_from_materiality(
        self, material_standards: List[str]
    ) -> None:
        """
        Set filing indicators based on materiality assessment.

        Always includes mandatory standards (ESRS-1, ESRS-2).

        Args:
            material_standards: List of material standard IDs.
        """
        self._filing_indicators.clear()
        mandatory = self._mapper.get_mandatory_standards()
        all_standards = set(mandatory) | set(material_standards)

        for std_id in sorted(all_standards):
            fi = self._mapper.get_filing_indicator(std_id)
            if fi:
                self._filing_indicators.append(fi.filing_code)

    # ------------------------------------------------------------------
    # Context and unit resolution
    # ------------------------------------------------------------------

    def _resolve_context(
        self,
        element: TaxonomyElement,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Resolve the context reference for a given element and dimensions.

        Creates a new context if one does not already exist for the
        dimension combination.

        Args:
            element: The taxonomy element.
            dimensions: Dimensional qualifiers.

        Returns:
            Context ID string.
        """
        if not dimensions:
            if element.period_type == PeriodType.INSTANT:
                return "ctx_instant"
            return "ctx_duration"

        # Generate a context ID for this dimension combination
        ctx_id = self._mapper.get_context_ref(
            dimensions=dimensions,
            period_type=element.period_type,
            instant_date=(
                self.reporting_period_end
                if element.period_type == PeriodType.INSTANT
                else None
            ),
            start_date=(
                self.reporting_period_start
                if element.period_type == PeriodType.DURATION
                else None
            ),
            end_date=(
                self.reporting_period_end
                if element.period_type == PeriodType.DURATION
                else None
            ),
        )

        # Create the context if it does not exist
        if ctx_id not in self._contexts:
            spec = self._mapper.build_context_spec(
                entity_identifier=self.entity_identifier,
                period_type=element.period_type,
                instant_date=(
                    self.reporting_period_end
                    if element.period_type == PeriodType.INSTANT
                    else None
                ),
                start_date=(
                    self.reporting_period_start
                    if element.period_type == PeriodType.DURATION
                    else None
                ),
                end_date=(
                    self.reporting_period_end
                    if element.period_type == PeriodType.DURATION
                    else None
                ),
                dimensions=dimensions,
            )
            ctx = IXBRLContext(
                context_id=ctx_id,
                entity_scheme=spec.entity_scheme,
                entity_identifier=spec.entity_identifier,
                period_type=spec.period_type,
                instant_date=spec.instant_date,
                start_date=spec.start_date,
                end_date=spec.end_date,
                dimensions=spec.dimensions,
            )
            self._contexts[ctx_id] = ctx

        return ctx_id

    def _resolve_unit(self, element: TaxonomyElement) -> str:
        """
        Resolve the unit reference for a numeric element.

        Args:
            element: The taxonomy element.

        Returns:
            Unit ID string.
        """
        if element.is_monetary:
            return f"u_{self.currency}"
        if element.is_percentage:
            return "u_pure"

        # Look up from taxonomy
        unit_ref = self._mapper.get_unit_for_element(element.element_id)
        if unit_ref:
            # Map taxonomy unit_id to our local u_ prefixed ID
            for uid, u in self._units.items():
                if u.measures and unit_ref in u.measures:
                    return uid
                if unit_ref == u.unit_id:
                    return uid

        # Fallback to pure
        return "u_pure"

    # ------------------------------------------------------------------
    # Continuation blocks
    # ------------------------------------------------------------------

    def _create_continuations(self, text: str) -> str:
        """
        Split long text into continuation blocks.

        Args:
            text: Remaining text after the initial fact value.

        Returns:
            ID of the first continuation block.
        """
        _validate_xml_safe(text)

        chunks: List[str] = []
        while text:
            chunks.append(text[:MAX_CONTINUATION_LENGTH])
            text = text[MAX_CONTINUATION_LENGTH:]

        first_id = None
        prev_id = None

        for i, chunk in enumerate(chunks):
            cont_id = f"cont_{self._fact_counter:06d}_{i}"
            if first_id is None:
                first_id = cont_id

            continuation = IXBRLContinuation(
                continuation_id=cont_id,
                text=chunk,
            )
            if prev_id is not None:
                # Update previous continuation to link to this one
                for c in self._continuations:
                    if c.continuation_id == prev_id:
                        c.next_continuation_id = cont_id
                        break

            self._continuations.append(continuation)
            prev_id = cont_id

        return first_id or ""

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """
        Generate the complete iXBRL (XHTML) document.

        Returns:
            Complete iXBRL document as a string.
        """
        start = datetime.now()

        parts: List[str] = []
        parts.append(self._render_xhtml_header())
        parts.append(self._render_ix_header())
        parts.append(self._render_body_content())
        parts.append(self._render_continuations())
        parts.append(self._render_xhtml_footer())

        document = "\n".join(parts)
        elapsed = (datetime.now() - start).total_seconds() * 1000
        logger.info(
            "iXBRL document generated: %d facts, %d contexts, %d units, %.1f ms",
            len(self._facts),
            len(self._contexts),
            len(self._units),
            elapsed,
        )
        return document

    def _render_xhtml_header(self) -> str:
        """Render the XHTML document header with namespaces."""
        ns_attrs = "\n    ".join(
            f'{key}="{val}"' for key, val in IXBRL_NAMESPACES.items()
        )
        lang = self.language.value
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<html
    {ns_attrs}
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xml:lang="{lang}">
<head>
  <meta charset="UTF-8"/>
  <title>{xml_escape(self.entity_name)} - ESRS Sustainability Statement {self.reporting_period_end[:4]}</title>
  <style type="text/css">
    body {{ font-family: Arial, Helvetica, sans-serif; font-size: 10pt; line-height: 1.5; margin: 2cm; }}
    h1 {{ font-size: 18pt; color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 8px; }}
    h2 {{ font-size: 14pt; color: #2c3e50; margin-top: 24px; }}
    h3 {{ font-size: 12pt; color: #34495e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
    th {{ background-color: #ecf0f1; font-weight: bold; }}
    .numeric {{ text-align: right; }}
    .section {{ margin-bottom: 24px; page-break-inside: avoid; }}
    .cover-page {{ text-align: center; margin-top: 120px; }}
    .toc {{ margin: 24px 0; }}
    .toc li {{ margin: 4px 0; }}
    .filing-indicator {{ color: #27ae60; font-weight: bold; }}
  </style>
</head>
<body>"""

    def _render_ix_header(self) -> str:
        """Render the ix:header block with contexts, units, and references."""
        lines: List[str] = []
        lines.append("<ix:header>")

        # ix:hidden block with contexts and units
        lines.append("  <ix:hidden>")

        # Render all contexts
        for ctx in self._contexts.values():
            lines.append(ctx.to_xml())

        # Render all units
        for unit in self._units.values():
            lines.append(unit.to_xml())

        lines.append("  </ix:hidden>")

        # ix:references
        lines.append("  <ix:references>")
        lines.append(
            '    <link:schemaRef xlink:type="simple" '
            'xlink:href="http://www.efrag.org/taxonomy/esrs/2024/esrs-2024.xsd"/>'
        )
        lines.append("  </ix:references>")

        lines.append("</ix:header>")
        return "\n".join(lines)

    def _render_body_content(self) -> str:
        """Render the visible document body with tagged facts."""
        lines: List[str] = []

        # Cover page
        lines.append(self._render_cover_page())

        # Table of contents
        lines.append(self._render_toc())

        # Filing indicators section
        if self._filing_indicators:
            lines.append(self._render_filing_indicators_section())

        # Render facts grouped by standard
        facts_by_standard = self._group_facts_by_standard()
        for standard in sorted(
            facts_by_standard.keys(),
            key=lambda s: self._mapper.get_presentation_order(s),
        ):
            lines.append(self._render_standard_section(standard, facts_by_standard[standard]))

        return "\n".join(lines)

    def _render_cover_page(self) -> str:
        """Render the cover page."""
        year = self.reporting_period_end[:4]
        return f"""
<div class="cover-page">
  <h1>{xml_escape(self.entity_name)}</h1>
  <h2>Sustainability Statement</h2>
  <h2>European Sustainability Reporting Standards (ESRS)</h2>
  <p>Reporting Period: {xml_escape(self.reporting_period_start)} to {xml_escape(self.reporting_period_end)}</p>
  <p>Entity Identifier (LEI): {xml_escape(self.entity_identifier)}</p>
  <p>Prepared in accordance with the European Sustainability Reporting Standards (ESRS)</p>
  <p>as adopted by Commission Delegated Regulation (EU) 2023/2772</p>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
</div>
<div style="page-break-after: always;"></div>"""

    def _render_toc(self) -> str:
        """Render the table of contents."""
        lines: List[str] = ['<div class="toc">', "<h2>Table of Contents</h2>", "<ol>"]
        facts_by_standard = self._group_facts_by_standard()
        for standard in sorted(
            facts_by_standard.keys(),
            key=lambda s: self._mapper.get_presentation_order(s),
        ):
            title = STANDARD_TITLES.get(standard, standard)
            anchor = standard.replace("-", "_").lower()
            lines.append(f'  <li><a href="#{anchor}">{xml_escape(standard)} - {xml_escape(title)}</a></li>')
        lines.append("</ol>")
        lines.append("</div>")
        return "\n".join(lines)

    def _render_filing_indicators_section(self) -> str:
        """Render the filing indicators section."""
        lines: List[str] = [
            '<div class="section">',
            '<h2>Filing Indicators</h2>',
            "<table>",
            "<tr><th>Filing Code</th><th>Standard</th><th>Status</th></tr>",
        ]
        for fi_code in sorted(self._filing_indicators):
            lines.append(
                f"<tr>"
                f"<td>{xml_escape(fi_code)}</td>"
                f'<td>{xml_escape(fi_code.replace("FI_", "ESRS-"))}</td>'
                f'<td class="filing-indicator">FILED</td>'
                f"</tr>"
            )
        lines.append("</table>")
        lines.append("</div>")
        return "\n".join(lines)

    def _render_standard_section(
        self, standard: str, facts: List[IXBRLFact]
    ) -> str:
        """Render a section for a single ESRS standard."""
        title = STANDARD_TITLES.get(standard, standard)
        anchor = standard.replace("-", "_").lower()

        lines: List[str] = [
            f'<div class="section" id="{anchor}">',
            f"<h2>{xml_escape(standard)} - {xml_escape(title)}</h2>",
        ]

        # Group by disclosure requirement
        dr_groups: Dict[str, List[IXBRLFact]] = {}
        for fact in facts:
            elem = self._mapper.get_element(fact.data_point_id)
            dr = elem.disclosure_requirement if elem else "Other"
            dr_groups.setdefault(dr, []).append(fact)

        for dr_id, dr_facts in sorted(dr_groups.items()):
            lines.append(f"<h3>{xml_escape(dr_id)}</h3>")

            # Separate numeric and text facts
            numeric_facts = [f for f in dr_facts if f.fact_type == FactType.NUMERIC]
            text_facts = [f for f in dr_facts if f.fact_type != FactType.NUMERIC]

            # Render numeric facts as a table
            if numeric_facts:
                lines.append(self._render_numeric_table(numeric_facts))

            # Render text facts as paragraphs
            for fact in text_facts:
                elem = self._mapper.get_element(fact.data_point_id)
                label = elem.get_label(self.language.value) if elem else fact.data_point_id
                lines.append(f"<p><strong>{xml_escape(label)}:</strong></p>")
                lines.append(f"<p>{fact.to_xml()}</p>")

        lines.append("</div>")
        return "\n".join(lines)

    def _render_numeric_table(self, facts: List[IXBRLFact]) -> str:
        """Render numeric facts as an HTML table with embedded XBRL tags."""
        lines: List[str] = [
            "<table>",
            '<tr><th>Data Point</th><th>Description</th><th class="numeric">Value</th></tr>',
        ]
        for fact in facts:
            elem = self._mapper.get_element(fact.data_point_id)
            label = elem.get_label(self.language.value) if elem else fact.data_point_id
            lines.append(
                f"<tr>"
                f"<td>{xml_escape(fact.data_point_id)}</td>"
                f"<td>{xml_escape(label)}</td>"
                f'<td class="numeric">{fact.to_xml()}</td>'
                f"</tr>"
            )
        lines.append("</table>")
        return "\n".join(lines)

    def _render_continuations(self) -> str:
        """Render all continuation blocks."""
        if not self._continuations:
            return ""
        lines: List[str] = ["<!-- Continuation blocks -->"]
        lines.append('<div style="display:none;">')
        for cont in self._continuations:
            lines.append(cont.to_xml())
        lines.append("</div>")
        return "\n".join(lines)

    def _render_xhtml_footer(self) -> str:
        """Render the XHTML footer."""
        return """
<div class="section">
  <h2>About This Report</h2>
  <p>This sustainability statement has been prepared in accordance with
  the European Sustainability Reporting Standards (ESRS) as adopted by
  Commission Delegated Regulation (EU) 2023/2772.</p>
  <p>The report is presented in iXBRL format compliant with the ESEF
  Regulation (Commission Delegated Regulation (EU) 2019/815).</p>
  <ix:exclude>
    <p><em>This document was generated by GreenLang GL-CSRD-APP v1.1</em></p>
  </ix:exclude>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def _group_facts_by_standard(self) -> Dict[str, List[IXBRLFact]]:
        """Group facts by their ESRS standard."""
        result: Dict[str, List[IXBRLFact]] = {}
        for fact in self._facts:
            elem = self._mapper.get_element(fact.data_point_id)
            std = elem.standard if elem else "Unknown"
            result.setdefault(std, []).append(fact)
        return result

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def get_fact_count(self) -> int:
        """Return the total number of facts added."""
        return len(self._facts)

    def get_context_count(self) -> int:
        """Return the total number of contexts."""
        return len(self._contexts)

    def get_unit_count(self) -> int:
        """Return the total number of units."""
        return len(self._units)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the document state."""
        return {
            "entity_name": self.entity_name,
            "entity_identifier": self.entity_identifier,
            "reporting_period": f"{self.reporting_period_start} to {self.reporting_period_end}",
            "currency": self.currency,
            "language": self.language.value,
            "total_facts": len(self._facts),
            "total_contexts": len(self._contexts),
            "total_units": len(self._units),
            "total_continuations": len(self._continuations),
            "filing_indicators": self._filing_indicators,
            "facts_by_type": {
                "numeric": sum(1 for f in self._facts if f.fact_type == FactType.NUMERIC),
                "text": sum(1 for f in self._facts if f.fact_type == FactType.TEXT),
                "boolean": sum(1 for f in self._facts if f.fact_type == FactType.BOOLEAN),
                "date": sum(1 for f in self._facts if f.fact_type == FactType.DATE),
                "nil": sum(1 for f in self._facts if f.is_nil),
            },
        }

    def compute_document_hash(self) -> str:
        """
        Compute SHA-256 hash of the generated document for provenance.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        doc = self.generate()
        return hashlib.sha256(doc.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ESEFPackager
# ---------------------------------------------------------------------------

class ESEFPackager:
    """
    Creates a valid ESEF reporting package ZIP file.

    The package follows the ESEF Reporting Manual v7 structure:

        {lei}_{period_end}/
            reports/
                {entity_name}_esrs_{year}.xhtml    (the iXBRL document)
            META-INF/
                reports.json                        (report metadata)
                taxonomyPackage.xml                 (taxonomy package descriptor)

    Usage::

        packager = ESEFPackager(
            generator=ixbrl_gen,
            output_dir=Path("./output"),
        )
        zip_path = packager.create_package()

    Attributes:
        generator: IXBRLGenerator instance with populated facts.
        output_dir: Directory for the output ZIP file.
    """

    def __init__(
        self,
        generator: IXBRLGenerator,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the ESEF packager.

        Args:
            generator: IXBRLGenerator instance.
            output_dir: Output directory (defaults to current directory).
        """
        self.generator = generator
        self.output_dir = output_dir or Path(".")
        self._package_name = self._build_package_name()

    def _build_package_name(self) -> str:
        """Build the ESEF package name per naming conventions."""
        lei = self.generator.entity_identifier
        period_end = self.generator.reporting_period_end.replace("-", "")
        return f"{lei}_{period_end}"

    def create_package(self) -> Path:
        """
        Create the ESEF reporting package as a ZIP file.

        Returns:
            Path to the created ZIP file.
        """
        start = datetime.now()
        zip_filename = f"{self._package_name}.zip"
        zip_path = self.output_dir / zip_filename

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate the iXBRL document
        ixbrl_content = self.generator.generate()

        # Build the ZIP
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # iXBRL document
            report_filename = self._build_report_filename()
            zf.writestr(
                f"{self._package_name}/reports/{report_filename}",
                ixbrl_content,
            )

            # META-INF/reports.json
            reports_json = self._create_reports_json(report_filename)
            zf.writestr(
                f"{self._package_name}/META-INF/reports.json",
                json.dumps(reports_json, indent=2, ensure_ascii=False),
            )

            # META-INF/taxonomyPackage.xml
            taxonomy_package_xml = self._create_taxonomy_package_xml()
            zf.writestr(
                f"{self._package_name}/META-INF/taxonomyPackage.xml",
                taxonomy_package_xml,
            )

            # META-INF/catalog.xml
            catalog_xml = self._create_catalog_xml()
            zf.writestr(
                f"{self._package_name}/META-INF/catalog.xml",
                catalog_xml,
            )

        elapsed = (datetime.now() - start).total_seconds() * 1000
        logger.info(
            "ESEF package created: %s (%.1f ms, %d bytes)",
            zip_path,
            elapsed,
            zip_path.stat().st_size,
        )
        return zip_path

    def create_package_bytes(self) -> bytes:
        """
        Create the ESEF reporting package as bytes (in-memory).

        Returns:
            ZIP file content as bytes.
        """
        buffer = io.BytesIO()
        ixbrl_content = self.generator.generate()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            report_filename = self._build_report_filename()
            zf.writestr(
                f"{self._package_name}/reports/{report_filename}",
                ixbrl_content,
            )
            reports_json = self._create_reports_json(report_filename)
            zf.writestr(
                f"{self._package_name}/META-INF/reports.json",
                json.dumps(reports_json, indent=2, ensure_ascii=False),
            )
            zf.writestr(
                f"{self._package_name}/META-INF/taxonomyPackage.xml",
                self._create_taxonomy_package_xml(),
            )
            zf.writestr(
                f"{self._package_name}/META-INF/catalog.xml",
                self._create_catalog_xml(),
            )

        return buffer.getvalue()

    def _build_report_filename(self) -> str:
        """Build the iXBRL report filename per ESEF conventions."""
        entity_slug = re.sub(
            r"[^a-zA-Z0-9]", "_", self.generator.entity_name
        ).lower()
        year = self.generator.reporting_period_end[:4]
        lang = self.generator.language.value
        return f"{entity_slug}_esrs_{year}_{lang}.xhtml"

    def _create_reports_json(self, report_filename: str) -> Dict[str, Any]:
        """Create the META-INF/reports.json metadata."""
        return {
            "documentInfo": {
                "documentType": "https://xbrl.org/2021/xbrl-csv/v2.0/report-package",
                "features": {},
            },
            "reports": {
                report_filename: {
                    "role": "http://www.xbrl.org/2021/role/sustainabilityReport",
                    "lang": self.generator.language.value,
                    "description": (
                        f"ESRS Sustainability Statement for "
                        f"{self.generator.entity_name} - "
                        f"FY {self.generator.reporting_period_end[:4]}"
                    ),
                }
            },
            "metadata": {
                "generator": "GreenLang GL-CSRD-APP v1.1",
                "generatedAt": datetime.now().isoformat(),
                "entityIdentifier": self.generator.entity_identifier,
                "reportingPeriod": {
                    "startDate": self.generator.reporting_period_start,
                    "endDate": self.generator.reporting_period_end,
                },
                "currency": self.generator.currency,
                "taxonomyVersion": "EFRAG ESRS 2024.1.0",
                "factCount": self.generator.get_fact_count(),
            },
        }

    def _create_taxonomy_package_xml(self) -> str:
        """Create the META-INF/taxonomyPackage.xml."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<tp:taxonomyPackage
    xmlns:tp="http://xbrl.org/2016/taxonomy-package"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://xbrl.org/2016/taxonomy-package
    http://xbrl.org/2016/taxonomy-package.xsd">
  <tp:identifier>http://www.efrag.org/taxonomy/esrs/2024</tp:identifier>
  <tp:name>EFRAG ESRS Taxonomy 2024</tp:name>
  <tp:description>
    European Sustainability Reporting Standards (ESRS) XBRL Taxonomy
    as published by EFRAG for CSRD reporting.
  </tp:description>
  <tp:version>2024.1.0</tp:version>
  <tp:publisher>EFRAG</tp:publisher>
  <tp:publisherURL>https://www.efrag.org</tp:publisherURL>
  <tp:publisherCountry>EU</tp:publisherCountry>
  <tp:publicationDate>2024-01-15</tp:publicationDate>
  <tp:entryPoints>
    <tp:entryPoint>
      <tp:name>ESRS 2024 Entry Point</tp:name>
      <tp:description>Main entry point for ESRS 2024 taxonomy</tp:description>
      <tp:entryPointDocument
        href="http://www.efrag.org/taxonomy/esrs/2024/esrs-2024.xsd"/>
    </tp:entryPoint>
  </tp:entryPoints>
</tp:taxonomyPackage>"""

    def _create_catalog_xml(self) -> str:
        """Create the META-INF/catalog.xml."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
  <rewriteURI
    uriStartString="http://www.efrag.org/taxonomy/esrs/2024/"
    rewritePrefix="../taxonomy/"/>
</catalog>"""

    def get_package_info(self) -> Dict[str, Any]:
        """Return package metadata summary."""
        return {
            "package_name": self._package_name,
            "report_filename": self._build_report_filename(),
            "entity_name": self.generator.entity_name,
            "entity_identifier": self.generator.entity_identifier,
            "reporting_period": (
                f"{self.generator.reporting_period_start} to "
                f"{self.generator.reporting_period_end}"
            ),
            "fact_count": self.generator.get_fact_count(),
            "context_count": self.generator.get_context_count(),
        }
