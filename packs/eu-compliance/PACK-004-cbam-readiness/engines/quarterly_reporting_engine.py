# -*- coding: utf-8 -*-
"""
QuarterlyReportingEngine - PACK-004 CBAM Readiness Engine 3
=============================================================

Quarterly CBAM report assembly and XML generation engine. Handles the
full lifecycle of CBAM quarterly reports: period detection, goods entry
aggregation, EU CBAM Registry XML format generation, pre-submission
validation, version-controlled amendments, and deadline tracking.

Reporting Periods:
    - Q1: January 1 - March 31   (deadline: April 30)
    - Q2: April 1 - June 30      (deadline: July 31)
    - Q3: July 1 - September 30  (deadline: October 31)
    - Q4: October 1 - December 31 (deadline: January 31 following year)
    - Amendment deadline: 2 months after original submission

Report Statuses:
    - DRAFT: Initial assembly, not yet validated
    - VALIDATED: Passed all pre-submission checks
    - SUBMITTED: Submitted to EU CBAM Registry
    - AMENDED: Correction submitted (version > 1)

Zero-Hallucination:
    - All date/deadline calculations use deterministic datetime arithmetic
    - XML generation uses string templates with validated data
    - No LLM involvement in any report assembly or validation path
    - SHA-256 provenance hashing on every report

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportStatus(str, Enum):
    """Quarterly report lifecycle status."""

    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    AMENDED = "AMENDED"

class CalculationMethod(str, Enum):
    """Emission calculation method (mirrored for self-containment)."""

    ACTUAL = "actual"
    DEFAULT = "default"
    COUNTRY_DEFAULT = "country_default"

# ---------------------------------------------------------------------------
# Quarter/Period Constants
# ---------------------------------------------------------------------------

# Quarter start months (1-indexed)
_QUARTER_START_MONTHS: Dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}
_QUARTER_END_MONTHS: Dict[int, Tuple[int, int]] = {
    1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31),
}

# Submission deadline: last day of month following quarter end
# Q1 -> April 30, Q2 -> July 31, Q3 -> October 31, Q4 -> January 31 (next year)
_SUBMISSION_DEADLINES: Dict[int, Tuple[int, int, int]] = {
    1: (0, 4, 30),   # Same year, April 30
    2: (0, 7, 31),   # Same year, July 31
    3: (0, 10, 31),  # Same year, October 31
    4: (1, 1, 31),   # Next year, January 31
}

# Amendment deadline: 2 months after submission deadline
_AMENDMENT_OFFSET_DAYS: int = 60

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class QuarterlyPeriod(BaseModel):
    """CBAM quarterly reporting period with key dates.

    Represents a single quarter with start date, end date, submission
    deadline, and amendment deadline.
    """

    year: int = Field(
        ..., ge=2023, le=2050,
        description="Calendar year of the quarter",
    )
    quarter: int = Field(
        ..., ge=1, le=4,
        description="Quarter number (1-4)",
    )
    start_date: date = Field(
        ..., description="First day of the quarter",
    )
    end_date: date = Field(
        ..., description="Last day of the quarter",
    )
    submission_deadline: date = Field(
        ..., description="Submission deadline for the report",
    )
    amendment_deadline: date = Field(
        ..., description="Deadline for submitting amendments",
    )

class GoodsEntry(BaseModel):
    """Single goods entry in a CBAM quarterly report.

    Represents one line of imported goods aggregated by CN code, country
    of origin, and installation.
    """

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique entry identifier",
    )
    cn_code: str = Field(
        ..., min_length=4, max_length=12,
        description="Combined Nomenclature code",
    )
    goods_description: str = Field(
        "", max_length=500,
        description="Description of goods",
    )
    country_of_origin: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    installation_id: Optional[str] = Field(
        None, max_length=100,
        description="Production installation identifier",
    )
    quantity_tonnes: float = Field(
        ..., ge=0,
        description="Total quantity in metric tonnes",
    )
    direct_emissions: float = Field(
        ..., ge=0,
        description="Total direct emissions (tCO2e)",
    )
    indirect_emissions: float = Field(
        ..., ge=0,
        description="Total indirect emissions (tCO2e)",
    )
    total_embedded_emissions: float = Field(
        ..., ge=0,
        description="Total embedded emissions (tCO2e)",
    )
    calculation_method: CalculationMethod = Field(
        CalculationMethod.DEFAULT,
        description="Emission calculation method used",
    )
    specific_embedded_emissions: float = Field(
        0.0, ge=0,
        description="Specific embedded emissions (tCO2e/t product)",
    )

    @field_validator("country_of_origin")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.strip().upper()

class ImporterConfig(BaseModel):
    """Importer configuration for report header."""

    eori_number: str = Field(
        ..., min_length=5, max_length=17,
        description="EORI (Economic Operators Registration and Identification) number",
    )
    company_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Legal name of the importer",
    )
    address: str = Field(
        "", max_length=500,
        description="Registered address",
    )
    contact_name: str = Field(
        "", max_length=200,
        description="Contact person name",
    )
    contact_email: str = Field(
        "", max_length=200,
        description="Contact email address",
    )
    member_state: str = Field(
        "", min_length=0, max_length=2,
        description="EU member state code",
    )

class QuarterlyReport(BaseModel):
    """Complete CBAM quarterly report.

    Contains all data required for submission to the EU CBAM Registry,
    including importer information, goods entries, emission totals, and
    submission metadata.
    """

    report_id: str = Field(
        default_factory=_new_uuid,
        description="Unique report identifier",
    )
    period: QuarterlyPeriod = Field(
        ..., description="Reporting period",
    )
    importer_eori: str = Field(
        ..., min_length=5, max_length=17,
        description="EORI number of the importer",
    )
    importer_name: str = Field(
        "", max_length=300,
        description="Legal name of the importer",
    )
    goods_entries: List[GoodsEntry] = Field(
        default_factory=list,
        description="List of goods entries in the report",
    )
    total_embedded_emissions_tco2e: float = Field(
        0.0, ge=0,
        description="Total embedded emissions across all entries",
    )
    total_quantity_tonnes: float = Field(
        0.0, ge=0,
        description="Total quantity across all entries",
    )
    report_status: ReportStatus = Field(
        ReportStatus.DRAFT,
        description="Current status of the report",
    )
    xml_content: Optional[str] = Field(
        None, description="Generated XML content for submission",
    )
    submission_date: Optional[datetime] = Field(
        None, description="Date report was submitted",
    )
    amendment_version: int = Field(
        1, ge=1,
        description="Version number (1 = original, 2+ = amendment)",
    )
    original_report_id: Optional[str] = Field(
        None, description="ID of the original report if this is an amendment",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Report creation timestamp",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )

class ValidationError(BaseModel):
    """Single validation error in a quarterly report."""

    field: str = Field(
        ..., description="Field or section with the error",
    )
    severity: str = Field(
        "ERROR", description="ERROR or WARNING",
    )
    message: str = Field(
        ..., description="Human-readable error message",
    )
    code: str = Field(
        "", description="Machine-readable error code",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QuarterlyReportingEngine:
    """Quarterly CBAM report assembly and management engine.

    Handles the complete lifecycle of CBAM quarterly reports from period
    detection through goods entry aggregation, XML generation, validation,
    amendment, and deadline tracking.

    Zero-Hallucination Guarantees:
        - All date calculations use deterministic datetime arithmetic
        - XML generation uses validated data with string formatting
        - No LLM involvement in report assembly or validation
        - SHA-256 provenance hashing on every report

    Example:
        >>> engine = QuarterlyReportingEngine()
        >>> period = engine.detect_period(date(2027, 2, 15))
        >>> assert period.quarter == 1
        >>> assert period.year == 2027
    """

    def __init__(self) -> None:
        """Initialize QuarterlyReportingEngine."""
        self._reports: Dict[str, QuarterlyReport] = {}
        self._report_count: int = 0
        logger.info("QuarterlyReportingEngine initialized (v%s)", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_period(self, target_date: date) -> QuarterlyPeriod:
        """Auto-detect the CBAM quarterly period for a given date.

        Determines which quarter the date falls in and calculates all
        associated deadlines.

        Args:
            target_date: Date to detect the period for.

        Returns:
            QuarterlyPeriod with start, end, submission, and amendment dates.
        """
        year = target_date.year
        month = target_date.month

        # Determine quarter from month
        if month <= 3:
            quarter = 1
        elif month <= 6:
            quarter = 2
        elif month <= 9:
            quarter = 3
        else:
            quarter = 4

        return self._build_period(year, quarter)

    def assemble_report(
        self,
        period: QuarterlyPeriod,
        emission_results: List[Any],
        importer_config: ImporterConfig,
    ) -> QuarterlyReport:
        """Assemble a complete quarterly CBAM report.

        Takes a list of emission results and an importer configuration,
        aggregates goods entries by CN code / country / installation, and
        creates a draft report ready for validation and submission.

        Args:
            period: Reporting period.
            emission_results: List of emission result objects.
            importer_config: Importer identification and contact data.

        Returns:
            QuarterlyReport in DRAFT status.
        """
        self._report_count += 1

        # Aggregate goods entries
        entries = self.aggregate_goods_entries(emission_results)

        # Calculate totals
        total_emissions = Decimal("0")
        total_qty = Decimal("0")
        for entry in entries:
            total_emissions += _decimal(entry.total_embedded_emissions)
            total_qty += _decimal(entry.quantity_tonnes)

        report = QuarterlyReport(
            period=period,
            importer_eori=importer_config.eori_number,
            importer_name=importer_config.company_name,
            goods_entries=entries,
            total_embedded_emissions_tco2e=_round_val(total_emissions, 4),
            total_quantity_tonnes=_round_val(total_qty, 2),
            report_status=ReportStatus.DRAFT,
            amendment_version=1,
        )

        report.provenance_hash = _compute_hash(report)
        self._reports[report.report_id] = report

        logger.info(
            "Report assembled [%s]: Q%d/%d, %d entries, %.2f tCO2e, %.2f t",
            report.report_id,
            period.quarter,
            period.year,
            len(entries),
            report.total_embedded_emissions_tco2e,
            report.total_quantity_tonnes,
        )

        return report

    def aggregate_goods_entries(
        self,
        emission_results: List[Any],
    ) -> List[GoodsEntry]:
        """Aggregate emission results into goods entries.

        Groups emission results by CN code, country of origin, and
        installation ID. Sums quantities and emissions within each group.

        Args:
            emission_results: List of emission result objects with attributes:
                cn_code, country_of_origin, installation_id, quantity_tonnes,
                direct_emissions_tco2e, indirect_emissions_tco2e,
                total_embedded_emissions_tco2e, calculation_method_used.

        Returns:
            List of GoodsEntry, one per unique CN/country/installation group.
        """
        groups: Dict[str, Dict[str, Any]] = {}

        for r in emission_results:
            cn = getattr(r, "cn_code", "UNKNOWN")
            country = getattr(r, "country_of_origin", "XX")
            install = getattr(r, "installation_id", None) or "UNSPECIFIED"

            # Extract calculation method
            method_raw = getattr(r, "calculation_method_used", "default")
            if hasattr(method_raw, "value"):
                method_str = method_raw.value
            else:
                method_str = str(method_raw)

            key = f"{cn}|{country}|{install}"

            if key not in groups:
                groups[key] = {
                    "cn_code": cn,
                    "country_of_origin": country,
                    "installation_id": install if install != "UNSPECIFIED" else None,
                    "quantity": Decimal("0"),
                    "direct": Decimal("0"),
                    "indirect": Decimal("0"),
                    "total": Decimal("0"),
                    "method": method_str,
                    "description": getattr(r, "goods_description", ""),
                }

            groups[key]["quantity"] += _decimal(getattr(r, "quantity_tonnes", 0.0))
            groups[key]["direct"] += _decimal(getattr(r, "direct_emissions_tco2e", 0.0))
            groups[key]["indirect"] += _decimal(getattr(r, "indirect_emissions_tco2e", 0.0))
            groups[key]["total"] += _decimal(
                getattr(r, "total_embedded_emissions_tco2e", 0.0)
            )

        entries: List[GoodsEntry] = []
        for g in groups.values():
            qty = g["quantity"]
            total_em = g["total"]
            specific = total_em / qty if qty > 0 else Decimal("0")

            # Map calculation method string to enum
            try:
                calc_method = CalculationMethod(g["method"])
            except ValueError:
                calc_method = CalculationMethod.DEFAULT

            entries.append(
                GoodsEntry(
                    cn_code=g["cn_code"],
                    goods_description=g["description"],
                    country_of_origin=g["country_of_origin"],
                    installation_id=g["installation_id"],
                    quantity_tonnes=_round_val(qty, 2),
                    direct_emissions=_round_val(g["direct"]),
                    indirect_emissions=_round_val(g["indirect"]),
                    total_embedded_emissions=_round_val(total_em),
                    calculation_method=calc_method,
                    specific_embedded_emissions=_round_val(specific),
                )
            )

        # Sort by CN code then country for consistent output
        entries.sort(key=lambda e: (e.cn_code, e.country_of_origin))
        return entries

    def generate_xml(self, report: QuarterlyReport) -> str:
        """Generate EU CBAM Registry XML format for a quarterly report.

        Produces an XML document conforming to the EU CBAM Transitional
        Registry schema. The XML includes importer data, period information,
        and all goods entries with emission breakdowns.

        Args:
            report: QuarterlyReport to serialize.

        Returns:
            XML string ready for CBAM Registry submission.
        """
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<CBAMReport xmlns="urn:eu:cbam:report:v1" version="1.0">',
            '  <ReportHeader>',
            f'    <ReportId>{_xml_escape(report.report_id)}</ReportId>',
            f'    <ReportingYear>{report.period.year}</ReportingYear>',
            f'    <ReportingQuarter>{report.period.quarter}</ReportingQuarter>',
            f'    <PeriodStart>{report.period.start_date.isoformat()}</PeriodStart>',
            f'    <PeriodEnd>{report.period.end_date.isoformat()}</PeriodEnd>',
            f'    <SubmissionDeadline>{report.period.submission_deadline.isoformat()}</SubmissionDeadline>',
            f'    <AmendmentVersion>{report.amendment_version}</AmendmentVersion>',
            f'    <Status>{report.report_status.value}</Status>',
            f'    <CreatedAt>{report.created_at.isoformat()}</CreatedAt>',
            '  </ReportHeader>',
            '  <Importer>',
            f'    <EORI>{_xml_escape(report.importer_eori)}</EORI>',
            f'    <Name>{_xml_escape(report.importer_name)}</Name>',
            '  </Importer>',
            '  <GoodsEntries>',
        ]

        for entry in report.goods_entries:
            xml_lines.extend([
                '    <GoodsEntry>',
                f'      <EntryId>{_xml_escape(entry.entry_id)}</EntryId>',
                f'      <CNCode>{_xml_escape(entry.cn_code)}</CNCode>',
                f'      <Description>{_xml_escape(entry.goods_description)}</Description>',
                f'      <CountryOfOrigin>{_xml_escape(entry.country_of_origin)}</CountryOfOrigin>',
                f'      <InstallationId>{_xml_escape(entry.installation_id or "")}</InstallationId>',
                f'      <QuantityTonnes>{entry.quantity_tonnes:.4f}</QuantityTonnes>',
                f'      <DirectEmissions>{entry.direct_emissions:.6f}</DirectEmissions>',
                f'      <IndirectEmissions>{entry.indirect_emissions:.6f}</IndirectEmissions>',
                f'      <TotalEmbeddedEmissions>{entry.total_embedded_emissions:.6f}</TotalEmbeddedEmissions>',
                f'      <SpecificEmbeddedEmissions>{entry.specific_embedded_emissions:.6f}</SpecificEmbeddedEmissions>',
                f'      <CalculationMethod>{entry.calculation_method.value}</CalculationMethod>',
                '    </GoodsEntry>',
            ])

        xml_lines.extend([
            '  </GoodsEntries>',
            '  <Totals>',
            f'    <TotalQuantityTonnes>{report.total_quantity_tonnes:.4f}</TotalQuantityTonnes>',
            f'    <TotalEmbeddedEmissions>{report.total_embedded_emissions_tco2e:.6f}</TotalEmbeddedEmissions>',
            '  </Totals>',
            f'  <ProvenanceHash>{report.provenance_hash}</ProvenanceHash>',
            '</CBAMReport>',
        ])

        xml_content = "\n".join(xml_lines)

        # Store XML in report
        report.xml_content = xml_content

        logger.info(
            "XML generated for report [%s]: %d bytes, %d entries",
            report.report_id,
            len(xml_content),
            len(report.goods_entries),
        )

        return xml_content

    def validate_report(
        self,
        report: QuarterlyReport,
    ) -> List[ValidationError]:
        """Run pre-submission validation checks on a quarterly report.

        Performs comprehensive validation including:
            - EORI number format
            - Period completeness
            - Goods entry data integrity
            - Emission value plausibility
            - Calculation method consistency
            - Required fields present

        Args:
            report: QuarterlyReport to validate.

        Returns:
            List of ValidationError objects. Empty list means report is valid.
        """
        errors: List[ValidationError] = []

        # EORI validation
        if not report.importer_eori or len(report.importer_eori) < 5:
            errors.append(ValidationError(
                field="importer_eori",
                severity="ERROR",
                message="EORI number is missing or too short (minimum 5 characters)",
                code="EORI_INVALID",
            ))

        # Period validation
        if report.period.start_date >= report.period.end_date:
            errors.append(ValidationError(
                field="period",
                severity="ERROR",
                message="Period start date must be before end date",
                code="PERIOD_INVALID",
            ))

        # Goods entries validation
        if not report.goods_entries:
            errors.append(ValidationError(
                field="goods_entries",
                severity="ERROR",
                message="Report must contain at least one goods entry",
                code="NO_ENTRIES",
            ))

        for idx, entry in enumerate(report.goods_entries):
            prefix = f"goods_entries[{idx}]"

            # CN code check
            if not entry.cn_code or len(entry.cn_code) < 4:
                errors.append(ValidationError(
                    field=f"{prefix}.cn_code",
                    severity="ERROR",
                    message=f"Invalid CN code: '{entry.cn_code}'",
                    code="CN_CODE_INVALID",
                ))

            # Country check
            if not entry.country_of_origin or len(entry.country_of_origin) != 2:
                errors.append(ValidationError(
                    field=f"{prefix}.country_of_origin",
                    severity="ERROR",
                    message=f"Invalid country code: '{entry.country_of_origin}'",
                    code="COUNTRY_INVALID",
                ))

            # Quantity check
            if entry.quantity_tonnes <= 0:
                errors.append(ValidationError(
                    field=f"{prefix}.quantity_tonnes",
                    severity="ERROR",
                    message=f"Quantity must be > 0, got {entry.quantity_tonnes}",
                    code="QUANTITY_ZERO",
                ))

            # Emission plausibility (direct + indirect should equal total)
            expected_total = entry.direct_emissions + entry.indirect_emissions
            if abs(expected_total - entry.total_embedded_emissions) > 0.01:
                errors.append(ValidationError(
                    field=f"{prefix}.total_embedded_emissions",
                    severity="WARNING",
                    message=(
                        f"Total ({entry.total_embedded_emissions:.4f}) does not match "
                        f"direct + indirect ({expected_total:.4f})"
                    ),
                    code="EMISSION_SUM_MISMATCH",
                ))

            # Specific emissions consistency
            if entry.quantity_tonnes > 0:
                calculated_specific = entry.total_embedded_emissions / entry.quantity_tonnes
                if entry.specific_embedded_emissions > 0:
                    diff_pct = abs(
                        calculated_specific - entry.specific_embedded_emissions
                    ) / max(calculated_specific, 0.001) * 100
                    if diff_pct > 1.0:
                        errors.append(ValidationError(
                            field=f"{prefix}.specific_embedded_emissions",
                            severity="WARNING",
                            message=(
                                f"Specific emissions ({entry.specific_embedded_emissions:.4f}) "
                                f"differs from calculated ({calculated_specific:.4f}) by {diff_pct:.1f}%"
                            ),
                            code="SPECIFIC_MISMATCH",
                        ))

            # Emission intensity plausibility ranges
            if entry.quantity_tonnes > 0:
                intensity = entry.total_embedded_emissions / entry.quantity_tonnes
                if intensity > 50.0:
                    errors.append(ValidationError(
                        field=f"{prefix}.total_embedded_emissions",
                        severity="WARNING",
                        message=(
                            f"Emission intensity {intensity:.2f} tCO2e/t is unusually high; "
                            "please verify the data"
                        ),
                        code="INTENSITY_HIGH",
                    ))

        # Total consistency
        sum_qty = sum(e.quantity_tonnes for e in report.goods_entries)
        sum_emissions = sum(e.total_embedded_emissions for e in report.goods_entries)

        if abs(sum_qty - report.total_quantity_tonnes) > 0.01:
            errors.append(ValidationError(
                field="total_quantity_tonnes",
                severity="ERROR",
                message=(
                    f"Total quantity ({report.total_quantity_tonnes:.2f}) does not match "
                    f"sum of entries ({sum_qty:.2f})"
                ),
                code="TOTAL_QTY_MISMATCH",
            ))

        if abs(sum_emissions - report.total_embedded_emissions_tco2e) > 0.01:
            errors.append(ValidationError(
                field="total_embedded_emissions_tco2e",
                severity="ERROR",
                message=(
                    f"Total emissions ({report.total_embedded_emissions_tco2e:.4f}) does not match "
                    f"sum of entries ({sum_emissions:.4f})"
                ),
                code="TOTAL_EMISSIONS_MISMATCH",
            ))

        # Update status if no errors
        error_count = sum(1 for e in errors if e.severity == "ERROR")
        if error_count == 0:
            report.report_status = ReportStatus.VALIDATED
            logger.info(
                "Report [%s] passed validation (%d warnings)",
                report.report_id,
                len(errors),
            )
        else:
            logger.warning(
                "Report [%s] validation failed: %d errors, %d warnings",
                report.report_id,
                error_count,
                len(errors) - error_count,
            )

        return errors

    def create_amendment(
        self,
        original_report_id: str,
        updated_entries: Optional[List[GoodsEntry]] = None,
        reason: str = "",
    ) -> QuarterlyReport:
        """Create a version-controlled amendment to an existing report.

        Creates a new report based on the original, with an incremented
        version number and updated goods entries if provided.

        Args:
            original_report_id: ID of the report to amend.
            updated_entries: Replacement goods entries. If None, copies
                the original entries.
            reason: Reason for the amendment.

        Returns:
            New QuarterlyReport with amendment_version incremented.

        Raises:
            ValueError: If original report is not found.
        """
        original = self._reports.get(original_report_id)
        if original is None:
            raise ValueError(f"Original report not found: {original_report_id}")

        entries = updated_entries if updated_entries is not None else original.goods_entries
        new_version = original.amendment_version + 1

        # Recalculate totals
        total_emissions = Decimal("0")
        total_qty = Decimal("0")
        for entry in entries:
            total_emissions += _decimal(entry.total_embedded_emissions)
            total_qty += _decimal(entry.quantity_tonnes)

        amended = QuarterlyReport(
            period=original.period,
            importer_eori=original.importer_eori,
            importer_name=original.importer_name,
            goods_entries=entries,
            total_embedded_emissions_tco2e=_round_val(total_emissions, 4),
            total_quantity_tonnes=_round_val(total_qty, 2),
            report_status=ReportStatus.AMENDED,
            amendment_version=new_version,
            original_report_id=original_report_id,
        )

        amended.provenance_hash = _compute_hash(amended)
        self._reports[amended.report_id] = amended

        logger.info(
            "Amendment created [%s] v%d from [%s]: %s",
            amended.report_id,
            new_version,
            original_report_id,
            reason or "No reason specified",
        )

        return amended

    def check_deadline(
        self,
        period: QuarterlyPeriod,
        reference_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Check days until submission deadline and alert level.

        Args:
            period: QuarterlyPeriod to check.
            reference_date: Date to measure from. Defaults to today.

        Returns:
            Dictionary with:
                - days_until_deadline: Days remaining (negative = overdue)
                - days_until_amendment: Days until amendment deadline
                - alert_level: GREEN/YELLOW/ORANGE/RED/OVERDUE
                - submission_deadline: The deadline date
                - amendment_deadline: The amendment deadline date
        """
        if reference_date is None:
            reference_date = utcnow().date()

        days_sub = (period.submission_deadline - reference_date).days
        days_amend = (period.amendment_deadline - reference_date).days

        alert = self._deadline_alert_level(days_sub)

        result = {
            "quarter": f"Q{period.quarter}/{period.year}",
            "days_until_deadline": days_sub,
            "days_until_amendment": days_amend,
            "alert_level": alert,
            "submission_deadline": period.submission_deadline.isoformat(),
            "amendment_deadline": period.amendment_deadline.isoformat(),
            "reference_date": reference_date.isoformat(),
        }

        if days_sub < 0:
            logger.warning(
                "Deadline OVERDUE for Q%d/%d: %d days past due",
                period.quarter, period.year, abs(days_sub),
            )
        elif alert in ("RED", "ORANGE"):
            logger.warning(
                "Deadline approaching for Q%d/%d: %d days remaining (%s)",
                period.quarter, period.year, days_sub, alert,
            )

        return result

    def get_submission_history(
        self,
        importer_eori: str,
    ) -> List[QuarterlyReport]:
        """Get all reports for a specific importer.

        Args:
            importer_eori: EORI number to filter by.

        Returns:
            List of QuarterlyReport sorted by period (newest first).
        """
        reports = [
            r for r in self._reports.values()
            if r.importer_eori == importer_eori
        ]
        reports.sort(
            key=lambda r: (r.period.year, r.period.quarter, r.amendment_version),
            reverse=True,
        )
        return reports

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def report_count(self) -> int:
        """Number of reports assembled."""
        return self._report_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_period(self, year: int, quarter: int) -> QuarterlyPeriod:
        """Construct a QuarterlyPeriod for a given year and quarter."""
        start_month = _QUARTER_START_MONTHS[quarter]
        end_month, end_day = _QUARTER_END_MONTHS[quarter]
        year_offset, dead_month, dead_day = _SUBMISSION_DEADLINES[quarter]

        start_date = date(year, start_month, 1)
        end_date = date(year, end_month, end_day)
        submission_deadline = date(year + year_offset, dead_month, dead_day)
        amendment_deadline = submission_deadline + timedelta(days=_AMENDMENT_OFFSET_DAYS)

        return QuarterlyPeriod(
            year=year,
            quarter=quarter,
            start_date=start_date,
            end_date=end_date,
            submission_deadline=submission_deadline,
            amendment_deadline=amendment_deadline,
        )

    def _deadline_alert_level(self, days_remaining: int) -> str:
        """Determine deadline alert level from days remaining."""
        if days_remaining < 0:
            return "OVERDUE"
        if days_remaining <= 7:
            return "RED"
        if days_remaining <= 14:
            return "ORANGE"
        if days_remaining <= 30:
            return "YELLOW"
        return "GREEN"

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _xml_escape(value: Optional[str]) -> str:
    """Escape special XML characters in a string."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s
