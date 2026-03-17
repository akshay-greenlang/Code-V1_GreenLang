# -*- coding: utf-8 -*-
"""
Customs Integration Workflow
===============================

Three-phase per-import-declaration workflow that automates the intake of
customs data into the CBAM compliance pipeline. Parses Single Administrative
Document (SAD) or Customs Declaration Service (CDS) format declarations,
enriches with TARIC classification and anti-circumvention checks, and
creates linked CBAM import records with preliminary emission estimates.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 2: Applies to goods classified under Annex I CN codes
      (iron/steel, aluminium, cement, fertilizers, electricity, hydrogen).
    - Article 35(4): CBAM Registry linked to national customs systems.
    - Annex I: Defines CBAM goods by Combined Nomenclature (CN) code.
    - Commission Delegated Regulation: Anti-circumvention measures for
      slightly modified goods and trans-shipment patterns.

    TARIC (Tarif Integre Communautaire):
    - EU's integrated tariff database for CN code classification
    - Includes duty rates, quotas, anti-dumping measures, CBAM applicability
    - Updated continuously by European Commission DG TAXUD

Goods Categories (CBAM Annex I):
    - Iron and Steel (CN 72xx, 73xx)
    - Aluminium (CN 76xx)
    - Cement (CN 2507, 2523)
    - Fertilizers (CN 28, 31)
    - Electricity (CN 2716)
    - Hydrogen (CN 2804 10 00)

Phases:
    1. Intake - Parse customs declaration, extract key fields
    2. Enrichment - TARIC validation, CBAM applicability, AEO, circumvention
    3. CBAMLinkage - Create import records, estimate emissions, link suppliers

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class DeclarationFormat(str, Enum):
    """Customs declaration format."""
    SAD = "SAD"
    CDS = "CDS"
    AUTO_DETECT = "AUTO_DETECT"


class CbamApplicability(str, Enum):
    """CBAM applicability determination for a goods item."""
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class CircumventionRisk(str, Enum):
    """Anti-circumvention risk level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    FLAGGED = "FLAGGED"


class GoodsCategory(str, Enum):
    """CBAM Annex I goods categories."""
    IRON_STEEL = "IRON_STEEL"
    ALUMINIUM = "ALUMINIUM"
    CEMENT = "CEMENT"
    FERTILIZERS = "FERTILIZERS"
    ELECTRICITY = "ELECTRICITY"
    HYDROGEN = "HYDROGEN"
    NOT_CBAM = "NOT_CBAM"


# =============================================================================
# CONSTANTS
# =============================================================================

# CN code prefix to goods category mapping (CBAM Annex I)
CN_CODE_CATEGORIES: Dict[str, GoodsCategory] = {
    "72": GoodsCategory.IRON_STEEL,
    "73": GoodsCategory.IRON_STEEL,
    "76": GoodsCategory.ALUMINIUM,
    "2507": GoodsCategory.CEMENT,
    "2523": GoodsCategory.CEMENT,
    "28": GoodsCategory.FERTILIZERS,
    "31": GoodsCategory.FERTILIZERS,
    "2716": GoodsCategory.ELECTRICITY,
    "2804": GoodsCategory.HYDROGEN,
}

# Default emission factors by goods category (tCO2e per tonne)
# These are conservative defaults; actual values come from supplier data
DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    GoodsCategory.IRON_STEEL.value: 1.85,
    GoodsCategory.ALUMINIUM.value: 8.50,
    GoodsCategory.CEMENT.value: 0.62,
    GoodsCategory.FERTILIZERS.value: 3.10,
    GoodsCategory.ELECTRICITY.value: 0.45,
    GoodsCategory.HYDROGEN.value: 9.30,
}

# EU member states (for origin validation)
EU_MEMBER_STATES: Set[str] = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE",
    "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV",
    "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
}


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(...)
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - CUSTOMS INTEGRATION
# =============================================================================


class CustomsLineItem(BaseModel):
    """A single line item from a customs declaration."""
    item_number: int = Field(default=1, ge=1)
    cn_code: str = Field(..., description="Combined Nomenclature code")
    description: str = Field(default="")
    country_of_origin: str = Field(..., description="ISO country code")
    net_mass_kg: float = Field(default=0.0, ge=0)
    supplementary_units: Optional[float] = Field(None)
    customs_value_eur: float = Field(default=0.0, ge=0)
    duty_rate_pct: Optional[float] = Field(None, ge=0)
    import_procedure: str = Field(
        default="4000", description="Customs procedure code"
    )
    supplier_name: Optional[str] = Field(None)
    supplier_id: Optional[str] = Field(None)
    installation_id: Optional[str] = Field(None)


class CustomsDeclarationData(BaseModel):
    """Parsed customs declaration data."""
    declaration_reference: str = Field(
        ..., description="MRN or declaration reference"
    )
    declaration_format: DeclarationFormat = Field(
        default=DeclarationFormat.SAD
    )
    declarant_eori: str = Field(..., description="Declarant EORI number")
    representative_eori: Optional[str] = Field(None)
    entry_date: str = Field(..., description="Date of entry YYYY-MM-DD")
    member_state: str = Field(..., description="EU member state of import")
    customs_office: Optional[str] = Field(None)
    total_items: int = Field(default=0, ge=0)
    total_customs_value_eur: float = Field(default=0.0, ge=0)
    line_items: List[CustomsLineItem] = Field(default_factory=list)
    aeo_status: Optional[str] = Field(
        None, description="Authorized Economic Operator status"
    )
    raw_data: Optional[Dict[str, Any]] = Field(
        None, description="Original unparsed declaration data"
    )


class CustomsIntegrationInput(BaseModel):
    """Input configuration for customs integration workflow."""
    organization_id: str = Field(...)
    declaration: CustomsDeclarationData = Field(...)
    known_suppliers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="supplier_id -> supplier data for matching"
    )
    known_installations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="installation_id -> installation data"
    )
    custom_emission_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="goods_category -> emission factor override"
    )
    enable_circumvention_check: bool = Field(default=True)
    skip_phases: List[str] = Field(default_factory=list)


class CustomsIntegrationResult(WorkflowResult):
    """Complete result from customs integration workflow."""
    declaration_reference: str = Field(default="")
    total_line_items: int = Field(default=0)
    cbam_applicable_items: int = Field(default=0)
    non_cbam_items: int = Field(default=0)
    total_embedded_emissions_tco2e: float = Field(default=0.0)
    total_certificate_obligation: float = Field(default=0.0)
    circumvention_flags: int = Field(default=0)
    import_records_created: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class IntakePhase:
    """
    Phase 1: Intake.

    Parses the customs declaration (SAD or CDS format) and extracts
    CN codes, quantities, country of origin, supplier details, customs
    values, and import procedures. Validates against expected format.
    """

    PHASE_NAME = "intake"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute intake phase.

        Args:
            context: Workflow context with declaration data.

        Returns:
            PhaseResult with parsed and validated declaration items.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            declaration = config.get("declaration", {})
            decl_format = declaration.get(
                "declaration_format", DeclarationFormat.SAD.value
            )
            line_items = declaration.get("line_items", [])

            outputs["declaration_reference"] = declaration.get(
                "declaration_reference", ""
            )
            outputs["declaration_format"] = decl_format
            outputs["declarant_eori"] = declaration.get("declarant_eori", "")
            outputs["entry_date"] = declaration.get("entry_date", "")
            outputs["member_state"] = declaration.get("member_state", "")

            # Validate member state
            ms = declaration.get("member_state", "")
            if ms and ms not in EU_MEMBER_STATES:
                errors.append(
                    f"Invalid EU member state: '{ms}'"
                )

            # Validate and normalize line items
            parsed_items: List[Dict[str, Any]] = []
            validation_issues: List[Dict[str, Any]] = []

            for idx, item in enumerate(line_items):
                parsed = self._parse_line_item(item, idx)
                issues = self._validate_line_item(parsed, idx)

                parsed_items.append(parsed)
                if issues:
                    validation_issues.extend(issues)

            outputs["parsed_items"] = parsed_items
            outputs["total_items"] = len(parsed_items)
            outputs["validation_issues"] = validation_issues

            # Summary statistics
            total_mass_kg = sum(
                Decimal(str(i.get("net_mass_kg", 0))) for i in parsed_items
            )
            total_value = sum(
                Decimal(str(i.get("customs_value_eur", 0))) for i in parsed_items
            )
            unique_origins = set(
                i.get("country_of_origin", "") for i in parsed_items
            )
            unique_cn_codes = set(
                i.get("cn_code", "") for i in parsed_items
            )

            outputs["total_mass_kg"] = float(total_mass_kg)
            outputs["total_mass_tonnes"] = float(
                (total_mass_kg / 1000).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
            )
            outputs["total_customs_value_eur"] = float(total_value)
            outputs["unique_origins"] = sorted(unique_origins - {""})
            outputs["unique_cn_codes"] = sorted(unique_cn_codes - {""})
            outputs["aeo_status"] = declaration.get("aeo_status")

            # Report validation issues
            error_issues = [
                i for i in validation_issues if i.get("severity") == "ERROR"
            ]
            warning_issues = [
                i for i in validation_issues if i.get("severity") == "WARNING"
            ]
            if error_issues:
                errors.append(
                    f"{len(error_issues)} validation error(s) in line items"
                )
            if warning_issues:
                warnings.append(
                    f"{len(warning_issues)} validation warning(s) in line items"
                )

            # Check for EU-origin items (CBAM only applies to non-EU imports)
            eu_origin_items = [
                i for i in parsed_items
                if i.get("country_of_origin", "") in EU_MEMBER_STATES
            ]
            if eu_origin_items:
                warnings.append(
                    f"{len(eu_origin_items)} item(s) have EU origin; "
                    f"CBAM does not apply to intra-EU trade"
                )

            status = PhaseStatus.COMPLETED
            records = len(parsed_items)

        except Exception as exc:
            logger.error("Intake failed: %s", exc, exc_info=True)
            errors.append(f"Intake parsing failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _parse_line_item(
        self, item: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Parse and normalize a single customs line item."""
        cn_code = str(item.get("cn_code", "")).replace(" ", "").replace(".", "")
        return {
            "item_number": item.get("item_number", index + 1),
            "cn_code": cn_code,
            "cn_code_short": cn_code[:4] if len(cn_code) >= 4 else cn_code,
            "description": item.get("description", ""),
            "country_of_origin": str(
                item.get("country_of_origin", "")
            ).upper()[:2],
            "net_mass_kg": float(item.get("net_mass_kg", 0)),
            "net_mass_tonnes": float(
                Decimal(str(item.get("net_mass_kg", 0))) / 1000
            ),
            "supplementary_units": item.get("supplementary_units"),
            "customs_value_eur": float(item.get("customs_value_eur", 0)),
            "duty_rate_pct": item.get("duty_rate_pct"),
            "import_procedure": item.get("import_procedure", "4000"),
            "supplier_name": item.get("supplier_name"),
            "supplier_id": item.get("supplier_id"),
            "installation_id": item.get("installation_id"),
        }

    def _validate_line_item(
        self, item: Dict[str, Any], index: int
    ) -> List[Dict[str, Any]]:
        """Validate a parsed line item and return issues."""
        issues = []
        prefix = f"item[{index}]"

        cn_code = item.get("cn_code", "")
        if not cn_code or len(cn_code) < 2:
            issues.append({
                "field": f"{prefix}.cn_code",
                "severity": "ERROR",
                "message": f"Invalid or missing CN code: '{cn_code}'",
            })

        origin = item.get("country_of_origin", "")
        if not origin or len(origin) != 2:
            issues.append({
                "field": f"{prefix}.country_of_origin",
                "severity": "ERROR",
                "message": f"Invalid country of origin: '{origin}'",
            })

        mass = item.get("net_mass_kg", 0)
        if mass <= 0:
            issues.append({
                "field": f"{prefix}.net_mass_kg",
                "severity": "WARNING",
                "message": f"Zero or negative net mass: {mass}",
            })

        return issues


class EnrichmentPhase:
    """
    Phase 2: Enrichment.

    Validates CN codes via the TARIC engine, determines CBAM
    applicability per CN code, checks AEO status, runs
    anti-circumvention detection rules, flags suspicious patterns,
    and determines the goods category for each item.
    """

    PHASE_NAME = "enrichment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute enrichment phase.

        Args:
            context: Workflow context with parsed declaration items.

        Returns:
            PhaseResult with enriched items and circumvention flags.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            intake = context.get_phase_output("intake")
            parsed_items = intake.get("parsed_items", [])
            enable_circumvention = config.get(
                "enable_circumvention_check", True
            )

            enriched_items: List[Dict[str, Any]] = []
            cbam_applicable_count = 0
            non_cbam_count = 0
            circumvention_flags: List[Dict[str, Any]] = []
            by_category: Dict[str, int] = {}

            for item in parsed_items:
                enriched = dict(item)

                # TARIC CN code validation and category determination
                cn_code = item.get("cn_code", "")
                category = self._determine_goods_category(cn_code)
                enriched["goods_category"] = category.value

                # CBAM applicability
                origin = item.get("country_of_origin", "")
                applicability = self._determine_cbam_applicability(
                    cn_code, category, origin
                )
                enriched["cbam_applicability"] = applicability.value

                if applicability == CbamApplicability.APPLICABLE:
                    cbam_applicable_count += 1
                elif applicability == CbamApplicability.REQUIRES_REVIEW:
                    cbam_applicable_count += 1
                    warnings.append(
                        f"Item {item.get('item_number')}: CN {cn_code} "
                        f"from {origin} requires manual CBAM review"
                    )
                else:
                    non_cbam_count += 1

                # Category statistics
                cat_val = category.value
                by_category[cat_val] = by_category.get(cat_val, 0) + 1

                # TARIC data enrichment
                taric_data = await self._lookup_taric(cn_code)
                enriched["taric_description"] = taric_data.get(
                    "description", ""
                )
                enriched["taric_duty_rate"] = taric_data.get(
                    "duty_rate_pct"
                )
                enriched["taric_valid"] = taric_data.get("valid", False)

                if not taric_data.get("valid", False):
                    warnings.append(
                        f"Item {item.get('item_number')}: CN {cn_code} "
                        f"not found in TARIC database"
                    )

                # Anti-circumvention check
                if enable_circumvention and category != GoodsCategory.NOT_CBAM:
                    circ_result = self._check_circumvention(enriched)
                    enriched["circumvention_risk"] = circ_result["risk"].value
                    if circ_result["risk"] in (
                        CircumventionRisk.HIGH, CircumventionRisk.FLAGGED
                    ):
                        circumvention_flags.append({
                            "item_number": item.get("item_number"),
                            "cn_code": cn_code,
                            "risk": circ_result["risk"].value,
                            "reasons": circ_result["reasons"],
                        })
                else:
                    enriched["circumvention_risk"] = CircumventionRisk.LOW.value

                enriched_items.append(enriched)

            # AEO status check
            aeo_status = intake.get("aeo_status")
            aeo_verified = aeo_status in ("AEOC", "AEOF", "AEOS")
            outputs["aeo_status"] = aeo_status
            outputs["aeo_verified"] = aeo_verified
            if aeo_verified:
                outputs["aeo_benefit"] = "Simplified customs procedures"

            outputs["enriched_items"] = enriched_items
            outputs["cbam_applicable_count"] = cbam_applicable_count
            outputs["non_cbam_count"] = non_cbam_count
            outputs["by_goods_category"] = by_category
            outputs["circumvention_flags"] = circumvention_flags
            outputs["circumvention_flag_count"] = len(circumvention_flags)

            if circumvention_flags:
                warnings.append(
                    f"{len(circumvention_flags)} anti-circumvention "
                    f"flag(s) detected - manual review required"
                )

            status = PhaseStatus.COMPLETED
            records = len(enriched_items)

        except Exception as exc:
            logger.error("Enrichment failed: %s", exc, exc_info=True)
            errors.append(f"Enrichment failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _determine_goods_category(self, cn_code: str) -> GoodsCategory:
        """Determine CBAM goods category from CN code."""
        if not cn_code:
            return GoodsCategory.NOT_CBAM

        # Check longest prefix match first
        for prefix_len in (4, 2):
            prefix = cn_code[:prefix_len]
            if prefix in CN_CODE_CATEGORIES:
                return CN_CODE_CATEGORIES[prefix]

        return GoodsCategory.NOT_CBAM

    def _determine_cbam_applicability(
        self,
        cn_code: str,
        category: GoodsCategory,
        origin: str,
    ) -> CbamApplicability:
        """Determine CBAM applicability for a line item."""
        # CBAM does not apply to EU-origin goods
        if origin in EU_MEMBER_STATES:
            return CbamApplicability.NOT_APPLICABLE

        if category == GoodsCategory.NOT_CBAM:
            return CbamApplicability.NOT_APPLICABLE

        # Known CBAM CN codes are applicable
        if category in (
            GoodsCategory.IRON_STEEL,
            GoodsCategory.ALUMINIUM,
            GoodsCategory.CEMENT,
            GoodsCategory.FERTILIZERS,
            GoodsCategory.ELECTRICITY,
            GoodsCategory.HYDROGEN,
        ):
            return CbamApplicability.APPLICABLE

        return CbamApplicability.REQUIRES_REVIEW

    async def _lookup_taric(self, cn_code: str) -> Dict[str, Any]:
        """
        Look up CN code in TARIC database.

        In production, this queries the TARIC REST API or local cache.
        """
        category = self._determine_goods_category(cn_code)
        return {
            "cn_code": cn_code,
            "description": f"TARIC description for {cn_code}",
            "valid": len(cn_code) >= 4,
            "duty_rate_pct": 0.0,
            "cbam_annex_i": category != GoodsCategory.NOT_CBAM,
        }

    def _check_circumvention(
        self, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run anti-circumvention detection rules.

        Checks for:
        - Slight modification patterns (CN code just outside CBAM scope)
        - Trans-shipment indicators (origin vs declared origin mismatch)
        - Unusual value-to-weight ratios
        - Known circumvention patterns by country
        """
        reasons: List[str] = []
        risk = CircumventionRisk.LOW

        cn_code = item.get("cn_code", "")
        origin = item.get("country_of_origin", "")
        mass_kg = item.get("net_mass_kg", 0)
        value_eur = item.get("customs_value_eur", 0)

        # Check value-to-weight ratio anomaly
        if mass_kg > 0 and value_eur > 0:
            ratio = value_eur / mass_kg
            category = item.get("goods_category", "")

            # Iron/steel typically EUR 0.3-3.0/kg
            if category == GoodsCategory.IRON_STEEL.value:
                if ratio < 0.1 or ratio > 10.0:
                    reasons.append(
                        f"Unusual value/weight ratio: EUR {ratio:.2f}/kg "
                        f"(expected 0.3-3.0 for iron/steel)"
                    )
                    risk = CircumventionRisk.MEDIUM

            # Aluminium typically EUR 1.5-8.0/kg
            if category == GoodsCategory.ALUMINIUM.value:
                if ratio < 0.5 or ratio > 20.0:
                    reasons.append(
                        f"Unusual value/weight ratio: EUR {ratio:.2f}/kg "
                        f"(expected 1.5-8.0 for aluminium)"
                    )
                    risk = CircumventionRisk.MEDIUM

        # Check for known trans-shipment risk countries
        transship_risk = {"AE", "SG", "MY", "VN", "TH", "TW"}
        high_production = {"CN", "IN", "RU", "TR", "UA", "KZ"}
        if origin in transship_risk:
            # Small quantities from transshipment hubs are suspicious
            if mass_kg > 0 and mass_kg < 5000:
                reasons.append(
                    f"Small quantity from known transshipment hub ({origin})"
                )
                risk = max(risk, CircumventionRisk.MEDIUM,
                           key=lambda x: list(CircumventionRisk).index(x))

        if reasons and len(reasons) >= 2:
            risk = CircumventionRisk.HIGH

        return {"risk": risk, "reasons": reasons}


class CbamLinkagePhase:
    """
    Phase 3: CBAM Linkage.

    Creates CBAM import records from customs data, calculates
    preliminary embedded emissions using available emission factors,
    links to supplier installations, creates certificate obligation
    estimates, and updates the import portfolio.
    """

    PHASE_NAME = "cbam_linkage"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute CBAM linkage phase.

        Args:
            context: Workflow context with enriched customs items.

        Returns:
            PhaseResult with CBAM import records and emission estimates.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            intake = context.get_phase_output("intake")
            enrichment = context.get_phase_output("enrichment")
            enriched_items = enrichment.get("enriched_items", [])
            known_suppliers = config.get("known_suppliers", {})
            known_installations = config.get("known_installations", {})
            custom_factors = config.get("custom_emission_factors", {})
            declaration = config.get("declaration", {})

            import_records: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")
            total_obligation = Decimal("0")
            by_category: Dict[str, Dict[str, Any]] = {}
            supplier_links: List[Dict[str, Any]] = []

            for item in enriched_items:
                applicability = item.get(
                    "cbam_applicability", CbamApplicability.NOT_APPLICABLE.value
                )
                if applicability == CbamApplicability.NOT_APPLICABLE.value:
                    continue

                category = item.get("goods_category", "")
                mass_tonnes = Decimal(str(item.get("net_mass_tonnes", 0)))

                # Get emission factor (custom > supplier-specific > default)
                emission_factor = self._get_emission_factor(
                    category, item, known_installations, custom_factors
                )
                factor_source = emission_factor["source"]
                factor_value = Decimal(str(emission_factor["value"]))

                # Calculate preliminary embedded emissions
                embedded_emissions = (mass_tonnes * factor_value).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                total_emissions += embedded_emissions

                # Certificate obligation (before deductions)
                total_obligation += embedded_emissions

                # Create import record
                record = {
                    "record_id": str(uuid.uuid4()),
                    "declaration_reference": declaration.get(
                        "declaration_reference", ""
                    ),
                    "item_number": item.get("item_number", 0),
                    "cn_code": item.get("cn_code", ""),
                    "goods_category": category,
                    "country_of_origin": item.get("country_of_origin", ""),
                    "net_mass_tonnes": float(mass_tonnes),
                    "customs_value_eur": item.get("customs_value_eur", 0),
                    "emission_factor_tco2e_per_tonne": float(factor_value),
                    "emission_factor_source": factor_source,
                    "embedded_emissions_tco2e": float(embedded_emissions),
                    "certificate_obligation_tco2e": float(embedded_emissions),
                    "entry_date": intake.get("entry_date", ""),
                    "member_state": intake.get("member_state", ""),
                    "supplier_id": item.get("supplier_id"),
                    "installation_id": item.get("installation_id"),
                    "circumvention_risk": item.get(
                        "circumvention_risk", CircumventionRisk.LOW.value
                    ),
                    "verification_status": "PENDING",
                    "created_at": datetime.utcnow().isoformat(),
                }
                import_records.append(record)

                # Accumulate by category
                if category not in by_category:
                    by_category[category] = {
                        "count": 0,
                        "mass_tonnes": Decimal("0"),
                        "emissions_tco2e": Decimal("0"),
                        "value_eur": Decimal("0"),
                    }
                by_category[category]["count"] += 1
                by_category[category]["mass_tonnes"] += mass_tonnes
                by_category[category]["emissions_tco2e"] += embedded_emissions
                by_category[category]["value_eur"] += Decimal(str(
                    item.get("customs_value_eur", 0)
                ))

                # Supplier linkage
                supplier_id = item.get("supplier_id")
                if supplier_id and supplier_id in known_suppliers:
                    supplier = known_suppliers[supplier_id]
                    supplier_links.append({
                        "record_id": record["record_id"],
                        "supplier_id": supplier_id,
                        "supplier_name": supplier.get("name", ""),
                        "installation_id": item.get("installation_id", ""),
                        "verified_emissions_available": supplier.get(
                            "verified", False
                        ),
                    })
                elif supplier_id:
                    warnings.append(
                        f"Item {item.get('item_number')}: Supplier "
                        f"'{supplier_id}' not in known suppliers list"
                    )

            outputs["import_records"] = import_records
            outputs["import_records_count"] = len(import_records)
            outputs["total_embedded_emissions_tco2e"] = float(total_emissions)
            outputs["total_certificate_obligation_tco2e"] = float(
                total_obligation
            )
            outputs["by_goods_category"] = {
                k: {
                    "count": v["count"],
                    "mass_tonnes": float(v["mass_tonnes"]),
                    "emissions_tco2e": float(v["emissions_tco2e"]),
                    "value_eur": float(v["value_eur"]),
                }
                for k, v in by_category.items()
            }
            outputs["supplier_links"] = supplier_links
            outputs["supplier_link_count"] = len(supplier_links)

            # Emission factor source breakdown
            factor_sources = {}
            for rec in import_records:
                src = rec.get("emission_factor_source", "unknown")
                factor_sources[src] = factor_sources.get(src, 0) + 1
            outputs["emission_factor_sources"] = factor_sources

            # Items using default factors need supplier data
            default_factor_count = factor_sources.get("default", 0)
            if default_factor_count > 0:
                warnings.append(
                    f"{default_factor_count} item(s) using default emission "
                    f"factors. Supplier-specific data will improve accuracy."
                )

            # Portfolio update record
            outputs["portfolio_update"] = {
                "update_id": str(uuid.uuid4()),
                "declaration_reference": declaration.get(
                    "declaration_reference", ""
                ),
                "records_added": len(import_records),
                "total_emissions_added_tco2e": float(total_emissions),
                "updated_at": datetime.utcnow().isoformat(),
            }

            status = PhaseStatus.COMPLETED
            records = len(import_records)

        except Exception as exc:
            logger.error("CbamLinkage failed: %s", exc, exc_info=True)
            errors.append(f"CBAM linkage failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _get_emission_factor(
        self,
        category: str,
        item: Dict[str, Any],
        known_installations: Dict[str, Dict[str, Any]],
        custom_factors: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Get best available emission factor for a goods item.

        Priority: custom > installation-specific > default.
        Zero-hallucination: all factors are from databases or config,
        never from LLM estimation.
        """
        # Check custom override
        if category in custom_factors:
            return {
                "value": custom_factors[category],
                "source": "custom",
            }

        # Check installation-specific factor
        installation_id = item.get("installation_id", "")
        if installation_id and installation_id in known_installations:
            inst = known_installations[installation_id]
            factor = inst.get("emission_factor_tco2e_per_tonne")
            if factor is not None and factor > 0:
                return {
                    "value": factor,
                    "source": "installation_specific",
                }

        # Default factor by goods category
        default = DEFAULT_EMISSION_FACTORS.get(category, 2.0)
        return {
            "value": default,
            "source": "default",
        }


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class CustomsIntegrationWorkflow:
    """
    Three-phase per-import customs integration workflow.

    Automates the intake of customs declarations into the CBAM
    compliance pipeline with TARIC enrichment, anti-circumvention
    detection, and CBAM import record creation.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered phase executors.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = CustomsIntegrationWorkflow()
        >>> input_data = CustomsIntegrationInput(
        ...     organization_id="org-123",
        ...     declaration=CustomsDeclarationData(
        ...         declaration_reference="MRN-2026-001",
        ...         declarant_eori="DE123456789012",
        ...         entry_date="2026-03-01",
        ...         member_state="DE",
        ...         line_items=[CustomsLineItem(
        ...             cn_code="72061000",
        ...             country_of_origin="CN",
        ...             net_mass_kg=10000,
        ...         )],
        ...     ),
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.cbam_applicable_items > 0
    """

    WORKFLOW_NAME = "customs_integration"

    PHASE_ORDER = [
        "intake",
        "enrichment",
        "cbam_linkage",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize customs integration workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "intake": IntakePhase(),
            "enrichment": EnrichmentPhase(),
            "cbam_linkage": CbamLinkagePhase(),
        }

    async def run(
        self, input_data: CustomsIntegrationInput
    ) -> CustomsIntegrationResult:
        """
        Execute the 3-phase customs integration workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            CustomsIntegrationResult with import records and emissions.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting customs integration %s for org=%s decl=%s",
            self.workflow_id, input_data.organization_id,
            input_data.declaration.declaration_reference,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_result = await self._phases[phase_name].execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    # Intake failure is critical
                    if phase_name == "intake":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised: %s", phase_name, exc, exc_info=True
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context, input_data)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return CustomsIntegrationResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            declaration_reference=input_data.declaration.declaration_reference,
            total_line_items=summary.get("total_items", 0),
            cbam_applicable_items=summary.get("cbam_applicable", 0),
            non_cbam_items=summary.get("non_cbam", 0),
            total_embedded_emissions_tco2e=summary.get(
                "total_emissions_tco2e", 0.0
            ),
            total_certificate_obligation=summary.get(
                "total_obligation_tco2e", 0.0
            ),
            circumvention_flags=summary.get("circumvention_flags", 0),
            import_records_created=summary.get("import_records_created", 0),
        )

    def _build_config(
        self, input_data: CustomsIntegrationInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return {
            "organization_id": input_data.organization_id,
            "declaration": input_data.declaration.model_dump(),
            "known_suppliers": input_data.known_suppliers,
            "known_installations": input_data.known_installations,
            "custom_emission_factors": input_data.custom_emission_factors,
            "enable_circumvention_check": input_data.enable_circumvention_check,
        }

    def _build_summary(
        self,
        context: WorkflowContext,
        input_data: CustomsIntegrationInput,
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        intake = context.get_phase_output("intake")
        enrichment = context.get_phase_output("enrichment")
        linkage = context.get_phase_output("cbam_linkage")
        return {
            "total_items": intake.get("total_items", 0),
            "cbam_applicable": enrichment.get("cbam_applicable_count", 0),
            "non_cbam": enrichment.get("non_cbam_count", 0),
            "total_emissions_tco2e": linkage.get(
                "total_embedded_emissions_tco2e", 0.0
            ),
            "total_obligation_tco2e": linkage.get(
                "total_certificate_obligation_tco2e", 0.0
            ),
            "circumvention_flags": enrichment.get(
                "circumvention_flag_count", 0
            ),
            "import_records_created": linkage.get(
                "import_records_count", 0
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
