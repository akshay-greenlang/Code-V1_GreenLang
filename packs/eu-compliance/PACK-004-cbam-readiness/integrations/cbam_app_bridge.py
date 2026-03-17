# -*- coding: utf-8 -*-
"""
CBAMAppBridge - Primary Bridge to GL-CBAM-APP v1.1 for CBAM Readiness Pack
============================================================================

This module provides the primary integration bridge between PACK-004 CBAM
Readiness Pack and the GL-CBAM-APP v1.1 application layer. It wraps each
GL-CBAM-APP module (certificate engine, quarterly engine, supplier portal,
de minimis engine, verification workflow, emission calculator) with
pack-compatible proxy interfaces.

Graceful degradation is enforced: if GL-CBAM-APP v1.1 is not available on
the import path, the bridge falls back to stub mode, providing safe default
responses so that pack health checks, demos, and integration tests continue
to function.

Example:
    >>> bridge = CBAMAppBridge()
    >>> cert_engine = bridge.get_certificate_engine()
    >>> price = cert_engine.calculate_price(ets_price=75.0, emissions_tco2=100.0)
    >>> print(f"Certificate obligation: {price} EUR")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Attempt to import GL-CBAM-APP v1.1 modules
# =============================================================================

_CBAM_APP_AVAILABLE = False
_cbam_app_modules: Dict[str, Any] = {}

try:
    from greenlang.apps.cbam import (
        engines as cbam_engines,
        workflows as cbam_workflows,
    )
    _CBAM_APP_AVAILABLE = True
    _cbam_app_modules["engines"] = cbam_engines
    _cbam_app_modules["workflows"] = cbam_workflows
    logger.info("GL-CBAM-APP v1.1 modules loaded successfully")
except ImportError:
    logger.info(
        "GL-CBAM-APP v1.1 not available on import path; "
        "CBAMAppBridge will operate in stub mode"
    )


# =============================================================================
# Data Models
# =============================================================================


class AppHealthStatus(BaseModel):
    """Health status of the GL-CBAM-APP connection."""
    app_available: bool = Field(default=False, description="Whether GL-CBAM-APP is importable")
    stub_mode: bool = Field(default=True, description="Whether running in stub mode")
    modules_loaded: List[str] = Field(default_factory=list, description="Loaded module names")
    version: str = Field(default="unknown", description="GL-CBAM-APP version")
    check_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Health check timestamp",
    )
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class CNCodeEntry(BaseModel):
    """A CN code entry from GL-CBAM-APP."""
    cn_code: str = Field(..., description="8-digit CN code")
    description: str = Field(default="", description="Goods description")
    category: str = Field(default="", description="CBAM goods category")
    unit: str = Field(default="tonnes", description="Reporting unit")


class EmissionFactorEntry(BaseModel):
    """An emission factor entry from GL-CBAM-APP."""
    factor_id: str = Field(..., description="Factor identifier")
    goods_category: str = Field(..., description="CBAM goods category")
    production_route: str = Field(default="default", description="Production route")
    factor_value: float = Field(..., description="Emission factor in tCO2/t product")
    source: str = Field(default="EU default", description="Factor data source")


class ComplianceRule(BaseModel):
    """A CBAM compliance rule from GL-CBAM-APP."""
    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., description="Rule display name")
    description: str = Field(default="", description="Rule description")
    severity: str = Field(default="error", description="Severity: error, warning, info")
    category: str = Field(default="", description="Rule category")
    article_reference: str = Field(default="", description="CBAM Regulation article reference")


class CountryCarbonPrice(BaseModel):
    """Country carbon price data from GL-CBAM-APP."""
    country_code: str = Field(..., description="ISO alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    price_eur_per_tco2: float = Field(default=0.0, description="Price in EUR/tCO2")
    scheme_name: str = Field(default="", description="Carbon pricing scheme name")
    deduction_eligible: bool = Field(default=False, description="Deductible under CBAM Art. 9")


# =============================================================================
# Proxy Classes
# =============================================================================


class CertificateEngineProxy:
    """Proxy for GL-CBAM-APP CertificateEngine.

    Wraps the certificate price calculation and obligation assessment
    functionality from GL-CBAM-APP.
    """

    def __init__(self, engine: Any = None) -> None:
        """Initialize the proxy.

        Args:
            engine: Real engine instance, or None for stub mode.
        """
        self._engine = engine
        self._stub_mode = engine is None

    def calculate_price(self, ets_price: float, emissions_tco2: float) -> float:
        """Calculate the CBAM certificate obligation in EUR.

        Args:
            ets_price: EU ETS price in EUR/tCO2.
            emissions_tco2: Total embedded emissions in tCO2.

        Returns:
            Certificate obligation amount in EUR.
        """
        if self._engine is not None:
            calc_fn = getattr(self._engine, "calculate_price", None)
            if calc_fn:
                return calc_fn(ets_price, emissions_tco2)

        return round(ets_price * emissions_tco2, 2)

    def calculate_obligation(
        self,
        embedded_emissions: float,
        certificate_price: float,
        free_allocation_deduction: float = 0.0,
        origin_carbon_price_deduction: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculate full certificate obligation with deductions.

        Args:
            embedded_emissions: Embedded emissions in tCO2.
            certificate_price: Certificate price in EUR/tCO2.
            free_allocation_deduction: Free allocation deduction in tCO2.
            origin_carbon_price_deduction: Origin carbon price deduction in EUR.

        Returns:
            Dictionary with obligation breakdown.
        """
        if self._engine is not None:
            fn = getattr(self._engine, "calculate_obligation", None)
            if fn:
                return fn(
                    embedded_emissions, certificate_price,
                    free_allocation_deduction, origin_carbon_price_deduction,
                )

        net_emissions = max(0.0, embedded_emissions - free_allocation_deduction)
        gross_obligation = round(net_emissions * certificate_price, 2)
        net_obligation = round(
            max(0.0, gross_obligation - origin_carbon_price_deduction), 2
        )

        return {
            "embedded_emissions_tco2": embedded_emissions,
            "free_allocation_deduction_tco2": free_allocation_deduction,
            "net_emissions_tco2": net_emissions,
            "certificate_price_eur": certificate_price,
            "gross_obligation_eur": gross_obligation,
            "origin_carbon_price_deduction_eur": origin_carbon_price_deduction,
            "net_obligation_eur": net_obligation,
            "certificates_required": int(net_emissions),
        }

    @property
    def is_stub(self) -> bool:
        """Whether this proxy is running in stub mode."""
        return self._stub_mode


class QuarterlyEngineProxy:
    """Proxy for GL-CBAM-APP QuarterlyReportingEngine."""

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._stub_mode = engine is None

    def generate_report(
        self,
        quarter: int,
        year: int,
        import_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a quarterly CBAM report.

        Args:
            quarter: Quarter number (1-4).
            year: Reporting year.
            import_data: List of import records.

        Returns:
            Quarterly report dictionary.
        """
        if self._engine is not None:
            fn = getattr(self._engine, "generate_report", None)
            if fn:
                return fn(quarter, year, import_data)

        total_mass = sum(
            float(item.get("net_mass_kg", 0.0)) for item in import_data
        )
        total_emissions = sum(
            float(item.get("embedded_emissions_tco2", 0.0)) for item in import_data
        )

        return {
            "quarter": quarter,
            "year": year,
            "total_imports": len(import_data),
            "total_mass_tonnes": round(total_mass / 1000.0, 2),
            "total_embedded_emissions_tco2": round(total_emissions, 2),
            "report_status": "draft",
            "generated_at": datetime.utcnow().isoformat(),
            "stub_mode": True,
        }

    @property
    def is_stub(self) -> bool:
        return self._stub_mode


class SupplierPortalProxy:
    """Proxy for GL-CBAM-APP SupplierPortal."""

    def __init__(self, portal: Any = None) -> None:
        self._portal = portal
        self._stub_mode = portal is None

    def register_supplier(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new supplier.

        Args:
            supplier_data: Supplier registration data.

        Returns:
            Registration confirmation.
        """
        if self._portal is not None:
            fn = getattr(self._portal, "register_supplier", None)
            if fn:
                return fn(supplier_data)

        return {
            "supplier_id": str(uuid4())[:12],
            "name": supplier_data.get("name", "Unknown"),
            "country": supplier_data.get("country", ""),
            "status": "registered",
            "registered_at": datetime.utcnow().isoformat(),
            "stub_mode": True,
        }

    def get_supplier_emissions(self, supplier_id: str) -> Dict[str, Any]:
        """Get emission data for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Supplier emission data.
        """
        if self._portal is not None:
            fn = getattr(self._portal, "get_supplier_emissions", None)
            if fn:
                return fn(supplier_id)

        return {
            "supplier_id": supplier_id,
            "has_verified_data": False,
            "default_values_used": True,
            "emissions_per_tonne": 0.0,
            "stub_mode": True,
        }

    @property
    def is_stub(self) -> bool:
        return self._stub_mode


class DeMinimisProxy:
    """Proxy for GL-CBAM-APP DeMinimisEngine."""

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._stub_mode = engine is None

    def assess(
        self,
        total_value_eur: float,
        total_mass_kg: float,
        cn_code: str = "",
    ) -> Dict[str, Any]:
        """Assess whether an import falls below the de minimis threshold.

        Per CBAM Article 2(3), consignments of a value not exceeding EUR 150
        are excluded, along with electricity imports below a threshold.

        Args:
            total_value_eur: Total customs value in EUR.
            total_mass_kg: Total mass in kg.
            cn_code: CN code for goods-specific thresholds.

        Returns:
            De minimis assessment result.
        """
        if self._engine is not None:
            fn = getattr(self._engine, "assess", None)
            if fn:
                return fn(total_value_eur, total_mass_kg, cn_code)

        is_below = total_value_eur < 150.0
        return {
            "total_value_eur": total_value_eur,
            "total_mass_kg": total_mass_kg,
            "cn_code": cn_code,
            "threshold_eur": 150.0,
            "is_below_threshold": is_below,
            "exemption_applicable": is_below,
            "stub_mode": True,
        }

    @property
    def is_stub(self) -> bool:
        return self._stub_mode


class VerificationProxy:
    """Proxy for GL-CBAM-APP VerificationWorkflow."""

    def __init__(self, workflow: Any = None) -> None:
        self._workflow = workflow
        self._stub_mode = workflow is None

    def start_verification(
        self,
        report_id: str,
        verifier_id: str = "",
    ) -> Dict[str, Any]:
        """Start a verification workflow for a CBAM report.

        Args:
            report_id: Report to verify.
            verifier_id: Assigned verifier.

        Returns:
            Verification workflow status.
        """
        if self._workflow is not None:
            fn = getattr(self._workflow, "start_verification", None)
            if fn:
                return fn(report_id, verifier_id)

        return {
            "verification_id": str(uuid4())[:12],
            "report_id": report_id,
            "verifier_id": verifier_id,
            "status": "initiated",
            "created_at": datetime.utcnow().isoformat(),
            "stub_mode": True,
        }

    def get_verification_status(self, verification_id: str) -> Dict[str, Any]:
        """Get the status of a verification workflow.

        Args:
            verification_id: Verification workflow ID.

        Returns:
            Current verification status.
        """
        if self._workflow is not None:
            fn = getattr(self._workflow, "get_verification_status", None)
            if fn:
                return fn(verification_id)

        return {
            "verification_id": verification_id,
            "status": "pending",
            "checks_completed": 0,
            "checks_total": 0,
            "stub_mode": True,
        }

    @property
    def is_stub(self) -> bool:
        return self._stub_mode


class EmissionCalculatorProxy:
    """Proxy for GL-CBAM-APP EmissionCalculator."""

    def __init__(self, calculator: Any = None) -> None:
        self._calculator = calculator
        self._stub_mode = calculator is None

    def calculate_embedded_emissions(
        self,
        goods_category: str,
        production_route: str,
        quantity_tonnes: float,
        specific_emission_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate embedded emissions for a goods import.

        Uses the specific emission factor if provided, otherwise falls
        back to EU default values per the CBAM Implementing Regulation.

        Args:
            goods_category: CBAM goods category.
            production_route: Production route (e.g., BF-BOF, EAF).
            quantity_tonnes: Quantity in metric tonnes.
            specific_emission_factor: Installation-specific factor if available.

        Returns:
            Emission calculation result.
        """
        if self._calculator is not None:
            fn = getattr(self._calculator, "calculate_embedded_emissions", None)
            if fn:
                return fn(
                    goods_category, production_route,
                    quantity_tonnes, specific_emission_factor,
                )

        # EU default emission factors by category (tCO2/t product)
        default_factors: Dict[str, float] = {
            "IRON_AND_STEEL": 1.85,
            "ALUMINIUM": 8.40,
            "CEMENT": 0.64,
            "FERTILISERS": 2.96,
            "HYDROGEN": 10.00,
            "ELECTRICITY": 0.0,  # Measured in tCO2/MWh, handled differently
        }

        factor = specific_emission_factor
        if factor is None:
            factor = default_factors.get(goods_category.upper(), 1.0)

        embedded = round(quantity_tonnes * factor, 4)

        return {
            "goods_category": goods_category,
            "production_route": production_route,
            "quantity_tonnes": quantity_tonnes,
            "emission_factor_tco2_per_tonne": factor,
            "factor_source": "specific" if specific_emission_factor else "EU_default",
            "embedded_emissions_tco2": embedded,
            "direct_emissions_tco2": round(embedded * 0.85, 4),
            "indirect_emissions_tco2": round(embedded * 0.15, 4),
            "stub_mode": True,
        }

    @property
    def is_stub(self) -> bool:
        return self._stub_mode


# =============================================================================
# Static Data for Stub Mode
# =============================================================================


def _get_stub_cn_codes() -> Dict[str, List[CNCodeEntry]]:
    """Return stub CN codes grouped by category."""
    stub_data: Dict[str, List[CNCodeEntry]] = {
        "IRON_AND_STEEL": [
            CNCodeEntry(cn_code="7201 10 11", description="Non-alloy pig iron", category="IRON_AND_STEEL"),
            CNCodeEntry(cn_code="7208 10 00", description="Flat-rolled products, hot-rolled", category="IRON_AND_STEEL"),
            CNCodeEntry(cn_code="7213 10 00", description="Bars and rods, hot-rolled", category="IRON_AND_STEEL"),
        ],
        "ALUMINIUM": [
            CNCodeEntry(cn_code="7601 10 00", description="Unwrought aluminium, not alloyed", category="ALUMINIUM"),
            CNCodeEntry(cn_code="7604 10 10", description="Aluminium bars and rods", category="ALUMINIUM"),
        ],
        "CEMENT": [
            CNCodeEntry(cn_code="2523 10 00", description="Cement clinkers", category="CEMENT"),
            CNCodeEntry(cn_code="2523 29 00", description="Other Portland cement", category="CEMENT"),
        ],
        "FERTILISERS": [
            CNCodeEntry(cn_code="2814 10 00", description="Anhydrous ammonia", category="FERTILISERS"),
            CNCodeEntry(cn_code="3102 10 10", description="Urea, >45% N", category="FERTILISERS"),
        ],
        "HYDROGEN": [
            CNCodeEntry(cn_code="2804 10 00", description="Hydrogen", category="HYDROGEN"),
        ],
        "ELECTRICITY": [
            CNCodeEntry(cn_code="2716 00 00", description="Electrical energy", category="ELECTRICITY", unit="MWh"),
        ],
    }
    return stub_data


def _get_stub_emission_factors() -> Dict[str, List[EmissionFactorEntry]]:
    """Return stub emission factors grouped by category."""
    return {
        "IRON_AND_STEEL": [
            EmissionFactorEntry(factor_id="EF-STEEL-BF-BOF", goods_category="IRON_AND_STEEL", production_route="BF-BOF", factor_value=2.10, source="EU default"),
            EmissionFactorEntry(factor_id="EF-STEEL-EAF", goods_category="IRON_AND_STEEL", production_route="EAF", factor_value=0.45, source="EU default"),
            EmissionFactorEntry(factor_id="EF-STEEL-DRI-EAF", goods_category="IRON_AND_STEEL", production_route="DRI-EAF", factor_value=1.10, source="EU default"),
        ],
        "ALUMINIUM": [
            EmissionFactorEntry(factor_id="EF-ALU-PRIMARY", goods_category="ALUMINIUM", production_route="primary_smelting", factor_value=8.40, source="EU default"),
            EmissionFactorEntry(factor_id="EF-ALU-SECONDARY", goods_category="ALUMINIUM", production_route="secondary_recycling", factor_value=0.50, source="EU default"),
        ],
        "CEMENT": [
            EmissionFactorEntry(factor_id="EF-CEMENT-CLINKER", goods_category="CEMENT", production_route="clinker", factor_value=0.84, source="EU default"),
            EmissionFactorEntry(factor_id="EF-CEMENT-PORTLAND", goods_category="CEMENT", production_route="portland", factor_value=0.64, source="EU default"),
        ],
        "FERTILISERS": [
            EmissionFactorEntry(factor_id="EF-FERT-AMMONIA", goods_category="FERTILISERS", production_route="ammonia_smr", factor_value=2.96, source="EU default"),
            EmissionFactorEntry(factor_id="EF-FERT-UREA", goods_category="FERTILISERS", production_route="urea", factor_value=3.50, source="EU default"),
            EmissionFactorEntry(factor_id="EF-FERT-NITRIC", goods_category="FERTILISERS", production_route="nitric_acid", factor_value=7.30, source="EU default"),
        ],
        "HYDROGEN": [
            EmissionFactorEntry(factor_id="EF-H2-SMR", goods_category="HYDROGEN", production_route="SMR", factor_value=10.00, source="EU default"),
            EmissionFactorEntry(factor_id="EF-H2-ELECTROLYSIS", goods_category="HYDROGEN", production_route="electrolysis", factor_value=0.00, source="EU default"),
        ],
        "ELECTRICITY": [
            EmissionFactorEntry(factor_id="EF-ELEC-DEFAULT", goods_category="ELECTRICITY", production_route="grid_average", factor_value=0.40, source="EU default"),
        ],
    }


def _get_stub_compliance_rules() -> List[ComplianceRule]:
    """Return the 50+ stub CBAM compliance rules."""
    rules = [
        # --- Registration & Declarant Rules ---
        ComplianceRule(rule_id="CBAM-REG-001", rule_name="Authorized Declarant Registration", description="Importer must be registered as an authorized CBAM declarant", severity="error", category="registration", article_reference="Art. 4"),
        ComplianceRule(rule_id="CBAM-REG-002", rule_name="EORI Number Validity", description="Importer EORI number must be valid and active", severity="error", category="registration", article_reference="Art. 5"),
        ComplianceRule(rule_id="CBAM-REG-003", rule_name="Member State Authorization", description="Declarant must be authorized by competent authority of member state", severity="error", category="registration", article_reference="Art. 5(1)"),
        ComplianceRule(rule_id="CBAM-REG-004", rule_name="Declarant Financial Guarantee", description="Financial guarantee must be provided", severity="error", category="registration", article_reference="Art. 5(4)"),
        ComplianceRule(rule_id="CBAM-REG-005", rule_name="Indirect Representative Authorization", description="Indirect customs representative must be authorized", severity="warning", category="registration", article_reference="Art. 5(3)"),
        # --- CN Code & Goods Rules ---
        ComplianceRule(rule_id="CBAM-CN-001", rule_name="CN Code Annex I Coverage", description="Imported goods CN code must be listed in CBAM Annex I", severity="error", category="cn_codes", article_reference="Art. 2(1)"),
        ComplianceRule(rule_id="CBAM-CN-002", rule_name="CN Code Format Validation", description="CN code must be valid 8-digit format", severity="error", category="cn_codes", article_reference="Annex I"),
        ComplianceRule(rule_id="CBAM-CN-003", rule_name="Goods Category Assignment", description="Each CN code must map to a CBAM goods category", severity="error", category="cn_codes", article_reference="Annex I"),
        ComplianceRule(rule_id="CBAM-CN-004", rule_name="Quantity Unit Consistency", description="Reported quantities must use correct unit for goods category", severity="error", category="cn_codes", article_reference="Annex III"),
        ComplianceRule(rule_id="CBAM-CN-005", rule_name="Origin Country Identification", description="Country of origin must be identified for all CBAM goods", severity="error", category="cn_codes", article_reference="Art. 6(2)"),
        # --- Emission Calculation Rules ---
        ComplianceRule(rule_id="CBAM-EMIT-001", rule_name="Direct Emissions Inclusion", description="Direct embedded emissions must be calculated and reported", severity="error", category="emissions", article_reference="Art. 7(1)"),
        ComplianceRule(rule_id="CBAM-EMIT-002", rule_name="Indirect Emissions Inclusion", description="Indirect embedded emissions must be included where applicable", severity="error", category="emissions", article_reference="Art. 7(3)"),
        ComplianceRule(rule_id="CBAM-EMIT-003", rule_name="Emission Factor Source", description="Emission factors must come from installation-specific data or EU defaults", severity="error", category="emissions", article_reference="Annex IV"),
        ComplianceRule(rule_id="CBAM-EMIT-004", rule_name="Installation Identification", description="Production installation must be identified for embedded emissions", severity="error", category="emissions", article_reference="Art. 7(2)"),
        ComplianceRule(rule_id="CBAM-EMIT-005", rule_name="Default Value Justification", description="Use of EU default values must be justified when actual data unavailable", severity="warning", category="emissions", article_reference="Annex IV"),
        ComplianceRule(rule_id="CBAM-EMIT-006", rule_name="Precursor Emissions Inclusion", description="Emissions from relevant precursors must be included", severity="error", category="emissions", article_reference="Art. 7(4)"),
        ComplianceRule(rule_id="CBAM-EMIT-007", rule_name="Monitoring Methodology", description="Calculation methodology must follow Annex IV rules", severity="error", category="emissions", article_reference="Annex IV"),
        ComplianceRule(rule_id="CBAM-EMIT-008", rule_name="Electricity Emissions Method", description="Electricity emissions must follow specified calculation method", severity="error", category="emissions", article_reference="Art. 7(3)"),
        ComplianceRule(rule_id="CBAM-EMIT-009", rule_name="Carbon Content Verification", description="Carbon content must be verifiable for applicable goods", severity="warning", category="emissions", article_reference="Annex III"),
        ComplianceRule(rule_id="CBAM-EMIT-010", rule_name="Aggregation Rules", description="Emissions must be correctly aggregated per installation and product", severity="error", category="emissions", article_reference="Annex IV"),
        # --- Certificate & Obligation Rules ---
        ComplianceRule(rule_id="CBAM-CERT-001", rule_name="Certificate Purchase Obligation", description="Certificates must be purchased to cover embedded emissions", severity="error", category="certificates", article_reference="Art. 22"),
        ComplianceRule(rule_id="CBAM-CERT-002", rule_name="Certificate Price Basis", description="Certificate price must be based on weekly average auction price", severity="error", category="certificates", article_reference="Art. 21"),
        ComplianceRule(rule_id="CBAM-CERT-003", rule_name="Certificate Surrender Deadline", description="Certificates must be surrendered by May 31 each year", severity="error", category="certificates", article_reference="Art. 22(1)"),
        ComplianceRule(rule_id="CBAM-CERT-004", rule_name="Certificate Account Balance", description="Certificate account must have sufficient balance for surrender", severity="error", category="certificates", article_reference="Art. 22(2)"),
        ComplianceRule(rule_id="CBAM-CERT-005", rule_name="Free Allocation Deduction", description="EU ETS free allocation for equivalent EU products must be deducted", severity="error", category="certificates", article_reference="Art. 31"),
        ComplianceRule(rule_id="CBAM-CERT-006", rule_name="Carbon Price Deduction", description="Carbon price effectively paid in origin country is deductible", severity="warning", category="certificates", article_reference="Art. 9"),
        ComplianceRule(rule_id="CBAM-CERT-007", rule_name="Certificate Validity Period", description="Certificates are valid for the year of purchase + 1 quarter", severity="warning", category="certificates", article_reference="Art. 23"),
        ComplianceRule(rule_id="CBAM-CERT-008", rule_name="Re-purchase Limit", description="Not more than 1/3 of certificates may be re-purchased by authority", severity="info", category="certificates", article_reference="Art. 23(2)"),
        # --- Reporting Rules ---
        ComplianceRule(rule_id="CBAM-RPT-001", rule_name="Quarterly Report Submission", description="CBAM quarterly report must be submitted by end of month following quarter", severity="error", category="reporting", article_reference="Art. 35"),
        ComplianceRule(rule_id="CBAM-RPT-002", rule_name="Annual Declaration Submission", description="Annual CBAM declaration must be submitted by May 31", severity="error", category="reporting", article_reference="Art. 6"),
        ComplianceRule(rule_id="CBAM-RPT-003", rule_name="Report Completeness", description="All mandatory fields must be completed in reports", severity="error", category="reporting", article_reference="Annex V"),
        ComplianceRule(rule_id="CBAM-RPT-004", rule_name="Report Data Consistency", description="Data across quarterly reports and annual declaration must be consistent", severity="error", category="reporting", article_reference="Art. 6(2)"),
        ComplianceRule(rule_id="CBAM-RPT-005", rule_name="Supporting Documentation", description="Reports must include supporting documentation for emission calculations", severity="warning", category="reporting", article_reference="Art. 6(3)"),
        ComplianceRule(rule_id="CBAM-RPT-006", rule_name="Correction Procedure", description="Corrections to submitted reports must follow specified procedure", severity="warning", category="reporting", article_reference="Art. 6(4)"),
        ComplianceRule(rule_id="CBAM-RPT-007", rule_name="Late Submission Penalty", description="Late submission attracts penalties per Art. 27", severity="error", category="reporting", article_reference="Art. 27"),
        # --- Verification Rules ---
        ComplianceRule(rule_id="CBAM-VER-001", rule_name="Accredited Verifier", description="Emissions data must be verified by an accredited verifier", severity="error", category="verification", article_reference="Art. 8"),
        ComplianceRule(rule_id="CBAM-VER-002", rule_name="Verification Completeness", description="Verification must cover all emission categories reported", severity="error", category="verification", article_reference="Art. 8(3)"),
        ComplianceRule(rule_id="CBAM-VER-003", rule_name="Verification Independence", description="Verifier must be independent of the operator and importer", severity="error", category="verification", article_reference="Art. 18"),
        ComplianceRule(rule_id="CBAM-VER-004", rule_name="Verification Report Content", description="Verification report must meet minimum content requirements", severity="error", category="verification", article_reference="Annex VI"),
        ComplianceRule(rule_id="CBAM-VER-005", rule_name="Material Misstatement Threshold", description="Materiality threshold for verification is 5%", severity="warning", category="verification", article_reference="Annex VI"),
        # --- Supplier & Installation Rules ---
        ComplianceRule(rule_id="CBAM-SUP-001", rule_name="Supplier Installation Identifier", description="Each supplier installation must have a unique identifier", severity="error", category="supplier", article_reference="Art. 10"),
        ComplianceRule(rule_id="CBAM-SUP-002", rule_name="Supplier Emission Data Provision", description="Suppliers must provide emission data per installation", severity="error", category="supplier", article_reference="Art. 10(1)"),
        ComplianceRule(rule_id="CBAM-SUP-003", rule_name="Supplier Data Completeness", description="Supplier emission data must cover all applicable emission sources", severity="error", category="supplier", article_reference="Art. 10(2)"),
        ComplianceRule(rule_id="CBAM-SUP-004", rule_name="Supplier Data Timeliness", description="Supplier data must be current (within reporting period)", severity="warning", category="supplier", article_reference="Art. 10(3)"),
        ComplianceRule(rule_id="CBAM-SUP-005", rule_name="Supplier Communication Record", description="Communications with suppliers requesting data must be documented", severity="info", category="supplier", article_reference="Art. 10(4)"),
        # --- De Minimis Rules ---
        ComplianceRule(rule_id="CBAM-MIN-001", rule_name="De Minimis Value Threshold", description="Consignments below EUR 150 value are excluded", severity="info", category="deminimis", article_reference="Art. 2(3)"),
        ComplianceRule(rule_id="CBAM-MIN-002", rule_name="De Minimis Mass Threshold", description="Consignments below 50kg net mass may qualify for exclusion", severity="info", category="deminimis", article_reference="Art. 2(3)"),
        ComplianceRule(rule_id="CBAM-MIN-003", rule_name="Split Consignment Prevention", description="Artificial splitting of consignments to avoid thresholds is prohibited", severity="error", category="deminimis", article_reference="Art. 2(4)"),
        # --- Data Quality Rules ---
        ComplianceRule(rule_id="CBAM-DQ-001", rule_name="Data Accuracy", description="Reported data must be accurate to within 5%", severity="error", category="data_quality", article_reference="Annex V"),
        ComplianceRule(rule_id="CBAM-DQ-002", rule_name="Data Completeness", description="All required data fields must be populated", severity="error", category="data_quality", article_reference="Annex V"),
        ComplianceRule(rule_id="CBAM-DQ-003", rule_name="Data Consistency Check", description="Cross-validation between data sources must be performed", severity="warning", category="data_quality", article_reference="Annex V"),
        ComplianceRule(rule_id="CBAM-DQ-004", rule_name="Audit Trail", description="Complete audit trail must be maintained for all data", severity="error", category="data_quality", article_reference="Art. 12"),
        ComplianceRule(rule_id="CBAM-DQ-005", rule_name="Record Retention", description="Records must be retained for at least 5 years", severity="error", category="data_quality", article_reference="Art. 12(2)"),
    ]
    return rules


def _get_stub_country_carbon_prices() -> Dict[str, CountryCarbonPrice]:
    """Return stub country carbon price data."""
    entries = {
        "GB": CountryCarbonPrice(country_code="GB", country_name="United Kingdom", price_eur_per_tco2=58.5, scheme_name="UK ETS", deduction_eligible=True),
        "CA": CountryCarbonPrice(country_code="CA", country_name="Canada", price_eur_per_tco2=54.4, scheme_name="Federal Carbon Tax", deduction_eligible=True),
        "NZ": CountryCarbonPrice(country_code="NZ", country_name="New Zealand", price_eur_per_tco2=39.2, scheme_name="NZ ETS", deduction_eligible=True),
        "KR": CountryCarbonPrice(country_code="KR", country_name="South Korea", price_eur_per_tco2=17.25, scheme_name="Korea ETS", deduction_eligible=True),
        "CN": CountryCarbonPrice(country_code="CN", country_name="China", price_eur_per_tco2=10.16, scheme_name="National ETS", deduction_eligible=True),
        "JP": CountryCarbonPrice(country_code="JP", country_name="Japan", price_eur_per_tco2=1.76, scheme_name="Carbon Tax", deduction_eligible=True),
        "ZA": CountryCarbonPrice(country_code="ZA", country_name="South Africa", price_eur_per_tco2=9.50, scheme_name="Carbon Tax", deduction_eligible=True),
        "CH": CountryCarbonPrice(country_code="CH", country_name="Switzerland", price_eur_per_tco2=127.2, scheme_name="Swiss ETS (linked)", deduction_eligible=True),
        "SG": CountryCarbonPrice(country_code="SG", country_name="Singapore", price_eur_per_tco2=23.0, scheme_name="Carbon Tax", deduction_eligible=True),
        "UA": CountryCarbonPrice(country_code="UA", country_name="Ukraine", price_eur_per_tco2=1.0, scheme_name="Carbon Tax", deduction_eligible=True),
    }
    return entries


# =============================================================================
# Main Bridge Class
# =============================================================================


class CBAMAppBridge:
    """Primary bridge to GL-CBAM-APP v1.1 for PACK-004.

    Wraps GL-CBAM-APP modules with pack-compatible interfaces. If GL-CBAM-APP
    is not available, the bridge operates in stub mode, providing safe default
    responses so that health checks and demos continue to function.

    Attributes:
        config: Optional configuration dictionary
        logger: Module-level logger
        _stub_mode: Whether running in stub mode
        _app_available: Whether GL-CBAM-APP is on the import path

    Example:
        >>> bridge = CBAMAppBridge()
        >>> calc = bridge.get_emission_calculator()
        >>> result = calc.calculate_embedded_emissions("IRON_AND_STEEL", "BF-BOF", 100.0)
        >>> assert result["embedded_emissions_tco2"] > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CBAM app bridge.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logger
        self._app_available = _CBAM_APP_AVAILABLE
        self._stub_mode = not _CBAM_APP_AVAILABLE
        self._initialized_at = datetime.utcnow()

        # Cache proxy instances
        self._certificate_engine: Optional[CertificateEngineProxy] = None
        self._quarterly_engine: Optional[QuarterlyEngineProxy] = None
        self._supplier_portal: Optional[SupplierPortalProxy] = None
        self._deminimis_engine: Optional[DeMinimisProxy] = None
        self._verification_workflow: Optional[VerificationProxy] = None
        self._emission_calculator: Optional[EmissionCalculatorProxy] = None

        mode = "live" if self._app_available else "stub"
        self.logger.info("CBAMAppBridge initialized in %s mode", mode)

    # -------------------------------------------------------------------------
    # Proxy Accessors
    # -------------------------------------------------------------------------

    def get_certificate_engine(self) -> CertificateEngineProxy:
        """Get the CertificateEngine proxy.

        Returns:
            CertificateEngineProxy wrapping GL-CBAM-APP or stub.
        """
        if self._certificate_engine is None:
            real_engine = None
            if self._app_available:
                engines = _cbam_app_modules.get("engines")
                if engines:
                    real_engine = getattr(engines, "CertificateEngine", None)
                    if callable(real_engine):
                        try:
                            real_engine = real_engine()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate CertificateEngine: %s", exc)
                            real_engine = None
            self._certificate_engine = CertificateEngineProxy(real_engine)
        return self._certificate_engine

    def get_quarterly_engine(self) -> QuarterlyEngineProxy:
        """Get the QuarterlyReportingEngine proxy.

        Returns:
            QuarterlyEngineProxy wrapping GL-CBAM-APP or stub.
        """
        if self._quarterly_engine is None:
            real_engine = None
            if self._app_available:
                engines = _cbam_app_modules.get("engines")
                if engines:
                    real_engine = getattr(engines, "QuarterlyReportingEngine", None)
                    if callable(real_engine):
                        try:
                            real_engine = real_engine()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate QuarterlyReportingEngine: %s", exc)
                            real_engine = None
            self._quarterly_engine = QuarterlyEngineProxy(real_engine)
        return self._quarterly_engine

    def get_supplier_portal(self) -> SupplierPortalProxy:
        """Get the SupplierPortal proxy.

        Returns:
            SupplierPortalProxy wrapping GL-CBAM-APP or stub.
        """
        if self._supplier_portal is None:
            real_portal = None
            if self._app_available:
                workflows = _cbam_app_modules.get("workflows")
                if workflows:
                    real_portal = getattr(workflows, "SupplierPortal", None)
                    if callable(real_portal):
                        try:
                            real_portal = real_portal()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate SupplierPortal: %s", exc)
                            real_portal = None
            self._supplier_portal = SupplierPortalProxy(real_portal)
        return self._supplier_portal

    def get_deminimis_engine(self) -> DeMinimisProxy:
        """Get the DeMinimisEngine proxy.

        Returns:
            DeMinimisProxy wrapping GL-CBAM-APP or stub.
        """
        if self._deminimis_engine is None:
            real_engine = None
            if self._app_available:
                engines = _cbam_app_modules.get("engines")
                if engines:
                    real_engine = getattr(engines, "DeMinimisEngine", None)
                    if callable(real_engine):
                        try:
                            real_engine = real_engine()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate DeMinimisEngine: %s", exc)
                            real_engine = None
            self._deminimis_engine = DeMinimisProxy(real_engine)
        return self._deminimis_engine

    def get_verification_workflow(self) -> VerificationProxy:
        """Get the VerificationWorkflow proxy.

        Returns:
            VerificationProxy wrapping GL-CBAM-APP or stub.
        """
        if self._verification_workflow is None:
            real_workflow = None
            if self._app_available:
                workflows = _cbam_app_modules.get("workflows")
                if workflows:
                    real_workflow = getattr(workflows, "VerificationWorkflow", None)
                    if callable(real_workflow):
                        try:
                            real_workflow = real_workflow()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate VerificationWorkflow: %s", exc)
                            real_workflow = None
            self._verification_workflow = VerificationProxy(real_workflow)
        return self._verification_workflow

    def get_emission_calculator(self) -> EmissionCalculatorProxy:
        """Get the EmissionCalculator proxy.

        Returns:
            EmissionCalculatorProxy wrapping GL-CBAM-APP or stub.
        """
        if self._emission_calculator is None:
            real_calc = None
            if self._app_available:
                engines = _cbam_app_modules.get("engines")
                if engines:
                    real_calc = getattr(engines, "EmissionCalculator", None)
                    if callable(real_calc):
                        try:
                            real_calc = real_calc()
                        except Exception as exc:
                            self.logger.warning("Failed to instantiate EmissionCalculator: %s", exc)
                            real_calc = None
            self._emission_calculator = EmissionCalculatorProxy(real_calc)
        return self._emission_calculator

    # -------------------------------------------------------------------------
    # Data Accessors
    # -------------------------------------------------------------------------

    def get_cn_codes(self) -> Dict[str, List[CNCodeEntry]]:
        """Get the full CN code database from GL-CBAM-APP.

        Returns:
            Dictionary mapping category to list of CNCodeEntry.
        """
        if self._app_available:
            engines = _cbam_app_modules.get("engines")
            if engines:
                fn = getattr(engines, "get_cn_codes", None)
                if fn:
                    try:
                        return fn()
                    except Exception as exc:
                        self.logger.warning("get_cn_codes failed: %s", exc)

        return _get_stub_cn_codes()

    def get_emission_factors(self) -> Dict[str, List[EmissionFactorEntry]]:
        """Get all emission factors from GL-CBAM-APP.

        Returns:
            Dictionary mapping category to list of EmissionFactorEntry.
        """
        if self._app_available:
            engines = _cbam_app_modules.get("engines")
            if engines:
                fn = getattr(engines, "get_emission_factors", None)
                if fn:
                    try:
                        return fn()
                    except Exception as exc:
                        self.logger.warning("get_emission_factors failed: %s", exc)

        return _get_stub_emission_factors()

    def get_country_carbon_pricing(self) -> Dict[str, CountryCarbonPrice]:
        """Get country carbon price data from GL-CBAM-APP.

        Returns:
            Dictionary mapping country code to CountryCarbonPrice.
        """
        if self._app_available:
            engines = _cbam_app_modules.get("engines")
            if engines:
                fn = getattr(engines, "get_country_carbon_pricing", None)
                if fn:
                    try:
                        return fn()
                    except Exception as exc:
                        self.logger.warning("get_country_carbon_pricing failed: %s", exc)

        return _get_stub_country_carbon_prices()

    def get_cbam_rules(self) -> List[ComplianceRule]:
        """Get the 50+ CBAM compliance rules from GL-CBAM-APP.

        Returns:
            List of ComplianceRule.
        """
        if self._app_available:
            engines = _cbam_app_modules.get("engines")
            if engines:
                fn = getattr(engines, "get_cbam_rules", None)
                if fn:
                    try:
                        return fn()
                    except Exception as exc:
                        self.logger.warning("get_cbam_rules failed: %s", exc)

        return _get_stub_compliance_rules()

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_app_health(self) -> AppHealthStatus:
        """Verify that GL-CBAM-APP v1.1 is available and operational.

        Returns:
            AppHealthStatus with availability, loaded modules, and version.
        """
        modules_loaded: List[str] = []
        version = "unknown"
        details: Dict[str, Any] = {}

        if self._app_available:
            for name, mod in _cbam_app_modules.items():
                if mod is not None:
                    modules_loaded.append(name)
            # Try to read version
            engines = _cbam_app_modules.get("engines")
            if engines:
                version = getattr(engines, "__version__", getattr(engines, "VERSION", "1.1.0"))
                details["engines_available"] = True
            workflows = _cbam_app_modules.get("workflows")
            if workflows:
                details["workflows_available"] = True
        else:
            details["reason"] = "GL-CBAM-APP not on import path"

        return AppHealthStatus(
            app_available=self._app_available,
            stub_mode=self._stub_mode,
            modules_loaded=modules_loaded,
            version=str(version),
            details=details,
        )


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
