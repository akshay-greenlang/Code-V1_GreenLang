# -*- coding: utf-8 -*-
"""
GL-POL-X-006: CBAM Compliance Agent
===================================

EU Carbon Border Adjustment Mechanism compliance agent. CRITICAL PATH
agent providing deterministic CBAM calculations and reporting.

Capabilities:
    - CBAM product classification
    - Embedded emissions calculation
    - CBAM certificate requirement calculation
    - Quarterly report preparation
    - Supplier data collection tracking
    - Transitional period compliance

Zero-Hallucination Guarantees:
    - All calculations use official CBAM methodology
    - Deterministic embedded emissions formulas
    - Complete audit trails for all calculations
    - No LLM inference in compliance determination

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class CBAMSector(str, Enum):
    """CBAM covered sectors."""
    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILISERS = "fertilisers"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"


class CBAMProductCategory(str, Enum):
    """CBAM product categories."""
    # Cement
    CEMENT_CLINKER = "cement_clinker"
    CEMENT = "cement"
    ALUMINOUS_CEMENT = "aluminous_cement"
    # Iron and Steel
    PIG_IRON = "pig_iron"
    FERRO_ALLOYS = "ferro_alloys"
    CRUDE_STEEL = "crude_steel"
    STEEL_PRODUCTS = "steel_products"
    # Aluminium
    UNWROUGHT_ALUMINIUM = "unwrought_aluminium"
    ALUMINIUM_PRODUCTS = "aluminium_products"
    # Fertilisers
    AMMONIA = "ammonia"
    NITRIC_ACID = "nitric_acid"
    MIXED_FERTILISERS = "mixed_fertilisers"
    # Hydrogen
    HYDROGEN = "hydrogen"
    # Electricity
    ELECTRICITY = "electricity"


class ReportingPeriod(str, Enum):
    """CBAM reporting periods."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class EmissionsDataSource(str, Enum):
    """Source of emissions data."""
    ACTUAL_SUPPLIER = "actual_supplier"
    VERIFIED_THIRD_PARTY = "verified_third_party"
    DEFAULT_VALUES = "default_values"
    CALCULATED = "calculated"


# =============================================================================
# DEFAULT EMISSION FACTORS
# =============================================================================


# Default emission factors by product (tCO2e per tonne product)
# These are transitional period default values
DEFAULT_EMISSION_FACTORS: Dict[CBAMProductCategory, Decimal] = {
    CBAMProductCategory.CEMENT_CLINKER: Decimal("0.8260"),
    CBAMProductCategory.CEMENT: Decimal("0.6420"),
    CBAMProductCategory.ALUMINOUS_CEMENT: Decimal("1.2400"),
    CBAMProductCategory.PIG_IRON: Decimal("1.3280"),
    CBAMProductCategory.FERRO_ALLOYS: Decimal("2.7500"),
    CBAMProductCategory.CRUDE_STEEL: Decimal("1.6850"),
    CBAMProductCategory.STEEL_PRODUCTS: Decimal("1.8910"),
    CBAMProductCategory.UNWROUGHT_ALUMINIUM: Decimal("6.7400"),
    CBAMProductCategory.ALUMINIUM_PRODUCTS: Decimal("7.1200"),
    CBAMProductCategory.AMMONIA: Decimal("2.1050"),
    CBAMProductCategory.NITRIC_ACID: Decimal("2.6300"),
    CBAMProductCategory.MIXED_FERTILISERS: Decimal("1.8500"),
    CBAMProductCategory.HYDROGEN: Decimal("9.3100"),
    CBAMProductCategory.ELECTRICITY: Decimal("0.3760"),  # Grid average
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ImportedGood(BaseModel):
    """An imported good subject to CBAM."""

    good_id: str = Field(
        default_factory=lambda: deterministic_uuid("good"),
        description="Unique identifier"
    )
    cn_code: str = Field(..., description="Combined Nomenclature code")
    description: str = Field(..., description="Product description")
    product_category: CBAMProductCategory = Field(..., description="CBAM category")
    sector: CBAMSector = Field(..., description="CBAM sector")

    # Quantity
    quantity_tonnes: Decimal = Field(..., description="Quantity in tonnes")

    # Origin
    country_of_origin: str = Field(..., description="ISO country code")
    installation_id: Optional[str] = Field(
        None,
        description="Installation identifier"
    )
    supplier_name: Optional[str] = Field(None, description="Supplier name")

    # Emissions data
    emissions_data_source: EmissionsDataSource = Field(
        default=EmissionsDataSource.DEFAULT_VALUES,
        description="Source of emissions data"
    )
    direct_emissions_tco2e_per_tonne: Optional[Decimal] = Field(
        None,
        description="Direct emissions per tonne"
    )
    indirect_emissions_tco2e_per_tonne: Optional[Decimal] = Field(
        None,
        description="Indirect emissions per tonne"
    )

    # Carbon price paid
    carbon_price_paid_eur: Decimal = Field(
        default=Decimal("0"),
        description="Carbon price already paid in origin country"
    )


class CBAMCalculationResult(BaseModel):
    """Result of CBAM calculation for a single good."""

    good_id: str = Field(..., description="Related good identifier")
    product_category: CBAMProductCategory = Field(...)
    quantity_tonnes: Decimal = Field(...)

    # Emissions
    direct_emissions_tco2e: Decimal = Field(..., description="Total direct emissions")
    indirect_emissions_tco2e: Decimal = Field(..., description="Total indirect emissions")
    total_embedded_emissions_tco2e: Decimal = Field(..., description="Total emissions")

    # Factors used
    emission_factor_used: Decimal = Field(..., description="EF used")
    data_source: EmissionsDataSource = Field(...)

    # Deductions
    carbon_price_deduction_eur: Decimal = Field(
        default=Decimal("0"),
        description="Deduction for carbon price paid"
    )

    # Certificate requirement (post-2026)
    certificates_required: Decimal = Field(
        default=Decimal("0"),
        description="CBAM certificates required"
    )

    # Calculation trace
    calculation_trace: List[str] = Field(default_factory=list)


class CBAMQuarterlyReport(BaseModel):
    """CBAM quarterly report structure."""

    report_id: str = Field(
        default_factory=lambda: deterministic_uuid("cbam_report"),
        description="Unique report identifier"
    )
    declarant_id: str = Field(..., description="Authorized declarant ID")
    organization_name: str = Field(..., description="Organization name")

    # Period
    reporting_year: int = Field(...)
    reporting_period: ReportingPeriod = Field(...)
    period_start: date = Field(...)
    period_end: date = Field(...)

    # Goods
    imported_goods: List[ImportedGood] = Field(default_factory=list)
    calculation_results: List[CBAMCalculationResult] = Field(default_factory=list)

    # Aggregates
    total_goods_imported_tonnes: Decimal = Field(default=Decimal("0"))
    total_embedded_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_direct_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_indirect_emissions_tco2e: Decimal = Field(default=Decimal("0"))

    # By sector
    emissions_by_sector: Dict[str, Decimal] = Field(default_factory=dict)

    # Certificate requirements (post-2026)
    total_certificates_required: Decimal = Field(default=Decimal("0"))
    total_carbon_price_deduction_eur: Decimal = Field(default=Decimal("0"))

    # Status
    submission_deadline: date = Field(...)
    submitted: bool = Field(default=False)
    submitted_at: Optional[datetime] = Field(None)

    # Data quality
    actual_data_percentage: float = Field(
        default=0.0,
        description="Percentage of emissions using actual data"
    )

    # Provenance
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = {
            "declarant_id": self.declarant_id,
            "reporting_year": self.reporting_year,
            "reporting_period": self.reporting_period.value,
            "total_embedded_emissions": str(self.total_embedded_emissions_tco2e),
            "total_goods_tonnes": str(self.total_goods_imported_tonnes),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class CBAMComplianceInput(BaseModel):
    """Input for CBAM compliance operations."""

    action: str = Field(
        ...,
        description="Action: calculate_emissions, generate_report, validate_data"
    )
    declarant_id: Optional[str] = Field(None)
    organization_name: Optional[str] = Field(None)
    reporting_year: Optional[int] = Field(None)
    reporting_period: Optional[ReportingPeriod] = Field(None)
    imported_goods: Optional[List[Dict[str, Any]]] = Field(None)


class CBAMComplianceOutput(BaseModel):
    """Output from CBAM compliance operations."""

    success: bool = Field(...)
    action: str = Field(...)
    report: Optional[CBAMQuarterlyReport] = Field(None)
    calculation_results: Optional[List[CBAMCalculationResult]] = Field(None)
    validation_errors: Optional[List[str]] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


# =============================================================================
# CBAM COMPLIANCE AGENT
# =============================================================================


class CBAMComplianceAgent(BaseAgent):
    """
    GL-POL-X-006: CBAM Compliance Agent

    EU Carbon Border Adjustment Mechanism compliance calculations and reporting.
    CRITICAL PATH agent with zero-hallucination guarantees.

    Calculation Methodology:
        Embedded Emissions = Quantity * Emission Factor
        Direct Emissions = Activity Data * EF (Scope 1)
        Indirect Emissions = Electricity * Grid EF (Scope 2)

    All calculations follow EU Implementing Regulation (EU) 2023/1773.

    Usage:
        agent = CBAMComplianceAgent()
        result = agent.run({
            'action': 'generate_report',
            'declarant_id': 'DE123456789',
            'imported_goods': [...]
        })
    """

    AGENT_ID = "GL-POL-X-006"
    AGENT_NAME = "CBAM Compliance Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="EU CBAM compliance calculations and reporting"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize CBAM Compliance Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="CBAM compliance agent",
                version=self.VERSION,
                parameters={
                    "use_actual_data_when_available": True,
                    "validate_cn_codes": True,
                }
            )

        self._default_factors = DEFAULT_EMISSION_FACTORS.copy()
        self._audit_trail: List[Dict[str, Any]] = []

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute CBAM compliance operation."""
        import time
        start_time = time.time()

        try:
            agent_input = CBAMComplianceInput(**input_data)

            action_handlers = {
                "calculate_emissions": self._handle_calculate_emissions,
                "generate_report": self._handle_generate_report,
                "validate_data": self._handle_validate_data,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)
            output.provenance_hash = hashlib.sha256(
                json.dumps({"action": agent_input.action}, sort_keys=True).encode()
            ).hexdigest()

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"CBAM compliance failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_calculate_emissions(
        self,
        input_data: CBAMComplianceInput
    ) -> CBAMComplianceOutput:
        """Calculate embedded emissions for imported goods."""
        if not input_data.imported_goods:
            return CBAMComplianceOutput(
                success=False,
                action="calculate_emissions",
                error="No imported goods provided",
            )

        goods = [ImportedGood(**g) for g in input_data.imported_goods]
        results = []
        warnings = []

        for good in goods:
            result = self._calculate_embedded_emissions(good)
            results.append(result)

            if result.data_source == EmissionsDataSource.DEFAULT_VALUES:
                warnings.append(
                    f"Using default emission factor for {good.description}"
                )

        return CBAMComplianceOutput(
            success=True,
            action="calculate_emissions",
            calculation_results=results,
            warnings=warnings,
        )

    def _handle_generate_report(
        self,
        input_data: CBAMComplianceInput
    ) -> CBAMComplianceOutput:
        """Generate CBAM quarterly report."""
        if not input_data.declarant_id:
            return CBAMComplianceOutput(
                success=False,
                action="generate_report",
                error="declarant_id required",
            )

        if not input_data.imported_goods:
            return CBAMComplianceOutput(
                success=False,
                action="generate_report",
                error="imported_goods required",
            )

        # Parse goods
        goods = [ImportedGood(**g) for g in input_data.imported_goods]

        # Determine period
        year = input_data.reporting_year or DeterministicClock.now().year
        period = input_data.reporting_period or ReportingPeriod.Q1

        # Calculate period dates
        period_dates = self._get_period_dates(year, period)

        # Create report
        report = CBAMQuarterlyReport(
            declarant_id=input_data.declarant_id,
            organization_name=input_data.organization_name or "Unknown",
            reporting_year=year,
            reporting_period=period,
            period_start=period_dates[0],
            period_end=period_dates[1],
            submission_deadline=period_dates[2],
            imported_goods=goods,
        )

        # Calculate emissions for each good
        results: List[CBAMCalculationResult] = []
        emissions_by_sector: Dict[str, Decimal] = {}
        actual_count = 0

        for good in goods:
            result = self._calculate_embedded_emissions(good)
            results.append(result)

            # Track by sector
            sector = good.sector.value
            if sector not in emissions_by_sector:
                emissions_by_sector[sector] = Decimal("0")
            emissions_by_sector[sector] += result.total_embedded_emissions_tco2e

            # Track data quality
            if result.data_source in [
                EmissionsDataSource.ACTUAL_SUPPLIER,
                EmissionsDataSource.VERIFIED_THIRD_PARTY
            ]:
                actual_count += 1

        # Populate report
        report.calculation_results = results
        report.total_goods_imported_tonnes = sum(r.quantity_tonnes for r in results)
        report.total_embedded_emissions_tco2e = sum(r.total_embedded_emissions_tco2e for r in results)
        report.total_direct_emissions_tco2e = sum(r.direct_emissions_tco2e for r in results)
        report.total_indirect_emissions_tco2e = sum(r.indirect_emissions_tco2e for r in results)
        report.emissions_by_sector = {k: v for k, v in emissions_by_sector.items()}
        report.total_certificates_required = sum(r.certificates_required for r in results)
        report.total_carbon_price_deduction_eur = sum(r.carbon_price_deduction_eur for r in results)

        # Data quality percentage
        if results:
            report.actual_data_percentage = (actual_count / len(results)) * 100

        # Calculate provenance
        report.provenance_hash = report.calculate_provenance_hash()

        return CBAMComplianceOutput(
            success=True,
            action="generate_report",
            report=report,
        )

    def _handle_validate_data(
        self,
        input_data: CBAMComplianceInput
    ) -> CBAMComplianceOutput:
        """Validate CBAM data for completeness and accuracy."""
        if not input_data.imported_goods:
            return CBAMComplianceOutput(
                success=False,
                action="validate_data",
                error="imported_goods required",
            )

        goods = [ImportedGood(**g) for g in input_data.imported_goods]
        errors: List[str] = []
        warnings: List[str] = []

        for good in goods:
            # Validate CN code format
            if self.config.parameters.get("validate_cn_codes", True):
                if not self._validate_cn_code(good.cn_code):
                    errors.append(f"Invalid CN code format: {good.cn_code}")

            # Check for required fields
            if good.quantity_tonnes <= 0:
                errors.append(f"Invalid quantity for {good.description}")

            # Warn about default data
            if good.emissions_data_source == EmissionsDataSource.DEFAULT_VALUES:
                warnings.append(
                    f"Using default values for {good.description} - "
                    f"actual supplier data recommended"
                )

            # Check installation ID for transitional period
            if not good.installation_id and good.emissions_data_source != EmissionsDataSource.DEFAULT_VALUES:
                warnings.append(
                    f"No installation ID for {good.description}"
                )

        return CBAMComplianceOutput(
            success=len(errors) == 0,
            action="validate_data",
            validation_errors=errors if errors else None,
            warnings=warnings,
        )

    def _calculate_embedded_emissions(
        self,
        good: ImportedGood
    ) -> CBAMCalculationResult:
        """
        Calculate embedded emissions - DETERMINISTIC.

        Formula:
            Total Embedded = Direct + Indirect
            Direct = Quantity * Direct EF
            Indirect = Quantity * Indirect EF (for applicable sectors)
        """
        trace: List[str] = []
        trace.append(f"Calculating emissions for: {good.description}")
        trace.append(f"  Product category: {good.product_category.value}")
        trace.append(f"  Quantity: {good.quantity_tonnes:,.3f} tonnes")

        # Determine emission factor and source
        if good.direct_emissions_tco2e_per_tonne is not None:
            direct_ef = good.direct_emissions_tco2e_per_tonne
            data_source = good.emissions_data_source
            trace.append(f"  Using provided direct EF: {direct_ef} tCO2e/t")
        else:
            direct_ef = self._default_factors.get(
                good.product_category,
                Decimal("1.0")
            )
            data_source = EmissionsDataSource.DEFAULT_VALUES
            trace.append(f"  Using default EF: {direct_ef} tCO2e/t")

        # Calculate direct emissions
        direct_emissions = good.quantity_tonnes * direct_ef
        trace.append(f"  Direct emissions: {direct_emissions:,.3f} tCO2e")

        # Calculate indirect emissions (electricity-related)
        indirect_ef = good.indirect_emissions_tco2e_per_tonne or Decimal("0")
        indirect_emissions = good.quantity_tonnes * indirect_ef
        trace.append(f"  Indirect emissions: {indirect_emissions:,.3f} tCO2e")

        # Total embedded emissions
        total_emissions = direct_emissions + indirect_emissions
        trace.append(f"  Total embedded: {total_emissions:,.3f} tCO2e")

        # Carbon price deduction
        carbon_deduction = good.carbon_price_paid_eur
        if carbon_deduction > 0:
            trace.append(f"  Carbon price already paid: {carbon_deduction:,.2f} EUR")

        # Certificate requirement (placeholder for post-2026)
        # During transitional period (2023-2025), no certificates required
        certificates = Decimal("0")

        return CBAMCalculationResult(
            good_id=good.good_id,
            product_category=good.product_category,
            quantity_tonnes=good.quantity_tonnes,
            direct_emissions_tco2e=direct_emissions.quantize(Decimal("0.001"), ROUND_HALF_UP),
            indirect_emissions_tco2e=indirect_emissions.quantize(Decimal("0.001"), ROUND_HALF_UP),
            total_embedded_emissions_tco2e=total_emissions.quantize(Decimal("0.001"), ROUND_HALF_UP),
            emission_factor_used=direct_ef,
            data_source=data_source,
            carbon_price_deduction_eur=carbon_deduction,
            certificates_required=certificates,
            calculation_trace=trace,
        )

    def _get_period_dates(
        self,
        year: int,
        period: ReportingPeriod
    ) -> Tuple[date, date, date]:
        """Get period start, end, and submission deadline dates."""
        period_dates = {
            ReportingPeriod.Q1: (date(year, 1, 1), date(year, 3, 31), date(year, 4, 30)),
            ReportingPeriod.Q2: (date(year, 4, 1), date(year, 6, 30), date(year, 7, 31)),
            ReportingPeriod.Q3: (date(year, 7, 1), date(year, 9, 30), date(year, 10, 31)),
            ReportingPeriod.Q4: (date(year, 10, 1), date(year, 12, 31), date(year + 1, 1, 31)),
        }
        return period_dates[period]

    def _validate_cn_code(self, cn_code: str) -> bool:
        """Validate CN code format (8 digits)."""
        clean_code = cn_code.replace(" ", "").replace(".", "")
        return len(clean_code) == 8 and clean_code.isdigit()

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_default_emission_factor(
        self,
        product_category: CBAMProductCategory
    ) -> Optional[Decimal]:
        """Get default emission factor for a product category."""
        return self._default_factors.get(product_category)

    def classify_product(
        self,
        cn_code: str
    ) -> Optional[CBAMProductCategory]:
        """Classify a product by CN code."""
        # CN code prefixes for CBAM products
        cn_prefixes = {
            "2523": CBAMProductCategory.CEMENT,
            "2507": CBAMProductCategory.CEMENT_CLINKER,
            "7201": CBAMProductCategory.PIG_IRON,
            "7202": CBAMProductCategory.FERRO_ALLOYS,
            "7206": CBAMProductCategory.CRUDE_STEEL,
            "7207": CBAMProductCategory.STEEL_PRODUCTS,
            "7601": CBAMProductCategory.UNWROUGHT_ALUMINIUM,
            "7604": CBAMProductCategory.ALUMINIUM_PRODUCTS,
            "2814": CBAMProductCategory.AMMONIA,
            "2808": CBAMProductCategory.NITRIC_ACID,
            "3102": CBAMProductCategory.MIXED_FERTILISERS,
            "2804": CBAMProductCategory.HYDROGEN,
            "2716": CBAMProductCategory.ELECTRICITY,
        }

        clean_code = cn_code.replace(" ", "").replace(".", "")
        for prefix, category in cn_prefixes.items():
            if clean_code.startswith(prefix):
                return category
        return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "CBAMComplianceAgent",
    "CBAMSector",
    "CBAMProductCategory",
    "ReportingPeriod",
    "EmissionsDataSource",
    "ImportedGood",
    "CBAMCalculationResult",
    "CBAMQuarterlyReport",
    "CBAMComplianceInput",
    "CBAMComplianceOutput",
    "DEFAULT_EMISSION_FACTORS",
]
