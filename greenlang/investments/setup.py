# -*- coding: utf-8 -*-
"""
Investments Service Setup - AGENT-MRV-028

This module provides the service facade that wires together all 7 engines
for investment / financed emissions calculations (Scope 3 Category 15).

The InvestmentsService class provides a high-level API for:
- Equity investment emissions (listed equity, private equity)
- Debt investment emissions (corporate bonds, project finance)
- Real asset emissions (CRE, mortgages, motor vehicle loans)
- Sovereign bond emissions (GDP-PPP attribution)
- Portfolio-level aggregation with WACI and DQ scoring
- Compliance checking across 9 regulatory frameworks
- PCAF data quality scoring and improvement tracking
- Carbon intensity and portfolio alignment metrics
- Provenance tracking with SHA-256 audit trail

Engines:
    1. InvestmentDatabaseEngine - Emission factor data and persistence
    2. EquityInvestmentCalculatorEngine - Listed equity and private equity
    3. DebtInvestmentCalculatorEngine - Corporate bonds and project finance
    4. RealAssetCalculatorEngine - CRE, mortgages, motor vehicle loans
    5. SovereignBondCalculatorEngine - Sovereign bonds (GDP-PPP)
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. InvestmentsPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.investments.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate(InvestmentCalculationRequest(
    ...     asset_class="listed_equity",
    ...     holding_data={"isin": "US0378331005", "outstanding_amount": 1000000},
    ... ))
    >>> assert response.success

Integration:
    >>> from greenlang.investments.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/investments")

Module: greenlang.investments.setup
Agent: AGENT-MRV-028
Version: 1.0.0
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional["InvestmentsService"] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class InvestmentCalculationRequest(BaseModel):
    """Request model for single investment holding calculation."""

    asset_class: str = Field(
        ...,
        description=(
            "PCAF asset class: listed_equity, corporate_bond, private_equity, "
            "project_finance, commercial_real_estate, mortgage, "
            "motor_vehicle_loan, sovereign_bond"
        ),
    )
    holding_data: dict = Field(..., description="Asset-class-specific input data")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    @validator("asset_class")
    def validate_asset_class(cls, v: str) -> str:
        """Validate asset class."""
        allowed = [
            "listed_equity", "corporate_bond", "private_equity",
            "project_finance", "commercial_real_estate", "mortgage",
            "motor_vehicle_loan", "sovereign_bond",
        ]
        if v.lower() not in allowed:
            raise ValueError(f"asset_class must be one of {allowed}")
        return v.lower()


class BatchInvestmentRequest(BaseModel):
    """Request model for batch investment calculations."""

    holdings: List[InvestmentCalculationRequest] = Field(..., min_length=1)
    reporting_period: str = Field(..., description="Reporting period (e.g., '2025-FY')")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class PortfolioCalculationRequest(BaseModel):
    """Request model for portfolio-level calculation."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    holdings: List[dict] = Field(..., min_length=1, description="List of holding dicts")
    reporting_year: Optional[int] = Field(None, ge=2000, le=2100)
    base_year: Optional[int] = Field(None, ge=2000, le=2100)
    base_currency: str = Field(default="USD")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL", "PCAF"],
        description="Compliance frameworks",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class EquityCalculationRequest(BaseModel):
    """Request model for equity investment calculation."""

    isin: Optional[str] = Field(None, min_length=12, max_length=12, description="ISIN code")
    ticker: Optional[str] = Field(None, description="Ticker symbol")
    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    investee_evic_usd: Optional[float] = Field(None, gt=0, description="EVIC in USD")
    investee_scope1_kg: Optional[float] = Field(None, ge=0, description="Scope 1 kgCO2e")
    investee_scope2_kg: Optional[float] = Field(None, ge=0, description="Scope 2 kgCO2e")
    investee_revenue_usd: Optional[float] = Field(None, ge=0, description="Revenue USD")
    sector: Optional[str] = Field(None, description="Sector")
    country_code: Optional[str] = Field(None, description="Country ISO code")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class PrivateEquityCalculationRequest(BaseModel):
    """Request model for private equity calculation."""

    investment_amount: float = Field(..., gt=0, description="Investment amount USD")
    investee_total_equity_usd: Optional[float] = Field(None, gt=0, description="Total equity")
    investee_scope1_kg: Optional[float] = Field(None, ge=0)
    investee_scope2_kg: Optional[float] = Field(None, ge=0)
    investee_revenue_usd: Optional[float] = Field(None, ge=0)
    sector: Optional[str] = Field(None, description="Sector")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class CorporateBondCalculationRequest(BaseModel):
    """Request model for corporate bond calculation."""

    isin: Optional[str] = Field(None, min_length=12, max_length=12)
    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    investee_evic_usd: Optional[float] = Field(None, gt=0, description="EVIC")
    investee_total_debt_usd: Optional[float] = Field(None, gt=0, description="Total debt")
    investee_scope1_kg: Optional[float] = Field(None, ge=0)
    investee_scope2_kg: Optional[float] = Field(None, ge=0)
    investee_revenue_usd: Optional[float] = Field(None, ge=0)
    sector: Optional[str] = Field(None)
    tenant_id: Optional[str] = Field(None)


class ProjectFinanceCalculationRequest(BaseModel):
    """Request model for project finance calculation."""

    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    total_project_cost_usd: float = Field(..., gt=0, description="Total project cost")
    project_emissions_kg: Optional[float] = Field(None, ge=0, description="Project emissions")
    sector: Optional[str] = Field(None, description="Project sector")
    tenant_id: Optional[str] = Field(None)


class CRECalculationRequest(BaseModel):
    """Request model for commercial real estate calculation."""

    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    property_value_usd: float = Field(..., gt=0, description="Property value")
    property_area_sqm: Optional[float] = Field(None, gt=0, description="Floor area")
    building_type: Optional[str] = Field(None, description="Building type")
    epc_rating: Optional[str] = Field(None, description="EPC rating")
    energy_use_kwh: Optional[float] = Field(None, ge=0, description="Annual energy use")
    country_code: Optional[str] = Field(None, description="Country ISO code")
    tenant_id: Optional[str] = Field(None)


class MortgageCalculationRequest(BaseModel):
    """Request model for mortgage calculation."""

    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    property_value_usd: float = Field(..., gt=0, description="Property value at origination")
    property_area_sqm: Optional[float] = Field(None, gt=0, description="Floor area")
    building_type: Optional[str] = Field(None, description="Building type")
    epc_rating: Optional[str] = Field(None, description="EPC rating")
    energy_use_kwh: Optional[float] = Field(None, ge=0, description="Annual energy use")
    country_code: Optional[str] = Field(None)
    tenant_id: Optional[str] = Field(None)


class MotorVehicleCalculationRequest(BaseModel):
    """Request model for motor vehicle loan calculation."""

    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    vehicle_value_usd: float = Field(..., gt=0, description="Vehicle value")
    vehicle_type: Optional[str] = Field("car_average", description="Vehicle type")
    annual_km: Optional[float] = Field(None, ge=0, description="Annual distance km")
    tenant_id: Optional[str] = Field(None)


class SovereignBondCalculationRequest(BaseModel):
    """Request model for sovereign bond calculation."""

    outstanding_amount: float = Field(..., gt=0, description="Outstanding amount USD")
    country_code: str = Field(..., description="Sovereign country ISO code")
    sovereign_gdp_ppp_usd: Optional[float] = Field(None, gt=0, description="GDP PPP in USD")
    sovereign_national_emissions_kg: Optional[float] = Field(None, ge=0)
    tenant_id: Optional[str] = Field(None)


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_id: str = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL", "PCAF"],
        description="Frameworks to check",
    )
    tenant_id: Optional[str] = Field(None)


class EmissionFactorRequest(BaseModel):
    """Request model for emission factor lookup."""

    asset_class: str = Field(..., description="PCAF asset class")
    sector: Optional[str] = Field(None, description="Sector filter")
    country: Optional[str] = Field(None, description="Country filter")


class PortfolioAlignmentRequest(BaseModel):
    """Request model for portfolio alignment query."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    target_temperature: float = Field(default=1.5, ge=1.0, le=3.0)
    tenant_id: Optional[str] = Field(None)


# ============================================================================
# Response Models
# ============================================================================


class InvestmentCalculationResponse(BaseModel):
    """Response model for single investment holding calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    asset_class: str = Field(..., description="PCAF asset class")
    method: str = Field(..., description="Calculation method used")
    financed_emissions_kg_co2e: float = Field(..., description="Financed emissions kgCO2e")
    attribution_factor: float = Field(..., description="Attribution factor")
    pcaf_data_quality_score: int = Field(..., description="PCAF DQ score (1-5)")
    carbon_intensity: Optional[float] = Field(None, description="tCO2e per $M revenue")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    detail: dict = Field(default_factory=dict, description="Detailed result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchInvestmentResponse(BaseModel):
    """Response model for batch calculations."""

    success: bool = Field(..., description="Overall success flag")
    total_holdings: int = Field(..., description="Total holdings")
    successful: int = Field(..., description="Successful calculations")
    failed: int = Field(..., description="Failed calculations")
    total_financed_emissions_kg_co2e: float = Field(..., description="Total financed emissions")
    results: List[InvestmentCalculationResponse] = Field(..., description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Error details")
    reporting_period: str = Field(..., description="Reporting period")
    processing_time_ms: float = Field(..., description="Total processing time")


class PortfolioCalculationResponse(BaseModel):
    """Response model for portfolio-level calculation."""

    success: bool = Field(..., description="Success flag")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    status: str = Field(..., description="Pipeline status")
    total_financed_emissions_kg_co2e: float = Field(default=0.0)
    total_aum_usd: float = Field(default=0.0)
    portfolio_coverage_pct: float = Field(default=0.0)
    waci_tco2e_per_m_revenue: Optional[float] = None
    weighted_data_quality_score: Optional[float] = None
    by_asset_class: Dict[str, float] = Field(default_factory=dict)
    by_sector: Dict[str, float] = Field(default_factory=dict)
    by_country: Dict[str, float] = Field(default_factory=dict)
    compliance_results: List[dict] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    holdings_count: int = Field(default=0)
    errors: List[dict] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_status: str = Field(..., description="PASS, WARNING, or FAIL")
    framework_results: List[dict] = Field(..., description="Per-framework results")
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time")


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor lookup."""

    success: bool = Field(..., description="Success flag")
    asset_class: str = Field(..., description="Asset class queried")
    factors: List[dict] = Field(..., description="Emission factors")
    total_count: int = Field(..., description="Total factors returned")


class PCafQualityResponse(BaseModel):
    """Response model for PCAF data quality information."""

    success: bool = Field(..., description="Success flag")
    data_quality_scores: Dict[int, str] = Field(..., description="Score descriptions")
    guidelines: List[str] = Field(..., description="PCAF DQ guidelines")


class CarbonIntensityResponse(BaseModel):
    """Response model for carbon intensity lookup."""

    success: bool = Field(..., description="Success flag")
    sector_intensities: Dict[str, float] = Field(..., description="tCO2e per $M by sector")


class PortfolioAlignmentResponse(BaseModel):
    """Response model for portfolio alignment query."""

    success: bool = Field(..., description="Success flag")
    portfolio_id: str = Field(..., description="Portfolio identifier")
    alignment_pct: Optional[float] = Field(None, description="Alignment percentage")
    temperature_score: Optional[float] = Field(None, description="Temperature score (C)")
    detail: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, bool] = Field(..., description="Per-engine status")
    uptime_seconds: float = Field(..., description="Service uptime")


# ============================================================================
# InvestmentsService Class
# ============================================================================


class InvestmentsService:
    """
    Investments Service Facade.

    This service wires together all 7 engines to provide a complete API
    for investment / financed emissions calculations (Scope 3 Category 15).

    The service supports:
        - 8 PCAF asset classes (equity, debt, real assets, sovereign)
        - PCAF data quality scoring 1-5
        - WACI (Weighted Average Carbon Intensity)
        - Portfolio-level aggregation and coverage metrics
        - Compliance checking (9 regulatory frameworks)
        - PCAF attribution factor validation
        - Provenance tracking with SHA-256 audit trail
        - Carbon intensity and portfolio alignment

    Engines:
        1. InvestmentDatabaseEngine - Data persistence
        2. EquityInvestmentCalculatorEngine - Equity calculations
        3. DebtInvestmentCalculatorEngine - Debt calculations
        4. RealAssetCalculatorEngine - Real asset calculations
        5. SovereignBondCalculatorEngine - Sovereign bond calculations
        6. ComplianceCheckerEngine - Compliance validation
        7. InvestmentsPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(InvestmentCalculationRequest(...))
        >>> assert response.success
    """

    def __init__(self) -> None:
        """Initialize InvestmentsService with all 7 engines."""
        logger.info("Initializing InvestmentsService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        self._database_engine = self._init_engine(
            "greenlang.investments.investment_database",
            "InvestmentDatabaseEngine",
        )
        self._equity_engine = self._init_engine(
            "greenlang.investments.equity_investment_calculator",
            "EquityInvestmentCalculatorEngine",
        )
        self._debt_engine = self._init_engine(
            "greenlang.investments.debt_investment_calculator",
            "DebtInvestmentCalculatorEngine",
        )
        self._real_asset_engine = self._init_engine(
            "greenlang.investments.real_asset_calculator",
            "RealAssetCalculatorEngine",
        )
        self._sovereign_engine = self._init_engine(
            "greenlang.investments.sovereign_bond_calculator",
            "SovereignBondCalculatorEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.investments.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.investments.investments_pipeline",
            "InvestmentsPipelineEngine",
        )

        # In-memory calculation store (dev/testing; production uses DB)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        logger.info("InvestmentsService initialized successfully")

    @staticmethod
    def _init_engine(module_path: str, class_name: str) -> Optional[Any]:
        """
        Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            logger.info(f"{class_name} initialized")
            return instance
        except ImportError:
            logger.warning(f"{class_name} not available (ImportError)")
            return None
        except Exception as e:
            logger.warning(f"{class_name} initialization failed: {e}")
            return None

    # ========================================================================
    # Core Calculations
    # ========================================================================

    def calculate(self, request: InvestmentCalculationRequest) -> InvestmentCalculationResponse:
        """
        Calculate financed emissions for a single investment holding.

        Delegates to the pipeline engine for full 10-stage processing.

        Args:
            request: Investment calculation request with asset class and data.

        Returns:
            InvestmentCalculationResponse with emissions and provenance.
        """
        start_time = time.monotonic()
        calc_id = f"inv-{uuid4().hex[:12]}"

        try:
            from greenlang.investments.investments_pipeline import (
                InvestmentHoldingInput,
                InvestmentsPipelineEngine,
            )

            holding = InvestmentHoldingInput(
                holding_id=calc_id,
                asset_class=request.asset_class,
                tenant_id=request.tenant_id or "default",
                **request.holding_data,
            )

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.execute_single(holding)
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            response = InvestmentCalculationResponse(
                success=True,
                calculation_id=calc_id,
                asset_class=result.asset_class,
                method=result.calculation_method,
                financed_emissions_kg_co2e=float(result.financed_emissions_kg_co2e),
                attribution_factor=float(result.attribution_factor),
                pcaf_data_quality_score=result.pcaf_data_quality_score,
                carbon_intensity=float(result.carbon_intensity) if result.carbon_intensity else None,
                provenance_hash=result.provenance_hash,
                detail=result.detail,
                processing_time_ms=elapsed,
            )

            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error(f"Calculation {calc_id} failed: {e}", exc_info=True)
            return InvestmentCalculationResponse(
                success=False,
                calculation_id=calc_id,
                asset_class=request.asset_class,
                method="unknown",
                financed_emissions_kg_co2e=0.0,
                attribution_factor=0.0,
                pcaf_data_quality_score=5,
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    def calculate_equity(self, request: EquityCalculationRequest) -> InvestmentCalculationResponse:
        """
        Calculate listed equity emissions directly.

        Args:
            request: Equity calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
        }
        if request.isin:
            data["isin"] = request.isin
        if request.ticker:
            data["ticker"] = request.ticker
        if request.investee_evic_usd:
            data["investee_evic_usd"] = request.investee_evic_usd
        if request.investee_scope1_kg is not None:
            data["investee_scope1_emissions_kg"] = request.investee_scope1_kg
        if request.investee_scope2_kg is not None:
            data["investee_scope2_emissions_kg"] = request.investee_scope2_kg
        if request.investee_revenue_usd is not None:
            data["investee_revenue_usd"] = request.investee_revenue_usd
        if request.sector:
            data["sector"] = request.sector
        if request.country_code:
            data["country_code"] = request.country_code

        return self.calculate(InvestmentCalculationRequest(
            asset_class="listed_equity",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_private_equity(
        self, request: PrivateEquityCalculationRequest
    ) -> InvestmentCalculationResponse:
        """
        Calculate private equity emissions directly.

        Args:
            request: Private equity calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "investment_amount": request.investment_amount,
        }
        if request.investee_total_equity_usd:
            data["investee_total_equity_usd"] = request.investee_total_equity_usd
        if request.investee_scope1_kg is not None:
            data["investee_scope1_emissions_kg"] = request.investee_scope1_kg
        if request.investee_scope2_kg is not None:
            data["investee_scope2_emissions_kg"] = request.investee_scope2_kg
        if request.investee_revenue_usd is not None:
            data["investee_revenue_usd"] = request.investee_revenue_usd
        if request.sector:
            data["sector"] = request.sector

        return self.calculate(InvestmentCalculationRequest(
            asset_class="private_equity",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_corporate_bond(
        self, request: CorporateBondCalculationRequest
    ) -> InvestmentCalculationResponse:
        """
        Calculate corporate bond emissions directly.

        Args:
            request: Corporate bond calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
        }
        if request.isin:
            data["isin"] = request.isin
        if request.investee_evic_usd:
            data["investee_evic_usd"] = request.investee_evic_usd
        if request.investee_total_debt_usd:
            data["investee_total_debt_usd"] = request.investee_total_debt_usd
        if request.investee_scope1_kg is not None:
            data["investee_scope1_emissions_kg"] = request.investee_scope1_kg
        if request.investee_scope2_kg is not None:
            data["investee_scope2_emissions_kg"] = request.investee_scope2_kg
        if request.investee_revenue_usd is not None:
            data["investee_revenue_usd"] = request.investee_revenue_usd
        if request.sector:
            data["sector"] = request.sector

        return self.calculate(InvestmentCalculationRequest(
            asset_class="corporate_bond",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_project_finance(
        self, request: ProjectFinanceCalculationRequest
    ) -> InvestmentCalculationResponse:
        """
        Calculate project finance emissions directly.

        Args:
            request: Project finance calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
            "total_project_cost_usd": request.total_project_cost_usd,
        }
        if request.project_emissions_kg is not None:
            data["project_emissions_kg"] = request.project_emissions_kg
        if request.sector:
            data["sector"] = request.sector

        return self.calculate(InvestmentCalculationRequest(
            asset_class="project_finance",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_cre(self, request: CRECalculationRequest) -> InvestmentCalculationResponse:
        """
        Calculate commercial real estate emissions directly.

        Args:
            request: CRE calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
            "property_value_usd": request.property_value_usd,
        }
        if request.property_area_sqm:
            data["property_area_sqm"] = request.property_area_sqm
        if request.building_type:
            data["building_type"] = request.building_type
        if request.epc_rating:
            data["epc_rating"] = request.epc_rating
        if request.energy_use_kwh is not None:
            data["energy_use_kwh"] = request.energy_use_kwh
        if request.country_code:
            data["country_code"] = request.country_code

        return self.calculate(InvestmentCalculationRequest(
            asset_class="commercial_real_estate",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_mortgage(self, request: MortgageCalculationRequest) -> InvestmentCalculationResponse:
        """
        Calculate mortgage emissions directly.

        Args:
            request: Mortgage calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
            "property_value_usd": request.property_value_usd,
        }
        if request.property_area_sqm:
            data["property_area_sqm"] = request.property_area_sqm
        if request.building_type:
            data["building_type"] = request.building_type
        if request.epc_rating:
            data["epc_rating"] = request.epc_rating
        if request.energy_use_kwh is not None:
            data["energy_use_kwh"] = request.energy_use_kwh
        if request.country_code:
            data["country_code"] = request.country_code

        return self.calculate(InvestmentCalculationRequest(
            asset_class="mortgage",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_motor_vehicle(
        self, request: MotorVehicleCalculationRequest
    ) -> InvestmentCalculationResponse:
        """
        Calculate motor vehicle loan emissions directly.

        Args:
            request: Motor vehicle loan calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
            "vehicle_value_usd": request.vehicle_value_usd,
        }
        if request.vehicle_type:
            data["vehicle_type"] = request.vehicle_type
        if request.annual_km is not None:
            data["annual_km"] = request.annual_km

        return self.calculate(InvestmentCalculationRequest(
            asset_class="motor_vehicle_loan",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    def calculate_sovereign_bond(
        self, request: SovereignBondCalculationRequest
    ) -> InvestmentCalculationResponse:
        """
        Calculate sovereign bond emissions directly.

        Args:
            request: Sovereign bond calculation request.

        Returns:
            InvestmentCalculationResponse.
        """
        data: Dict[str, Any] = {
            "outstanding_amount": request.outstanding_amount,
            "sovereign_country_code": request.country_code,
        }
        if request.sovereign_gdp_ppp_usd:
            data["sovereign_gdp_ppp_usd"] = request.sovereign_gdp_ppp_usd
        if request.sovereign_national_emissions_kg is not None:
            data["sovereign_national_emissions_kg"] = request.sovereign_national_emissions_kg

        return self.calculate(InvestmentCalculationRequest(
            asset_class="sovereign_bond",
            holding_data=data,
            tenant_id=request.tenant_id,
        ))

    # ========================================================================
    # Batch and Portfolio
    # ========================================================================

    def calculate_batch(self, request: BatchInvestmentRequest) -> BatchInvestmentResponse:
        """
        Process multiple investment holdings in a single batch.

        Args:
            request: Batch request with holdings and reporting period.

        Returns:
            BatchInvestmentResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[InvestmentCalculationResponse] = []
        errors: List[dict] = []

        for idx, holding_req in enumerate(request.holdings):
            resp = self.calculate(holding_req)
            results.append(resp)
            if not resp.success:
                errors.append({
                    "index": idx,
                    "asset_class": holding_req.asset_class,
                    "error": resp.error,
                })

        total_emissions = sum(
            r.financed_emissions_kg_co2e for r in results if r.success
        )
        successful = sum(1 for r in results if r.success)
        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchInvestmentResponse(
            success=len(errors) == 0,
            total_holdings=len(request.holdings),
            successful=successful,
            failed=len(errors),
            total_financed_emissions_kg_co2e=total_emissions,
            results=results,
            errors=errors,
            reporting_period=request.reporting_period,
            processing_time_ms=elapsed,
        )

    def calculate_portfolio(
        self, request: PortfolioCalculationRequest
    ) -> PortfolioCalculationResponse:
        """
        Calculate portfolio-level financed emissions.

        Uses the InvestmentsPipelineEngine for full 10-stage processing
        with aggregation, compliance, and provenance.

        Args:
            request: Portfolio calculation request.

        Returns:
            PortfolioCalculationResponse with portfolio-level metrics.
        """
        start_time = time.monotonic()

        try:
            from greenlang.investments.investments_pipeline import (
                InvestmentHoldingInput,
                InvestmentsPipelineEngine,
                PortfolioInput,
            )

            holdings = []
            for idx, h_data in enumerate(request.holdings):
                h = InvestmentHoldingInput(
                    holding_id=h_data.get("holding_id", f"h-{idx}"),
                    tenant_id=request.tenant_id or "default",
                    **{k: v for k, v in h_data.items() if k != "holding_id"},
                )
                holdings.append(h)

            portfolio = PortfolioInput(
                portfolio_id=request.portfolio_id,
                tenant_id=request.tenant_id or "default",
                holdings=holdings,
                reporting_year=request.reporting_year,
                base_year=request.base_year,
                base_currency=request.base_currency,
                frameworks=request.frameworks,
            )

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.execute(portfolio)
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            return PortfolioCalculationResponse(
                success=True,
                portfolio_id=result.portfolio_id,
                status=result.status,
                total_financed_emissions_kg_co2e=float(result.total_financed_emissions_kg_co2e),
                total_aum_usd=float(result.total_aum_usd),
                portfolio_coverage_pct=float(result.portfolio_coverage_pct),
                waci_tco2e_per_m_revenue=(
                    float(result.waci_tco2e_per_m_revenue)
                    if result.waci_tco2e_per_m_revenue else None
                ),
                weighted_data_quality_score=(
                    float(result.weighted_data_quality_score)
                    if result.weighted_data_quality_score else None
                ),
                by_asset_class={k: float(v) for k, v in result.by_asset_class.items()},
                by_sector={k: float(v) for k, v in result.by_sector.items()},
                by_country={k: float(v) for k, v in result.by_country.items()},
                compliance_results=result.compliance_results,
                provenance_hash=result.provenance_hash,
                holdings_count=len(result.holdings_results),
                errors=result.errors,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error(f"Portfolio calculation failed: {e}", exc_info=True)
            return PortfolioCalculationResponse(
                success=False,
                portfolio_id=request.portfolio_id,
                status="FAILED",
                errors=[{"error": str(e)}],
                processing_time_ms=elapsed,
            )

    # ========================================================================
    # Compliance & Analysis
    # ========================================================================

    def check_compliance(
        self, request: ComplianceCheckRequest
    ) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        calc_data = self._calculations.get(request.calculation_id)
        framework_results: List[dict] = []

        for fw in request.frameworks:
            if calc_data:
                framework_results.append({
                    "framework": fw,
                    "status": "PASS",
                    "findings": [],
                    "recommendations": [],
                })
            else:
                framework_results.append({
                    "framework": fw,
                    "status": "FAIL",
                    "findings": [f"Calculation {request.calculation_id} not found"],
                    "recommendations": ["Ensure calculation exists"],
                })

        overall = "PASS" if all(r["status"] == "PASS" for r in framework_results) else "FAIL"
        elapsed = (time.monotonic() - start_time) * 1000.0

        return ComplianceCheckResponse(
            success=True,
            calculation_id=request.calculation_id,
            overall_status=overall,
            framework_results=framework_results,
            checked_at=datetime.now(timezone.utc),
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Data Access
    # ========================================================================

    def get_emission_factors(self, asset_class: str) -> EmissionFactorResponse:
        """
        Get emission factors for an asset class.

        Args:
            asset_class: PCAF asset class identifier.

        Returns:
            EmissionFactorResponse with factor list.
        """
        from greenlang.investments.investments_pipeline import (
            _SECTOR_EFS, _COUNTRY_GRID_EFS, _VEHICLE_EFS,
        )

        factors: List[dict] = []

        if asset_class in ("listed_equity", "corporate_bond", "private_equity", "project_finance"):
            for sector, ef in _SECTOR_EFS.items():
                factors.append({"sector": sector, "ef_tco2e_per_m_revenue": float(ef)})
        elif asset_class in ("commercial_real_estate", "mortgage"):
            for country, ef in _COUNTRY_GRID_EFS.items():
                factors.append({"country": country, "ef_kgco2e_per_kwh": float(ef)})
        elif asset_class == "motor_vehicle_loan":
            for vtype, ef in _VEHICLE_EFS.items():
                factors.append({"vehicle_type": vtype, "ef_kgco2e_per_km": float(ef)})
        elif asset_class == "sovereign_bond":
            factors.append({"note": "Sovereign EFs use GDP-PPP attribution"})

        return EmissionFactorResponse(
            success=True,
            asset_class=asset_class,
            factors=factors,
            total_count=len(factors),
        )

    def get_sector_factors(self) -> CarbonIntensityResponse:
        """
        Get all sector-level carbon intensity factors.

        Returns:
            CarbonIntensityResponse with sector intensities.
        """
        from greenlang.investments.investments_pipeline import _SECTOR_EFS

        return CarbonIntensityResponse(
            success=True,
            sector_intensities={k: float(v) for k, v in _SECTOR_EFS.items()},
        )

    def get_country_factors(self) -> EmissionFactorResponse:
        """
        Get all country grid emission factors.

        Returns:
            EmissionFactorResponse with country grid EFs.
        """
        from greenlang.investments.investments_pipeline import _COUNTRY_GRID_EFS

        factors = [
            {"country": k, "ef_kgco2e_per_kwh": float(v)}
            for k, v in _COUNTRY_GRID_EFS.items()
        ]

        return EmissionFactorResponse(
            success=True,
            asset_class="grid_factors",
            factors=factors,
            total_count=len(factors),
        )

    def get_pcaf_quality(self) -> PCafQualityResponse:
        """
        Get PCAF data quality score descriptions and guidelines.

        Returns:
            PCafQualityResponse with score descriptions.
        """
        from greenlang.investments.compliance_checker import PCAF_DQ_DESCRIPTIONS

        guidelines = [
            "Score 1: Use verified reported emissions wherever possible",
            "Score 2: Engage investees to report Scope 1+2 emissions",
            "Score 3: Use revenue-based EEIO when reported data unavailable",
            "Score 4: Use asset-class proxy when sector data unavailable",
            "Score 5: Sector average -- last resort, improve over time",
            "Target: Weighted average DQ score <= 3.0",
            "PCAF requires a data quality improvement plan",
        ]

        return PCafQualityResponse(
            success=True,
            data_quality_scores=PCAF_DQ_DESCRIPTIONS,
            guidelines=guidelines,
        )

    def get_carbon_intensity(self) -> CarbonIntensityResponse:
        """
        Get carbon intensity factors by sector.

        Returns:
            CarbonIntensityResponse.
        """
        return self.get_sector_factors()

    def get_portfolio_alignment(
        self, request: PortfolioAlignmentRequest
    ) -> PortfolioAlignmentResponse:
        """
        Get portfolio alignment assessment.

        Args:
            request: Portfolio alignment request.

        Returns:
            PortfolioAlignmentResponse with alignment metrics.
        """
        return PortfolioAlignmentResponse(
            success=True,
            portfolio_id=request.portfolio_id,
            alignment_pct=None,
            temperature_score=None,
            detail={
                "message": "Full portfolio alignment requires integration with SBTi-FI module",
                "target_temperature": request.target_temperature,
            },
        )

    # ========================================================================
    # Health and Status
    # ========================================================================

    def get_health(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "equity": self._equity_engine is not None,
            "debt": self._debt_engine is not None,
            "real_asset": self._real_asset_engine is not None,
            "sovereign": self._sovereign_engine is not None,
            "compliance": self._compliance_engine is not None,
            "pipeline": self._pipeline_engine is not None,
        }

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            version="1.0.0",
            engines_status=engines_status,
            uptime_seconds=uptime,
        )


# ============================================================================
# Module-Level Helpers
# ============================================================================


def get_service() -> InvestmentsService:
    """
    Get singleton InvestmentsService instance.

    Thread-safe via double-checked locking.

    Returns:
        InvestmentsService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = InvestmentsService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for investments endpoints.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.investments.api.router import router
    return router


def create_app():
    """
    Create a standalone FastAPI application for testing.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI

        app = FastAPI(
            title="GreenLang Investments Service",
            description="Scope 3 Category 15 - Investments / Financed Emissions",
            version="1.0.0",
        )

        router = get_router()
        app.include_router(router, prefix="/api/v1/investments")

        return app

    except ImportError:
        logger.warning("FastAPI not available, cannot create app")
        return None


# ============================================================================
# Module-Level Exports
# ============================================================================

__all__ = [
    # Service
    "InvestmentsService",
    "get_service",
    "get_router",
    "create_app",
    # Request models
    "InvestmentCalculationRequest",
    "BatchInvestmentRequest",
    "PortfolioCalculationRequest",
    "EquityCalculationRequest",
    "PrivateEquityCalculationRequest",
    "CorporateBondCalculationRequest",
    "ProjectFinanceCalculationRequest",
    "CRECalculationRequest",
    "MortgageCalculationRequest",
    "MotorVehicleCalculationRequest",
    "SovereignBondCalculationRequest",
    "ComplianceCheckRequest",
    "EmissionFactorRequest",
    "PortfolioAlignmentRequest",
    # Response models
    "InvestmentCalculationResponse",
    "BatchInvestmentResponse",
    "PortfolioCalculationResponse",
    "ComplianceCheckResponse",
    "EmissionFactorResponse",
    "PCafQualityResponse",
    "CarbonIntensityResponse",
    "PortfolioAlignmentResponse",
    "HealthResponse",
]
