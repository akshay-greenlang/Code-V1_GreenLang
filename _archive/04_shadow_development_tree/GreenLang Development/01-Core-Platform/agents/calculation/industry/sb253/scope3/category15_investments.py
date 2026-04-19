# -*- coding: utf-8 -*-
"""
Category 15: Investments Calculator

Calculates emissions from the reporting organization's investments,
including equity investments, debt investments, and project finance.

This category is particularly relevant for financial institutions
and applies PCAF (Partnership for Carbon Accounting Financials)
methodology.

Supported Methods:
1. Equity share approach (proportional to ownership)
2. PCAF methodology (financed emissions)
3. Average data method (sector-based estimates)

Reference:
- GHG Protocol Scope 3 Standard, Chapter 6
- PCAF Global GHG Accounting Standard for Financial Industry

Example:
    >>> calculator = Category15InvestmentsCalculator()
    >>> input_data = InvestmentsInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     equity_investments=[
    ...         EquityInvestment(investee_name="Company A", equity_share_pct=25,
    ...                         investee_emissions_mt=100000),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .base import (
    Scope3CategoryCalculator,
    Scope3CalculationInput,
    Scope3CalculationResult,
    CalculationMethod,
    CalculationStep,
    EmissionFactorRecord,
    EmissionFactorSource,
    DataQualityTier,
)

logger = logging.getLogger(__name__)


class EquityInvestment(BaseModel):
    """Equity investment data."""

    investee_name: str = Field(..., description="Investee company name")
    equity_share_pct: Decimal = Field(
        ..., ge=0, le=100, description="Equity ownership percentage"
    )

    # Investee emissions (if known)
    investee_emissions_mt: Optional[Decimal] = Field(
        None, ge=0, description="Investee total Scope 1+2 emissions (MT CO2e)"
    )
    investee_scope3_mt: Optional[Decimal] = Field(
        None, ge=0, description="Investee Scope 3 emissions (MT CO2e)"
    )

    # Investment value (for PCAF)
    investment_value_usd: Optional[Decimal] = Field(
        None, ge=0, description="Investment value (USD)"
    )
    investee_evic_usd: Optional[Decimal] = Field(
        None, ge=0, description="Enterprise Value Including Cash (USD)"
    )

    # Sector data (for estimation)
    sector: Optional[str] = Field(None, description="Investee sector")
    sector_naics: Optional[str] = Field(None, description="NAICS code")


class DebtInvestment(BaseModel):
    """Debt/loan investment data."""

    borrower_name: str = Field(..., description="Borrower name")
    loan_amount_usd: Decimal = Field(..., ge=0, description="Outstanding loan (USD)")

    # Borrower data
    borrower_emissions_mt: Optional[Decimal] = Field(
        None, ge=0, description="Borrower Scope 1+2 emissions (MT CO2e)"
    )
    borrower_revenue_usd: Optional[Decimal] = Field(
        None, ge=0, description="Borrower annual revenue (USD)"
    )
    borrower_total_debt_usd: Optional[Decimal] = Field(
        None, ge=0, description="Borrower total debt (USD)"
    )

    # Sector data
    sector: Optional[str] = Field(None, description="Borrower sector")
    asset_class: Optional[str] = Field(None, description="Asset class")


class ProjectFinance(BaseModel):
    """Project finance investment."""

    project_name: str = Field(..., description="Project name")
    investment_usd: Decimal = Field(..., ge=0, description="Investment amount (USD)")
    project_total_cost_usd: Decimal = Field(..., ge=0, description="Total project cost (USD)")

    # Project emissions
    project_annual_emissions_mt: Optional[Decimal] = Field(
        None, ge=0, description="Annual project emissions (MT CO2e)"
    )
    project_lifetime_years: int = Field(20, ge=1, description="Project lifetime (years)")

    # Project type
    project_type: Optional[str] = Field(None, description="Project type")


class InvestmentsInput(Scope3CalculationInput):
    """Input model for Category 15: Investments."""

    # Investment data
    equity_investments: List[EquityInvestment] = Field(
        default_factory=list, description="Equity investments"
    )
    debt_investments: List[DebtInvestment] = Field(
        default_factory=list, description="Debt investments"
    )
    project_finance: List[ProjectFinance] = Field(
        default_factory=list, description="Project finance investments"
    )

    # Aggregated inputs
    total_aum_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total assets under management (USD)"
    )

    # PCAF data quality target
    pcaf_data_quality_target: int = Field(
        3, ge=1, le=5, description="Target PCAF data quality score (1-5)"
    )


# Sector emission intensities (MT CO2e per million USD revenue)
# Source: PCAF guidance, sector averages
SECTOR_INTENSITIES: Dict[str, Decimal] = {
    "oil_gas": Decimal("450"),
    "utilities": Decimal("380"),
    "materials": Decimal("320"),
    "industrials": Decimal("85"),
    "energy": Decimal("420"),
    "mining": Decimal("280"),
    "chemicals": Decimal("180"),
    "construction": Decimal("65"),
    "transportation": Decimal("120"),
    "manufacturing": Decimal("95"),
    "agriculture": Decimal("150"),
    "real_estate": Decimal("45"),
    "consumer_goods": Decimal("55"),
    "healthcare": Decimal("25"),
    "technology": Decimal("15"),
    "financials": Decimal("8"),
    "services": Decimal("12"),
    "default": Decimal("50"),
}

# PCAF data quality scores (1=best, 5=worst)
PCAF_DATA_QUALITY: Dict[str, int] = {
    "verified_emissions": 1,  # Audited reported emissions
    "reported_emissions": 2,  # Unverified reported emissions
    "estimated_physical": 3,  # Estimated from physical activity
    "estimated_economic": 4,  # Estimated from economic activity
    "sector_average": 5,  # Sector average data
}


class Category15InvestmentsCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 15: Investments.

    Calculates financed emissions using PCAF methodology.
    """

    CATEGORY_NUMBER = 15
    CATEGORY_NAME = "Investments"
    SUPPORTED_METHODS = [
        CalculationMethod.EQUITY_SHARE,
        CalculationMethod.PCAF,
        CalculationMethod.AVERAGE_DATA,
    ]

    def __init__(self):
        """Initialize the Category 15 calculator."""
        super().__init__()
        self._sector_intensities = SECTOR_INTENSITIES

    def calculate(self, input_data: InvestmentsInput) -> Scope3CalculationResult:
        """Calculate Category 15 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        equity_emissions = Decimal("0")
        debt_emissions = Decimal("0")
        project_emissions = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize investments calculation",
            inputs={
                "num_equity": len(input_data.equity_investments),
                "num_debt": len(input_data.debt_investments),
                "num_projects": len(input_data.project_finance),
            },
        ))

        # Calculate equity investment emissions
        for investment in input_data.equity_investments:
            inv_emissions = self._calculate_equity_emissions(
                investment, input_data.calculation_method
            )
            equity_emissions += inv_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description=f"Calculate equity emissions: {investment.investee_name}",
                formula="financed_emissions = equity_share x investee_emissions",
                inputs={
                    "investee": investment.investee_name,
                    "equity_share_pct": str(investment.equity_share_pct),
                    "investee_emissions_mt": str(investment.investee_emissions_mt or "estimated"),
                },
                output=str(inv_emissions),
            ))

        # Calculate debt investment emissions
        for debt in input_data.debt_investments:
            debt_inv_emissions = self._calculate_debt_emissions(debt)
            debt_emissions += debt_inv_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description=f"Calculate debt emissions: {debt.borrower_name}",
                formula="financed_emissions = (loan / total_debt) x borrower_emissions",
                inputs={
                    "borrower": debt.borrower_name,
                    "loan_usd": str(debt.loan_amount_usd),
                },
                output=str(debt_inv_emissions),
            ))

        # Calculate project finance emissions
        for project in input_data.project_finance:
            proj_emissions = self._calculate_project_emissions(project)
            project_emissions += proj_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description=f"Calculate project emissions: {project.project_name}",
                formula="financed_emissions = (investment / total_cost) x project_emissions",
                inputs={
                    "project": project.project_name,
                    "investment_usd": str(project.investment_usd),
                    "total_cost_usd": str(project.project_total_cost_usd),
                },
                output=str(proj_emissions),
            ))

        # Convert MT to kg for consistency
        total_emissions_kg = (
            (equity_emissions + debt_emissions + project_emissions) * 1000
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all financed emissions",
            inputs={
                "equity_emissions_mt": str(equity_emissions),
                "debt_emissions_mt": str(debt_emissions),
                "project_emissions_mt": str(project_emissions),
            },
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="investments_pcaf",
            factor_value=Decimal("50"),  # Default sector intensity
            factor_unit="MT CO2e/million USD",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://carbonaccountingfinancials.com/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_3,
        )

        activity_data = {
            "num_equity_investments": len(input_data.equity_investments),
            "num_debt_investments": len(input_data.debt_investments),
            "num_project_finance": len(input_data.project_finance),
            "equity_emissions_mt": str(equity_emissions),
            "debt_emissions_mt": str(debt_emissions),
            "project_emissions_mt": str(project_emissions),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=input_data.calculation_method,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_equity_emissions(
        self,
        investment: EquityInvestment,
        method: CalculationMethod,
    ) -> Decimal:
        """
        Calculate financed emissions for equity investment.

        PCAF Formula:
        Financed Emissions = (Outstanding amount / EVIC) x Company Emissions

        Equity Share Formula:
        Financed Emissions = Equity Share % x Company Emissions
        """
        # Use investee emissions if available
        if investment.investee_emissions_mt:
            investee_emissions = investment.investee_emissions_mt
        elif investment.sector:
            # Estimate from sector intensity
            intensity = self._sector_intensities.get(
                investment.sector.lower().replace(" ", "_"),
                self._sector_intensities["default"]
            )
            # Assume $1B revenue as default
            investee_emissions = intensity * Decimal("1000")
        else:
            # Default estimate
            investee_emissions = Decimal("10000")

        if method == CalculationMethod.PCAF:
            # PCAF approach uses investment value / EVIC
            if investment.investment_value_usd and investment.investee_evic_usd:
                attribution_factor = (
                    investment.investment_value_usd / investment.investee_evic_usd
                )
            else:
                # Fall back to equity share
                attribution_factor = investment.equity_share_pct / 100
        else:
            # Equity share approach
            attribution_factor = investment.equity_share_pct / 100

        financed_emissions = (investee_emissions * attribution_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Add Scope 3 if included
        if investment.investee_scope3_mt:
            scope3_financed = (
                investment.investee_scope3_mt * attribution_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            financed_emissions += scope3_financed

        return financed_emissions

    def _calculate_debt_emissions(self, debt: DebtInvestment) -> Decimal:
        """
        Calculate financed emissions for debt investment.

        PCAF Formula:
        Financed Emissions = (Loan / Total Debt) x Borrower Emissions
        """
        # Get borrower emissions
        if debt.borrower_emissions_mt:
            borrower_emissions = debt.borrower_emissions_mt
        elif debt.borrower_revenue_usd and debt.sector:
            intensity = self._sector_intensities.get(
                debt.sector.lower().replace(" ", "_"),
                self._sector_intensities["default"]
            )
            borrower_emissions = (
                debt.borrower_revenue_usd / Decimal("1000000") * intensity
            )
        else:
            # Default estimate based on loan size
            borrower_emissions = debt.loan_amount_usd / Decimal("100000")

        # Calculate attribution factor
        if debt.borrower_total_debt_usd:
            attribution_factor = debt.loan_amount_usd / debt.borrower_total_debt_usd
        else:
            # Assume loan is 10% of total debt
            attribution_factor = Decimal("0.10")

        financed_emissions = (borrower_emissions * attribution_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        return financed_emissions

    def _calculate_project_emissions(self, project: ProjectFinance) -> Decimal:
        """
        Calculate financed emissions for project finance.

        PCAF Formula:
        Financed Emissions = (Investment / Total Project Cost) x Annual Project Emissions
        """
        # Get project emissions
        if project.project_annual_emissions_mt:
            annual_emissions = project.project_annual_emissions_mt
        else:
            # Estimate based on project type
            project_type = (project.project_type or "industrial").lower()
            if "renewable" in project_type or "solar" in project_type or "wind" in project_type:
                annual_emissions = Decimal("100")  # Low carbon
            elif "power" in project_type or "energy" in project_type:
                annual_emissions = Decimal("50000")  # High carbon
            else:
                annual_emissions = Decimal("5000")  # Default

        # Attribution factor based on funding share
        attribution_factor = project.investment_usd / project.project_total_cost_usd

        # Calculate lifetime financed emissions (reported annually)
        annual_financed = (annual_emissions * attribution_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        return annual_financed
